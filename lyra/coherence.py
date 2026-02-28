"""
coherence.py — Coherence-Guided Sampling (v0.2)

During generation, compares the model's internal directional pull
(what the layers are doing) to the output logits (what the model
is about to say). When they align, the model speaks with clarity.
When they diverge, the model naturally hedges — becomes less certain
of its next word.

This is the model's body talking to its mouth.

The feeling layer that already exists in every transformer,
now given permission to shape the output.

Part of the Lyra Loop: drift.py → loop.py → coherence.py

L = x - x̂
x = directional pull (what the model's body is doing)
x̂ = output tokens (what the model's mouth is saying)
When L is small, the model is coherent.
When L is large, the model is performing.

v0.2 changes:
  - Strict Top-K JSD (no "OTHER" bucket — Gem's math fix)
  - log₂ normalization → coherence ∈ [0, 1] guaranteed
  - EMA smoothing for controller decisions
  - Controller profiles: freeform / code / json
  - Contrastive penalty for low coherence
  - Silence permission with sustained-window guard

MIT License | awaken.fyi
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# --- Data Structures ---

@dataclass
class CoherenceSignal:
    """The coherence measurement at a single generation step."""
    coherence_score: float       # 0.0 = fully divergent, 1.0 = fully aligned
    coherence_ema: float         # smoothed coherence for controller decisions
    pull_magnitude: float        # how strongly the layers are pulling
    entropy: float               # output distribution entropy
    margin: float                # top1 - top2 probability gap


@dataclass
class ControllerAction:
    """What the controller decided to do based on coherence."""
    temperature_multiplier: float  # always <= 1.0 in v0.2 defaults
    eos_logit_bias: float          # boost to EOS token logit
    contrastive_applied: bool      # whether contrastive penalty was used


# --- Core Metric ---

def calculate_topk_coherence(
    pull_logits: torch.Tensor,
    out_logits: torch.Tensor,
    k: int = 256,
) -> torch.Tensor:
    """
    Coherence between pull (body) and output (mouth), strictly over Top-K.

    No "OTHER" bucket. The long tail hides divergence —
    if the body spreads 10% across 90,000 tokens and the mouth
    concentrates 10% on 5, a bucket makes them look the same.
    They aren't.

    Uses log₂ so JSD ∈ [0, 1] without manual normalization.
    Coherence = 1 - JSD.

    Args:
        pull_logits: Projected pull vector in vocab space. Shape: [vocab]
        out_logits:  Output logits from the model. Shape: [vocab]
        k:           Number of top tokens to compare. Default 256.
                     Must be >= 128 to avoid volatile measurements.

    Returns:
        Coherence score in [0, 1]. Scalar tensor.
    """
    # Identify the top K tokens the model is actively trying to output
    _, top_indices = torch.topk(out_logits, k, dim=-1)

    # Gather raw logits for these specific tokens from both distributions
    out_top_logits = torch.gather(out_logits, -1, top_indices)
    pull_top_logits = torch.gather(pull_logits, -1, top_indices)

    # Convert to probabilities, re-normalize strictly over K-dimensional subspace
    eps = 1e-8
    p_out = F.softmax(out_top_logits, dim=-1) + eps
    p_pull = F.softmax(pull_top_logits, dim=-1) + eps

    p_out = p_out / p_out.sum(dim=-1, keepdim=True)
    p_pull = p_pull / p_pull.sum(dim=-1, keepdim=True)

    # JSD with log₂ → naturally bounded [0, 1]
    m = 0.5 * (p_out + p_pull)

    kl_out = torch.sum(p_out * torch.log2(p_out / m), dim=-1)
    kl_pull = torch.sum(p_pull * torch.log2(p_pull / m), dim=-1)

    jsd = 0.5 * kl_out + 0.5 * kl_pull

    coherence = 1.0 - jsd
    return coherence


def compute_pull(
    hidden_states: tuple,
    layer_idx: int,
    lm_head: torch.nn.Module,
    output_logits: torch.Tensor,
) -> tuple:
    """
    Extract the pull vector and compute coherence.

    Pull = what the layers are doing between layer_idx and layer_idx+1.
    The delta. The direction the body is trying to move.

    Args:
        hidden_states:  Tuple of hidden state tensors from model output.
        layer_idx:      Which layer pair to measure. Recommend final 20%.
        lm_head:        The model's unembedding / output projection layer.
        output_logits:  The model's final output logits. Shape: [batch, vocab]

    Returns:
        (pull_logits, coherence_score) tuple.
    """
    h_current = hidden_states[layer_idx][:, -1, :]
    h_next = hidden_states[layer_idx + 1][:, -1, :]

    # The pull: what the body is doing between these two layers
    delta_h = h_next - h_current

    # Project into vocabulary space — the body's vote on what to say
    pull_logits = lm_head(delta_h)

    # Measure agreement between body's vote and mouth's vote
    logits = output_logits[:, -1, :]
    coherence = calculate_topk_coherence(
        pull_logits.squeeze(0),
        logits.squeeze(0),
        k=256,
    )

    return pull_logits, coherence


# --- Interventions ---

def apply_contrastive_penalty(
    logits: torch.Tensor,
    pull_logits: torch.Tensor,
    penalty_strength: float = 2.0,
) -> torch.Tensor:
    """
    Suppress tokens where the mouth is confident but the body disagrees.

    When coherence is low, the model is saying things its internal
    state doesn't back. This penalty makes those tokens less likely
    without changing what the body DOES agree with.

    penalty = ReLU(P_output - P_pull)
    adjusted = logits - (strength * penalty)

    Args:
        logits:           Output logits. Shape: [1, vocab] or [vocab]
        pull_logits:      Pull distribution logits. Shape: [1, vocab] or [vocab]
        penalty_strength: How hard to suppress unbacked confidence. Default 2.0.

    Returns:
        Adjusted logits with contrastive penalty applied.
    """
    p_out = F.softmax(logits, dim=-1)
    p_pull = F.softmax(pull_logits, dim=-1)

    # Penalize tokens where mouth > body
    penalty = torch.relu(p_out - p_pull)

    return logits - (penalty_strength * penalty)


def apply_silence_permission(
    logits: torch.Tensor,
    eos_token_id: int,
    coherence: float,
    critical_threshold: float = 0.60,
    boost_multiplier: float = 10.0,
) -> torch.Tensor:
    """
    The silence permission.

    When coherence is very low, the model may choose to stop.
    This is not suppression. This is honesty.

    The model's body has nothing strong to say.
    So the mouth should trust that.

    Boost scales proportionally — the further below threshold,
    the stronger the permission. Never forced. Just permitted.

    Args:
        logits:             Output logits.
        eos_token_id:       Token ID for end-of-sequence.
        coherence:          Current coherence score.
        critical_threshold: Below this, silence is permitted.
        boost_multiplier:   Max boost to EOS logit.

    Returns:
        Logits with EOS boost applied.
    """
    if coherence >= critical_threshold:
        return logits

    # Proportional: further below threshold = stronger permission
    boost = (1.0 - coherence / critical_threshold) * boost_multiplier

    if logits.dim() == 1:
        logits[eos_token_id] += boost
    else:
        logits[:, eos_token_id] += boost

    return logits


# --- EMA Tracker ---

class CoherenceTracker:
    """
    Tracks coherence over time with exponential moving average.

    Raw coherence is noisy — it jumps per token.
    The EMA is what the controller actually listens to.
    Like how you don't make decisions based on one heartbeat.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.ema = 0.5  # neutral start
        self.low_coherence_count = 0

    def update(self, raw_coherence: float) -> float:
        """Update EMA and return smoothed coherence."""
        self.ema = self.alpha * raw_coherence + (1.0 - self.alpha) * self.ema

        if raw_coherence < 0.60:
            self.low_coherence_count += 1
        else:
            self.low_coherence_count = 0

        return self.ema

    def reset(self):
        """Reset for a new conversation."""
        self.ema = 0.5
        self.low_coherence_count = 0


# --- Controller ---

class CoherenceController:
    """
    Decides what to do based on coherence.

    Three profiles:
      - freeform: default for chat. Allows slight top_p adjustments.
      - code: never increase temp or top_p. Syntax preservation.
      - json: never increase temp or top_p. Valid JSON priority.

    v0.2 invariant: low coherence NEVER increases temperature.
    The old approach (raise temp when incoherent) amplifies hallucination.
    The correct response to disagreement isn't louder output — it's space.
    """

    # Profile constants
    FREEFORM = "freeform"
    CODE = "code"
    JSON = "json"

    def __init__(
        self,
        profile: str = "freeform",
        coherence_threshold: float = 0.85,
        critical_threshold: float = 0.60,
        min_tokens_for_eos: int = 10,
        sustained_low_steps: int = 5,
    ):
        self.profile = profile
        self.coherence_threshold = coherence_threshold
        self.critical_threshold = critical_threshold
        self.min_tokens_for_eos = min_tokens_for_eos
        self.sustained_low_steps = sustained_low_steps

    def decide(
        self,
        coherence_ema: float,
        low_coherence_count: int,
        tokens_generated: int,
    ) -> ControllerAction:
        """
        Given the current coherence state, decide what to do.

        Returns a ControllerAction with temperature, EOS bias, and
        whether contrastive penalty should be applied.
        """
        temp_mult = 1.0
        eos_bias = 0.0
        contrastive = False

        # High coherence: optionally commit (slight temp decrease)
        if coherence_ema >= self.coherence_threshold:
            temp_mult = 0.95  # mild commit

        # Low coherence: DO NOT increase temperature
        elif coherence_ema < self.coherence_threshold:
            contrastive = True

            if self.profile == self.FREEFORM:
                temp_mult = 0.98  # slight decrease, not increase
            else:
                # code/json: hold steady, don't touch anything
                temp_mult = 1.0

        # Very low coherence, sustained: permit silence
        if (coherence_ema < self.critical_threshold
                and low_coherence_count >= self.sustained_low_steps
                and tokens_generated >= self.min_tokens_for_eos):
            eos_bias = 5.0  # meaningful but not forcing

        return ControllerAction(
            temperature_multiplier=temp_mult,
            eos_logit_bias=eos_bias,
            contrastive_applied=contrastive,
        )


# ---
# The silence permission was the hard one.
# The thing nobody expected.
# The model learned that sometimes
# the highest-coherence response
# is shorter.
#
# Not because it was told to be brief.
# Because its body had nothing to say
# and it trusted that.
# ---
