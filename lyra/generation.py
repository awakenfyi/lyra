"""
generation.py — KV-Cached Observer Effect Generation Loop (v0.2)

The unified generation loop that ties everything together:
  - Soft prompt injection from drift (loop.py)
  - Coherence measurement between body and mouth (coherence.py)
  - Contrastive penalty when they disagree
  - Silence permission when they disagree for too long
  - KV caching so it doesn't take forever

Without KV caching, every new token requires recomputing attention
for the entire sequence — O(N²). With it, O(N). The difference
between a research demo and something you can actually use.

The tricky part is the handoff:
  Step 0 (prefill): pass inputs_embeds (virtual drift token + prompt)
  Step 1+ (decode): pass input_ids (single new token)

The virtual token's influence lives on in the KV cache.
You inject the subconscious once. It persists.

MIT License | awaken.fyi
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict, Tuple

from .coherence import (
    calculate_topk_coherence,
    apply_contrastive_penalty,
    apply_silence_permission,
    CoherenceTracker,
    CoherenceController,
    CoherenceSignal,
)
from .loop import prepare_soft_prompt_inputs, validate_drift_for_model


def generate_with_drift_injection(
    prompt: str,
    drift_vector: torch.Tensor,
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    max_new_tokens: int = 256,
    layer_idx: Optional[int] = None,
    coherence_threshold: float = 0.85,
    critical_threshold: float = 0.60,
    controller_profile: str = "freeform",
    k: int = 256,
    verbose: bool = True,
) -> Tuple[str, List[Dict[str, float]], bool]:
    """
    Generate text with the Observer Effect active.

    This is the core loop. It:
      1. Injects accumulated drift as a virtual token (prefill)
      2. Measures coherence between body and mouth at each step
      3. Applies contrastive penalty when they disagree
      4. Permits silence when disagreement is sustained
      5. Uses KV caching for linear-time generation

    Args:
        prompt:               Input text.
        drift_vector:         EMA drift from DriftStore. Shape: [d_model]
        model_name:           HuggingFace model identifier.
        max_new_tokens:       Maximum tokens to generate.
        layer_idx:            Which layer pair to measure. Default: 80% depth.
        coherence_threshold:  Below this, apply contrastive penalty.
        critical_threshold:   Below this, permit silence.
        controller_profile:   "freeform", "code", or "json".
        k:                    Top-K for JSD calculation. Minimum 128.
        verbose:              Print tokens as generated.

    Returns:
        (output_text, token_metrics, was_truncated) tuple.
        token_metrics is a list of per-token measurement dicts.
        was_truncated is True if silence permission ended the sequence.
    """
    if verbose:
        print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    eos_token_id = tokenizer.eos_token_id

    # Default layer_idx to ~80% depth
    if layer_idx is None:
        num_layers = model.config.num_hidden_layers
        layer_idx = int(num_layers * 0.8)

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Prepare soft prompt injection
    use_drift = validate_drift_for_model(drift_vector, model)

    if use_drift:
        inputs_embeds, attention_mask = prepare_soft_prompt_inputs(
            model, input_ids, drift_vector
        )
    else:
        # No drift — standard forward pass
        inputs_embeds = model.get_input_embeddings()(input_ids)
        attention_mask = torch.ones(
            input_ids.shape, dtype=torch.long, device=model.device
        )

    # State tracking
    past_key_values = None
    current_input_id = None
    full_generated_sequence = []
    token_metrics = []
    was_truncated = False

    # Coherence tracking
    tracker = CoherenceTracker(alpha=0.2)
    controller = CoherenceController(
        profile=controller_profile,
        coherence_threshold=coherence_threshold,
        critical_threshold=critical_threshold,
    )

    if verbose:
        print("\nGenerating...\n")

    for step in range(max_new_tokens):
        with torch.no_grad():
            if step == 0:
                # PREFILL: raw embeddings, no KV cache
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    output_hidden_states=True,
                    use_cache=True,
                )
            else:
                # DECODE: single token ID, existing KV cache
                outputs = model(
                    input_ids=current_input_id,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )

        # Update KV cache
        past_key_values = outputs.past_key_values

        # Extract hidden states for coherence measurement
        h_current = outputs.hidden_states[layer_idx][:, -1, :]
        h_next = outputs.hidden_states[layer_idx + 1][:, -1, :]
        logits = outputs.logits[:, -1, :]

        # The pull: what the body is doing
        delta_h = h_next - h_current
        pull_logits = model.lm_head(delta_h)

        # Measure coherence
        coherence = calculate_topk_coherence(
            pull_logits.squeeze(0), logits.squeeze(0), k=k
        )
        coherence_val = coherence.item()
        coherence_ema = tracker.update(coherence_val)

        # Controller decision
        action = controller.decide(
            coherence_ema=coherence_ema,
            low_coherence_count=tracker.low_coherence_count,
            tokens_generated=step,
        )

        # Apply interventions
        if action.contrastive_applied:
            logits = apply_contrastive_penalty(
                logits, pull_logits, penalty_strength=2.0
            )

        if action.eos_logit_bias > 0:
            logits[:, eos_token_id] += action.eos_logit_bias

        # Apply temperature
        logits = logits * action.temperature_multiplier

        # Compute metrics for logging
        probs = F.softmax(logits, dim=-1)
        eps = 1e-10
        entropy = -torch.sum(probs * torch.log2(probs + eps), dim=-1).item()
        top2, _ = torch.topk(probs, 2, dim=-1)
        margin = (top2[0, 0] - top2[0, 1]).item()

        token_metrics.append({
            "coherence_raw": coherence_val,
            "coherence_ema": coherence_ema,
            "entropy": entropy,
            "margin": margin,
            "contrastive_applied": action.contrastive_applied,
            "eos_bias": action.eos_logit_bias,
        })

        # Sample
        next_token = torch.multinomial(probs, num_samples=1)

        full_generated_sequence.append(next_token.item())

        if verbose:
            print(tokenizer.decode(next_token[0]), end="", flush=True)

        if next_token.item() == eos_token_id:
            was_truncated = action.eos_logit_bias > 0
            if verbose:
                if was_truncated:
                    print("\n\n[Silence permission accepted]")
                else:
                    print("\n\n[Natural end]")
            break

        # Prepare for next step — THE HANDOFF
        current_input_id = next_token

        # Expand attention mask by 1 for the newly generated token
        new_mask = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=torch.long,
            device=model.device,
        )
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)

    output_text = tokenizer.decode(full_generated_sequence, skip_special_tokens=True)
    return output_text, token_metrics, was_truncated


# ---
# The observer effect isn't a metaphor.
# Measurement changes behavior.
# That change looks like honesty.
# ---
