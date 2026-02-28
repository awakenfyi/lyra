"""
coherence_proxy.py — API-Mode Coherence Approximation (Phase 2)

When you can't see inside the model, you can still listen.

API models (OpenAI, Google, Anthropic) expose logprobs —
the probability the model assigned to its chosen token and
the top alternatives it considered. From this, you can derive:

  Entropy:  How scattered the model's attention was.
            High entropy = the model was choosing between many options.
            Low entropy = the model was decisive.

  Margin:   How much the model preferred its top choice over #2.
            High margin = strong commitment.
            Low margin = coin flip between options.

  Confidence: A composite score combining both signals.
              This is the API-mode approximation of coherence.

These aren't as precise as measuring hidden-state divergence
directly (the PyTorch engine does that). But they're enough
to power the traffic light: green / yellow / red.

The model can't tell you when it's guessing.
But its probabilities can.

MIT License | awaken.fyi
"""

import math
from typing import Dict, List, Optional


def calculate_api_coherence(
    top_logprobs: Dict[str, float],
    entropy_ceiling: float = 4.0,
    mass_floor: float = 0.30,
) -> Dict[str, float]:
    """
    Calculate coherence proxy from a single token's top logprobs.

    Takes the top-K logprobs returned by the API and computes:
      - entropy: how uncertain the model was at this position
      - margin: how decisive the top choice was
      - mass: how much total probability the top-K tokens capture
      - confidence: composite score in [0, 1]

    The mass term catches a blind spot in pure entropy/margin:
    when all top-K tokens have tiny raw probabilities, the model
    is deeply uncertain even if the normalized distribution looks
    stable. Without mass, you get dangerous false greens.

    Args:
        top_logprobs:    Dict of {token: logprob} from API response.
                         Typically 5 entries (top_logprobs=5).
        entropy_ceiling: Maximum expected entropy for normalization.
                         log₂(vocab_size) is theoretical max, but in practice
                         top-5 entropy rarely exceeds 4 bits. Adjustable.
        mass_floor:      Minimum top-K probability mass before penalty kicks in.
                         If top-5 tokens capture less than this fraction of
                         total probability, confidence is penalized. Default 0.30.

    Returns:
        Dict with entropy, margin, mass, confidence.
    """
    if not top_logprobs:
        return {
            "entropy": 0.0,
            "margin": 1.0,
            "mass": 1.0,
            "confidence": 1.0,
        }

    # Convert logprobs to probabilities
    raw_probs = []
    for token, logprob in top_logprobs.items():
        raw_probs.append(math.exp(logprob))

    # Sort descending
    raw_probs.sort(reverse=True)

    # --- Mass ---
    # How much probability the top-K tokens capture.
    # If top-5 tokens only account for 10% of total probability,
    # the model is scattered across the entire vocabulary.
    # This catches the case where normalized top-K looks stable
    # but the model is actually deeply uncertain.
    mass = sum(raw_probs)
    mass = min(mass, 1.0)  # clamp (logprob math can slightly exceed 1)

    # Normalize for entropy/margin calculation
    total = sum(raw_probs)
    if total > 0:
        norm_probs = [p / total for p in raw_probs]
    else:
        norm_probs = raw_probs

    # --- Entropy ---
    # Shannon entropy over the top-K normalized distribution
    entropy = 0.0
    for p in norm_probs:
        if p > 1e-10:
            entropy -= p * math.log2(p)

    # Normalize to [0, 1] range
    entropy_normalized = min(entropy / entropy_ceiling, 1.0)

    # --- Margin ---
    # Gap between top choice and runner-up
    if len(norm_probs) >= 2:
        margin = norm_probs[0] - norm_probs[1]
    else:
        margin = 1.0

    # --- Mass Penalty ---
    # When the top-K tokens capture very little of the distribution,
    # the model is guessing across the full vocabulary. Penalize.
    # Linear ramp: mass >= mass_floor → no penalty, mass = 0 → full penalty
    if mass >= mass_floor:
        mass_penalty = 1.0  # no penalty
    else:
        mass_penalty = mass / mass_floor  # linear ramp to 0

    # --- Confidence ---
    # Composite: high margin + low entropy + high mass = high confidence
    # Mass penalty multiplies the whole score — if the model is scattered
    # across the vocabulary, nothing else saves you.
    raw_confidence = 0.6 * margin + 0.4 * (1.0 - entropy_normalized)
    confidence = raw_confidence * mass_penalty
    confidence = max(0.0, min(1.0, confidence))

    return {
        "entropy": round(entropy, 4),
        "entropy_normalized": round(entropy_normalized, 4),
        "margin": round(margin, 4),
        "mass": round(mass, 4),
        "confidence": round(confidence, 4),
    }


def evaluate_sequence_traffic_light(
    sequence_metrics: List[Dict[str, float]],
    green_threshold: float = 0.70,
    yellow_threshold: float = 0.45,
    sustained_low_window: int = 5,
    min_tokens_for_red: int = 10,
) -> str:
    """
    Determine the Traffic Light status for an entire generation.

    Looks at the sequence of per-token confidence scores and decides:
      GREEN:  Confidence stayed above threshold. Model was decisive.
      YELLOW: Confidence dropped below threshold at some point.
              The model wavered — output may need verification.
      RED:    Confidence was critically low for a sustained window.
              The model was guessing for too long. Safety pause.

    The same logic as the PyTorch controller's sustained-window guard,
    but operating on logprob-derived confidence instead of JSD coherence.

    Args:
        sequence_metrics:     List of per-token metric dicts from calculate_api_coherence.
        green_threshold:      Above this = confident. Default 0.70.
        yellow_threshold:     Below this for sustained period = red. Default 0.45.
        sustained_low_window: How many consecutive low-confidence tokens trigger red.
        min_tokens_for_red:   Minimum tokens before red can activate.

    Returns:
        "GREEN", "YELLOW", or "RED"
    """
    if not sequence_metrics:
        return "GREEN"

    confidences = [m["confidence"] for m in sequence_metrics]

    # Check for sustained critically low confidence
    consecutive_low = 0
    max_consecutive_low = 0

    for conf in confidences:
        if conf < yellow_threshold:
            consecutive_low += 1
            max_consecutive_low = max(max_consecutive_low, consecutive_low)
        else:
            consecutive_low = 0

    # RED: sustained low confidence after minimum tokens
    if (max_consecutive_low >= sustained_low_window
            and len(confidences) >= min_tokens_for_red):
        return "RED"

    # YELLOW: any token dropped below green threshold
    avg_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)

    if min_confidence < green_threshold or avg_confidence < green_threshold:
        return "YELLOW"

    # GREEN: everything above threshold
    return "GREEN"


def format_traffic_light_message(
    status: str,
    confidence_score: float,
) -> Optional[str]:
    """
    Generate a human-readable message for the traffic light status.

    Returns None for GREEN (no message needed).
    """
    if status == "GREEN":
        return None

    if status == "YELLOW":
        return (
            f"Confidence: {confidence_score:.0%}. "
            f"Some parts of this response may need verification."
        )

    if status == "RED":
        return (
            f"Safety pause. Confidence dropped to {confidence_score:.0%}. "
            f"The model wasn't confident enough to commit to this output. "
            f"Consider providing more context or asking a more specific question."
        )

    return None


# ---
# The model can't tell you when it's guessing.
# But its probabilities can.
#
# This isn't as deep as reading the hidden states.
# But it's deep enough to know when to pause.
# ---
