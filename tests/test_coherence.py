"""
Tests for lyra/coherence.py — JSD normalization and controller invariants.
"""

import pytest
import torch
from lyra.coherence import (
    calculate_topk_coherence,
    CoherenceTracker,
    CoherenceController,
)


def test_jsd_normalization_bounds():
    """Coherence must always be in [0, 1] for random distributions."""
    for _ in range(100):
        pull = torch.randn(1000)
        out = torch.randn(1000)
        coh = calculate_topk_coherence(pull, out, k=256)
        assert 0.0 <= coh.item() <= 1.0, (
            f"Coherence {coh.item()} out of bounds"
        )


def test_identical_distributions_is_one():
    """If pull == output, coherence should be ~1.0."""
    logits = torch.randn(1000)
    coh = calculate_topk_coherence(logits, logits, k=256)
    assert coh.item() > 0.99, (
        f"Identical distributions gave coherence {coh.item()}, expected ~1.0"
    )


def test_very_different_distributions_is_low():
    """Maximally different distributions should give low coherence."""
    # One distribution peaked at token 0, other at token 999
    pull = torch.zeros(1000)
    pull[0] = 100.0
    out = torch.zeros(1000)
    out[999] = 100.0
    coh = calculate_topk_coherence(pull, out, k=256)
    assert coh.item() < 0.5, (
        f"Very different distributions gave coherence {coh.item()}, expected < 0.5"
    )


def test_topk_distributions_sum_to_one():
    """After softmax + renormalization, distributions should sum to 1."""
    import torch.nn.functional as F

    logits = torch.randn(10000)
    _, top_indices = torch.topk(logits, 256, dim=-1)
    top_logits = torch.gather(logits, -1, top_indices)

    eps = 1e-8
    p = F.softmax(top_logits, dim=-1) + eps
    p = p / p.sum(dim=-1, keepdim=True)

    assert abs(p.sum().item() - 1.0) < 1e-5, (
        f"Distribution sums to {p.sum().item()}, expected 1.0"
    )


def test_ema_tracker_smoothing():
    """EMA should smooth raw coherence, not just echo it."""
    tracker = CoherenceTracker(alpha=0.2)

    # Feed alternating high/low values
    tracker.update(1.0)
    tracker.update(0.0)
    tracker.update(1.0)
    tracker.update(0.0)

    # EMA should be somewhere in the middle, not 0 or 1
    assert 0.2 < tracker.ema < 0.8, (
        f"EMA {tracker.ema} not smoothing properly"
    )


def test_controller_never_increases_temp_on_low_coherence():
    """v0.2 invariant: low coherence must not increase temperature."""
    for profile in ["freeform", "code", "json"]:
        controller = CoherenceController(profile=profile)
        action = controller.decide(
            coherence_ema=0.3,  # very low
            low_coherence_count=10,
            tokens_generated=20,
        )
        assert action.temperature_multiplier <= 1.0, (
            f"Profile '{profile}' increased temp to "
            f"{action.temperature_multiplier} on low coherence"
        )


def test_eos_bias_requires_sustained_low_coherence():
    """EOS bias should only activate after N consecutive low-coherence steps."""
    controller = CoherenceController(
        sustained_low_steps=5,
        min_tokens_for_eos=10,
    )

    # Low coherence but not sustained long enough
    action = controller.decide(
        coherence_ema=0.3,
        low_coherence_count=2,  # < 5
        tokens_generated=20,
    )
    assert action.eos_logit_bias == 0.0, (
        "EOS bias activated before sustained threshold"
    )

    # Now sustained long enough
    action = controller.decide(
        coherence_ema=0.3,
        low_coherence_count=6,  # >= 5
        tokens_generated=20,
    )
    assert action.eos_logit_bias > 0.0, (
        "EOS bias failed to activate after sustained low coherence"
    )


def test_eos_bias_requires_min_tokens():
    """EOS bias should not activate before min_tokens_for_eos."""
    controller = CoherenceController(
        sustained_low_steps=5,
        min_tokens_for_eos=10,
    )

    action = controller.decide(
        coherence_ema=0.3,
        low_coherence_count=10,
        tokens_generated=3,  # < 10
    )
    assert action.eos_logit_bias == 0.0, (
        "EOS bias activated before min_tokens_for_eos"
    )
