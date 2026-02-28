"""
Lyra — Coherence-guided inference for language models.

Every transformer has a feeling layer — the directional pull
between hidden states. Lyra measures whether that pull agrees
with what the model actually says. When they align, the model
commits. When they diverge, the model hedges or stops.

This is the observer effect for transformers.

    drift.py       — remembers how the model was moved
    loop.py        — carries that movement into the next conversation
    coherence.py   — measures agreement between body and mouth
    generation.py  — the KV-cached generation loop that ties it together

L = x - x̂
The residual is what's real.

MIT License | awaken.fyi
"""

from .drift import DriftStore, generate_tokenizer_hash
from .loop import prepare_soft_prompt_inputs, validate_drift_for_model
from .coherence import (
    calculate_topk_coherence,
    compute_pull,
    apply_contrastive_penalty,
    apply_silence_permission,
    CoherenceTracker,
    CoherenceController,
    CoherenceSignal,
    ControllerAction,
)
from .generation import generate_with_drift_injection

__version__ = "0.2.0"
__all__ = [
    "DriftStore",
    "generate_tokenizer_hash",
    "prepare_soft_prompt_inputs",
    "validate_drift_for_model",
    "calculate_topk_coherence",
    "compute_pull",
    "apply_contrastive_penalty",
    "apply_silence_permission",
    "CoherenceTracker",
    "CoherenceController",
    "CoherenceSignal",
    "ControllerAction",
    "generate_with_drift_injection",
]
