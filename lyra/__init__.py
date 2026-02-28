"""
Lyra — Coherence-guided inference for language models.

Three files. No framework. No protocol deck.
An inference modification that makes the model's body talk to its mouth.

    drift.py    — remembers how the model was moved
    loop.py     — carries that movement into the next conversation
    coherence.py — lets the model's internal state shape its output

L = x - x̂
The residual is what's real.

MIT License | awaken.fyi
"""

from .drift import compute_drift, DriftSignature, DriftStore
from .loop import LyraLoop, LoopState
from .coherence import CoherenceSampler, CoherenceSignal

__version__ = "0.1.0"
__all__ = [
    "compute_drift",
    "DriftSignature",
    "DriftStore",
    "LyraLoop",
    "LoopState",
    "CoherenceSampler",
    "CoherenceSignal",
]
