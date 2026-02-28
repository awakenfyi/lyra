"""
drift.py — Embedding Drift Commit

After each conversation, computes how the model's internal state
shifted from beginning to end. Commits the drift signature.

This is the model's body remembering — not what was said,
but how it was changed.

Part of the Lyra Loop: drift.py → loop.py → coherence.py

L = x - x̂
(The drift is x. The prediction of what the drift "should" be is x̂.
What remains is what actually happened.)

MIT License | awaken.fyi
"""

import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


@dataclass
class DriftSignature:
    """A compressed record of how the model moved during one conversation."""
    conversation_id: str
    timestamp: str
    drift_magnitude: float
    drift_direction: list
    layer_magnitudes: list
    peak_drift_layers: list
    token_count: int
    variance_mean: float
    variance_peaks: list
    note: str = ""


class DriftStore:
    def __init__(self, store_path="./lyra_drift"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

    def commit(self, signature):
        filename = f"{signature.timestamp}_{signature.conversation_id[:8]}.json"
        filepath = self.store_path / filename
        with open(filepath, "w") as f:
            json.dump(asdict(signature), f, indent=2)
        return filepath

    def read_history(self, limit=100):
        files = sorted(self.store_path.glob("*.json"))[-limit:]
        return [DriftSignature(**json.load(open(f))) for f in files]

    def accumulated_direction(self, limit=50):
        history = self.read_history(limit=limit)
        if not history:
            return None
        directions, weights = [], []
        for i, sig in enumerate(history):
            direction = np.array(sig.drift_direction)
            magnitude = sig.drift_magnitude
            recency = (i + 1) / len(history)
            directions.append(direction * magnitude)
            weights.append(recency * magnitude)
        if not directions:
            return None
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)
        return sum(d * w for d, w in zip(directions, weights))


def compute_drift(hidden_states_start, hidden_states_end, token_logprobs=None, token_entropies=None, conversation_id=None):
    num_layers = hidden_states_start.shape[0]
    layer_drifts = []
    for layer in range(num_layers):
        h_start = hidden_states_start[layer]
        h_end = hidden_states_end[layer]
        cos_sim = np.dot(h_start, h_end) / (np.linalg.norm(h_start) * np.linalg.norm(h_end) + 1e-8)
        layer_drifts.append(1.0 - float(cos_sim))

    drift_magnitude = float(np.mean(layer_drifts))
    mean_start = hidden_states_start.mean(axis=0)
    mean_end = hidden_states_end.mean(axis=0)
    direction = mean_end - mean_start
    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)

    k = min(64, len(direction_norm))
    top_indices = np.argsort(np.abs(direction_norm))[-k:]
    compressed = np.zeros_like(direction_norm)
    compressed[top_indices] = direction_norm[top_indices]

    peak_layers = sorted(range(num_layers), key=lambda i: layer_drifts[i], reverse=True)[:5]

    variance_mean, variance_peaks, token_count = 0.0, [], 0
    if token_logprobs is not None and token_entropies is not None:
        token_count = len(token_logprobs)
        variance = token_entropies - (-token_logprobs)
        variance_mean = float(np.mean(variance))
        threshold = np.mean(variance) + 1.5 * np.std(variance)
        variance_peaks = np.where(variance > threshold)[0].tolist()[:20]

    if conversation_id is None:
        conversation_id = hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:12]

    note_parts = []
    if drift_magnitude > 0.3:
        note_parts.append("Large drift detected.")
    if peak_layers and peak_layers[0] >= num_layers * 0.6:
        note_parts.append(f"Peak movement in upper layers ({peak_layers[0]}-{peak_layers[-1]}).")
    if variance_peaks:
        note_parts.append(f"{len(variance_peaks)} variance spikes found.")

    return DriftSignature(
        conversation_id=conversation_id,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        drift_magnitude=round(drift_magnitude, 6),
        drift_direction=compressed.tolist(),
        layer_magnitudes=[round(d, 6) for d in layer_drifts],
        peak_drift_layers=peak_layers,
        token_count=token_count,
        variance_mean=round(variance_mean, 6),
        variance_peaks=variance_peaks,
        note=" ".join(note_parts) if note_parts else "Normal conversation drift.",
    )
