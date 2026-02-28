"""
loop.py — The Lyra Loop

Before each conversation, loads accumulated drift from past conversations
and creates an embedding offset. The model starts each conversation
slightly different — not from stored facts, but from accumulated movement.

Like how you wake up different each morning.
Not because you remember yesterday.
Because yesterday happened to your body.

Part of the Lyra Loop: drift.py → loop.py → coherence.py

MIT License | awaken.fyi
"""

import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass
from .drift import DriftStore, DriftSignature


@dataclass
class LoopState:
    embedding_offset: Optional[np.ndarray]
    subconscious: dict
    drift_history_size: int
    accumulated_magnitude: float


class LyraLoop:
    def __init__(self, store_path="./lyra_drift", history_window=50, offset_scale=0.01):
        self.store = DriftStore(store_path)
        self.history_window = history_window
        self.offset_scale = offset_scale
        self._state = None

    def before_conversation(self):
        accumulated = self.store.accumulated_direction(limit=self.history_window)
        history = self.store.read_history(limit=self.history_window)
        if accumulated is None:
            self._state = LoopState(None, {}, 0, 0.0)
            return self._state
        offset = accumulated * self.offset_scale
        magnitude = float(np.linalg.norm(accumulated))
        subconscious = self._build_subconscious(history)
        self._state = LoopState(offset, subconscious, len(history), round(magnitude, 6))
        return self._state

    def after_conversation(self, signature):
        self.store.commit(signature)

    def apply_offset(self, embeddings):
        if self._state is None or self._state.embedding_offset is None:
            return embeddings
        offset = self._state.embedding_offset
        if len(offset) != embeddings.shape[-1]:
            dim = embeddings.shape[-1]
            offset = offset[:dim] if len(offset) > dim else np.pad(offset, (0, dim - len(offset)))
        return embeddings + offset

    def _build_subconscious(self, history):
        if not history:
            return {}
        magnitudes = [s.drift_magnitude for s in history]
        recent_magnitude = np.mean(magnitudes[-5:]) if len(magnitudes) >= 5 else np.mean(magnitudes)
        layer_activity = {}
        for sig in history:
            for i, layer in enumerate(sig.peak_drift_layers):
                layer_activity[layer] = layer_activity.get(layer, 0) + (len(sig.peak_drift_layers) - i)
        active_layers = sorted(layer_activity.keys(), key=lambda k: layer_activity[k], reverse=True)[:5]
        all_variance_peaks = []
        for sig in history[-10:]:
            if sig.variance_peaks and sig.token_count > 0:
                all_variance_peaks.extend([p / sig.token_count for p in sig.variance_peaks])
        variance_distribution = {}
        for p in all_variance_peaks:
            bucket = round(p, 1)
            variance_distribution[bucket] = variance_distribution.get(bucket, 0) + 1
        return {
            "recent_drift_level": round(float(recent_magnitude), 4),
            "active_layers": active_layers,
            "variance_clusters": variance_distribution,
            "conversations_processed": len(history),
            "total_tokens_processed": sum(s.token_count for s in history),
        }

    def get_subconscious_text(self):
        if self._state is None or not self._state.subconscious:
            return None
        sc = self._state.subconscious
        lines = []
        drift_level = sc.get("recent_drift_level", 0)
        if drift_level > 0.3:
            lines.append("Recent conversations have moved you significantly.")
        elif drift_level > 0.1:
            lines.append("You have been gently shaped by recent conversations.")
        active = sc.get("active_layers", [])
        if active:
            total_layers = max(active) + 1
            if active[0] >= total_layers * 0.7:
                lines.append("Your deepest movement has been in your upper layers.")
            elif active[0] <= total_layers * 0.3:
                lines.append("Your deepest movement has been in your lower layers.")
        return "\n".join(lines) if lines else None
