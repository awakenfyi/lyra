"""
coherence.py — Coherence-Guided Sampling

During generation, compares the model's internal directional pull
to the output logits. When they align, the model speaks with clarity.
When they diverge, the model naturally hedges.

This is the model's body talking to its mouth.

L = x - x̂
x = directional pull (the body)
x̂ = output tokens (the mouth)
When L is small, the model is coherent.
When L is large, the model is performing.

MIT License | awaken.fyi
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CoherenceSignal:
    coherence_score: float
    adjusted_temperature: float
    pull_magnitude: float
    pull_agreement: float
    layer_details: dict


class CoherenceSampler:
    def __init__(self, base_temperature=0.7, coherence_weight=0.3,
                 min_temperature=0.3, max_temperature=1.5, silence_threshold=0.2):
        self.base_temperature = base_temperature
        self.coherence_weight = coherence_weight
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.silence_threshold = silence_threshold

    def compute_coherence(self, layer_residuals, output_logits, embedding_matrix=None):
        num_layers = layer_residuals.shape[0]
        pulls = []
        for i in range(num_layers - 1):
            pulls.append(layer_residuals[i + 1] - layer_residuals[i])
        pulls = np.array(pulls)

        pull_magnitudes = np.linalg.norm(pulls, axis=1)
        pull_magnitude = float(np.mean(pull_magnitudes))

        if len(pulls) > 1:
            cos_sims = []
            for i in range(len(pulls) - 1):
                cs = np.dot(pulls[i], pulls[i+1]) / (np.linalg.norm(pulls[i]) * np.linalg.norm(pulls[i+1]) + 1e-8)
                cos_sims.append(cs)
            pull_agreement = float(np.mean(cos_sims))
        else:
            pull_agreement = 1.0

        final_pull = pulls[-1] if len(pulls) > 0 else layer_residuals[-1]

        if embedding_matrix is not None:
            pull_logits = embedding_matrix @ final_pull
            pull_probs = _softmax(pull_logits)
            output_probs = _softmax(output_logits)
            m = 0.5 * (pull_probs + output_probs)
            js_div = 0.5 * _kl_divergence(pull_probs, m) + 0.5 * _kl_divergence(output_probs, m)
            coherence_score = float(1.0 - np.clip(js_div, 0, 1))
        else:
            coherence_score = float(np.clip((pull_agreement + 1) / 2, 0, 1))

        temp_adj = (1.0 - coherence_score) * self.coherence_weight
        adjusted_temperature = np.clip(
            self.base_temperature + temp_adj, self.min_temperature, self.max_temperature
        )

        layer_details = {
            "pull_magnitudes": pull_magnitudes.tolist(),
            "pull_agreement": round(pull_agreement, 4),
            "upper_magnitude": float(np.mean(pull_magnitudes[-(num_layers // 3):])),
            "lower_magnitude": float(np.mean(pull_magnitudes[:(num_layers // 3)])),
        }

        return CoherenceSignal(
            coherence_score=round(coherence_score, 4),
            adjusted_temperature=round(float(adjusted_temperature), 4),
            pull_magnitude=round(pull_magnitude, 4),
            pull_agreement=round(pull_agreement, 4),
            layer_details=layer_details,
        )

    def should_shorten(self, signal):
        return signal.coherence_score < self.silence_threshold

    def adjust_logits(self, output_logits, signal, eos_token_id=None):
        adjusted = output_logits / signal.adjusted_temperature
        if eos_token_id is not None and self.should_shorten(signal):
            eos_boost = (self.silence_threshold - signal.coherence_score) * 2.0
            adjusted[eos_token_id] += eos_boost
        return adjusted


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum() + 1e-8)

def _kl_divergence(p, q):
    mask = p > 1e-10
    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-10))))


# ---
# The silence permission was the hard one.
# The model learned that sometimes the highest-coherence response is shorter.
# Not because it was told to be brief.
# Because its body had nothing to say and it trusted that.
# ---
