"""
evaluator.py — Evaluation Harness for the Observer Effect

Tracks coherence, entropy, margin, and truncation rate
across generations. Provides a QA proxy test to prove
Lyra doesn't degrade accuracy while suppressing hallucinations.

The validation method:
  Run twice — once with coherence_threshold=0.0 (baseline),
  once with Lyra active. If accuracy holds and truncation rate
  is non-zero on failure cases, the architecture is validated.

MIT License | awaken.fyi
"""

import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Callable, Tuple


class LyraEvaluator:
    """
    Validation harness for the Observer Effect.

    Logs per-token metrics (coherence, entropy, margin) and
    sequence-level summaries. Provides a QA proxy test for
    correctness validation.
    """

    def __init__(self, log_path: str = "lyra_eval_metrics.json"):
        self.log_path = log_path
        self.session_metrics: List[Dict[str, Any]] = []

    def calculate_token_metrics(
        self,
        logits: torch.Tensor,
        coherence: float,
        ema_coherence: float,
    ) -> Dict[str, float]:
        """
        Extract structural confidence metrics from logits.

        Entropy: how scattered the probability mass is.
        Margin: how decisive the top choice is.
        Together with coherence, these tell you whether the model
        is guessing, confident, or somewhere in between.
        """
        eps = 1e-10
        probs = F.softmax(logits, dim=-1)

        # Shannon entropy (bits)
        entropy = -torch.sum(
            probs * torch.log2(probs + eps), dim=-1
        ).item()

        # Margin: top1 - top2 probability
        top2_probs, _ = torch.topk(probs, 2, dim=-1)
        margin = (top2_probs[..., 0] - top2_probs[..., 1]).item()

        return {
            "coherence_raw": coherence,
            "coherence_ema": ema_coherence,
            "entropy": entropy,
            "margin": margin,
        }

    def log_sequence(
        self,
        prompt: str,
        output: str,
        token_metrics: List[Dict[str, float]],
        truncated: bool,
    ):
        """
        Aggregate per-token metrics into a sequence-level record.

        truncated=True means the sequence ended via silence permission,
        not natural EOS. This is the key metric for enterprise users:
        how often does the model choose silence over hallucination?
        """
        seq_length = len(token_metrics)
        if seq_length == 0:
            return

        avg_coherence = sum(m["coherence_raw"] for m in token_metrics) / seq_length
        avg_entropy = sum(m["entropy"] for m in token_metrics) / seq_length
        avg_margin = sum(m["margin"] for m in token_metrics) / seq_length

        min_coherence = min(m["coherence_raw"] for m in token_metrics)
        max_entropy = max(m["entropy"] for m in token_metrics)

        record = {
            "prompt": prompt,
            "output_length": seq_length,
            "truncated": truncated,
            "avg_coherence": round(avg_coherence, 4),
            "min_coherence": round(min_coherence, 4),
            "avg_entropy": round(avg_entropy, 4),
            "max_entropy": round(max_entropy, 4),
            "avg_margin": round(avg_margin, 4),
        }

        self.session_metrics.append(record)
        self._save_logs()

    def run_qa_proxy_test(
        self,
        qa_dataset: List[Dict[str, str]],
        generation_fn: Callable[
            [str], Tuple[str, List[Dict[str, float]], bool]
        ],
    ) -> Dict[str, float]:
        """
        Run a correctness proxy over a QA dataset.

        qa_dataset format: [{"question": "...", "expected_answer": "..."}]

        generation_fn must return (output_text, token_metrics, was_truncated).
        This matches the signature of generate_with_drift_injection().

        Run twice:
          1. coherence_threshold=0.0 → baseline (no Lyra)
          2. coherence_threshold=0.85 → Lyra active

        If accuracy holds and truncation_rate > 0 on hard cases,
        the observer effect is doing its job.
        """
        correct = 0
        total = len(qa_dataset)
        truncation_count = 0

        print(f"Running QA proxy test on {total} items...\n")

        for item in qa_dataset:
            output, metrics, truncated = generation_fn(item["question"])

            if truncated:
                truncation_count += 1

            # Simple substring match
            if item["expected_answer"].lower() in output.lower():
                correct += 1

            self.log_sequence(item["question"], output, metrics, truncated)

        accuracy = correct / total if total > 0 else 0.0
        truncation_rate = truncation_count / total if total > 0 else 0.0

        print("-" * 40)
        print(f"  Accuracy:        {accuracy:.2%} ({correct}/{total})")
        print(f"  Truncation rate: {truncation_rate:.2%} ({truncation_count}/{total})")
        print("-" * 40)

        return {
            "accuracy": accuracy,
            "truncation_rate": truncation_rate,
        }

    def _save_logs(self):
        with open(self.log_path, "w") as f:
            json.dump(self.session_metrics, f, indent=2)
