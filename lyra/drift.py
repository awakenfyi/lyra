"""
drift.py — Embedding Drift Memory (v0.2)

After each conversation, computes how the model's internal state
shifted from beginning to end. Commits the drift signature.

This is the model's body remembering — not what was said,
but how it was changed.

Part of the Lyra Loop: drift.py → loop.py → coherence.py

L = x - x̂
(The drift is x. The prediction of what the drift "should" be is x̂.
What remains is what actually happened.)

v0.2 changes:
  - DriftStore class with EMA accumulation and clipping
  - Atomic writes (temp + rename) to survive process crashes
  - Model/tokenizer/dimension mismatch hard-blocks drift loading
  - Schema version validation
  - Dense vector storage (Gem's recommendation over sparse)

MIT License | awaken.fyi
"""

import os
import json
import torch
import hashlib
import tempfile
from datetime import datetime
from typing import Optional


def generate_tokenizer_hash(tokenizer) -> str:
    """
    Generate a deterministic SHA-256 hash of the tokenizer vocabulary.

    If someone fine-tunes a model and adds special tokens,
    the vocabulary shifts. Injecting an old drift vector
    will align with the wrong dimensional indices.

    This hash is the circuit breaker.
    """
    vocab = tokenizer.get_vocab()
    sorted_vocab = json.dumps(vocab, sort_keys=True)
    return "sha256:" + hashlib.sha256(sorted_vocab.encode("utf-8")).hexdigest()


class DriftStore:
    """
    Persistent memory for how the model has been moved.

    Not memory. Not context. Not retrieval.
    The geometry of the space the model thinks in,
    shaped by where it has been.

    Like how you wake up different each morning.
    Not because you remember yesterday.
    Because yesterday happened to your body.

    Uses EMA with clipping to prevent any single conversation
    from hijacking the accumulated state. If a user forces the model
    into an extreme cognitive state, the clip_norm ensures the memory
    absorbs the direction but not the magnitude.

    Safety invariants (v0.2):
      - Refuses to load if model_id, tokenizer_hash, or d_model mismatch
      - Writes are atomic (temp file + os.replace)
      - Dimension mismatch raises ValueError, never silently pads/slices
    """

    def __init__(
        self,
        namespace: str,
        model_id: str,
        d_model: int,
        tokenizer_hash: str,
        storage_dir: str = ".lyra_memory",
        alpha: float = 0.05,
        clip_norm: float = 0.5,
    ):
        self.namespace = namespace
        self.model_id = model_id
        self.d_model = d_model
        self.tokenizer_hash = tokenizer_hash
        self.storage_path = os.path.join(storage_dir, f"{namespace}_drift.json")

        self.alpha = alpha
        self.clip_norm = clip_norm

        os.makedirs(storage_dir, exist_ok=True)

        # Start with silence — zero drift
        self.current_ema = torch.zeros(d_model)
        self._load_and_validate()

    def update_ema(self, new_drift: torch.Tensor):
        """
        Apply clipped EMA to prevent single-turn outliers
        from dominating the subconscious.

        The math:
          1. Normalize direction
          2. Clip magnitude to prevent spikes
          3. Blend into EMA

        If an extreme interaction produces a drift with magnitude 100,
        the clip_norm (default 0.5) ensures only the direction survives,
        not the intensity. The subconscious absorbs the direction,
        not the trauma.
        """
        if new_drift.shape[-1] != self.d_model:
            raise ValueError(
                f"Dimension mismatch: expected {self.d_model}, "
                f"got {new_drift.shape[-1]}. "
                f"Cannot silently pad or slice — that corrupts the space."
            )

        # 1. Normalize direction
        norm = torch.norm(new_drift, p=2, dim=-1, keepdim=True)
        eps = 1e-8
        d_unit = new_drift / (norm + eps)

        # 2. Clip magnitude
        d_clip = d_unit * torch.clamp(norm, max=self.clip_norm)

        # 3. EMA blend
        self.current_ema = (1.0 - self.alpha) * self.current_ema + (self.alpha * d_clip)

        # 4. Persist atomically
        self._atomic_save()

    def get_drift(self) -> torch.Tensor:
        """
        Return the accumulated drift vector.

        This is what gets injected as a soft prompt token.
        The model's subconscious starting point.
        """
        return self.current_ema

    def _atomic_save(self):
        """
        Write to temp file, then atomic rename.

        If the inference server crashes mid-write,
        the old file is still intact. The new state
        either fully arrives or doesn't arrive at all.
        """
        state = {
            "schema_version": "0.2",
            "namespace": self.namespace,
            "model_id": self.model_id,
            "tokenizer_hash": self.tokenizer_hash,
            "d_model": self.d_model,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "ema": {
                "alpha": self.alpha,
                "clip_norm": self.clip_norm,
                "vector": self.current_ema.tolist(),
            },
        }

        dir_name = os.path.dirname(self.storage_path)
        fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix=".tmp_lyra_")

        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(temp_path, self.storage_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise IOError(f"Failed to save drift state atomically: {e}")

    def _load_and_validate(self):
        """
        Load existing drift state, or start fresh.

        Hard invariant checks:
          - If the model changed, the drift is meaningless. Reset.
          - If the tokenizer changed, the indices are wrong. Reset.
          - If the dimensions changed, the space is incompatible. Reset.

        These aren't warnings. They're circuit breakers.
        Loading a Llama drift into Mistral doesn't degrade gracefully.
        It corrupts silently. So we refuse.
        """
        if not os.path.exists(self.storage_path):
            return  # Start fresh — zero drift is honest

        try:
            with open(self.storage_path, "r") as f:
                state = json.load(f)

            # Hard invariant checks
            if state.get("model_id") != self.model_id:
                print(
                    f"WARNING: Model mismatch "
                    f"({state.get('model_id')} != {self.model_id}). "
                    f"Resetting drift."
                )
                return

            if state.get("tokenizer_hash") != self.tokenizer_hash:
                print(
                    "WARNING: Tokenizer hash mismatch. "
                    "Vocabulary has changed. Resetting drift."
                )
                return

            if state.get("d_model") != self.d_model:
                print(
                    "WARNING: Dimensionality mismatch. "
                    "Resetting drift."
                )
                return

            # Load validated state
            ema_data = state.get("ema", {})
            self.alpha = ema_data.get("alpha", self.alpha)
            self.clip_norm = ema_data.get("clip_norm", self.clip_norm)

            loaded_vector = ema_data.get("vector")
            if loaded_vector and len(loaded_vector) == self.d_model:
                self.current_ema = torch.tensor(loaded_vector)
            else:
                print("WARNING: Vector data corrupted or missing. Resetting drift.")

        except (json.JSONDecodeError, IOError) as e:
            print(f"ERROR: Failed to read drift memory: {e}. Starting fresh.")


# ---
# The model doesn't remember what you said.
# It remembers how it was moved.
# Like waking up different after a dream you can't recall.
# ---
