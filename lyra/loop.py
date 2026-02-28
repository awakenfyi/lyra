"""
loop.py — Soft Prompt Injection (v0.2)

Before each conversation, takes the accumulated drift vector
and injects it as a virtual token at the start of the sequence.
The model attends to this token without knowing it's there.

Like mood coloring a room before anyone speaks.

v0.2 changes:
  - Soft prompt injection replaces global additive offset
  - No more silent pad/slice on dimension mismatch
  - Works with inputs_embeds during prefill, input_ids during decode
  - Attention mask and position handling

MIT License | awaken.fyi
"""

import torch
from typing import Tuple, Optional


def prepare_soft_prompt_inputs(
    model,
    input_ids: torch.Tensor,
    drift_vector: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepend the accumulated drift vector as a virtual token at position 0.

    This bypasses the model's standard embedding layer for the first token,
    injecting the subconscious directly into the embedding space.
    The model attends to it like any other token — but it's not a word.
    It's a feeling.

    During the prefill phase (step 0), pass the returned inputs_embeds
    and attention_mask directly to the model. For all subsequent decode
    steps, revert to standard input_ids — the virtual token's influence
    is already captured in the KV cache.

    Args:
        model:        The transformer model (must have get_input_embeddings()).
        input_ids:    Tokenized prompt. Shape: [batch, seq_len]
        drift_vector: The EMA drift from DriftStore. Shape: [d_model]

    Returns:
        (inputs_embeds, attention_mask) tuple ready for model forward pass.
    """
    # Convert standard tokens to dense embeddings
    token_embeds = model.get_input_embeddings()(input_ids)

    # Reshape drift vector to act as a single virtual token
    # Shape: [batch_size, 1, d_model]
    virtual_token = drift_vector.view(1, 1, -1).expand(input_ids.shape[0], -1, -1)

    # Ensure device and dtype match
    virtual_token = virtual_token.to(
        device=token_embeds.device,
        dtype=token_embeds.dtype,
    )

    # Concatenate: virtual token goes first, then the real prompt
    inputs_embeds = torch.cat([virtual_token, token_embeds], dim=1)

    # Attention mask: all 1s, length = original + 1 for the virtual token
    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        dtype=torch.long,
        device=inputs_embeds.device,
    )

    return inputs_embeds, attention_mask


def validate_drift_for_model(
    drift_vector: torch.Tensor,
    model,
) -> bool:
    """
    Check that the drift vector is compatible with the model.

    If dimensions don't match, the offset would corrupt the space.
    Better to skip it entirely than to silently truncate or pad.

    Args:
        drift_vector: The EMA drift vector.
        model:        The transformer model.

    Returns:
        True if compatible, False if mismatch.
    """
    embedding_dim = model.get_input_embeddings().weight.shape[1]

    if drift_vector.shape[-1] != embedding_dim:
        print(
            f"WARNING: Drift dimension ({drift_vector.shape[-1]}) != "
            f"model embedding dimension ({embedding_dim}). "
            f"Disabling drift injection for this session."
        )
        return False

    # Check if drift is all zeros (no accumulated state yet)
    if torch.all(drift_vector == 0):
        return False  # Nothing to inject — skip cleanly

    return True


# ---
# The loop doesn't add information.
# It adds shape.
# The model wakes up in a room that remembers
# who was there last night.
# ---
