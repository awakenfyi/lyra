"""
Tests for lyra/drift.py — EMA clipping, atomic writes, model mismatch protection.
"""

import os
import json
import pytest
import torch
from lyra.drift import DriftStore


@pytest.fixture
def store_params(tmp_path):
    """Provides a fresh, isolated configuration for each test."""
    return {
        "namespace": "test_user_01",
        "model_id": "meta-llama/Meta-Llama-3-8B",
        "d_model": 64,  # Small dimension for fast testing
        "tokenizer_hash": "sha256:dummyhash123",
        "storage_dir": str(tmp_path / ".lyra_memory"),
        "alpha": 0.1,
        "clip_norm": 0.5,
    }


@pytest.fixture
def store(store_params):
    return DriftStore(**store_params)


def test_initialization_is_zero(store, store_params):
    """A new namespace should initialize with absolute zero drift."""
    drift = store.get_drift()
    assert torch.all(drift == 0.0), "Initial drift vector is not zeroed."
    assert drift.shape[-1] == store_params["d_model"]


def test_ema_clipping_bounds(store):
    """An extreme outlier shift must be clipped so it cannot hijack the memory."""
    extreme_vector = torch.ones(store.d_model) * 100.0

    store.update_ema(extreme_vector)
    updated_drift = store.get_drift()

    # Max magnitude from one update: alpha * clip_norm = 0.1 * 0.5 = 0.05
    resulting_magnitude = torch.norm(updated_drift, p=2, dim=-1).item()

    assert resulting_magnitude <= (store.alpha * store.clip_norm) + 1e-5, (
        f"EMA clipping failed. Magnitude {resulting_magnitude} exceeds bounds."
    )


def test_atomic_save_and_reload(store, store_params):
    """State must persist correctly across instances."""
    vector = torch.randn(store.d_model)
    store.update_ema(vector)
    saved_drift = store.get_drift()

    # Re-instantiate pointing to same directory
    reloaded_store = DriftStore(**store_params)
    reloaded_drift = reloaded_store.get_drift()

    assert torch.allclose(saved_drift, reloaded_drift, atol=1e-6), (
        "Reloaded drift vector does not match saved state."
    )


def test_model_mismatch_resets_drift(store, store_params):
    """If the model changes, old drift vectors must be zeroed."""
    vector = torch.randn(store.d_model)
    store.update_ema(vector)

    # Tamper with JSON to simulate model swap
    with open(store.storage_path, "r") as f:
        state = json.load(f)
    state["model_id"] = "mistralai/Mistral-7B-v0.1"
    with open(store.storage_path, "w") as f:
        json.dump(state, f)

    # Reload — mismatch should trigger hard reset
    reloaded_store = DriftStore(**store_params)
    assert torch.all(reloaded_store.get_drift() == 0.0), (
        "Store failed to reset drift after model_id mismatch."
    )


def test_tokenizer_hash_mismatch_resets_drift(store, store_params):
    """If the tokenizer vocabulary changes, old indices are invalid."""
    vector = torch.randn(store.d_model)
    store.update_ema(vector)

    with open(store.storage_path, "r") as f:
        state = json.load(f)
    state["tokenizer_hash"] = "sha256:different_hash_entirely"
    with open(store.storage_path, "w") as f:
        json.dump(state, f)

    reloaded_store = DriftStore(**store_params)
    assert torch.all(reloaded_store.get_drift() == 0.0), (
        "Store failed to reset drift after tokenizer_hash mismatch."
    )


def test_dimension_mismatch_raises_error(store):
    """Passing a vector with wrong d_model should crash, not silently fail."""
    wrong_dim_vector = torch.randn(store.d_model + 10)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        store.update_ema(wrong_dim_vector)


def test_multiple_updates_converge(store):
    """Repeated updates with the same direction should converge, not explode."""
    direction = torch.randn(store.d_model)

    magnitudes = []
    for _ in range(100):
        store.update_ema(direction)
        mag = torch.norm(store.get_drift(), p=2).item()
        magnitudes.append(mag)

    # Should converge to a stable value, not grow unbounded
    assert magnitudes[-1] < 10.0, (
        f"EMA magnitude {magnitudes[-1]} after 100 updates — not converging."
    )
    # Last 10 values should be close to each other
    recent = magnitudes[-10:]
    assert max(recent) - min(recent) < 0.01, (
        "EMA not converging after 100 identical updates."
    )
