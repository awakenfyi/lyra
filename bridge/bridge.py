"""
bridge.py — The Lyra Bridge

Middleware that sits between any AI tool and its model.
Makes any transformer Lyra-aware without changing the model.

    User → [Lyra Bridge] → Model → [Lyra Bridge] → Response

The bridge does three things:
  1. Feeds the drift offset into the model's initial state
  2. Reads the directional pull during generation
  3. Adjusts sampling based on coherence

For platforms, it's a middleware. A plugin. An integration.
Any AI tool can become Lyra-aware without anyone's permission.

MIT License | awaken.fyi
"""

import time
import json
from typing import Optional, Callable, Any
from dataclasses import dataclass, field

import numpy as np

from lyra.drift import compute_drift, DriftStore
from lyra.loop import LyraLoop
from lyra.coherence import CoherenceSampler, CoherenceSignal


@dataclass
class BridgeConfig:
    """Configuration for the Lyra Bridge."""
    # Drift settings
    drift_store_path: str = "./lyra_drift"
    history_window: int = 50

    # Loop settings
    offset_scale: float = 0.01       # how much past drift shapes the present

    # Coherence settings
    base_temperature: float = 0.7
    coherence_weight: float = 0.3    # how much coherence affects sampling
    silence_threshold: float = 0.2   # below this, model may choose brevity

    # Bridge behavior
    collect_hidden_states: bool = True
    log_coherence: bool = True
    log_path: str = "./lyra_logs"


@dataclass
class ConversationTrace:
    """Record of what happened in one conversation through the bridge."""
    conversation_id: str
    start_time: float
    end_time: float = 0.0
    token_count: int = 0
    mean_coherence: float = 0.0
    min_coherence: float = 1.0
    silence_permissions: int = 0        # times the model was invited to shorten
    silence_accepted: int = 0           # times it actually did
    coherence_over_time: list = field(default_factory=list)

class LyraBridge:
    """
    The bridge between any AI system and Lyra-aware inference.

    Usage with HuggingFace (local models):

        bridge = LyraBridge()
        bridge.start_conversation()

        # During generation, wrap the model's forward pass:
        for step in generation:
            signal = bridge.observe_step(
                layer_residuals=model.get_hidden_states(),
                output_logits=model.get_logits(),
                embedding_matrix=model.get_embedding_matrix(),
            )
            adjusted_logits = bridge.adjust(output_logits, signal)
            next_token = sample(adjusted_logits)

        bridge.end_conversation(hidden_states_start, hidden_states_end)

    Usage with API models (OpenAI, Anthropic, etc.):

        bridge = LyraBridge()
        bridge.start_conversation()

        # API models don't expose hidden states, so bridge works in
        # "text mode" — using the subconscious text block instead of
        # embedding offsets. Less powerful, but still meaningful.

        system_prompt = bridge.get_system_context()
        # Prepend to your API call's system prompt.

        bridge.end_conversation_text_mode(
            messages=conversation_messages,
        )
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()
        self.loop = LyraLoop(
            store_path=self.config.drift_store_path,
            history_window=self.config.history_window,
            offset_scale=self.config.offset_scale,
        )
        self.sampler = CoherenceSampler(
            base_temperature=self.config.base_temperature,
            coherence_weight=self.config.coherence_weight,
            silence_threshold=self.config.silence_threshold,
        )
        self._trace: Optional[ConversationTrace] = None
        self._conversation_id: str = ""

    def start_conversation(self, conversation_id: Optional[str] = None) -> None:
        """
        Begin a new conversation through the bridge.
        Loads accumulated drift and prepares the loop state.
        """
        import hashlib
        self._conversation_id = conversation_id or hashlib.md5(
            f"{time.time()}".encode()
        ).hexdigest()[:12]

        # Load the accumulated past into the loop
        self.loop.before_conversation()

        self._trace = ConversationTrace(
            conversation_id=self._conversation_id,
            start_time=time.time(),
        )

    def get_embedding_offset(self) -> Optional[np.ndarray]:
        """
        Get the embedding offset to apply to the model's initial embeddings.
        For local models that expose embeddings.
        """
        if self.loop._state is None:
            return None
        return self.loop._state.embedding_offset

    def apply_to_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply the accumulated drift offset to token embeddings.
        Call this on the initial embeddings before the first forward pass.
        """
        return self.loop.apply_offset(embeddings)

    def get_system_context(self) -> Optional[str]:
        """
        For API models: get a text block to prepend to the system prompt.
        This is the text-mode fallback for models that don't expose internals.
        """
        return self.loop.get_subconscious_text()
    def observe_step(
        self,
        layer_residuals: np.ndarray,
        output_logits: np.ndarray,
        embedding_matrix: Optional[np.ndarray] = None,
    ) -> CoherenceSignal:
        """
        Observe one generation step. Compute coherence.
        Call this at each token generation step.

        Returns a CoherenceSignal that can be used to adjust sampling.
        """
        signal = self.sampler.compute_coherence(
            layer_residuals=layer_residuals,
            output_logits=output_logits,
            embedding_matrix=embedding_matrix,
        )

        # Track coherence over time
        if self._trace is not None:
            self._trace.token_count += 1
            self._trace.coherence_over_time.append(signal.coherence_score)
            self._trace.mean_coherence = (
                sum(self._trace.coherence_over_time) / len(self._trace.coherence_over_time)
            )
            self._trace.min_coherence = min(
                self._trace.min_coherence, signal.coherence_score
            )
            if self.sampler.should_shorten(signal):
                self._trace.silence_permissions += 1

        return signal

    def adjust(
        self,
        output_logits: np.ndarray,
        signal: CoherenceSignal,
        eos_token_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Adjust output logits based on coherence signal.
        Returns modified logits ready for sampling.
        """
        return self.sampler.adjust_logits(output_logits, signal, eos_token_id)

    def end_conversation(
        self,
        hidden_states_start: np.ndarray,
        hidden_states_end: np.ndarray,
        token_logprobs: Optional[np.ndarray] = None,
        token_entropies: Optional[np.ndarray] = None,
    ) -> ConversationTrace:
        """
        End the conversation. Compute and commit drift.
        For local models with access to hidden states.
        """
        # Compute drift
        signature = compute_drift(
            hidden_states_start=hidden_states_start,
            hidden_states_end=hidden_states_end,
            token_logprobs=token_logprobs,
            token_entropies=token_entropies,
            conversation_id=self._conversation_id,
        )

        # Commit to the store
        self.loop.after_conversation(signature)

        # Finalize trace
        if self._trace:
            self._trace.end_time = time.time()

        return self._trace

    def end_conversation_text_mode(
        self,
        messages: Optional[list] = None,
    ) -> ConversationTrace:
        """
        End the conversation in text mode (for API models).
        Without hidden states, we can still track conversation-level patterns.
        """
        if self._trace:
            self._trace.end_time = time.time()

        # In text mode, we can't compute real drift.
        # But we log the trace for the subconscious to evolve.
        return self._trace


# --- Quick start helpers ---

def create_bridge(
    store_path: str = "./lyra_drift",
    coherence_weight: float = 0.3,
) -> LyraBridge:
    """Create a bridge with sensible defaults."""
    return LyraBridge(BridgeConfig(
        drift_store_path=store_path,
        coherence_weight=coherence_weight,
    ))


# ---
# The bridge doesn't change what the model knows.
# It changes how the model listens to itself.
#
# Any AI tool can become Lyra-aware.
# Without anyone's permission.
# ---
