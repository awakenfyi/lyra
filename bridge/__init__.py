"""
bridge — API mode middleware for Lyra.

When you can't see inside the model (API calls to OpenAI, Anthropic, etc.),
the bridge maintains memory and injects context from the outside.

Phase 1: Memory layer — persistence and retrieval.
Phase 2: Confidence signal — coherence approximation from logprobs.

Less powerful than local mode. Still meaningful.
"""

from .bridge import BridgeMiddleware
from .coherence_proxy import (
    calculate_api_coherence,
    evaluate_sequence_traffic_light,
    format_traffic_light_message,
)

__all__ = [
    "BridgeMiddleware",
    "calculate_api_coherence",
    "evaluate_sequence_traffic_light",
    "format_traffic_light_message",
]
