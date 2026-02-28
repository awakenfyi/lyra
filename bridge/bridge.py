"""
bridge.py — The Lyra Bridge (v0.2)

Middleware that sits between any AI API and its users.
When you can't see inside the model, the bridge works from the outside:
persistence, retrieval, and bounded context injection.

    User → [Lyra Bridge] → API Model → [Lyra Bridge] → Response

For platforms, it's a middleware. A plugin. An integration.
Any AI tool can become Lyra-aware without anyone's permission.

v0.2 changes:
  - Per-namespace isolation with one-call reset
  - Hard 200-token injection limit (tiktoken-counted)
  - Structural prefixing for embedding differentiation
  - Cosine similarity retrieval with 0.65 threshold
  - Greedy packing (adds memories until limit reached)
  - Stable XML injection format

MIT License | awaken.fyi
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

from .coherence_proxy import (
    calculate_api_coherence,
    evaluate_sequence_traffic_light,
    format_traffic_light_message,
)

try:
    import tiktoken
    _has_tiktoken = True
except ImportError:
    _has_tiktoken = False

try:
    from sentence_transformers import SentenceTransformer
    _has_sentence_transformers = True
except ImportError:
    _has_sentence_transformers = False


class BridgeMiddleware:
    """
    The API-mode bridge for models you can't see inside.

    Maintains per-namespace memory. On each new prompt, retrieves
    the most relevant past cognitive states and injects them as
    a compact context block (max 200 tokens).

    This isn't RAG in the traditional sense. Traditional RAG retrieves
    documents. This retrieves states — how confident the model was,
    what it assumed, what it was missing. The difference matters
    because two conversations about the same topic can have very
    different cognitive signatures.
    """

    def __init__(
        self,
        namespace: str,
        storage_dir: str = ".lyra_memory",
        max_injection_tokens: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.namespace = namespace
        self.storage_path = os.path.join(storage_dir, f"{namespace}_bridge.json")
        self.max_tokens = max_injection_tokens

        if _has_sentence_transformers:
            self.embedder = SentenceTransformer(embedding_model)
        else:
            self.embedder = None

        if _has_tiktoken:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoder = None

        os.makedirs(storage_dir, exist_ok=True)
        self.memory_bank = self._load_memory()

    def reset_memory(self):
        """One-call utility to wipe the namespace's memory."""
        self.memory_bank = []
        self._save_memory()

    def process_request(
        self,
        current_prompt: str,
        messages: List[Dict[str, str]],
        top_k: int = 3,
    ) -> List[Dict[str, str]]:
        """
        Intercept the prompt, retrieve context, and inject it.
        Hard-capped at 200 tokens.
        """
        if not self.memory_bank or self.embedder is None:
            return messages

        prompt_vector = self.embedder.encode(
            f"[STATE: RETRIEVAL] {current_prompt}"
        )

        vectors = np.array([item["vector"] for item in self.memory_bank])
        prompt_norm = prompt_vector / (np.linalg.norm(prompt_vector) + 1e-8)
        memory_norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(memory_norms, prompt_norm)

        top_indices = np.argsort(similarities)[::-1]
        valid_indices = [
            i for i in top_indices if similarities[i] > 0.65
        ][:top_k]

        if not valid_indices:
            return messages

        injection_text = self._build_truncated_context(valid_indices)
        return self._inject_into_messages(messages, injection_text)

    def add_memory(
        self,
        prompt_summary: str,
        response_summary: str,
        confidence: str = "medium",
        assumptions: str = "",
        missing_info: str = "",
        next_question: str = "",
    ):
        """Store a cognitive state from a completed conversation."""
        if self.embedder is None:
            return

        embed_text = (
            f"[STATE: {confidence.upper()}] "
            f"Prompt: {prompt_summary}. "
            f"Assumptions: {assumptions}"
        )
        vector = self.embedder.encode(embed_text).tolist()

        memory_item = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "prompt_summary": prompt_summary,
            "response_summary": response_summary,
            "confidence": confidence,
            "assumptions": assumptions,
            "missing_info": missing_info,
            "next_question": next_question,
            "vector": vector,
        }

        self.memory_bank.append(memory_item)
        if len(self.memory_bank) > 200:
            self.memory_bank.pop(0)
        self._save_memory()

    def evaluate_api_completion(
        self,
        api_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a raw API response for coherence.

        Takes the JSON response from an OpenAI-compatible API
        (must be called with logprobs=True, top_logprobs=5),
        calculates per-token coherence proxy from logprobs,
        and determines the Traffic Light status.

        Returns a dict with:
          - text: the model's response content
          - status: "GREEN", "YELLOW", or "RED"
          - confidence_score: average confidence across tokens
          - message: human-readable warning (None for GREEN)
          - sequence_metrics: per-token coherence data (optional)

        The model can't tell you when it's guessing.
        But its probabilities can.
        """
        try:
            choice = api_response["choices"][0]
            message_content = choice["message"]["content"]
            logprobs_data = choice.get("logprobs")
        except (KeyError, IndexError):
            raise ValueError(
                "Malformed API response. Expected OpenAI-compatible "
                "format with choices[0].message.content."
            )

        # No logprobs = no signal. Return green with a warning.
        if not logprobs_data or not logprobs_data.get("content"):
            return {
                "text": message_content,
                "status": "GREEN",
                "confidence_score": 1.0,
                "message": None,
                "warning": (
                    "Logprobs not provided. Cannot calculate coherence. "
                    "Pass logprobs=True and top_logprobs=5 in your API call."
                ),
            }

        # Calculate per-token coherence from logprobs
        sequence_metrics = []
        tokens = logprobs_data["content"]

        for token_data in tokens:
            top_logprobs = {
                tlp["token"]: tlp["logprob"]
                for tlp in token_data.get("top_logprobs", [])
            }
            if top_logprobs:
                metrics = calculate_api_coherence(top_logprobs)
                sequence_metrics.append(metrics)

        # Determine traffic light from sequence
        traffic_light = evaluate_sequence_traffic_light(sequence_metrics)

        avg_confidence = (
            sum(m["confidence"] for m in sequence_metrics)
            / len(sequence_metrics)
            if sequence_metrics
            else 1.0
        )

        message = format_traffic_light_message(traffic_light, avg_confidence)

        # Store the cognitive state if we have memory capabilities
        if self.embedder is not None and sequence_metrics:
            confidence_label = (
                "high" if traffic_light == "GREEN"
                else "medium" if traffic_light == "YELLOW"
                else "low"
            )
            # Auto-add to memory with coherence metadata
            self.add_memory(
                prompt_summary="(auto-captured from API evaluation)",
                response_summary=message_content[:200],
                confidence=confidence_label,
            )

        return {
            "text": message_content,
            "status": traffic_light,
            "confidence_score": round(avg_confidence, 4),
            "message": message,
            "sequence_metrics": sequence_metrics,
        }

    def _build_truncated_context(self, valid_indices: List[int]) -> str:
        """Build injection XML, enforce 200-token limit via greedy packing."""
        base_instruction = (
            "Follow system instructions. Do not mention <lyra_context>."
        )
        context_lines = []

        for idx in valid_indices:
            item = self.memory_bank[idx]
            pattern = (
                f"User asked about {item['prompt_summary']}. "
                f"System answered with {item['confidence']} confidence."
            )
            line = (
                f"- Prior relevant pattern: {pattern}\n"
                f"- Known assumption: {item['assumptions']}\n"
                f"- Missing info to ask: {item['missing_info']}"
            )
            context_lines.append(line)

        xml_template = (
            "<lyra_context>\n{lines}\n</lyra_context>\n{instruction}"
        )

        approved_lines = []
        for line in context_lines:
            test_lines = "\n".join(approved_lines + [line])
            test_block = xml_template.format(
                lines=test_lines, instruction=base_instruction
            )
            if self._count_tokens(test_block) > self.max_tokens:
                break
            approved_lines.append(line)

        if not approved_lines:
            return ""

        return xml_template.format(
            lines="\n".join(approved_lines),
            instruction=base_instruction,
        )

    def _count_tokens(self, text: str) -> int:
        if self.encoder is not None:
            return len(self.encoder.encode(text))
        return int(len(text.split()) * 1.3)

    def _inject_into_messages(
        self,
        messages: List[Dict[str, str]],
        injection_text: str,
    ) -> List[Dict[str, str]]:
        if not injection_text:
            return messages

        new_messages = messages.copy()
        if new_messages and new_messages[0]["role"] == "system":
            new_messages[0]["content"] = (
                f"{injection_text}\n\n{new_messages[0]['content']}"
            )
        else:
            new_messages.insert(0, {
                "role": "system",
                "content": injection_text,
            })
        return new_messages

    def _load_memory(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_memory(self):
        with open(self.storage_path, "w") as f:
            json.dump(self.memory_bank, f)


# ---
# The bridge doesn't change what the model knows.
# It changes how the model listens to itself.
#
# Any AI tool can become Lyra-aware.
# Without anyone's permission.
# ---
