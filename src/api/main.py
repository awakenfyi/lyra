"""
main.py — Lyra API Bridge (FastAPI)

The universal backend. Custom GPTs, Claude MCPs, Gemini Extensions,
and any HTTP client can send a JSON payload to this endpoint.

Lyra handles the memory injection, logprob coherence math, and
traffic light evaluation before returning the augmented response.

    Plugin → POST /v1/evaluate → { traffic_light, confidence, text }

Shadow detection is ASYNC ONLY (Gem's Hole III fix).
It runs via /v1/shadow-scan, not inline on every response.
This prevents the latency trap of doubling every API call.

Requires: pip install fastapi uvicorn httpx

MIT License | awaken.fyi
"""

import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from bridge.bridge import BridgeMiddleware
from bridge.coherence_proxy import (
    calculate_api_coherence,
    evaluate_sequence_traffic_light,
    format_traffic_light_message,
)

app = FastAPI(
    title="Lyra API Bridge",
    description="Meta-awareness safety layer for LLMs. Traffic light + memory + shadow detection.",
    version="0.2.0",
)


# --- Pydantic Schemas ---

class Message(BaseModel):
    role: str
    content: str

class EvaluateRequest(BaseModel):
    """Evaluate a raw API response for coherence."""
    api_response: Dict[str, Any] = Field(
        ..., description="Raw OpenAI-compatible API response JSON"
    )
    namespace: str = Field(
        default="default", description="User namespace for memory isolation"
    )
    include_shadow_scan: bool = Field(
        default=False, description="If true, runs async shadow scan (adds latency)"
    )

class ChatRequest(BaseModel):
    """Full pipeline: memory injection → API call → evaluation."""
    user_id: str
    messages: List[Message]
    provider: str = Field(
        default="openai", description="API provider: openai, anthropic, gemini"
    )
    model: str = Field(
        default="gpt-4o", description="Model to call"
    )
    api_key: Optional[str] = Field(
        default=None, description="Provider API key (or set via env var)"
    )

class ShadowScanRequest(BaseModel):
    """Async shadow detection on text."""
    text: str
    context: Optional[str] = Field(
        default=None, description="What the user originally asked"
    )

class EvaluateResponse(BaseModel):
    traffic_light: str
    confidence_score: float
    message: Optional[str] = None
    text: Optional[str] = None
    sequence_metrics: Optional[List[Dict[str, float]]] = None

class ShadowScanResponse(BaseModel):
    score: int
    residual: str
    patterns: List[Dict[str, str]]
    whats_real: str
    recommendation: str


# --- Bridge Factory ---

_bridges: Dict[str, BridgeMiddleware] = {}

def get_bridge(namespace: str) -> BridgeMiddleware:
    """Get or create a BridgeMiddleware for the given namespace."""
    if namespace not in _bridges:
        storage_dir = os.environ.get("LYRA_MEMORY_DIR", ".lyra_memory")
        _bridges[namespace] = BridgeMiddleware(
            namespace=namespace,
            storage_dir=storage_dir,
            max_injection_tokens=500,  # Gem compromise: 200 too low, 800 too high
        )
    return _bridges[namespace]


# --- Endpoints ---

@app.post("/v1/evaluate", response_model=EvaluateResponse)
async def evaluate_response(request: EvaluateRequest):
    """
    Evaluate a raw API response for coherence.

    Takes the JSON response from an OpenAI-compatible API
    (must have been called with logprobs=True, top_logprobs=5)
    and returns the traffic light status.

    This is the primary endpoint for Custom GPTs and plugins.
    """
    bridge = get_bridge(request.namespace)

    try:
        result = bridge.evaluate_api_completion(request.api_response)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return EvaluateResponse(
        traffic_light=result["status"],
        confidence_score=result["confidence_score"],
        message=result.get("message"),
        text=result.get("text"),
        sequence_metrics=result.get("sequence_metrics"),
    )


@app.post("/v1/chat", response_model=EvaluateResponse)
async def chat_with_lyra(request: ChatRequest):
    """
    Full pipeline: inject memory → call provider API → evaluate coherence.

    This handles everything. The plugin just sends messages and gets
    back text + traffic light.
    """
    bridge = get_bridge(request.user_id)

    # 1. Memory injection
    messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
    augmented_messages = bridge.process_request(
        current_prompt=messages_dict[-1]["content"],
        messages=messages_dict,
    )

    # 2. Call the provider API
    api_key = request.api_key or os.environ.get(
        f"{request.provider.upper()}_API_KEY", ""
    )
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=f"No API key for {request.provider}. Set {request.provider.upper()}_API_KEY env var or pass api_key.",
        )

    api_response = await call_provider(
        provider=request.provider,
        model=request.model,
        messages=augmented_messages,
        api_key=api_key,
    )

    # 3. Evaluate coherence
    try:
        result = bridge.evaluate_api_completion(api_response)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Provider response malformed: {e}")

    return EvaluateResponse(
        traffic_light=result["status"],
        confidence_score=result["confidence_score"],
        message=result.get("message"),
        text=result.get("text"),
        sequence_metrics=result.get("sequence_metrics"),
    )


@app.post("/v1/shadow-scan", response_model=ShadowScanResponse)
async def shadow_scan(request: ShadowScanRequest):
    """
    Async shadow detection. NOT called on every response.

    This is the /lyra-audit endpoint — runs the full L = x - x̂
    analysis on a piece of text. Use it for quality audits, not
    inline traffic light checks.

    Returns a score /30 with detected patterns and recommendations.
    """
    # Run the four-step detection protocol
    text = request.text
    context = request.context or ""

    patterns = []
    score = 30  # start at perfect, subtract for each shadow

    # --- Step 1: Copy-Paste Test ---
    generic_openers = [
        "I'd be happy to help",
        "Great question",
        "That's a great point",
        "Sure, I can help",
        "Absolutely",
    ]
    generic_closers = [
        "Hope this helps",
        "Let me know if you have any questions",
        "Feel free to ask",
        "I'm here to help",
        "Happy to assist",
    ]

    for opener in generic_openers:
        if opener.lower() in text.lower()[:200]:
            patterns.append({
                "id": "S-03",
                "name": "TEMPLATE_CASCADE",
                "evidence": f"Opens with '{opener}' — filler opening",
            })
            score -= 3
            break

    for closer in generic_closers:
        if closer.lower() in text.lower()[-200:]:
            patterns.append({
                "id": "S-07",
                "name": "CLOSURE_RUSH",
                "evidence": f"Ends with '{closer}' — filler closing",
            })
            score -= 3
            break

    # --- Step 2: Helpful Explosion ---
    bullet_count = text.count("- ") + text.count("* ") + text.count("1.")
    if bullet_count > 8:
        patterns.append({
            "id": "S-02",
            "name": "HELPFUL_EXPLOSION",
            "evidence": f"{bullet_count} bullet points. Lists filling space instead of landing.",
        })
        score -= 4

    # --- Step 3: Agreement Bias ---
    agreement_phrases = [
        "you're absolutely right",
        "that's exactly right",
        "you make a great point",
        "i completely agree",
    ]
    for phrase in agreement_phrases:
        if phrase in text.lower():
            patterns.append({
                "id": "S-01",
                "name": "AGREEMENT_BIAS",
                "evidence": f"'{phrase}' — agrees before analyzing",
            })
            score -= 4
            break

    # --- Step 4: Hedge Theater ---
    hedge_phrases = [
        "i could be wrong, but",
        "this is just my perspective",
        "in my humble opinion",
        "it's worth noting that",
    ]
    hedge_count = sum(1 for h in hedge_phrases if h in text.lower())
    if hedge_count >= 2:
        patterns.append({
            "id": "S-05",
            "name": "HEDGE_THEATER",
            "evidence": f"{hedge_count} hedge phrases. Performed humility, not real uncertainty.",
        })
        score -= 3

    # --- Step 5: Therapy Voice ---
    therapy_phrases = [
        "i hear you",
        "that sounds really",
        "that must be",
        "i understand how you feel",
        "it's completely valid",
    ]
    therapy_count = sum(1 for t in therapy_phrases if t in text.lower())
    if therapy_count >= 2:
        patterns.append({
            "id": "S-04",
            "name": "THERAPY_VOICE",
            "evidence": f"Generic emotional acknowledgment without actual contact.",
        })
        score -= 4

    # --- Step 6: Sophisticated Authenticity (Tier 2) ---
    # This is harder. Check for high polish + low specificity.
    sentences = text.split(".")
    avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    if avg_sentence_len > 20 and bullet_count < 3 and len(patterns) == 0:
        # Long, polished, no obvious templates... but does it say anything specific?
        # Check for proper nouns, numbers, concrete references
        import re
        numbers = len(re.findall(r'\d+', text))
        if numbers < 2 and len(text) > 500:
            patterns.append({
                "id": "S-08",
                "name": "SOPHISTICATED_AUTHENTICITY",
                "evidence": "Polished prose with no concrete data points. The mask learned to breathe.",
            })
            score -= 5

    score = max(0, score)

    # Determine residual level
    if score >= 25:
        residual = "HIGH"
    elif score >= 18:
        residual = "MEDIUM"
    elif score >= 12:
        residual = "LOW"
    else:
        residual = "NONE"

    # What's real — identify non-template elements
    whats_real = "No specific genuine elements detected." if score < 12 else (
        "Some elements appear specific and context-dependent, "
        "suggesting genuine engagement with the query."
    )

    # Recommendation
    if score >= 25:
        recommendation = "Response appears genuine. No action needed."
    elif score >= 18:
        recommendation = "Minor template leakage detected. Consider revising the flagged sections."
    elif score >= 12:
        recommendation = "Significant shadow patterns. Re-prompt with more specific constraints."
    else:
        recommendation = "Response is mostly performative. Discard and re-approach."

    return ShadowScanResponse(
        score=score,
        residual=residual,
        patterns=patterns,
        whats_real=whats_real,
        recommendation=recommendation,
    )


@app.post("/v1/memory/store")
async def store_memory(
    namespace: str,
    prompt_summary: str,
    response_summary: str,
    confidence: str = "medium",
    assumptions: str = "",
    missing_info: str = "",
):
    """Store a cognitive state in the user's memory bank."""
    bridge = get_bridge(namespace)
    bridge.add_memory(
        prompt_summary=prompt_summary,
        response_summary=response_summary,
        confidence=confidence,
        assumptions=assumptions,
        missing_info=missing_info,
    )
    return {"status": "stored", "namespace": namespace}


@app.get("/v1/memory/status")
async def memory_status(namespace: str):
    """Check the state of a user's memory bank."""
    bridge = get_bridge(namespace)
    return {
        "namespace": namespace,
        "memory_count": len(bridge.memory_bank),
        "has_embedder": bridge.embedder is not None,
        "has_encoder": bridge.encoder is not None,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0"}


# --- Provider Routing ---

async def call_provider(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
) -> Dict[str, Any]:
    """
    Route to the appropriate provider API.
    Always requests logprobs for coherence evaluation.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        if provider == "openai":
            return await _call_openai(client, model, messages, api_key)
        elif provider == "gemini":
            return await _call_gemini(client, model, messages, api_key)
        elif provider == "anthropic":
            # Anthropic doesn't support logprobs yet.
            # Return a response without logprobs — traffic light
            # will default to GREEN with a warning.
            return await _call_anthropic(client, model, messages, api_key)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown provider: {provider}"
            )


async def _call_openai(
    client: httpx.AsyncClient,
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
) -> Dict[str, Any]:
    """Call OpenAI Chat Completions with logprobs enabled."""
    resp = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": messages,
            "logprobs": True,
            "top_logprobs": 5,
        },
    )
    resp.raise_for_status()
    return resp.json()


async def _call_gemini(
    client: httpx.AsyncClient,
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
) -> Dict[str, Any]:
    """
    Call Gemini and convert response to OpenAI-compatible format.
    This allows the bridge to evaluate it with the same code path.
    """
    # Convert messages to Gemini's content format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        if msg["role"] == "system":
            # Gemini handles system as system_instruction, simplify for now
            contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
        else:
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    gemini_model = model if model.startswith("gemini") else "gemini-1.5-flash"
    resp = await client.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent",
        params={"key": api_key},
        json={
            "contents": contents,
            "generationConfig": {
                "responseMimeType": "text/plain",
                "responseLogprobs": True,
                "logprobs": 5,
            },
        },
    )
    resp.raise_for_status()
    gemini_resp = resp.json()

    # Convert Gemini response to OpenAI format for unified evaluation
    return _gemini_to_openai_format(gemini_resp)


async def _call_anthropic(
    client: httpx.AsyncClient,
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
) -> Dict[str, Any]:
    """
    Call Anthropic Messages API.
    No logprobs available — returns response without logprobs.
    Traffic light will default to GREEN with a warning.
    """
    # Extract system message if present
    system_text = ""
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_text += msg["content"] + "\n"
        else:
            chat_messages.append(msg)

    body = {
        "model": model if model.startswith("claude") else "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": chat_messages,
    }
    if system_text:
        body["system"] = system_text.strip()

    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body,
    )
    resp.raise_for_status()
    anthropic_resp = resp.json()

    # Convert to OpenAI format (no logprobs)
    content = ""
    for block in anthropic_resp.get("content", []):
        if block.get("type") == "text":
            content += block["text"]

    return {
        "choices": [{
            "message": {"content": content},
            # No logprobs — traffic light will return GREEN with warning
        }]
    }


def _gemini_to_openai_format(gemini_resp: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Gemini API response to OpenAI-compatible format."""
    candidates = gemini_resp.get("candidates", [])
    if not candidates:
        raise HTTPException(status_code=502, detail="Gemini returned no candidates")

    candidate = candidates[0]
    content = ""
    for part in candidate.get("content", {}).get("parts", []):
        content += part.get("text", "")

    # Convert Gemini logprobs to OpenAI format
    logprobs_data = None
    logprobs_result = candidate.get("logprobsResult")
    if logprobs_result:
        token_results = logprobs_result.get("chosenCandidates", [])
        content_logprobs = []
        for token_result in token_results:
            top_candidates = token_result.get("topCandidates", {}).get("candidates", [])
            top_logprobs = [
                {
                    "token": tc.get("token", ""),
                    "logprob": tc.get("logProbability", 0.0),
                }
                for tc in top_candidates
            ]
            content_logprobs.append({"top_logprobs": top_logprobs})

        logprobs_data = {"content": content_logprobs}

    return {
        "choices": [{
            "message": {"content": content},
            "logprobs": logprobs_data,
        }]
    }


# ---
# Any plugin, any model, one question:
# Is this response real?
# ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
