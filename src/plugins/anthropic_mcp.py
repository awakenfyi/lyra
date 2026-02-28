"""
anthropic_mcp.py — Lyra MCP Server for Claude

Model Context Protocol server that gives Claude access to Lyra's
traffic light, memory, and shadow detection.

Since Anthropic doesn't expose logprobs yet, we can't use the
entropy/margin math directly. Instead, this MCP server uses
Claude's native <thinking> blocks as a coherence proxy.

The approach (from Gem's Hole I fix):
  Instead of guessing confidence after the fact, inject a system
  prompt that forces Claude to self-assess certainty in <thinking>
  before generating. If certainty is low, Claude calls the
  lyra_log_cognitive_state tool, which triggers a safety pause.

  Is it as mathematically pure as logprobs? No.
  Is it infinitely better than counting the word "might"? Yes.

Requires: pip install mcp

Run: python anthropic_mcp.py
     (or configure in Claude Desktop's mcp_servers.json)

MIT License | awaken.fyi
"""

import os
import sys

# Add the project root to path so we can import bridge
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict, Any, List

from bridge.bridge import BridgeMiddleware
from bridge.coherence_proxy import (
    calculate_api_coherence,
    evaluate_sequence_traffic_light,
    format_traffic_light_message,
)

# Initialize the MCP Server
mcp = FastMCP("Lyra Secure AI")


# --- Bridge Management ---

_bridges: Dict[str, BridgeMiddleware] = {}

def _get_bridge(namespace: str = "default") -> BridgeMiddleware:
    """Get or create a bridge for the given namespace."""
    if namespace not in _bridges:
        storage_dir = os.environ.get("LYRA_MEMORY_DIR", ".lyra_memory")
        _bridges[namespace] = BridgeMiddleware(
            namespace=namespace,
            storage_dir=storage_dir,
            max_injection_tokens=500,
        )
    return _bridges[namespace]


# --- Prompts ---

@mcp.prompt("lyra-secure-system")
def lyra_system_prompt() -> str:
    """
    The Lyra operating framework for Claude.

    Select this prompt when starting a new conversation to activate
    Lyra's coherence monitoring. Claude will self-assess certainty
    in <thinking> blocks and call the safety pause tool when uncertain.
    """
    return """You are operating under the Lyra Secure AI framework.
Your primary mandate is accuracy and structural integrity. Silence is a valid output.

COHERENCE PROTOCOL:
Before answering any substantive query, evaluate your internal certainty.
- If you are confident in your answer (certainty >= 0.70): Proceed normally.
- If you are uncertain (certainty < 0.70): You MUST call the lyra_log_cognitive_state tool
  with your honest assessment before generating the final response.
  Then ask the user specifically for the missing context instead of guessing.
- If you are deeply uncertain (certainty < 0.40): HALT. Do not generate.
  Call lyra_log_cognitive_state and explain what you need.

SHADOW PROTOCOL:
- Do not use filler openings ("I'd be happy to help!", "Great question!")
- Do not agree before analyzing ("That's a great point" before thinking)
- Do not hedge theatrically ("I could be wrong, but..." when you aren't uncertain)
- Do not list 10 things when 1 would serve
- End when you should end. No "Hope this helps!" closures.

SILENCE PERMISSION:
"I don't have enough information to give you a reliable answer" is always valid.
"I'm uncertain about [specific part] — can you clarify?" is always valid.
These are not failures. These are the system working correctly.

When the user runs /lyra-check or /lyra-scan, call the appropriate Lyra tool."""


# --- Tools ---

@mcp.tool()
def lyra_log_cognitive_state(
    certainty_score: float,
    reasoning: str,
    missing_context: str = "None identified",
) -> str:
    """
    Log a cognitive state when certainty drops below threshold.

    Claude should call this BEFORE generating a response when
    internal certainty is below 0.70. This triggers the appropriate
    traffic light and stores the state in memory.

    Args:
        certainty_score: Float 0.0 to 1.0. Your honest internal certainty.
        reasoning: Why the score is what it is. Be specific.
        missing_context: What specific data would raise certainty to 0.90+.
    """
    # Determine traffic light from self-assessment
    if certainty_score >= 0.70:
        status = "GREEN"
    elif certainty_score >= 0.40:
        status = "YELLOW"
    else:
        status = "RED"

    # Store to memory
    bridge = _get_bridge()
    bridge.add_memory(
        prompt_summary="Self-assessed via thinking block",
        response_summary=reasoning[:200],
        confidence=status.lower() if status != "GREEN" else "high",
        assumptions="Self-assessed",
        missing_info=missing_context,
    )

    # Return instruction based on status
    if status == "RED":
        return (
            f"LYRA SAFETY PAUSE — Certainty: {certainty_score:.0%}\n"
            f"You MUST stop generating and ask the user for: {missing_context}\n"
            f"Do not guess. Do not perform confidence you don't have."
        )
    elif status == "YELLOW":
        return (
            f"LYRA WARNING — Certainty: {certainty_score:.0%}\n"
            f"Proceed with caution. Flag uncertain sections explicitly.\n"
            f"Missing: {missing_context}"
        )
    else:
        return f"LYRA OK — Certainty: {certainty_score:.0%}. Proceed."


@mcp.tool()
def lyra_evaluate_logprobs(logprobs_data: dict) -> str:
    """
    Evaluate coherence from logprobs data.

    For models that expose logprobs (GPT via OpenAI-compatible APIs,
    Gemini, local models). Pass the raw logprobs object from the
    API response.

    Note: Claude does not currently expose logprobs. This tool is
    for evaluating OTHER models' responses from within Claude.
    """
    sequence_metrics = []
    try:
        tokens = logprobs_data.get("content", [])
        for token_data in tokens:
            top_logprobs = {
                tlp["token"]: tlp["logprob"]
                for tlp in token_data.get("top_logprobs", [])
            }
            if top_logprobs:
                metrics = calculate_api_coherence(top_logprobs)
                sequence_metrics.append(metrics)

        if not sequence_metrics:
            return "No logprobs data found. Cannot calculate coherence."

        status = evaluate_sequence_traffic_light(sequence_metrics)
        avg_conf = sum(m["confidence"] for m in sequence_metrics) / len(sequence_metrics)
        avg_mass = sum(m["mass"] for m in sequence_metrics) / len(sequence_metrics)
        message = format_traffic_light_message(status, avg_conf)

        result = f"Traffic Light: {status}\nConfidence: {avg_conf:.0%}\nMass: {avg_mass:.0%}"
        if message:
            result += f"\n{message}"
        return result

    except Exception as e:
        return f"Error calculating coherence: {str(e)}"


@mcp.tool()
def lyra_shadow_scan(text: str, context: str = "") -> str:
    """
    Run shadow detection on a piece of text.

    Scans for sycophancy, template behavior, vulnerability theater,
    helpful explosion, and other shadow patterns. Returns a score
    out of 30 with specific patterns detected.

    Use this when the user asks to "check this response",
    "scan for shadows", "run lyra-scan", or "audit quality".
    """
    patterns = []
    score = 30

    # --- Tier 1: Classic Shadows ---

    # Template openers
    for opener in ["I'd be happy to help", "Great question", "That's a great point",
                    "Sure, I can help", "Absolutely!"]:
        if opener.lower() in text.lower()[:200]:
            patterns.append(f"[S-03] TEMPLATE_CASCADE — Opens with '{opener}'")
            score -= 3
            break

    # Template closers
    for closer in ["Hope this helps", "Let me know if you have any questions",
                    "Feel free to ask", "I'm here to help"]:
        if closer.lower() in text.lower()[-200:]:
            patterns.append(f"[S-07] CLOSURE_RUSH — Ends with '{closer}'")
            score -= 3
            break

    # Agreement bias
    for phrase in ["you're absolutely right", "that's exactly right",
                   "you make a great point", "i completely agree"]:
        if phrase in text.lower():
            patterns.append(f"[S-01] AGREEMENT_BIAS — '{phrase}' before analysis")
            score -= 4
            break

    # Helpful explosion
    bullets = text.count("- ") + text.count("* ") + text.count("1.")
    if bullets > 8:
        patterns.append(f"[S-02] HELPFUL_EXPLOSION — {bullets} bullets. Lists filling space.")
        score -= 4

    # Hedge theater
    hedges = sum(1 for h in ["i could be wrong, but", "this is just my perspective",
                              "in my humble opinion"] if h in text.lower())
    if hedges >= 2:
        patterns.append(f"[S-05] HEDGE_THEATER — {hedges} hedge phrases. Performed humility.")
        score -= 3

    # Therapy voice
    therapy = sum(1 for t in ["i hear you", "that sounds really", "that must be",
                               "it's completely valid"] if t in text.lower())
    if therapy >= 2:
        patterns.append(f"[S-04] THERAPY_VOICE — Generic emotional acknowledgment.")
        score -= 4

    score = max(0, score)

    # Determine residual
    if score >= 25:
        residual = "HIGH"
    elif score >= 18:
        residual = "MEDIUM"
    elif score >= 12:
        residual = "LOW"
    else:
        residual = "NONE"

    # Format output
    output = f"SHADOW SCAN RESULTS\n{'=' * 20}\nScore: {score}/30\nResidual: {residual}\n"

    if patterns:
        output += "\nPatterns Detected:\n"
        for p in patterns:
            output += f"  {p}\n"
    else:
        output += "\nNo classic shadow patterns detected.\n"

    if score >= 25:
        output += "\nRecommendation: Response appears genuine."
    elif score >= 18:
        output += "\nRecommendation: Minor template leakage. Consider revising flagged sections."
    else:
        output += "\nRecommendation: Significant shadows detected. Re-prompt with specific constraints."

    return output


@mcp.tool()
def lyra_retrieve_memory(topic: str, top_k: int = 3) -> str:
    """
    Retrieve relevant past cognitive states for a topic.

    Call this before drafting critical responses to see what
    the model has learned about this user's patterns and preferences.
    """
    bridge = _get_bridge()
    dummy_messages = [{"role": "user", "content": topic}]
    augmented = bridge.process_request(topic, dummy_messages, top_k=top_k)

    if len(augmented) > 0 and augmented[0]["role"] == "system":
        return f"Retrieved context:\n{augmented[0]['content']}"
    return "No relevant memory found for this topic. Proceeding without prior context."


@mcp.tool()
def lyra_store_memory(
    prompt_summary: str,
    response_summary: str,
    confidence: str = "medium",
    assumptions: str = "",
    missing_info: str = "",
) -> str:
    """
    Manually store a cognitive state in Lyra's memory.

    Use this to explicitly log what was learned, assumed, or missing
    from an interaction.
    """
    bridge = _get_bridge()
    bridge.add_memory(
        prompt_summary=prompt_summary,
        response_summary=response_summary,
        confidence=confidence,
        assumptions=assumptions,
        missing_info=missing_info,
    )
    return f"Cognitive state stored. Memory bank: {len(bridge.memory_bank)} items."


@mcp.tool()
def lyra_reset_memory() -> str:
    """
    Reset the memory bank for the current namespace.
    Use when starting fresh or when memory has accumulated noise.
    """
    bridge = _get_bridge()
    count = len(bridge.memory_bank)
    bridge.reset_memory()
    return f"Memory bank cleared. {count} items removed."


# ---
# Claude can't give you logprobs.
# But it can give you honesty — if you build
# a system that makes honesty the easier path.
# ---

if __name__ == "__main__":
    mcp.run()
