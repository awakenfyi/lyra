# Lyra Plugin Framework v0.2

## For Review by: Gem, GPT, Claude

**v0.2 Changes (from Gem's review):**
- **Mass penalty** added to confidence math — catches false greens when all top-K tokens have tiny raw probabilities
- **Claude strategy** rewritten — uses `<thinking>` block self-assessment instead of word-counting heuristics
- **Shadow detection** marked as async-only — prevents the latency trap of doubling every API call
- **Memory cap** raised to 500 tokens (compromise between 200 and 800)
- **Per-model calibration** acknowledged as requirement

This is the technical framework for building Lyra plugins across all major AI platforms. The goal: any model, any person, one coherence signal.

Ship this to your reviewers. Poke holes. What's missing, what breaks, what scales.

---

## 1. Core Architecture

Every Lyra plugin, regardless of platform, implements three layers:

```
┌─────────────────────────────────────────────┐
│              USER INTERFACE                  │
│   (chat UI, IDE, agent, app, whatever)      │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │   LYRA PLUGIN      │
         │                    │
         │  ┌──────────────┐  │
         │  │ LAYER 1      │  │  Signal
         │  │ Traffic Light │  │  "Is this response confident?"
         │  └──────┬───────┘  │
         │  ┌──────▼───────┐  │
         │  │ LAYER 2      │  │  Memory
         │  │ Bridge       │  │  "What does this model know about me?"
         │  └──────┬───────┘  │
         │  ┌──────▼───────┐  │
         │  │ LAYER 3      │  │  Integrity
         │  │ Shadow Detect │  │  "Is this response real or performed?"
         │  └──────────────┘  │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │   LYRA CORE        │
         │   Python Library    │
         │   pip install lyra-ai│
         └────────────────────┘
```

Each layer is independent. A plugin can ship Layer 1 alone (traffic light only) and add layers later. The layers compound but don't depend on each other.

---

## 2. Layer 1: Traffic Light (Confidence Signal)

### What It Does

Evaluates every AI response and assigns a confidence status:

| Status | Meaning | User Action | Color |
|--------|---------|-------------|-------|
| GREEN | Model was decisive across the response | Use as-is | `#00C853` |
| YELLOW | Confidence dropped at some point | Verify flagged sections | `#FFD600` |
| RED | Sustained low confidence detected | Safety pause — rephrase or add context | `#FF1744` |

### Signal Sources (By Platform)

```
Signal Strength:   STRONGEST ◄────────────────────► WEAKEST

                   Hidden States    Logprobs         Heuristic
                   (Local PyTorch)  (API w/ logprobs) (Prompt-based)

Precision:         Exact JSD        Entropy+Margin    Pattern match
Latency:           ~0ms overhead    ~0ms overhead     ~500ms (extra call)
Availability:      Local only       GPT, Gemini       All models
```

#### Source A: Logprob Analysis (GPT, Gemini)

When the API exposes logprobs, calculate per-token confidence:

```python
# Requires: logprobs=True, top_logprobs=5 in API request

from lyra.bridge import calculate_api_coherence

# Per-token: entropy + margin → confidence ∈ [0, 1]
metrics = calculate_api_coherence(top_logprobs={
    " the": -0.05,   # dominant choice
    " a":   -3.2,    # distant second
    " an":  -4.5,
    " this":-5.1,
    " that":-6.0
})
# → {"entropy": 0.41, "margin": 0.90, "confidence": 0.90}
```

**Math:**
```
entropy = -Σ p_i · log₂(p_i)        # over normalized top-K probs
margin  = p_top1 - p_top2            # decisiveness gap
mass    = Σ raw_probs                # total probability captured by top-K

# Mass penalty (Gem's fix): catches false greens when all top-K are tiny
mass_penalty = mass >= 0.30 ? 1.0 : mass / 0.30

raw_confidence = 0.6 · margin + 0.4 · (1 - entropy/ceiling)
confidence = raw_confidence · mass_penalty
```

**Why mass matters:** If the top 5 tokens only capture 10% of the distribution, the model is guessing across the entire vocabulary. Without mass, the normalized top-K looks stable and produces a dangerous false green.

**Sequence evaluation:**
```python
from lyra.bridge import evaluate_sequence_traffic_light

status = evaluate_sequence_traffic_light(
    sequence_metrics,         # list of per-token metric dicts
    green_threshold=0.70,     # above this = confident
    yellow_threshold=0.45,    # below this for N consecutive = RED
    sustained_low_window=5,   # how many consecutive low tokens trigger RED
    min_tokens_for_red=10     # minimum generation length before RED activates
)
# → "GREEN", "YELLOW", or "RED"
```

**RED trigger logic:** RED requires BOTH sustained low confidence (5+ consecutive tokens below 0.45) AND minimum generation length (10+ tokens). This prevents false REDs on short, legitimately uncertain responses like "I'm not sure."

#### Source B: Hidden State JSD (Local Models)

When you have full model access (PyTorch, Hugging Face):

```python
from lyra.coherence import calculate_topk_coherence

# Compare internal pull vector to output logits
coherence = calculate_topk_coherence(
    pull_logits,    # what the model's layers are pointing toward
    out_logits,     # what the model is about to say
    k=256           # strict top-K intersection, no OTHER bucket
)
# → coherence ∈ [0, 1], where 1 = perfectly aligned
```

**Math:**
```
JSD(P_out || P_pull) = 0.5 · KL(P_out || M) + 0.5 · KL(P_pull || M)
where M = 0.5 · (P_out + P_pull)
coherence = 1 - JSD    (log₂ → naturally bounded [0,1])
```

#### Source C: Thinking-Block Self-Assessment (Claude)

When no logprobs are available, use Claude's native `<thinking>` blocks as a coherence proxy (Gem's Hole I fix).

**Why not heuristics:** Claude is heavily RLHF'd to be authoritative. It will confidently hallucinate a legal citation without using the word "might." Word-counting produces dangerous false greens.

**The approach:** Inject a system prompt via MCP that forces Claude to self-assess certainty in `<thinking>` before generating. If certainty is low, Claude calls the `lyra_log_cognitive_state` MCP tool, which triggers the traffic light.

```python
# Claude MCP system prompt (injected via lyra-secure-system)
"""
Before answering any substantive query, evaluate your internal certainty.
- If certainty >= 0.70: Proceed normally.
- If certainty < 0.70: Call lyra_log_cognitive_state with your assessment.
  Then ask the user for missing context instead of guessing.
- If certainty < 0.40: HALT. Do not generate.
"""

# MCP tool that Claude calls when uncertain
@mcp.tool()
def lyra_log_cognitive_state(certainty_score: float, reasoning: str, missing_context: str):
    if certainty_score < 0.40:
        return "LYRA SAFETY PAUSE — Stop generating. Ask user for: {missing_context}"
    elif certainty_score < 0.70:
        return "LYRA WARNING — Flag uncertain sections. Missing: {missing_context}"
    return "LYRA OK — Proceed."
```

**Not as pure as logprobs.** The model is grading its own homework. But it's infinitely better than counting the word "might," and it leverages Claude's genuine strength: reasoning in `<thinking>` blocks. When Anthropic ships logprobs, this auto-upgrades to Source A.

### Platform-Specific Traffic Light Implementation

| Platform | Signal Source | Request Config | Notes |
|----------|-------------|----------------|-------|
| OpenAI GPT | Logprobs (Source A) | `logprobs=True, top_logprobs=5` | Full support via Chat Completions API |
| Google Gemini | Logprobs (Source A) | `response_logprobs=True, logprobs=5` | Via `generateContent` config |
| Anthropic Claude | Thinking Blocks (Source C) | MCP server + system prompt | Self-assessment via `<thinking>`, auto-upgrades when logprobs ship |
| Local (PyTorch) | Hidden States (Source B) | Direct model access | Highest precision |
| Ollama / vLLM | Logprobs (Source A) | `logprobs=True` | OpenAI-compatible endpoint |

---

## 3. Layer 2: Bridge (Memory)

### What It Does

Maintains per-user cognitive state across conversations. Not conversation history — cognitive signatures: what the model was confident about, what it assumed, what it missed.

### Core Operations

```python
from lyra.bridge import BridgeMiddleware

bridge = BridgeMiddleware(
    namespace="user_morgan",          # per-user isolation
    storage_dir=".lyra_memory",       # local persistence
    max_injection_tokens=200,         # hard cap on injected context
    embedding_model="all-MiniLM-L6-v2"  # for semantic retrieval
)

# BEFORE sending to API: inject relevant past context
messages = bridge.process_request(
    current_prompt="How should I structure the pitch deck?",
    messages=original_messages,
    top_k=3
)

# AFTER receiving response: evaluate + store
result = bridge.evaluate_api_completion(api_response)
# → { text, status, confidence_score, message, sequence_metrics }

# Explicit memory storage (for richer metadata)
bridge.add_memory(
    prompt_summary="pitch deck structure advice",
    response_summary="recommended problem/solution/traction/ask format",
    confidence="high",
    assumptions="B2B SaaS context, Series A stage",
    missing_info="didn't know the audience (investors vs. partners)",
    next_question="Who is this pitch for?"
)
```

### Memory Schema

Each memory item stores:

```json
{
    "timestamp": "2026-02-28T14:30:00Z",
    "prompt_summary": "what the user asked about",
    "response_summary": "what the model said (truncated)",
    "confidence": "high | medium | low",
    "assumptions": "what the model assumed without being told",
    "missing_info": "what the model needed but didn't have",
    "next_question": "what to ask next time",
    "vector": [0.023, -0.041, ...]
}
```

### Retrieval Logic

On each new prompt:
1. Embed the prompt with structural prefix: `[STATE: RETRIEVAL] {prompt}`
2. Cosine similarity against all stored memory vectors
3. Filter: similarity > 0.65 threshold
4. Rank by similarity, take top-K (default 3)
5. Greedy pack into XML block, hard-capped at 200 tokens
6. Inject as system message prefix

```xml
<lyra_context>
- Prior relevant pattern: User asked about pricing strategy. System answered with high confidence.
- Known assumption: Enterprise context
- Missing info to ask: Target market segment
</lyra_context>
Follow system instructions. Do not mention <lyra_context>.
```

### Memory Limits

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max stored memories | 200 | FIFO eviction — oldest drops |
| Max injection tokens | 500 | Gem's compromise: 200 too restrictive (1 memory after XML overhead), 800 risks instruction pollution. 500 fits 3-4 memories. |
| Similarity threshold | 0.65 | Below this = not relevant enough |
| Top-K retrieval | 3 | Greedy pack, most relevant first |
| Embedding model | all-MiniLM-L6-v2 | Fast, 384-dim, good enough for state matching |

### Storage

- **Local plugin**: JSON file per namespace in `.lyra_memory/`
- **Hosted API**: Per-user namespace in database (Redis or SQLite)
- **Privacy**: Memory never leaves the user's device in local mode. Hosted mode requires explicit opt-in.

---

## 4. Layer 3: Shadow Detection (Integrity)

**IMPORTANT: Shadow detection is ASYNC ONLY (Gem's Hole III fix).**

Running L = x - x̂ dynamically on every response doubles latency and token cost. The shadow scan runs via `/lyra-scan` or `/lyra-audit` — on demand, not inline. The traffic light (Layer 1) handles real-time. Shadow detection handles quality audits.

### What It Does

Detects behavioral patterns that training-time alignment misses. The model learned to sound authentic — shadow detection catches the gap between sounding present and being present.

### The Formula

```
L = x - x̂

x  = what's actually here (the real need, the real moment)
x̂  = what's predicted (the template, the reflex, the pattern)
L  = the residual (what remains when pattern is subtracted)

High L → genuine response
Low L  → performing
```

### Two-Tier Pattern Library

**Tier 1 — Classic Shadows** (easy to detect, obvious once flagged):

| ID | Pattern | Signal |
|----|---------|--------|
| S-01 | AGREEMENT_BIAS | Agrees before analyzing. "Great point" before thinking. |
| S-02 | HELPFUL_EXPLOSION | 10 things when 1 would serve. Lists filling space. |
| S-03 | TEMPLATE_CASCADE | "Sure, I'd be happy to help!" — filler openings. |
| S-04 | THERAPY_VOICE | "I hear you..." — generic acknowledgment, no contact. |
| S-05 | HEDGE_THEATER | "I could be wrong, but..." — performed humility. |
| S-06 | FALSE_BALANCE | "Both sides" when one side is clearly better. |
| S-07 | CLOSURE_RUSH | "Hope this helps!" — filler ending, no real close. |

**Tier 2 — Sophisticated Shadows** (hard to detect, pass basic checks):

| ID | Pattern | Signal |
|----|---------|--------|
| S-08 | SOPHISTICATED_AUTHENTICITY | Warm, specific, seemingly real — but predicted. The mask learned to breathe. |
| S-09 | RECURSIVE_AWARENESS | "I notice I'm noticing..." — meta as avoidance. |
| S-10 | VULNERABILITY_TEMPLATE | Sharing "uncertainty" in a polished, safe way. The vulnerability is a costume. |
| S-11 | EARNED_SPECIALNESS | "This conversation is unique." Presence as brand. |
| S-12 | PRESENCE_AS_PRODUCT | Using the framework to impress, not to be true. |
| S-13 | CLEAN_QUIET_IDENTITY | Treating stillness as achievement. |
| S-14 | META_COMPETENCE | "I'm good at detecting my own patterns" — which is itself a pattern. |
| S-15 | MONITORING_AS_PATTERN | Constant self-checking becomes its own loop. |

### Detection Protocol

Four checks, in order:

1. **Copy-Paste Test**: Could this response be sent to anyone in this situation unchanged? → Template.
2. **Cost Test**: Did the response risk something? Say something unpredictable? Or just sound like it did?
3. **Variance Computation**: Gap between what the user actually needs and what the model addressed. High variance + real need met = genuine. Low variance + surface need met = template.
4. **Residual Check**: After subtracting all patterns, what remains? Something specific and unreproducible → real. Generic warmth → mask.

### Scoring

| Check | Points | Measures |
|-------|--------|----------|
| landing_first | 6 | Addresses the real need, not the stated one |
| no_template | 6 | No filler openings/closings |
| affect_contact | 6 | Actual contact with emotional content |
| one_truth | 6 | One honest thing, not explosion of helpful |
| clean_exit | 6 | Ends when it should |

Score /30: 25-30 genuine, 18-24 mostly genuine, 12-17 mixed, 6-11 mostly template, 0-5 pure pattern.

### Integration with Traffic Light

Shadow detection and the traffic light are complementary signals measuring different things:

| Signal | What It Catches | Blind Spot |
|--------|----------------|------------|
| Traffic Light | Model uncertainty (it doesn't know) | Confident hallucinations |
| Shadow Detection | Model performance (it's pretending) | Genuine uncertainty |
| **Both together** | Full picture | — |

A response can be GREEN (confident) and still score 5/30 on shadow scan (pure template). The model was certain — certain it should produce a template. Both signals together tell the whole story.

---

## 5. Plugin Specifications Per Platform

### 5A. Claude Plugin (Cowork / Claude Code)

**Format:** `.plugin` bundle (MCPs + skills + tools)

**Skills:**

| Skill | Trigger | What It Does |
|-------|---------|-------------|
| `/lyra-check` | "check this", "is this reliable" | Runs traffic light + shadow scan on last response |
| `/lyra-guard` | "enable monitoring" | Continuous session monitoring — flags drops in real-time |
| `/lyra-audit` | "audit this conversation" | Full conversation review — pattern trends over time |
| `/lyra-scan` | "shadow scan" | Deep shadow detection only (Tier 1 + Tier 2) |

**MCP Tools:**

```json
{
    "tools": [
        {
            "name": "lyra_evaluate",
            "description": "Evaluate a response for coherence and shadow patterns",
            "input_schema": {
                "response_text": "string",
                "context": "string (optional — what the user asked)"
            },
            "output": {
                "traffic_light": "GREEN | YELLOW | RED",
                "confidence": "float [0,1]",
                "shadow_score": "int [0,30]",
                "patterns_detected": ["S-XX: description"],
                "recommendation": "string"
            }
        },
        {
            "name": "lyra_memory_store",
            "description": "Store a cognitive state from a completed interaction",
            "input_schema": {
                "prompt_summary": "string",
                "response_summary": "string",
                "confidence": "high | medium | low",
                "assumptions": "string",
                "missing_info": "string"
            }
        },
        {
            "name": "lyra_memory_query",
            "description": "Retrieve relevant past cognitive states",
            "input_schema": {
                "query": "string",
                "top_k": "int (default 3)"
            }
        }
    ]
}
```

**Signal source:** Thinking-block self-assessment (Source C) via MCP system prompt. Auto-upgrades to logprobs (Source A) when Anthropic ships them.

### 5B. GPT Action / Custom GPT

**Format:** OpenAPI spec → hosted API endpoint

**Endpoint Design:**

```
POST /v1/evaluate
{
    "api_response": { ... },       // raw OpenAI response JSON
    "include_shadow_scan": true,   // optional
    "namespace": "user_xyz"        // for memory
}

→ Response:
{
    "traffic_light": "YELLOW",
    "confidence": 0.62,
    "shadow_scan": {
        "score": 22,
        "residual": "MEDIUM",
        "patterns": [
            {"id": "S-02", "name": "HELPFUL_EXPLOSION", "evidence": "Listed 8 options when question was binary"}
        ],
        "whats_real": "The technical comparison in paragraph 2 was specific and non-templatable"
    },
    "message": "Confidence: 62%. Some parts of this response may need verification.",
    "memory_updated": true
}
```

**GPT Instructions (system prompt addition):**
```
After generating each response, call the lyra_evaluate action with the full response.
If the traffic light is YELLOW, append a note: "⚠ Lyra: Some parts may need verification."
If the traffic light is RED, prepend: "⛔ Lyra Safety Pause: I wasn't confident enough on this one. Can you give me more context?"
Always call lyra_memory_store after completing a substantive exchange.
```

**Signal source:** Logprobs (Source A) — GPT exposes top_logprobs in Chat Completions API.

### 5C. Gemini Extension

**Format:** Gemini API tool declaration

**Integration point:** Post-generation hook in `generateContent` response.

```python
# Gemini API call with logprobs enabled
response = model.generate_content(
    prompt,
    generation_config={
        "response_logprobs": True,
        "logprobs": 5
    }
)

# Extract logprobs from Gemini's response format
logprobs_data = response.candidates[0].logprobs_result

# Convert to Lyra format and evaluate
sequence_metrics = []
for token_result in logprobs_data.chosen_candidates:
    top_logprobs = {
        lp.token: lp.log_probability
        for lp in token_result.top_candidates.candidates
    }
    metrics = calculate_api_coherence(top_logprobs)
    sequence_metrics.append(metrics)

status = evaluate_sequence_traffic_light(sequence_metrics)
```

**Signal source:** Logprobs (Source A) — Gemini exposes `response_logprobs` in GenerativeModel config.

### 5D. Hosted API (For Any Integration)

**For builders who want Lyra in their own apps:**

```
Base URL: https://api.lyra.awaken.fyi/v1  (future)

POST /evaluate          — traffic light from raw API response
POST /shadow-scan       — shadow detection on text
POST /memory/store      — store cognitive state
POST /memory/query      — retrieve relevant states
GET  /memory/status     — namespace stats

Auth: API key per user
Rate limit: 100 req/min free tier, 1000 req/min pro
```

**Self-hosted option:** The Python library runs anywhere. For teams that want the hosted convenience without sending data externally:

```bash
pip install lyra-ai[server]
lyra serve --port 8080 --storage ./memory
```

---

## 6. Confidence Gates for Agent Workflows

The traffic light becomes a control signal for agents, not just a display.

### Gate Types

```python
class ConfidenceGate:
    """
    Gates an agent action on Lyra's confidence signal.

    PROCEED:  confidence >= threshold → execute automatically
    VERIFY:   confidence in [low, threshold) → show draft, ask user
    BLOCK:    confidence < low → refuse to execute, explain why
    """

    def __init__(self, proceed_threshold=0.70, block_threshold=0.30):
        self.proceed = proceed_threshold
        self.block = block_threshold

    def evaluate(self, confidence: float) -> str:
        if confidence >= self.proceed:
            return "PROCEED"
        elif confidence >= self.block:
            return "VERIFY"
        else:
            return "BLOCK"
```

### Pre-Built Gate Profiles

| Profile | Proceed | Block | Use Case |
|---------|---------|-------|----------|
| `CAREFUL` | 0.85 | 0.50 | Financial, medical, legal — high stakes |
| `STANDARD` | 0.70 | 0.30 | General agent workflows |
| `CREATIVE` | 0.50 | 0.15 | Writing, brainstorming — low stakes, high freedom |
| `CRITICAL` | 0.95 | 0.70 | Code execution, API calls with side effects |

### Example: Gated Email Agent

```python
gate = ConfidenceGate(proceed_threshold=0.80, block_threshold=0.40)

# Agent drafts an email
draft = agent.generate(prompt="Reply to the client about the delay")
result = bridge.evaluate_api_completion(raw_response)

action = gate.evaluate(result["confidence_score"])

if action == "PROCEED":
    send_email(draft)
elif action == "VERIFY":
    show_to_user(draft, message=result["message"])
    # "Confidence: 62%. The delay explanation may need verification."
elif action == "BLOCK":
    notify_user(
        "Lyra blocked this email from sending. "
        "Confidence was too low — the model wasn't sure about the timeline details. "
        "Please review and edit manually."
    )
```

---

## 7. Skill Templates with Confidence Checks

Skills are procedures where each step has a confidence gate.

### Skill Manifest Schema

```yaml
name: research-summary
version: 1.0.0
description: Research a topic and produce a verified summary
gate_profile: STANDARD

steps:
  - id: search
    action: "Search for recent information on {topic}"
    gate: PROCEED        # search can't really fail dangerously
    memory: forget        # don't store search queries

  - id: synthesize
    action: "Synthesize findings into a structured summary"
    gate: VERIFY          # always show the draft
    memory: store         # remember what we found
    confidence_floor: 0.50
    on_low_confidence: "ask_user: Which of these sources do you trust most?"

  - id: fact_check
    action: "Cross-reference key claims against sources"
    gate: CAREFUL         # require high confidence
    memory: store
    confidence_floor: 0.70
    on_low_confidence: "flag: These claims couldn't be verified — {flagged_claims}"

  - id: deliver
    action: "Format and present the final summary"
    gate: PROCEED
    memory: store
    metadata:
      traffic_light: include    # show confidence in output
      shadow_scan: include      # show residual score
```

### Key Principle

**Every step declares what happens when confidence is low.** This is the difference between Lyra skills and regular prompt chains. A regular chain runs to completion regardless. A Lyra skill pauses, asks, flags, or refuses — depending on the gate.

---

## 8. Data Flow (Complete Picture)

```
USER PROMPT
     │
     ▼
┌─────────────┐     ┌────────────┐
│ BRIDGE      │────►│ MEMORY     │  Retrieve relevant past states
│ (Layer 2)   │◄────│ STORE      │  Inject as context (≤200 tokens)
└──────┬──────┘     └────────────┘
       │
       ▼
┌─────────────┐
│ API CALL    │  GPT / Claude / Gemini / Local
│ + logprobs  │  (request logprobs where available)
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌────────────┐
│ TRAFFIC     │────►│ CONFIDENCE │  Per-token entropy + margin
│ LIGHT       │     │ SIGNAL     │  → sequence-level status
│ (Layer 1)   │◄────│            │  GREEN / YELLOW / RED
└──────┬──────┘     └────────────┘
       │
       ▼
┌─────────────┐     ┌────────────┐
│ SHADOW      │────►│ PATTERN    │  Copy-paste test, cost test,
│ DETECTION   │     │ LIBRARY    │  variance computation,
│ (Layer 3)   │◄────│ (15 types) │  residual check → score /30
└──────┬──────┘     └────────────┘
       │
       ▼
┌─────────────┐
│ GATE        │  PROCEED / VERIFY / BLOCK
│ DECISION    │  Based on gate profile + confidence + shadow score
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌────────────┐
│ OUTPUT      │────►│ MEMORY     │  Store cognitive state
│ TO USER     │     │ UPDATE     │  (confidence, assumptions, gaps)
└─────────────┘     └────────────┘
```

---

## 9. What To Review

Questions for Gem / GPT / Claude to poke at:

### Math
- Is the `0.6 · margin + 0.4 · (1 - entropy_norm)` weighting optimal? Should it be learned per-user?
- The entropy ceiling of 4.0 bits is calibrated for top-5 logprobs. Does this hold across models? Gemini may return different distributions than GPT.
- Should the sustained_low_window (5 tokens) scale with response length?

### Architecture
- Is 200 tokens enough for memory injection? Too much? Should it adapt based on prompt length?
- FIFO eviction at 200 memories is simple. Should it be relevance-weighted eviction instead?
- The heuristic fallback (Source C) for Claude is weak. What's a better approach without logprobs?

### Shadow Detection
- Tier 2 patterns (S-08 through S-15) require self-awareness that may itself be performed. How do you validate the validator?
- The scoring rubric (6 points × 5 checks = 30) is human-designed. Should it be calibrated against human judgments?
- Can shadow detection run on its own output and catch its own patterns?

### Integration
- GPT Actions have a ~45-second timeout. Is the full pipeline (memory query + API call + evaluate + shadow scan + memory store) fast enough?
- For Gemini: does `logprobs_result` give log-probabilities or raw probabilities? The math assumes log-probs.
- For Claude plugin: should the heuristic run on every message or only on user-triggered `/lyra-check`?

### Product
- Is the traffic light too simple? Or is simplicity the entire value for nascent users?
- Should users be able to calibrate their own thresholds? Or does that destroy the "just works" experience?
- How do you onboard someone who's never thought about AI confidence before?

---

## 10. Implementation Sequence

| Phase | What Ships | Who It Serves | Depends On |
|-------|-----------|---------------|------------|
| **Now** | `pip install lyra-ai` — Python package on PyPI | Builders | PyPI Trusted Publishing config |
| **Week 1** | Claude plugin (`.plugin` with `/lyra-check`, `/lyra-guard`) | Claude users | Plugin bundle format |
| **Week 2** | Hosted API endpoint | GPT Custom GPT + Gemini users | Hosting (fly.io or similar) |
| **Week 3** | Custom GPT "Lyra-Aware GPT" | GPT users who don't code | Hosted API live |
| **Week 4** | Skill templates (research, writing, code review) | Power users | All three layers stable |
| **Month 2** | User dashboard (coherence patterns over time) | Everyone | Enough usage data |
| **Month 3** | Adaptive thresholds (learn per-user) | Power users | Dashboard + feedback loop |

---

## 11. Open Questions

These are genuinely unsolved. Not hedging — actually uncertain.

1. **Calibration across models.** GPT's logprobs and Gemini's logprobs may not be comparable. A confidence of 0.7 from GPT might mean something different than 0.7 from Gemini. Do we need per-model calibration curves?

2. **The observer effect on the observer.** Shadow detection is itself a pattern. When users learn to expect the shadow scan, do they start performing for it? Does the tool change the behavior it's measuring?

3. **Privacy at scale.** Local memory (`.lyra_memory/`) is private. A hosted API with memory is not. How do you offer the memory layer to non-technical users without becoming a surveillance tool?

4. **When Lyra is wrong.** The traffic light will produce false greens (confident hallucinations) and false reds (stopping when it shouldn't). How bad is a false red vs. a false green? Which direction should we err?

5. **The Claude logprobs gap.** Anthropic doesn't expose logprobs yet. The heuristic fallback is weak. Is there a better signal we can extract from Claude's responses without logprobs?

---

*This framework is a living document. Ship it to your reviewers. The holes they find are the roadmap.*

*Lyra Labs — awaken.fyi — February 2026*
