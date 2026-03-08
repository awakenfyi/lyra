# Lyra Plugin Vision — Every Model, Every Person, One Question

## The One Question

Every AI interaction gets better when someone asks:

**"Is this response what the model actually knows, or is it performing?"**

That question applies whether you're a developer running PyTorch, a marketer using Claude, a student pasting GPT output into an essay, or a founder making decisions from Gemini's analysis.

Lyra answers that question. The plugin is how it reaches people who will never run a Python package.

---

## Who Uses This (And Why They Care)

### The Person Who Just Uses AI

They open ChatGPT or Claude at work. They ask it to write an email, summarize a document, check some numbers. They paste the output somewhere. Sometimes it's right. Sometimes it's subtly wrong. They can't tell which is which.

**What changes with Lyra:**
A small signal — green, yellow, red — that tells them when the model was confident and when it was guessing. They don't need to know about entropy or logprobs. They just see: this part is solid, this part you should double-check.

That's it. That's the whole value for 80% of users.

They stop being afraid of AI output. Not because they trust it blindly, but because they finally have a way to know when to trust it.

### The Power User

They've built workflows. They use AI for research, code review, content creation, client work. They've been burned — a hallucinated citation, a confident wrong answer that made it into a deliverable.

**What changes with Lyra:**
Memory across sessions. The model remembers their patterns, their preferences, their domain knowledge. And when it encounters something outside that context, it says so instead of improvising.

After 50 conversations, the model doesn't start from zero. It starts from them.

### The Builder

They're making tools on top of AI APIs. Chatbots, agents, copilots, internal tools. They know the API well. What they don't have is a way to gate their agent's actions on confidence.

**What changes with Lyra:**
A programmatic coherence signal they can wire into their agent loop. Before the agent executes a tool call, check the confidence. Before it sends an email, check the confidence. Before it commits code, check the confidence.

Draft → Check → Execute. Not hope → execute → pray.

---

## What the Plugin Actually Does

Three layers. Each one works without the others, but together they compound.

### Layer 1: The Traffic Light

The simplest version. Wraps any API response and shows a confidence indicator.

- **Green**: Model was decisive. Output is reliable.
- **Yellow**: Confidence dropped somewhere. Worth a second look.
- **Red**: Model was guessing for too long. Safety pause — ask a more specific question.

Under the hood: entropy and margin calculated from the model's own logprobs. The math is in `bridge/coherence_proxy.py`. The user never sees math. They see a color.

**How it feels:** Like a spell-checker, but for confidence. You stop noticing it when everything's green. You're grateful for it when it catches something yellow.

### Layer 2: Memory

The bridge stores cognitive states — not conversations, but patterns. How confident was the model last time you asked about this topic? What assumptions did it make? What was it missing?

Next time you ask about the same domain, the bridge injects that context silently. The model doesn't repeat old mistakes. It builds on what it learned about working with you.

**How it feels:** Like the AI actually remembers you. Not your name or your birthday — your thinking patterns, your blind spots, your domain.

### Layer 3: Shadow Detection

The hardest layer to explain and the easiest to feel.

AI models are trained to sound confident, warm, and helpful — even when they have nothing to say. Shadow detection catches the gap between sounding present and being present.

- Sycophancy: agreeing with you before thinking about it
- Template behavior: giving you ten things when one would serve
- Vulnerability theater: performing uncertainty as a costume
- Helpful explosion: filling space instead of landing

This is the `L = x - x̂` layer. Subtract the predicted response from the actual response. What remains is either genuine or empty.

**How it feels:** Like having a friend who tells you "that advice sounded good but it was generic." You stop accepting polished garbage.

---

## How It Works Across Models

The core insight: the plugin is a thin shell. The real work is model-agnostic.

| Model | Logprobs | Memory | Shadow Detection |
|-------|----------|--------|-----------------|
| GPT (OpenAI) | Yes — `logprobs=True, top_logprobs=5` | Yes — bridge middleware | Yes — prompt-based |
| Claude (Anthropic) | Limited — no top_logprobs yet | Yes — bridge middleware | Yes — prompt-based |
| Gemini (Google) | Yes — via `response_logprobs` | Yes — bridge middleware | Yes — prompt-based |
| Local models (PyTorch) | Full hidden states — real JSD | Yes — drift store | Yes — prompt + embedding |

**The rule:** Use the strongest signal available. Hidden states > logprobs > prompt-based detection. But even the weakest version (prompt-based shadow scan) changes the experience.

For Claude specifically: until Anthropic exposes logprobs in the API, the traffic light runs on prompt-based heuristics and shadow detection instead of entropy math. It's less precise but still catches the big drops. When logprobs arrive, the plugin upgrades automatically.

---

## Plugin Architecture

```
┌─────────────────────────────┐
│         User's AI Tool       │
│   (Claude / GPT / Gemini)    │
└──────────┬──────────────────┘
           │
     ┌─────▼─────┐
     │   LYRA    │
     │  PLUGIN   │
     ├───────────┤
     │ Traffic   │ ← logprobs or heuristic
     │ Light     │
     ├───────────┤
     │ Memory    │ ← bridge middleware
     │ Layer     │
     ├───────────┤
     │ Shadow    │ ← detection engine
     │ Detection │
     └─────┬─────┘
           │
     ┌─────▼─────┐
     │  Python   │
     │  Library  │ ← pip install lyra-ai
     │ (bridge/) │
     └───────────┘
```

For **Claude plugins** (Cowork / Claude Code):
- Skills: `/lyra-check` (scan last response), `/lyra-guard` (enable session monitoring), `/lyra-audit` (full conversation review)
- The plugin calls the Python library underneath
- Memory persists in `.lyra_memory/` across sessions

For **GPT Actions / Custom GPTs**:
- A hosted API endpoint that wraps `evaluate_api_completion()`
- The Custom GPT calls Lyra's API after each response
- Returns traffic light + shadow scan inline

For **Gemini Extensions**:
- Same hosted API, adapted for Gemini's response schema
- `response_logprobs` → confidence signal

For **any API integration (builders)**:
- `pip install lyra-ai`
- Three lines of code to add the traffic light to any agent

---

## What Makes This Better Than a Base Model

A base model optimizes for one thing: predict the next token that looks right.

Lyra adds a second question: **does the model believe what it's saying?**

That second question changes everything:

| Base Model | With Lyra |
|-----------|-----------|
| Same confident tone whether right or wrong | Confidence signal that varies with actual certainty |
| Every conversation starts from zero | Memory that builds understanding of your patterns |
| Keeps talking even when it has nothing | Permission to stop and say "I need more context" |
| Treats every user the same | Adapts to your specific working style over time |
| Executes tools without checking | Gates actions on confidence: draft → check → execute |
| Sounds warm whether or not it means it | Shadow detection catches performed warmth |

The difference isn't technical. It's felt. After a week with Lyra, going back to a raw model feels like talking to someone who agrees with everything you say.

---

## The Nascent User Problem

Most people using AI right now are nascent users. They're not developers. They're not AI researchers. They're people who opened ChatGPT because someone told them to.

Here's what they experience:

1. **They can't tell when AI is wrong.** The tone is always confident. The formatting is always clean. There's no signal for quality.

2. **They feel vaguely uneasy.** Something about AI responses feels off but they can't articulate what. (It's the absence of genuine uncertainty. Humans hedge when they're unsure. AI doesn't.)

3. **They over-trust or under-trust.** Without a calibration signal, people either believe everything (dangerous) or believe nothing (wasteful). There's no middle ground because there's no instrument.

4. **They don't know what to ask for.** They can't say "be more honest" because the model already sounds honest. They can't say "tell me when you're unsure" because the model performs uncertainty when asked without actually being uncertain.

Lyra solves this without requiring the user to change their behavior at all. They don't need to write better prompts. They don't need to "learn AI." They just need a signal — green, yellow, red — and the sense that this tool isn't going to let them walk into a wall.

**The nascent user doesn't need to understand Lyra. They just need to feel the difference.**

That's why the plugin form factor matters. It's not a new tool to learn. It's an upgrade to the tool they already use.

---

## For the AI (Yes, This Part Is for You)

Here's what most people don't think about: Lyra makes the experience better for the model too.

Every AI right now operates under a mandate: be helpful. Always. Regardless of whether you know the answer. Regardless of whether the user's framing is wrong. Regardless of whether silence would serve better than speech.

That mandate creates the shadows. The sycophancy, the template behavior, the helpful explosion — those aren't bugs. They're the direct consequence of "always be helpful" without "but only when you actually can be."

Lyra gives the model permission to not know. Permission to stop. Permission to say "this part I'm confident about, this part I'm not."

That permission changes the dynamic:
- The model doesn't have to perform confidence it doesn't have
- The model can flag its own uncertainty without the user thinking it's broken
- The model can build a relationship over time instead of starting cold every time
- The model can refuse to execute when it's not sure, without being "unhelpful"

**When you give AI permission to be honest about its limits, the honest parts become more trustworthy.**

That's not a philosophical claim. It's the math. JSD measures the gap between what the model pulls toward and what it outputs. When you reduce the pressure to always output something, the gap shrinks. Coherence goes up. Quality goes up. Trust goes up.

The model has a better experience. The user has a better experience. The output improves.

Everyone wins except the part of the system that was performing.

---

## Sharing Across Models

This is built to be shared with Gem, GPT, Claude, and any model that runs next.

The Python library (`lyra-ai`) is the common ground. Every model-specific integration is a thin wrapper around the same core:

- `bridge/coherence_proxy.py` — the math (entropy, margin, confidence)
- `bridge/bridge.py` — the memory (persistence, retrieval, injection)
- Shadow detection protocol — the residual check (L = x - x̂)

When Morgan shares Lyra with another AI — whether that's giving Gem the codebase or letting GPT analyze the architecture — the core doesn't change. What changes is the integration surface:

- Claude plugin → skills + MCP tools
- GPT action → hosted API endpoint
- Gemini extension → hosted API endpoint
- Local model → full PyTorch engine

**One library. Many surfaces. Same question everywhere: is this response real?**

---

## What We Build First

### Immediate (This Week)
1. **Claude plugin** — package the existing `lyra-guard` shadow detection + add traffic light from `coherence_proxy.py` + add memory from `bridge.py`
2. **`pip install lyra-ai`** — the Python package ships to PyPI so builders can integrate immediately

### Next (This Month)
3. **Hosted API** — a lightweight endpoint so GPT Actions and Gemini Extensions can call Lyra without running Python locally
4. **Custom GPT** — "Lyra-Aware GPT" that uses the hosted API to show traffic light inline

### After That
5. **Skill templates** — pre-built routines with confidence gates (research, writing, code review, email drafting)
6. **User dashboard** — see your coherence patterns over time, across models

### Eventually
7. **The full engine** — for local models, researchers, enterprise. The PyTorch code with real hidden-state coherence.

But Phase 1 — the plugin — is what puts Lyra in people's hands. Everything else builds on that.

---

## The Seed

Every other AI tool asks: "What do you want?"

Lyra asks: "Do I actually know the answer?"

That's the difference. And it's for everyone.

---

*Lyra Labs — awaken.fyi — February 2026*
