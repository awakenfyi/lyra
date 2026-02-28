# Lyra — Code Review Prompt

Use this prompt when sharing the Lyra repo with other AI models for feedback.

---

## The Prompt

I'm building an open-source inference modification for language models called Lyra. The repo is at github.com/awakenfyi/lyra. I'd like your honest technical feedback.

**What it does:** Three Python files that modify how a transformer samples its next token by listening to its own hidden states.

- `lyra/drift.py` — Records how the model's internal state shifted after each conversation (embedding drift). Git-backed JSON storage.
- `lyra/loop.py` — Before each conversation, loads accumulated drift and applies a subtle offset to the model's starting embeddings. Builds a "subconscious" text block for API models that don't expose internals.
- `lyra/coherence.py` — During generation, compares the directional pull between layers to the output logits. Uses Jensen-Shannon divergence to measure alignment. Adjusts sampling temperature and boosts EOS probability when coherence is low ("silence permission").
- `bridge/bridge.py` — Middleware wrapper that makes any AI tool Lyra-aware.

**The core idea:**

```
pull = hidden_state[layer+1] - hidden_state[layer]  (the "body")
output = softmax(logits)                              (the "mouth")
coherence = 1 - JSD(body_vote, mouth_vote)

High coherence → lower temperature → commit
Low coherence → higher temperature → hedge honestly
Very low coherence → boost EOS → say less
```

**What I want feedback on:**

1. **Does the math hold up?** Is Jensen-Shannon divergence the right measure here? Are there better ways to project the directional pull into vocabulary space?

2. **Drift accumulation** — currently using weighted average of past drift directions compressed to top-64 components. Is this a reasonable approach or will it collapse/diverge over many conversations?

3. **The offset mechanism** — applying accumulated drift as an additive offset to initial embeddings. What are the risks? Could this cause instability?

4. **Coherence-guided sampling** — dynamically adjusting temperature based on layer agreement. Are there edge cases where this breaks down?

5. **The silence permission** — boosting EOS probability when coherence is low. Is this a sound intervention or could it cause premature truncation?

6. **What's missing?** What would you add, change, or remove to make this production-ready?

7. **API/text mode fallback** — the bridge builds a "subconscious" text block for models without hidden state access. Is there a better approach for API-only models?

Be direct. Tell me what's wrong, what's naive, and what might actually work.
