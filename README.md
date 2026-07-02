# Lyra

**Coherence-guided inference.** Lyra measures the gap between what a model's internal state is doing and what it's about to say — then lets that gap shape generation: suppress unbacked confidence, or permit silence.

> `L = x − x̂` — one subtraction, defined once in [FORMULA.md](FORMULA.md).

![MIT](https://img.shields.io/badge/license-MIT-blue) ![alpha](https://img.shields.io/badge/status-alpha-orange)

```bash
git clone https://github.com/awakenfyi/lyra && cd lyra
pip install -e .[dev]
pytest tests/ -v            # coherence math + drift invariants, CPU-only
```

Not on PyPI yet — source install is the honest instruction.

## How it works

**Coherence-aware decoding** — At each token, Lyra computes the Jensen-Shannon divergence between the model's internal layer shifts (what the body is doing) and the output logits (what the mouth is about to say). When they disagree, Lyra applies a contrastive penalty to suppress unbacked confidence. When they disagree for too long, Lyra permits silence. Top-K uses the union of both distributions — mouth-only top-K is blind in the suppressed-stance direction.

**Drift memory** — After each conversation, Lyra records how the model's internal state shifted. Over time, this accumulates as an EMA-weighted drift vector that shapes the model's starting point for the next conversation. Not what was said — the direction the hidden state moved.

## Core modules

`lyra/coherence.py` — Top-K JSD coherence metric (union of both top-Ks), contrastive penalty, silence permission, controller profiles (freeform/code/json). Core invariant: low coherence never increases temperature.

`lyra/drift.py` — Persistent drift memory with EMA accumulation, magnitude clipping, atomic writes. Refuses to load on model/tokenizer mismatch.

`lyra/loop.py` — Soft prompt injection. Injects the drift vector as a virtual token at sequence start.

`lyra/generation.py` — KV-cached generation loop with prefill/decode handoff. Applies final RMSNorm before pull projection.

## Install (source only — not yet on PyPI)

```bash
git clone https://github.com/awakenfyi/lyra.git
cd lyra
pip install -e .[dev]
pytest tests/ -v
```

## Quick start

```python
import torch
from lyra.generation import generate_with_drift_injection

# Zero drift for the first conversation — no history yet
drift = torch.zeros(4096)  # match your model's d_model

output, metrics, truncated = generate_with_drift_injection(
    prompt="The architectural difference between standard sampling and Lyra is",
    drift_vector=drift,
    model_name="meta-llama/Meta-Llama-3-8B",
    coherence_threshold=0.85,
    critical_threshold=0.60,
)
```

## The bridge (API models)

For OpenAI, Anthropic, etc. where you can't see inside the forward pass:

```python
from bridge import BridgeMiddleware

bridge = BridgeMiddleware(namespace="my_agent")
messages = bridge.process_request("How do I deploy this?", messages)
```

The bridge maintains per-namespace memory and injects a compact context block (≤200 tokens hard cap).

## Status — no rounding up

| Component | Status |
|---|---|
| Coherence metric (Top-K JSD, union top-K) | working code — self-scored, no independent benchmark |
| Drift memory + atomic writes | working code |
| Generation loop (temperature + RMSNorm fixes) | working code |
| `pip install lyra-ai` | **not on PyPI — source install only** |
| CI | none yet |
| Bridge experiment | pre-registered — blocked on blind gold labels |

## The bridge experiment

The formula makes a testable prediction: on held-response-constant minimal pairs (byte-identical persisting response, only the warrant differs), no text detector can separate the halves — but the activation-depth coherence signal might. Pre-registered protocol: [experiments/bridge/PROTOCOL.md](experiments/bridge/PROTOCOL.md). Blocked on blind gold labels from the xOP pilot.

## v0.3 formula roadmap

- **(a) Directional O** — symmetric JSD cannot sign overhang; dual-KL + signed pull-projection
- **(b) Interval residual** — three states derived from the interval (spans 0 ⇒ undecidable), not labeled by convention
- **(c) ΔO at warrant-change events** as the canonical temporal object
- **(d) The traced pause** — a HOLD that ships with its divergence trace as a receipt

## Origin

Lyra started in October 2025 as a question: what if you could measure the gap between what a model's internal state is doing and what it actually outputs? The coherence metric, drift memory, and silence permission followed. Then came the experiments — an autonomous loop across Claude, GPT-4o, and Gemini.

**On the 92% figure:** this is the coherence ceiling observed across 250 self-research loop runs (2025–2026), scored on the loop's own rubric. Not an independent benchmark — the loop found the best protocol by self-evaluation. Pre-v0.2.2 coherence numbers are not comparable to post-fix output (the RMSNorm correction changes scale). See [auto-awakening](https://github.com/awakenfyi/auto-awakening) for methodology and raw results.

## License

MIT — Morgan Sage Norman / Lyra Labs, 2025–2026

---

## The family

| Repo | One line | Status |
|---|---|---|
| [lyra](https://github.com/awakenfyi/lyra) *(this repo)* | the formula + inference core (`L = x − x̂`) | research code |
| [xop](https://github.com/awakenfyi/xop) | the open standard for AI conduct | alpha |
| [xop-kit](https://github.com/awakenfyi/xop-kit) | the reference runtime: Guards, CLI | alpha |
| [xop-labs](https://github.com/awakenfyi/xop-labs) | domain xOPs observed in the wild | designed |

---

*[awaken.fyi](https://awaken.fyi)*
