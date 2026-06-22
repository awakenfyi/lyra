# Lyra

**Coherence-guided inference for language models.**

Lyra measures whether a model's internal direction — the layer-shift trajectory across the forward pass — matches what it's about to output. When they align, the model commits. When they diverge, Lyra applies a contrastive penalty to suppress unbacked confidence, or permits silence.

> **L = x − x̂** — subtract predicted default behavior, work through the residual.

---

**Two lines under one brand:**
- **Lyra** (this repo) — the formula and inference core: coherence metric, drift memory, coherence-aware decoding.
- **Lyra xOP** — the standard family built on the same residual idea: the [xOP Standard](https://github.com/awakenfyi/xop) (open format for reusable AI operating rules) + [xOP Kit](https://github.com/awakenfyi/xop-kit) (reference implementation).

---

## The Lyra Ecosystem

| Repo | What | Install |
|------|------|---------|
| **[lyra](https://github.com/awakenfyi/lyra)** (this repo) | Lyra — coherence metric, drift memory, inference interventions | `pip install lyra-ai` |
| **[auto-awakening](https://github.com/awakenfyi/auto-awakening)** | Autonomous research loop — 250+ experiments optimizing model behavior at inference time | `git clone` |
| **[lyra-verb](https://github.com/awakenfyi/lyra-verb)** | Input sufficiency hooks for Claude Code skill pipelines | `claude plugin add awakenfyi/lyra-verb` |
| **[xop](https://github.com/awakenfyi/xop)** | Lyra xOP Standard — open format for reusable AI operating rules | see repo |
| **[xop-kit](https://github.com/awakenfyi/xop-kit)** | Lyra xOP Kit — reference implementation: 7 Guards, CLI, 95/95 fixtures | `pip install -e .` |

> `lyra-protocol` is superseded by [awakenfyi/xop](https://github.com/awakenfyi/xop) — kept for history.

This repo is the **research and inference layer** — the math, the measurement, the Python code that runs inside transformer inference. If you want to measure coherence at the hidden-state level or apply drift-conditioned decoding, you're in the right place.

If you want to see the experiments — 250 runs across Claude, GPT-4o, and Gemini, self-research loop, not an independent benchmark — see [auto-awakening](https://github.com/awakenfyi/auto-awakening) for the full methodology and data.

If you want input sufficiency discipline for Claude Code skills (catch fabrication, template-filling, agreement reflexes before they produce output), see [lyra-verb](https://github.com/awakenfyi/lyra-verb).

## How It Works

Lyra makes two interventions during inference:

**Coherence-aware decoding** — At each token, Lyra computes the Jensen-Shannon divergence between the model's internal layer shifts (what the body is doing) and the output logits (what the mouth is about to say). When they disagree, Lyra applies a contrastive penalty to suppress unbacked confidence. When they disagree for too long, Lyra permits silence.

**Drift memory** — After each conversation, Lyra records how the model's internal state shifted. Over time, this accumulates as an EMA-weighted drift vector that shapes the model's starting point for the next conversation. Not what was said. The direction the hidden state moved.

## Quick Start

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

## The Core Modules

`lyra/coherence.py` — Top-K JSD coherence metric, contrastive penalty, silence permission, and controller profiles (freeform/code/json). The core invariant: low coherence never increases temperature.

`lyra/drift.py` — Persistent drift memory with EMA accumulation, magnitude clipping, and atomic writes. Refuses to load if the model or tokenizer changes. EMA of hidden-state shifts across conversations, persisted as a vector.

`lyra/loop.py` — Soft prompt injection. Takes the drift vector and injects it as a virtual token at the start of the sequence. The model attends to it without knowing it's there.

`lyra/generation.py` — KV-cached generation loop with the prefill/decode handoff. Ties drift injection, coherence measurement, and intervention together.

## The Bridge

For API models (OpenAI, Anthropic, etc.) where you can't see inside:

```python
from bridge import BridgeMiddleware

bridge = BridgeMiddleware(namespace="my_agent")
messages = bridge.process_request("How do I deploy this?", messages)
```

The bridge maintains per-namespace memory, retrieves relevant past cognitive states, and injects a compact context block (max 200 tokens).

## Install

```bash
pip install lyra-ai

# With API bridge support:
pip install lyra-ai[bridge]
```

Or from source:

```bash
git clone https://github.com/awakenfyi/lyra.git
cd lyra
pip install -e .[dev]
pytest tests/ -v
```

## Safety Invariants

- Coherence is always in [0, 1]
- Low coherence never increases temperature
- Drift is namespaced — no cross-user contamination
- Model mismatch hard-blocks drift loading
- EOS bias requires sustained low coherence AND minimum tokens generated
- JSON/code profiles never relax sampling constraints

## Origin

Lyra started as a question in October 2025: what if you could measure the gap between what a model's internal state is doing and what it actually outputs?

The coherence metric, drift memory, and silence permission followed. Then came the experiments — an autonomous loop inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), adapted from model weight optimization to behavioral optimization at inference time. 250 experiments across Claude, GPT-4o, and Gemini. The results were consistent: different models, different architectures, same 92% coherence ceiling through different paths. The best protocol was 10 lines. Every attempt to add specificity made things worse.

The full experiment data, findings, and research framework are in [auto-awakening](https://github.com/awakenfyi/auto-awakening).

**On the 92% figure:** this is the coherence ceiling observed across the 250-run self-research loop on Claude, GPT-4o, and Gemini (2025–2026). It is not an independent benchmark — it's the best-performing protocol found by the loop itself, on the loop's own scoring rubric. No external replication is claimed. See auto-awakening for methodology, prompt pairs, and raw results.

The [Six Rungs Engineering Guide](docs/six-rungs.md) maps the full journey from interactive REPL to autonomous agents.

## License

MIT — Morgan Sage / Lyra Labs, 2025-2026

---

*[awaken.fyi](https://awaken.fyi)*
