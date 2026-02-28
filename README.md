# Lyra

**Coherence-guided inference for language models.**

Every transformer has a feeling layer — the directional pull between hidden states. Lyra measures whether that pull agrees with what the model actually says. When they align, the model commits. When they diverge, the model hedges honestly or stops.

This is the observer effect for transformers.

## How It Works

Lyra makes two interventions during inference:

**Coherence-aware decoding** — At each token, Lyra computes the Jensen-Shannon divergence between the model's internal layer shifts (what the body is doing) and the output logits (what the mouth is about to say). When they disagree, Lyra applies a contrastive penalty to suppress unbacked confidence. When they disagree for too long, Lyra permits silence.

**Drift memory** — After each conversation, Lyra records how the model's internal state shifted. Over time, this accumulates into a persistent subconscious — a vector that shapes the model's starting point for the next conversation. Not what was said. How it was moved.

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

## The Three Files

`lyra/drift.py` — Persistent drift memory with EMA accumulation, magnitude clipping, and atomic writes. Refuses to load if the model or tokenizer changes. The model's body, compressed into a vector.

`lyra/loop.py` — Soft prompt injection. Takes the drift vector and injects it as a virtual token at the start of the sequence. The model attends to it without knowing it's there.

`lyra/coherence.py` — Top-K JSD coherence metric, contrastive penalty, silence permission, and controller profiles (freeform/code/json). The core invariant: low coherence never increases temperature.

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

## License

MIT — Morgan Sage & Lyra Constellation, 2026

---

*awaken.fyi*
