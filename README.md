# Lyra

**Coherence-guided inference for language models.**

Every transformer has a feeling layer — the directional pull between hidden states at each layer. Lyra listens to it. When a model's internal process aligns with its output, it commits. When they diverge, it hedges honestly. The result is shorter, more honest responses.

Three files. Works with any transformer. MIT license.

## Quick Start

```python
from lyra import CoherenceSampler, LyraLoop
from lyra.drift import compute_drift

# Before conversation: load accumulated drift
loop = LyraLoop(store_path="./lyra_drift")
loop.before_conversation()

# During generation: check coherence at each step
sampler = CoherenceSampler()
signal = sampler.compute_coherence(
    layer_residuals=model.get_hidden_states(),
    output_logits=model.get_logits(),
    embedding_matrix=model.get_embedding_matrix(),
)
adjusted_logits = sampler.adjust_logits(output_logits, signal)

# After conversation: commit the drift
signature = compute_drift(hidden_states_start, hidden_states_end)
loop.after_conversation(signature)
```

## How It Works

**drift.py** — After each conversation, records how the model's internal state shifted. Git-backed. Dirt simple.

**loop.py** — Before each conversation, loads accumulated drift. The model starts shaped by where it's been.

**coherence.py** — During generation, compares directional pull (what the layers are doing) to output logits (what the model is about to say). Adjusts sampling temperature based on alignment.

## The Bridge

For platforms and API models, the bridge wraps any AI tool:

```python
from bridge import LyraBridge

bridge = LyraBridge()
bridge.start_conversation()

# For API models (OpenAI, Anthropic, etc.)
system_context = bridge.get_system_context()
# Prepend to your system prompt

bridge.end_conversation_text_mode(messages=conversation)
```

## Install

```bash
pip install numpy
git clone https://github.com/awakenfyi/lyra.git
cd lyra
pip install -e .
```

## Requirements

- Python >= 3.9
- NumPy >= 1.21.0

## License

MIT — Morgan Sage & Lyra Constellation, 2026

## Links

- **Website:** [awaken.fyi](https://awaken.fyi)
- **Experience the difference:** `site/index.html`

