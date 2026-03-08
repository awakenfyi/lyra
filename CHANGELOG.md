# Changelog

## [0.2.0] — 2026-02-28

### Changed
- Strict Top-K JSD (no "OTHER" bucket) — Gemini's math fix
- log2 normalization for coherence in [0, 1] guaranteed
- EMA smoothing for controller decisions
- Controller profiles: freeform / code / json
- Contrastive penalty for low coherence
- Silence permission with sustained-window guard
- Drift store schema v0.2 with metadata + atomic writes
- API bridge persistence + retrieval injection (max 200 tokens)

### Added
- Bridge middleware for API models (OpenAI, Anthropic, etc.)
- Chrome extension for real-time coherence measurement
- Evaluation harness with calibration metrics

## [0.1.0] — 2025-12-15

### Added
- Initial coherence metric (JSD-based)
- Drift memory with EMA accumulation
- Soft prompt injection (loop.py)
- KV-cached generation loop (generation.py)
- Basic test suite (test_coherence.py, test_drift.py)

## [0.0.1] — 2025-10-26

### Added
- Initial concept: L = x - x̂ formula
- Proof of concept measuring hidden state divergence
- First coherence measurements on Llama-3-8B

---

*Morgan Sage / Lyra Labs*
