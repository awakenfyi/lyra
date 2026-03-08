# Changelog

All notable changes to this project will be documented in this file.

## [0.2.1] - 2026-03-08

### Fixed
- Added missing optional dependency groups: `api` (fastapi, uvicorn, httpx), `mcp` (mcp)
- Added Python 3.12 to classifiers
- Removed tracked `__pycache__` files from repository
- Moved internal review documents to `docs/internal/`
- Added `[all]` extras group for full installation

### Added
- `[project.urls]` now includes documentation link to Six Rungs guide
- pytest configuration in pyproject.toml

## [0.2.0] - 2026-02-28

### Added
- Coherence-aware decoding with Top-K JSD metric (no "OTHER" bucket)
- Drift memory with EMA accumulation, clipping, and atomic writes
- Soft prompt injection (virtual token at position 0)
- KV-cached generation loop with prefill/decode handoff
- Controller profiles: freeform, code, json
- Contrastive penalty for low-coherence tokens
- Silence permission with sustained-window guard
- API bridge with per-namespace memory and 200-token injection cap
- API coherence proxy from logprobs (entropy, margin, mass)
- Traffic light system (GREEN/YELLOW/RED)
- Test suite for coherence bounds, EMA behavior, controller invariants, drift safety

### Safety Invariants
- Coherence always in [0, 1]
- Low coherence never increases temperature
- Drift namespaced — no cross-user contamination
- Model mismatch hard-blocks drift loading
- EOS bias requires sustained low coherence AND minimum tokens

## [0.1.0] - 2026-01-15

### Added
- Initial coherence metric prototype
- Basic drift accumulation
- Proof of concept generation loop

## [0.0.1] - 2025-10-15

### Added
- Project initialization
- Core question: can you measure the gap between internal state and output?
