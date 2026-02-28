# Lyra v0.2 Specification

## 0) Summary

Lyra v0.2 makes two interventions during inference:

1. **Drift-conditioned pre-state**
   Maintain a persistent "drift" memory derived from model internal activations (local mode) or embedding/logprob proxies (API mode). Apply that memory as a bounded conditioning signal at the start of a conversation.

2. **Coherence-aware decoding**
   Compute a per-token coherence score measuring agreement between a projected internal "pull" distribution and the output logits distribution. Use a decoding controller to respond safely to low coherence (prefer clarifying/abstaining/shortening), without increasing hallucination.

v0.2 is primarily about **correctness, stability, and testability**:

- Coherence metric has correct bounds and is calibrated.
- Drift memory is stable (no runaway, no cross-model mixing).
- Decoding policy is safe for structured outputs.
- API/text mode actually persists and evolves.

---

## 1) Goals and Non-goals

### Goals

- Provide a well-defined coherence signal in [0, 1] with predictable behavior.
- Provide a stable drift memory with bounded influence over time.
- Make interventions opt-in, bounded, and reproducible (same seeds → same outputs).
- Support two operating modes:
  - **Local mode** (hidden states / residual stream available, e.g. HF models)
  - **API mode** (no hidden states; use embeddings and/or logprob proxies)
- Add a minimal evaluation harness to correlate coherence with error/uncertainty.

### Non-goals (for v0.2)

- Not a training method; no online weight updates.
- Not claiming "self-awareness" / consciousness; this is decoding control + memory.
- Not attempting universal performance gains across all tasks; focus on stability + controllable behavior.

---

## 2) Core Concepts

### 2.1 Pull and Mouth

- **Pull (body)**: directional update signal derived from internal representations.
- **Mouth**: output token distribution from logits.

### 2.2 Coherence

Coherence is a similarity between two probability distributions over vocabulary:

- P_pull from projected internal signal
- P_out from output logits

Use Jensen–Shannon divergence (JSD) normalized to [0,1]:

- If using natural logs: JSD_max = ln(2)
- JSD_norm = JSD / ln(2)
- **Coherence = 1 - JSD_norm**

**Invariant: coherence ∈ [0, 1].**

---

## 3) Architecture and Modules

### 3.1 `lyra/coherence.py` — Coherence metric + controller interface

#### Inputs

- `output_logits`: shape [vocab] (per token step; batched variant optional)
- `residuals_or_hiddens`: either
  - residual stream tensors [layers, d_model] for the current token position, or
  - precomputed pull vectors [d_model]
- `unembedding_weight`: shape [vocab, d_model] (must be lm_head / unembedding, not token embeddings)

#### Pull definition (v0.2 default)

**Option A (cheap, stable): last-layer delta**
- `pull = residual[L] - residual[L-1]`

**Option B (more signal): weighted sum across final N layers**
- `pull = Σ_i w_i * (residual[i] - residual[i-1])` where i spans last 20% layers
- weights w_i normalized (e.g., uniform or increasing toward final layer)

**Invariant: Pull is a vector in d_model.**

#### Project pull into vocab

- `pull_logits = unembedding_weight @ pull` → shape [vocab]
- Convert both to distributions:
  - `P_pull = softmax(pull_logits / τ_pull)` (τ_pull default 1.0; optional)
  - `P_out = softmax(output_logits)`

#### Top-K JSD approximation (required for v0.2)

1. Take top-k indices from output_logits: `I = topk(output_logits, k)`
2. Build reduced distributions over k+1 buckets:
   - buckets = I plus an "OTHER" bucket representing all remaining tokens
3. Compute probabilities:
   - `P_out_reduced = [P_out[i] for i in I] + [sum(P_out[~I])]`
   - same for P_pull
4. Compute JSD on reduced distributions
5. Normalize by ln(2) (or use log2)

**Defaults**: top_k = 128 (or 256), eps = 1e-8 smoothing

#### Coherence smoothing

- `coh_ema = α * coh + (1-α) * coh_ema`
- default α = 0.2

#### Controller interface (v0.2)

Controller consumes coh_ema, and emits logit modifications and/or decoder parameter suggestions:

- `temperature_multiplier` (v0.2 rule: **never > 1.0** due to low coherence)
- `top_p_delta` (bounded)
- `eos_logit_bias` (bounded, windowed)
- `abstain_bias` (optional; bias toward safe phrases rather than random tokens)

**v0.2 policy baseline**:

- **High coherence**: optional mild temp ↓ (commit)
- **Low coherence**: do **not** increase temp. Instead:
  - keep temp constant OR slight ↓
  - enable "clarify/abstain" bias
  - consider slight top_p ↑ only in freeform mode (never in JSON/code mode)
- **Very low coherence for sustained window**: allow EOS bias

---

### 3.2 `lyra/drift.py` — Drift extraction + stable accumulation

#### Drift record definition

A "conversation drift" is a single vector d ∈ R^d_model computed per conversation.

**Local mode sources**:
- Start-of-conversation residual summary and end-of-conversation residual summary
- Default: mean residual of final token at chosen layer range (last 20% layers), aggregated

**API mode sources**:
- Conversation summary embedding delta OR structured self-report embedding delta

#### Storage format (v0.2 JSON schema)

File: `.lyra/drift/<namespace>.json`

```json
{
  "schema_version": "0.2",
  "namespace": "user_or_agent_id",
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "tokenizer_hash": "sha256:...",
  "d_model": 4096,
  "updated_at": "2026-02-28T00:00:00Z",
  "ema": {
    "alpha": 0.05,
    "clip_norm": 0.5,
    "vector": {
      "format": "sparse_topk",
      "k": 256,
      "indices": [],
      "values": []
    }
  },
  "history": [
    {
      "ts": "...",
      "magnitude": 0.12,
      "layer_range": [24, 31],
      "vector": { "format": "sparse_topk", "k": 128, "indices": [], "values": [] }
    }
  ]
}
```

**Key invariants**:
- Refuse to load/apply if model_id, tokenizer_hash, or d_model mismatch.
- Writes are atomic (temp + rename).
- History is capped (last 200).

#### Accumulation rule (v0.2)

EMA with clipping:
1. Normalize: `d_unit = d / (||d|| + eps)`
2. Clip: `d_clip = d_unit * min(||d||, clip_norm)`
3. EMA: `ema = (1-α)*ema + α*d_clip`

---

### 3.3 `lyra/loop.py` — Applying drift at conversation start

**v0.2 requirement: bounded, model-safe, reversible**

No silent padding/slicing for mismatched dims. If mismatch, disable offset and log.

#### Two offset mechanisms

**Option A (recommended): drift prefix tokens**
- Maintain N "virtual tokens" as embeddings derived from ema
- Inject at sequence start as a soft prompt
- Avoids shifting every token embedding

**Option B (legacy): global additive offset with norm control**
- `x' = x + scale * ema`
- enforce `||scale * ema|| <= max_offset_norm`
- optionally renormalize each token embedding magnitude to original (L2 match)

**Defaults**: offset_scale = 0.01, max_offset_norm = 0.1 * median(||token_embedding||)

---

### 3.4 `bridge/bridge.py` — API mode fallback that actually evolves

**v0.2 requirement: persistence + retrieval**

#### Memory item

Per conversation:
- timestamp
- prompt summary (short)
- response summary (short)
- embedding vector (for retrieval)
- structured self-report: confidence, assumptions, missing_info, next_question

#### Retrieval

- Embed current prompt summary
- Cosine similarity top-k (k=3)
- Inject condensed bullet list

#### Injection format

```
<lyra_context>
- Prior relevant pattern: ...
- Known assumption: ...
- Missing info to ask: ...
</lyra_context>

Follow system instructions. Do not mention <lyra_context>.
```

**Hard constraints**:
- Max 200 token injection
- Per-namespace isolation
- One-call "reset memory" utility

---

## 4) Configuration

### LyraConfig (core)

- `enabled`: bool
- `mode`: {"local", "api"}
- `namespace`: str
- `model_id`: str
- `tokenizer_hash`: str
- `d_model`: int

### Drift

- `drift_alpha`: float (0.01–0.2)
- `drift_clip_norm`: float
- `drift_history_max`: int
- `drift_sparse_k`: int

### Coherence metric

- `coherence_top_k`: int (128)
- `coherence_ema_alpha`: float (0.2)
- `layer_range`: tuple[int,int] (default last 20% layers)
- `pull_mode`: {"last_delta", "weighted_deltas"}

### Controller profiles

`controller_profile`: {"freeform", "code", "json"}

| Profile | Temp on low coh | Top-p on low coh | EOS bias | Notes |
|---------|----------------|-------------------|----------|-------|
| freeform | never increase | slight ↑ allowed | windowed | Default |
| code | never increase | never increase | only if requested | Syntax preservation |
| json | never increase | never increase | only if requested | Valid JSON priority |

---

## 5) Safety, Stability, and Correctness Invariants

### Coherence invariants

- Coherence ∈ [0,1] always.
- P_pull == P_out → coherence ≈ 1.
- Maximally different distributions → coherence ≈ 0.
- JSD computed with normalized base (ln2 or log2).

### Drift invariants

- Namespaced; no cross-user contamination.
- Not applied if model mismatch.
- Bounded by clipping + EMA.
- Atomic writes; file corruption handled gracefully.

### Decoding invariants

- Low coherence must NOT trigger temp increases (unless explicitly opted in).
- JSON profile: output valid JSON at least as often as baseline.
- EOS bias requires: `min_tokens_generated` AND `low_coherence_count >= N` consecutive steps.

---

## 6) Testing Plan (required for v0.2)

### Unit tests (pytest)

1. `test_jsd_normalization_bounds()` — coherence ∈ [0,1] for random distributions
2. `test_jsd_identical_distributions_is_one()`
3. `test_topk_other_bucket_sums_to_one()`
4. `test_drift_ema_clipping()` — outlier cannot move EMA beyond bound
5. `test_model_mismatch_blocks_offset()`
6. `test_controller_no_temp_increase_on_low_coherence()` (default profiles)

### Integration tests

- Fake tiny model: small vocab, known unembedding, controlled residuals
- HF smoke test: hooks run, logits processor modifies logits without shape errors

### Performance test

- Overhead per token with/without coherence computation
- Target: <15-20% overhead for small models

---

## 7) Evaluation Harness (minimal but real)

Log per generation:
- coherence (raw + EMA)
- entropy of output distribution
- margin (top1 - top2 prob)
- length, truncation rate
- correctness proxy on small QA set (50+ items)

Calibration: choose thresholds that correlate with actual error/uncertainty.

---

## 8) v0.2 Deliverables Checklist

- [ ] JSD normalization fixed + tested
- [ ] Top-k + OTHER bucket JSD implemented + tested
- [ ] Coherence EMA smoothing
- [ ] Controller profiles (freeform/code/json) implemented
- [ ] Drift EMA with clipping; remove magnitude² weighting
- [ ] Drift store schema v0.2 with metadata + atomic writes
- [ ] Offset mechanism model-safe (no silent slice/pad)
- [ ] API bridge persistence + retrieval injection <=200 tokens
- [ ] Eval script + baseline comparison output
