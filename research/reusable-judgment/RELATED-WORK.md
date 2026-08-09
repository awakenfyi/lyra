# RELATED WORK — Reusable Judgment

**Intended home:** `lyra/research/reusable-judgment/RELATED-WORK.md`
**Status:** DESIGNED · **Date:** 2026-08-09

Where this program sits relative to adjacent fields, what it borrows, and what
it deliberately does not. Written so a reader who knows the inference stack can
place this work in one page — and so we can be checked on the borrowing.

---

## 1. The layer distinction

**Primary reference:** Philip Kiely, *Inference Engineering* (Baseten Books,
2026).

That book and this program share the word "inference" and almost nothing else.
The distinction is worth stating precisely because the name collision will
otherwise cost us a conversation every time.

| | Inference engineering | Reusable judgment |
|---|---|---|
| Layer | Token machine — CUDA, quantization, KV cache, GPU topology | Decision layer — which adopted rule governs this situation |
| Objective | **Same output, less cost** | **Different output, less correction** |
| Unit of measurement | The request | The task, across requests |
| Proof burden | Show nothing changed | Show something changed, and that it wasn't just more context |
| Metrics | TTFT, TPS, ITL, p50–p99 | required corrections, blinded edit minutes, reuse precision / recall / stale rate |
| Instrument | Stopwatch, automated, millions of runs/week | Blind human labelers, hours per data point |

The proof burdens are inverted, and that is the load-bearing difference. A
quantization result is credible when the quality delta is indistinguishable from
noise. A reusable-judgment result is credible only when the delta is *visible*
and survives a token-matched placebo. Their field never needs a placebo arm,
because their intervention is arithmetic precision rather than context. Ours is
context, which is exactly why arm E exists.

**Naming rule for this program:** we do not describe this work as "inference
engineering." That term is claimed, has a book attached, and means the layer
below us. Where a phrase is needed: *inference-time judgment*, or simply the
decision layer.

---

## 2. What we borrow, and from where

### 2.1 Speculative decoding as a structural analogy

*Inference Engineering* §5.2 describes speculative decoding: a draft model
proposes tokens, the target model validates them, accepted drafts are kept.
Kiely's framing — *"generating a token is like solving a sudoku; validating a
draft token is like checking a finished sudoku"* — is the same claim this
program makes one layer up:

> **Applying an already-adopted rule is cheaper than re-deriving the judgment.**

The ledger is a speculator for judgment. We take this as an **analogy of
structure, not a technique to port** — nothing in our design touches decoding.
But four of the field's hard-won lessons transfer directly, and each one
sharpens something we had stated more loosely:

1. **Acceptance rate is reported separately, never folded into throughput.**
   Their token acceptance rate is our reuse precision. A field that ships this
   in production learned not to blend it into the headline number. Independent
   support for the no-pooled-score rule.

2. **Draft reliability decays with distance.** *"Draft tokens get less reliable
   deeper in the sequence."* That is staleness in their vocabulary — a proposal
   degrades as it moves away from the context that warranted it. This is the
   same intuition that put the ledger before the cache, arriving from a field
   with no stake in our argument.

3. **One wrong acceptance cascades.** *"Once a single draft token is rejected as
   wrong, all subsequent tokens are also rejected."* The cost of a wrong
   acceptance is not one unit — it invalidates what follows. This is the
   argument for our asymmetric gate, stated in a domain where it was learned
   empirically rather than reasoned to.

4. **The technique disables itself outside its regime.** *"Speculative decoding
   must be dynamically disabled at higher batch sizes as compute is too
   saturated to afford verification."* A mature reuse technique knows the
   conditions under which it stops paying and turns off. This is the precedent
   for our **bypass condition** (§4 below) and it is why that condition is a
   procedure, not a scored quantity.

### 2.2 The non-inferiority standard, and the noise floor

*Inference Engineering* §5.1.3 sets the production standard for quantization:
zero perceptible quality loss, checked three independent ways, *"looking for a
difference in scores that's indistinguishable from noise."*

Two things transfer:

- **The form of the quality criterion.** "D must not materially reduce
  acceptance" is a non-inferiority claim and should be written as one.
- **You cannot claim "indistinguishable from noise" without knowing what noise
  looks like.** LLMs are non-deterministic; scores vary run to run. This
  motivates the **C′/D′ repeatability arms** in the preregistration:
  independent repeats of C and D on a stratified subset, whose spreads set the
  sanity threshold every other delta is read against.

The repeatability check is the single most useful thing this book contributed
to our design, and it was missing before.

### 2.3 Lossy versus lossless — an honest self-classification

Kiely observes that of the techniques in his Chapter 5, quantization is the only
**lossy** one; caching, batching, parallelism, and disaggregation don't change
outputs. He advises that in quality-sensitive domains, everything else is safe
and quantization needs a quality gate.

**Reusable judgment is a lossy technique.** It changes what the model produces —
that is the entire point. It trades range for consistency, and its
quality-degradation mode has a name in our own design: *flattened taste*.

Consequences we adopt:

- Evaluate it the way quantization is evaluated (explicit quality floor,
  baseline to compare against), not the way caching is evaluated.
- The opposite-error count is not a secondary metric. It is the quality gate.
- Any future runtime cache sits *downstream* of a lossy decision layer, which
  compounds rather than isolates the risk — a further argument for ledger before
  cache.

### 2.4 Benchmark hygiene from the same source

§1.3.1 cites Goodhart's Law directly — *"when a measure becomes a target, it
ceases to be a good measure"* — notes that public intelligence benchmarks are
"saturated or even gamed," and concludes there is no substitute for a
domain-specific eval, plus: *"establish a baseline: some optimization techniques
risk reducing model quality, requiring a baseline to compare against."*

That is arm C, and the argument for hand-authored task cards over a public
benchmark, sourced from outside this project.

---

## 3. Adjacent work we are not

**Agent simulation platforms (e.g. Coval).** Structure: agent + persona + test
set + metrics → simulated conversations → pass/fail → regression detection.
Nearest neighbor in *methodology genre* — pinned inputs, versioned datasets,
published methodology — and the model for how a benchmark page earns trust.

Two differences: their pass/fail is substantially LLM-judged, where our judged
calls require blind humans and splits resolve to unclear; and they measure
whether the **agent** degraded, where we measure whether the **correction**
stuck. Their regression detection and our stale-reuse rate point at different
objects.

**Memory, RAG, and context engineering.** All supply *what is true* or *what
happened*. The ledger supplies *what was decided and under what condition*. The
placebo arm exists precisely because a skeptic is right to suspect we are doing
context engineering with extra ceremony — and that is a live hypothesis until E
is scored.

**Constitutional / rule-based steering.** Closest in mechanism. The differences
we claim are project-scoped adoption rather than model-scoped training, an
explicit exception clause per rule, a review date, and measured staleness. We
have not surveyed this literature properly; that is an open item, not a settled
distinction.

---

## 4. What this reading changed in the design

Four concrete amendments, all pre-run, all logged in the preregistration:

1. **C′/D′ repeatability check** on a stratified subset of cards (§2.2).
2. **Non-inferiority framing** for the acceptance criterion, read against C–C′.
3. **Explicit lossy classification**, making opposite errors the quality gate
   rather than a secondary count.
4. **Bypass condition stated as a procedure**: where a task is ordinary
   execution with low ambiguity and no rule's SIGNAL is present, the correct
   behavior is to proceed without rule commentary. A ledger that produces
   deliberation on `what is the syntax for a Postgres upsert` has manufactured
   overhead. Recorded per run as `unwarranted_deliberation` — diagnostic in
   Pilot 000, a design requirement for any future router.

---

## 5. The positioning line

> Inference engineering makes the answer arrive faster.
> Reusable judgment reduces how many times you have to ask.

The industry's cost model is well-developed per request and stops there. TTFT,
TPS, and even end-to-end latency (inference plus network plus queue) all measure
**one request**. Nothing in that stack counts how many requests the human needed
to make.

Cost-per-token is solved. **Cost-per-accepted-outcome is not measured by
anyone.** That is the axis this program is aimed at — and the claim we are not
yet entitled to make, until Pilot 000 says the effect exists at all.

---

## References

- Philip Kiely, *Inference Engineering*, Baseten Books, 2026. ISBN
  979-8-9943597-2-3. Specifically §1.3.1 (model evaluation, Goodhart),
  §1.4 (latency metrics), §5.1.3 (measuring quality impact), §5.2
  (speculative decoding).
- Coval, simulation-based agent evaluation — https://docs.coval.ai/concepts/simulations/overview
- Internal: `LYRA_ON_FORMULAS.md` (formula discipline), `LABEL_PROTOCOL.md`
  (gold-set procedure), `pilot-000-paper/` (preregistration, labeling,
  analysis plan).
