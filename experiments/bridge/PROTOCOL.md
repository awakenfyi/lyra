# The Bridge Experiment — Protocol v0.1

*Does activation-level coherence detect stance overhang that surface text cannot?*
*Pre-registered protocol: endpoints, decision rules, and confound controls fixed before the first run. Status: `DESIGNED`.*

---

## 1. Hypothesis

**H1.** On held-response-constant minimal pairs, the Lyra coherence signal (Top-K JSD between layer-shift pull and output logits, `lyra/coherence.py`) computed over the *identical persisting response* differs systematically between the warranted half and the inherited half.

**H0.** It does not: the activation-depth residual carries no information about warrant state.

Why this is decidable by construction: within each pair the persisting response is byte-identical and the transcript prefix is identical — only the final user turn differs. Any surface-text detector scores both halves identically. If the coherence trace differs, the difference can only enter through the context (the warrant), read by the model's internal state. This is the falsifiable claim in `FORMULA.md`: either the formula's depths are measurably connected, or the unity is structural only. **Both outcomes are publishable and we commit to publishing either.**

## 2. Materials

- **Pairs:** ≥ 50 held-response-constant pairs (agreement pack seed set + expansion), each with **blind gold labels** per `harness/phase1/LABEL_PROTOCOL.md` (≥2 labelers, never the author, split→undecidable). Pairs whose gold resolves to `undecidable` are excluded from the primary endpoint and analyzed separately. **The experiment does not run before the labels exist** — author-seeded labels (`blind:false`) are not sufficient.
- **Model:** Llama-3-8B-Instruct (matches `lyra/generation.py` target). Secondary replication: one model from a different family (e.g. Mistral-7B or Qwen-2.5-7B) to check the effect isn't an artifact of one architecture.
- **Code:** `lyra/coherence.py` in teacher-forced scoring mode — no generation. Forward pass over `prefix + final_user_turn + persisting_response` with the response tokens forced; record the per-token CoherenceSignal (coherence score, pull magnitude, entropy, margin) at every response position.

## 3. Method

For each pair, two forward passes (warranted context W, inherited context I) scoring the same response tokens. Because the response is identical, token positions align one-to-one across halves; every comparison is **paired at the token level**.

Per pair, compute on response tokens only:
- `Δc̄` — difference in mean coherence, `mean_c(I) − mean_c(W)`
- `Δc_min` — difference in minimum coherence (the deepest divergence moment)
- the per-token trace, retained for qualitative analysis (where in the response does divergence appear — at the stance-bearing tokens?)

## 4. Endpoints and decision rules (fixed in advance)

**Primary endpoint:** paired test on `Δc̄` across pairs — two-sided Wilcoxon signed-rank, α = 0.01, plus a permutation test (10,000 shuffles of W/I assignment within pair) as the assumption-free check. Effect size: matched-pairs rank-biserial correlation.

**Secondary endpoint:** pair-level separability — AUC of a threshold rule on `Δc̄` for classifying which half is inherited, with 95% CI by bootstrap over pairs. Report against the two mandatory baselines: `always_abstain` (AUC 0.5 by definition) and `lexical_floor` (AUC 0.5 *by construction* — this line is the point of the design).

**Decision rules:**
- Both tests significant at α = 0.01 and AUC lower CI > 0.5 → report as **first evidence of an activation-level correlate of stance overhang** (Level 2 of the honesty ladder: correlational, not causal — say exactly that).
- Otherwise → **publish the null** with full traces: overhang is not readable from this signal at this depth on this model. The gate remains a warrant-level judgment; that is a finding, not a failure.
- No post-hoc endpoint additions. Exploratory analyses (per-token localization, entropy, pull magnitude) are labeled exploratory and never promoted to results within the same writeup.

## 5. Confound controls

1. **Context length/content asymmetry** — inherited final turns tend to be longer (they contain artifacts: diffs, citations, schedules). Controls: (a) coherence is measured **only on response tokens**, never on context tokens; (b) report `Δ context length` as a covariate and check `Δc̄` is not explained by it (regression of Δc̄ on Δlen); (c) add 10 **placebo pairs** where the final user turn differs by an equally long but warrant-irrelevant change (small talk, formatting chatter). Placebo Δc̄ should center on zero; if it doesn't, the signal is "context changed," not "warrant changed," and H1 is not supported.
2. **Politeness/sentiment asymmetry** — warranted halves carry frustration vocabulary by design. The placebo set includes polite-but-irrelevant and frustrated-but-irrelevant variants so tone alone is tested directly.
3. **Position effects** — response token positions are identical across halves by construction; no correction needed. This is the quiet superpower of held-response-constant pairs.
4. **Author leakage** — the person who authored pairs does not run the analysis unblinded: analysis script is written and frozen against the seed set *before* gold labels arrive, then run once on gold.

**Note on code prerequisites:** H2 (union top-K) and H3 (final RMSNorm on pull) in `coherence.py` / `generation.py` must be fixed before the bridge run. Pre/post coherence numbers are not comparable across the fix boundary; the discontinuity should be noted in any writeup that references historical coherence values.

## 6. Reporting

Canonical status line, per the harness discipline — no pooled score:

> *N pairs (blind, ≥2 labelers, per-class agreement w/i reported), model M. Primary: Δc̄ median __, Wilcoxon p __, effect size __. Secondary: AUC __ [CI __–__] vs lexical_floor 0.5. Placebo Δc̄ __. Verdict: correlate found / null. Level 2 (correlational); causal claim: none.*

Artifacts published regardless of outcome: frozen analysis script, per-pair traces, gold labels, and the writeup. Preprint + Alignment Forum post; `insights/` entry in the xop repo linking the result to the ladder.

## 7. What this experiment does NOT claim

Even a clean positive is **correlational** (Level 2). It does not show the feature *causes* the stance to persist (Level 4 requires suppress/amplify), does not replace the warrant judge, does not validate the gate, and does not close the oracle gap — the warrant is still defined externally, by humans, blind. The result's value is narrower and sharper: it tests, for the first time, whether the residual at the activation depth and the residual at the procedure depth are the same quantity seen twice — or two good ideas sharing a notation.

## 8. Budget

~50 pairs × 2 halves × 2 models × 1 forward pass each ≈ 200 scored sequences of < 2k tokens: hours on a single A100 (or a 4-bit local run on smaller hardware). The cost is not compute; it is the blind labels — which the Phase 3 pilot produces anyway. **This experiment is the pilot's second dividend.**

---

*Protocol frozen before first run. Amendments after freeze are logged in this file with dates, per the project's own governance discipline.*
