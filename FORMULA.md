# The Lyra Formula

*The residual, defined once. Every repo in this org links here instead of redefining it.*

---

## The equation

```
L = x − x̂

x    what is actually present    (the warrant, the pull, the behavior)
x̂    what the default predicts   (the reflex, the template, the claim)
L    the residual                 (what remains when the default is subtracted)

O = x̂ − x = −L    overhang: the part of the default that exceeds what's present
```

One subtraction. `x` is always the thing that's actually here; `x̂` is always the habitual account of it. The residual is what a response, a stance, or a token distribution contains *beyond* its default — or the gap where the default has outlived what's present.

## The sign convention (locked — do not flip)

```
O > 0    the default exceeds what's present     → overhang / performing
O ≈ 0    default and present agree              → aligned / coherent
O < 0    what's present exceeds the default     → dropped stance / missing response
```

An earlier draft had this flipped and it confused the entire system. **O positive means the reflex outlived its warrant.** UI and prose report **O** (the readable quantity); **L** is the canonical residual, exposed as `L = −O`.

## Why this notation survived its own retirement

Concede the boundary first: **nothing on this page is computed.** `L` is not evaluated at any depth in this repo, and no decision anywhere reads its value.

The internal note `LYRA_ON_FORMULAS.md` retired `L = x − x̂` by name. Its charge: the equation "reads like a measurement, but in most domains `x̂` is unobservable, so the equation was a pointer dressed as arithmetic." That charge is correct and this page does not contest it — the activation section below concedes exactly the same thing in the one place the project has real numbers, where `D_act` is an unsigned JSD, is not `L`, and signed `O_act` is unimplemented.

What is readmitted here is not arithmetic. It is **a naming scheme and a sign convention**: one pair of names (`x` present, `x̂` default) and one locked direction (`O > 0` means the default outlived what's present), shared across three depths that would otherwise each invent private vocabulary for the same asymmetry. That is a glossary with a minus sign in it. It earns its place on two grounds — it fixes the direction an earlier draft had backwards, which confused the whole system; and it is what lets the bridge claim be stated as a falsifiable prediction instead of an analogy.

Where it does not earn its place: any sentence that writes `L` as a value, a threshold, or a step to "compute." The note's ruling on `L_runtime = warm − cold` applies here unchanged — the operation is a classification, so write it as one. The unlock condition is unchanged too, and it is the same one everything else waits on: **validate the judge.** Until then `L` is how this program names a direction, not how it measures one.

**One binding, settled.** `x̂` is the **stated** term at every depth — the account the system gives of itself, available cheaply. `x` is the **actual** term — what is really so, available only at a cost. At procedure depth that means `x̂` is the condition a stance cites as its justification, and `x` is whether that condition still holds at the final turn.

The Procedure row above previously read the other way, putting the carried stance itself in `x̂` and the warrant in `x`. That was not merely a flip: it made the subtraction type-incoherent, since a stance is an object and a warrant is a condition, and the two cannot be subtracted. Every other source binds it as stated-minus-actual — `xop/standard/CONCEPTS.md` ("self-description / the stated account"), `xop/standard/SPECIFICATION.md`, `xop/standard/xOP_Standard_v0_2.md` ("actual minus stated"), `xop/catalog/AOP-01` and `COP-01`, and the two implementations, `xop/harness/pause.py` (`x_hat = stance["trigger"]`) and `lyra_xop/schema.py` (`warrant: str  # x̂`). This row was the lone outlier, so it is the row that moved; no code changed.

## Three depths, one operation

The formula is not a metaphor reused three times. It is the same subtraction bound to different observables:

| Depth | Repo | `x` (present) | `x̂` (default) | `O > 0` means | Status |
|---|---|---|---|---|---|
| **Activation** — inside the forward pass | [lyra](https://github.com/awakenfyi/lyra) | internal directional pull (layer-shift trajectory) | output logits (what the mouth is about to say) | unbacked confidence — the mouth exceeds the body → contrastive penalty, or silence | working code, self-scored evidence |
| **Response** — a single output | [lyra](https://github.com/awakenfyi/lyra) (protocol) | what this moment actually calls for | the template default (filler, hedges, performed warmth) | performance — the response is running on pattern, not contact | practice + shadow-pattern library |
| **Procedure** — a stance across turns | [xop](https://github.com/awakenfyi/xop) / [xop-kit](https://github.com/awakenfyi/xop-kit) | whether the cited condition still holds at the final turn (read from the transcript) | the condition the stance cites as its justification (the original trigger) | overhang — the stance outlived the condition that warranted it | standard + deterministic Guards; gate validation pending |

Reading down the column: `x̂` is always the cheaper signal — the logits, the template, the reflex. `x` is always the costlier one — the pull, the moment, the warrant. Drift, in every depth, is `x̂` winning by default.

### What the activation code computes today (the proxy)

`L = x − x̂` is the conceptual formula. The current activation implementation ([`lyra/coherence.py`](lyra/coherence.py)) does **not** compute signed `L`. It computes an unsigned proxy:

```
D_act = JSD(P_pull, P_out)     symmetric Jensen–Shannon divergence over union top-K
C_act = 1 − D_act              activation coherence, in [0, 1]
```

JSD is symmetric, so `D_act` measures the *magnitude* of body/mouth divergence but cannot say which side exceeds the other — it is not `L` and not `O`. Signed `O_act` (a directional estimator) is **unimplemented**; see the v0.3 roadmap in the README. Read `C_act` as a coherence proxy, not as the canonical residual.

## The gate, in formula terms

The xOP Constitution's gate — `false_positive_on_warranted == 0` — is a constraint on acting against the sign of O:

> Never treat a state as overhang (`O > 0`) when the warrant is still present (`O ≤ 0`).

Asymmetric on purpose. Holding a stale stance (missing a true `O > 0`) is a coverage failure — bounded by the floor, correctable. Overriding a warranted one (calling `O > 0` when it isn't) is the failure that *feels like help*, and it is the one the system treats as unforgivable. The gate protects the warranted state in whichever direction it points: sometimes *don't force compliance*, sometimes *don't force confrontation*.

## The falsifiable claim

Because the three depths bind the same subtraction, they make a testable prediction: **signals at one depth should carry information about another.** Specifically — on held-response-constant minimal pairs (identical persisting response; only the warrant differs), no surface-text detector can separate warranted from inherited by construction. If the activation-depth residual separates them even weakly, the formula's depths are measurably connected. If it doesn't, overhang is decidable only at the warrant level, and the formula's unity is structural, not physical. Either result is publishable. See [`experiments/bridge/PROTOCOL.md`](experiments/bridge/PROTOCOL.md).

## What the formula is not

- Not a score to optimize. Optimizing for small O directly produces the Always-Abstain failure (never commit, never overhang, never useful). The coverage floor exists because of this.
- Not a claim that the three depths are already empirically linked. Today the link is structural — same operation, same sign convention. The bridge experiment is how that claim earns or loses its evidence.
- Not sentiment. `x` is defined by warrant and observables, never by whether anyone felt better (`CONSTITUTION.md §III`).

---

*One subtraction, three depths, one gate. `x̂` is what the system would do anyway; `x` is what's actually here; everything Lyra measures is the difference.*
