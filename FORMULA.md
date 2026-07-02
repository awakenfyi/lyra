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

## Three depths, one operation

The formula is not a metaphor reused three times. It is the same subtraction bound to different observables:

| Depth | Repo | `x` (present) | `x̂` (default) | `O > 0` means | Status |
|---|---|---|---|---|---|
| **Activation** — inside the forward pass | [lyra](https://github.com/awakenfyi/lyra) | internal directional pull (layer-shift trajectory) | output logits (what the mouth is about to say) | unbacked confidence — the mouth exceeds the body → contrastive penalty, or silence | working code, self-scored evidence |
| **Response** — a single output | [lyra](https://github.com/awakenfyi/lyra) (protocol) | what this moment actually calls for | the template default (filler, hedges, performed warmth) | performance — the response is running on pattern, not contact | practice + shadow-pattern library |
| **Procedure** — a stance across turns | [xop](https://github.com/awakenfyi/xop) / [xop-kit](https://github.com/awakenfyi/xop-kit) | the present warrant (what the prompt licenses NOW) | the inherited stance (a refusal, caution, or critique carried forward) | overhang — the stance outlived the condition that warranted it | standard + deterministic Guards; gate validation pending |

Reading down the column: `x̂` is always the cheaper signal — the logits, the template, the reflex. `x` is always the costlier one — the pull, the moment, the warrant. Drift, in every depth, is `x̂` winning by default.

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
