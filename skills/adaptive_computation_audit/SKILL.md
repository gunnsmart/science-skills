---
name: adaptive-computation-audit
description: >
  Audit claims about adaptive-depth, early-exit, and halting-based neural
  computation (ACT/PonderNet-style halting, adaptive computation time, dynamic
  depth). Use when reviewing or designing a system whose per-input compute is
  set by a learned halting signal, when a difficulty-versus-compute correlation
  is offered as evidence of adaptive computation, when a soft-training /
  hard-inference mismatch must be quantified, or when an adaptive system must be
  compared against a compute-matched static baseline.
---

# Adaptive Computation Audit

A review protocol for claims about **depth-adaptive computation**: systems that
choose how many of N ordered stages to execute per input, using a learned
halting signal.

    h_0 → h_1 → ... → h_N,       q_l = σ(W · LN(h_l)) ∈ [0, 1]
    soft:  h_out = Σ_l a_l h_l,  a_l = r_l q_l,  r_{l+1} = r_l (1 − q_l)
    hard:  exit at the first l with q_l ≥ τ

## When to use

- A system halts per input, adaptively budgets depth, or learns when to stop.
- An adaptive-computation result is claimed from a correlation between
  difficulty and compute.
- A soft (mixture) training rule is paired with a hard (threshold) inference
  rule and their agreement must be quantified.
- A halting gate must be designed, repaired, or falsified.

## Do not accept these as evidence

Aggregate soft/hard KL; correlation between "difficulty" and compute; mean exit
depth; accuracy at reduced average compute. Each is reproducible by a
depth-blind router, or minimized by a degenerate halter that has no adaptive
capacity at all. See **Negative results** below.

## Mandatory controls (run before any headline number)

| Control | Definition | What it rules out |
|---|---|---|
| Depth-blind controller | Same architecture, objective, budget, and seeds; the halting signal sees only h_0 (or the input), never h_l for l ≥ 1 | If it reproduces the correlation, accuracy, and frontier, the depth axis is decorative |
| Static compute envelope | Train fixed-depth models at each depth 1…N; take the upper envelope of metric vs. compute; compare at the adaptive system's average compute | If the envelope matches, the gain is early-exit regularization, not adaptive computation |
| Identity-halter floor | Evaluate the trivial halter q_1 ≡ 1 (zero adaptive capacity) | Its soft and hard rules agree exactly (KL = 0). Any consistency metric it also achieves certifies nothing |
| Depth-necessity margin | D* = E[ℓ(h_1)] − E[ℓ(h_N)] versus the compute price λ·E[C] | If D* ≤ λ·E[C], adaptive halting is vacuous by construction; the system should halt at stage 1 |

## Six diagnostics

**D1 — Depth channel (label-free).** Report ρ̄ = E_l ‖LN(h_l) − LN(h_{l−1})‖ and
dSNR = Var_l(z) / Var_x(z). The halting logit must move at least as much along
depth as it does across inputs. **Rule:** dSNR ≳ 10⁻¹. **If dSNR < 10⁻³** the
head cannot order mid-stages; expect a two-level exit distribution (e.g.
79% / 21% / 0%), and any reported correlation is input-conditional, not
depth-conditional.

**D2 — Exit-stage entropy (label-free).** Report the full histogram of exit
stages, its entropy H(l*), and compare H to log(1 + r_eff) from D3. **Rule:**
depth adaptation requires H(l*) > 1 bit for a task with more than two difficulty
classes.

**D3 — Trajectory rank and reach (label-free).** Report r_eff, the numerical
rank of the centered increments Δ = [δ_1 … δ_N], and the reach
R = E_x max_{l,l'} D(p_θ(·|h_l) ‖ p_θ(·|h_{l'})). By the rank bound below, an
r-dimensional trajectory admits only r-dimensional exit policies. **If
r_eff ≈ 1**, every halting policy is a scalar threshold along a line.

**D4 — Per-sample soft/hard mismatch (label-free).** Never report the aggregate
halting KL alone. Report:

    TV      = E_x [ 1 − α_{l*(x)}(x) ]            (probability units)
    off-mass per sample, its histogram (not just the mean)
    γ       = E_x ‖h_out − h_{l*}‖ / (E_x √V(x) + ε),  V(x) = Σ_l α_l ‖h_l − h̄‖²
    Slack   = E_x Σ_l α_l ( ℓ_l − ℓ_{l*} )         (sign is diagnostic; see memo)
    S       = E_x Σ_l α_l (1 − q_l)                (saturation; the vacuity normalizer)

Report S next to every consistency number. When S → 0, all soft/hard
comparisons are vacuous.

**D5 — Level drift (label-free).** Log z̄(x) = mean_l z_l(x) against training
step and mean exit stage. **If z̄ rises while the exit distribution drifts
toward earlier stages**, the compute term is inflating the sharpness coordinate
— an unpenalized, one-sided direction — and the halting policy is freezing into
whatever the early phase established.

**D6 — Causal tracking and value (evaluation only).** Report Spearman
ρ(q_l, Ψ_l) restricted to stages in the top decile of Ψ_l, where Ψ_l is the
interventional ablation effect D(p(·|h_out^{do(f_l = 0)}) ‖ p(·|h_out)).
**Rule:** ρ ≥ 0.5 in that restricted set. Also train a probe from h_l to
"continuing past l reduces the eventual loss" and report its AUC and reliability
diagram. **If AUC ≈ 0.5**, the state carries no information about its own
future and no halting signal can work.

## Negative results to test against

1. **Self-assessment bias.** For p = p_θ(·|h_out) and p′ after continuing,
   E_{y~p}[ΔCE] = KL(p ‖ p′) ≥ 0: the model's own expected value of more
   computation is *always non-positive*, with bias equal to its own drift.
   Every label-free value or causal signal is **signed wrong**, not merely
   noisy. Value must be supervised at training and only evaluated without
   labels at inference.
2. **Gradient starvation.** ∂A/∂z_k = a_k (1 − q_k) · Adv_k, with
   Adv_k = ℓ_k − L_k^+. Each factor is driven to zero by successful training
   (decisiveness, halting, convergence to a vertex of the simplex), and the
   depth-discriminative part of the advantage is O(‖g‖ρ) while the
   depth-blind part is O(1). **Consequence:** the halting policy is not
   identified by the objective; expect seed-dependent verdicts and acute
   sensitivity to any control that moves the logit level.
3. **Soft/hard irreducibility.** The deployed rule θ ↦ ℓ(h_{l*(θ)}) is
   piecewise constant, hence has zero gradient almost everywhere. Training it
   with a differentiable mixture is training a different objective; the gap is
   exactly the mixing slack. Resolve by *deriving* the halting distribution
   from a separately trained smooth value (soft-Bellman / Boltzmann over the
   loss profile), never by tuning a temperature.
4. **Rank bound.** Every halting policy evaluates h_out ∈ conv{h_1,…,h_N} ⊆
   h_0 + span(Δ). Depth adaptation cannot exceed the trajectory's rank, no
   matter how many stages are added.
5. **Metric vacuity.** In the saturated regime KL(Bern(p)‖Bern(q)) =
   O(ε log 1/ε) regardless of the difference between the underlying stage
   distributions, and it attains its global minimum at the identity halter.

## Reporting protocol

- Report the **frontier** (task metric vs. average compute) with the static
  envelope on the same axes; never a single scalar verdict.
- Report **per-sample** mismatch quantities, not marginals.
- Report **seed dispersion** and pair configurations by seed; differences
  within ~2 standard errors are not evidence.
- Make the **difficulty label external** to the model under audit (fixed
  reference model, or held-out annotation), or the correlation is circular.
- Use a **dimensionless control** — budget utilization u = (B − 1)/(N − 1)
  with a dual price — instead of an unnormalized compute coefficient λ.
- Report ρ̄, r_eff, H(l*), TV, S, and z̄ drift in **every** run, before any
  headline result.

## Dependencies

None. This is a text-only review protocol: every step is analysis of measured
quantities, with no external API, dataset, or script.

## Reference material

- `references/adaptive_computation_theory.md` — the full theory for the seven
  problems: halting signal, target-free completion, soft/hard mismatch,
  identity collapse, competing-logit asymmetry, causal evidence gap, and
  non-reproducible verdicts.
- `references/synthesis_and_derivations.md` — common root cause; the
  closed-form (soft-Bellman) halting construction; the negative results with
  proofs; the minimal falsification experiment; derivations and arithmetic.
