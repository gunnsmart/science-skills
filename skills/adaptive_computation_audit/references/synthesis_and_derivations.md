# Cross-Cutting Synthesis

## Q1. Common root

**One sentence.** Halting is a decision along an axis whose learning signal is three to four orders of magnitude weaker than the signal along every other available axis — trained by an objective that is linear in the decision and therefore has *zero* gradient at its own optimum — inside a parameterization where the only strongly-priced direction is one the task cannot see, and measured by statistics that are minimized by the mechanism's own collapse.

### The four causes

**RC1 — Signal hierarchy (the primary cause).**
The halting gradient is ∂A/∂z_k = a_k(1−q_k)·Adv_k (§1.2, M2′; Appendix A.2). The factors that carry *depth* information are small, each for an independent reason:

| Factor | Value at the reported operating point | Driven to zero by |
|---|---|---|
| decision margin (1−q) | ≤ 0.17 | the model's own decisiveness (success) |
| depth-discriminative advantage ⟨g, δ_k − δ̄⟩ | O(ρ) ≈ 10⁻³·‖g‖ | the trajectory's rank collapse (§4) |
| mass at stage k, a_k | ≤ 0.83, and 0 for k > l\* | the compute price and the halting itself |
| **product** | **≈ 10⁻⁴·‖g‖** | **all of the above** |

Meanwhile: the input-conditional part of the same gradient is O(1) (it is the variation of z across *samples*, not across depth), and the compute term's gradient in the level direction is λ·q_k·(remaining depth) ≈ 10⁻²–10⁻¹ (§5.2). **The controller learns what its gradient can see: a depth-blind, input-conditional router.** Every reported correlation between "difficulty" and compute is consistent with that, and none of it is evidence for depth-adaptive computation.

**RC2 — Self-extinguishing signal.**
The soft objective is linear in the halting mass, so its inner optimum is a vertex of the simplex (Appendix A.6): the optimal policy is *deterministic*, and at any deterministic policy every halting logit has exactly zero gradient. The signal that trains halting is therefore proportional to the *current regret*, i.e., it is large only while the policy is bad and vanishes as the policy improves. Whatever halting structure is learned is learned early, then frozen; from that point on the halting behavior is set by initialization, the regularizer, and the early trajectory — which is exactly what "hyperparameter sensitivity" and "each run gives a different verdict" describe.

**RC3 — Price asymmetry on a coordinate the task cannot see.**
The logit level (the sharpening coordinate, §5.2) has a task-side gradient of order the regret — small, and small *because* the trajectory is rank-deficient — and a compute-side gradient that is non-vanishing, one-signed, and grows with the remaining depth. Nothing in the objective defends that coordinate. The system drifts toward the static one-stage model, halting earlier and earlier, with its accuracy apparently intact because the first stage was already good (P3: the mixture is confined to a rank-deficient hull, so "early" and "late" states are nearly the same state).

**RC4 — Measurement vacuum.**
The instruments cannot see the phenomenon. Aggregate halting KL is minimized by the degenerate halter (§3, R3) and is bounded by ε log(1/ε) in the saturated regime whatever the stage distributions do (P5). The completion target has a run-dependent scale (§2, D2). The reported "difficulty" correlation may be partly circular if difficulty is measured from the same model's behavior. And the correlation is a scalar verdict where the decision-relevant object is a frontier. **The field has been optimizing against an instrument whose optimum is the absence of the mechanism.**

### Which problems are actually the same problem

Seven problems reduce to **four technical problems**, with three of the seven being restatements:

| Problem | Primary cause | Secondary | Fix (structural) |
|---|---|---|---|
| 1. Halting | RC1 + RC2 | RC3 | value-derived halting; contrastive features; logit-space losses; budget constraint with dual price |
| 2. Target-free completion | RC-value: it is a *prediction*, not a functional (§2) | RC4 | supervised value head (labels at training, none at inference); regret metric |
| 3. Soft–hard mismatch | RC4 (vacuous metric) | RC3 | report (TV, off-mass, trajectory spread, mixing slack, deployment-Δ); derive the halting so consistency is structural |
| 4. Identity collapse | RC-representation: rank deficit | LN blindness (§4, Reason 4) | normalized increments; propagate-effect diagnostics; over-depth diagnosis; step normalization |
| 5. Asymmetric leverage | RC3 (level coordinate) | over-parameterized controls | per-sample logit normalization; contrastive head input |
| 6. Causal gap | RC-value: self-assessment is structurally biased (§6.4) | RC4 | interventional Ψ; supervised critic; causality at inference, value at training |
| 7. "Chaos" | RC4 (frontier vs point) | RC1 ill-conditioning + two attractors | dimensionless control; frontier reporting; response-coordinate collapse |

The equivalences: **2 ≡ 6** (both are "the value of computation is a supervised prediction, and no self-referential signal can substitute for it"); **3 ≡ 5** (both are consequences of the saturated regime produced by the level coordinate — one is its metric signature, the other its parameterization); **1 is the union of all four**; and **7 is not a phenomenon at all — it is the signature of measuring a curve at points.**

---

## Q2. Unifying principle

**The halting decision is an optimal-stopping decision, so the halting probability must be a statistic of a value function — not a gate trained against a cost.**

### The principle

Define, for a fixed input and its label, the per-stage loss profile ℓ_l. The optimal stopping problem is

    stop at stage l  ⟺  ℓ_l ≤ W_{l+1} + λ·(expected remaining depth)

where W_{l+1} is the *best achievable loss from stage l+1 onward* (the value of continuing) and λ is the price per stage. The sufficient statistic for the halting decision is the **advantage**

    Adv_l = W_{l+1} + λ·E[remaining l] − ℓ_l       (continue if positive, stop if negative)

Every design failure in this memo is a violation of that statement:

- a gate trained against a *cost* rather than a *value* has no way to represent "stop here because this is the trough of the profile" (the objective it sees is a *rate*, and rates are depth-blind);
- a gate whose input is LN(h_l) cannot see the motion that carries the profile (Problem 4);
- a gate trained directly through a threshold has a vanishing and ill-conditioned gradient (RC1, RC2, and the soft–hard dilemma);
- a gate evaluated by an aggregate halting metric cannot be certified (RC4).

### The construction, and why it is exact

The striking fact is that once the halting probabilities are derived from the value, the correct halting distribution is *available in closed form*. Let W_{l+1} = −τ log Σ_{k>l} e^{−ℓ_k/τ} be the soft-min of the remaining losses (τ the value resolution), and let the cost enter as an additive λ per stage inside that soft-min. Choosing

    q_l = σ( (W_{l+1} − ℓ_l) / τ )

makes the induced halting mass exactly the Boltzmann distribution over stages:

    a_l = r_l q_l = e^{−ℓ_l/τ} / Σ_{k=1}^{N} e^{−ℓ_k/τ}      (Appendix A.6, proven by induction)

Three classical objects collapse into one another:

- the stick-breaking halting distribution,
- the soft-Bellman (log-sum-exp) value recursion,
- a softmax attention over depth with logits −ℓ_l/τ.

Consequences, all of which are exactly the properties whose absence caused the seven problems:

1. **The halting logit is a contrast, never a level.** (W_{l+1} − ℓ_l) is invariant to adding a constant to the loss profile and to the logits; the sharpening coordinate that RC3 exploits does not exist in this parameterization.
2. **The signal is depth-discriminative by construction.** The target is a *difference between two depths*, so it cannot be produced by an input-conditional router that ignores depth.
3. **The soft–hard gap is a theorem, not a tuned quantity.** The soft rule is a Boltzmann policy over stages and the hard rule is its argmax; the per-sample disagreement is exactly 1 − max_l a_l in probability units. Since H(a) ≤ −log max_l a_l, we get 1 − max_l a_l ≤ 1 − e^{−H(a)}: vanishing entropy implies vanishing mismatch, and the entropy of the derived distribution is a single interpretable number that *controls* the gap instead of a swept temperature hoping for it. τ ↓ 0 (relative to the spread of the loss profile) is the limit in which the two rules coincide.
4. **The gradient does not starve.** The gradient flows through the value regression — a smooth, well-conditioned target (the per-stage losses are smooth functions of the parameters) — instead of through the halting probability, whose gradient vanishes at its own optimum. The stop/continue decision is *derived*, so no gradient needs to flow through it.
5. **Calibration is measurable and becomes the single point of failure.** The halting head is a predictor of a difference in per-stage losses; its reliability diagram is checkable, and the deployment criterion (does the predicted profile's argmin match the realized profile's argmin) is a one-number report.

**The catch, and it is the same catch as Problem 6.** The value must be *predicted* from the current state before the future is known. At training that is a supervised regression on the model's own multi-depth losses (labels available, no depth annotation needed, fully differentiable, single phase, no sampling). At inference the prediction is used label-free. The failure mode is therefore exactly the one isolated in §6.4: if the predicted advantage is uncorrelated with the true advantage, the derived policy is arbitrary, and Prop 6.1 guarantees that no *self-referential* substitute can repair it. **"The value of computation is a supervised quantity" is the irreducible content of the whole problem cluster.**

### What the principle resolves, in one list

| Symptom | Resolution |
|---|---|
| unbounded hyperparameter tuning (P1, P7) | no temperature to sweep (τ enters the data as a resolution, and the alternative—a budget constraint with a dual price—removes λ); no level coordinate to inflate |
| non-monotone in N (P1) | the halting statistic is a *difference of values*, defined at any N; and over-depth shows up as W_{l+1} ≈ ℓ_l (a flat profile), which is measurable rather than mysterious |
| completion requires y (P2) | completion *is* the value; it is learned with labels at training and evaluated without them |
| soft/hard mismatch (P3) | becomes the entropy of a Boltzmann policy, a controlled and interpretable quantity |
| identity collapse (P4) | the value profile is trivial when the trajectory is rank-deficient, so the same diagnostic exposes it; and the fix (contrastive features) is a precondition the principle makes explicit |
| logit asymmetry (P5) | the halting logit is a contrast; no bounded-vs-unbounded race exists |
| causal gap (P6) | causality is measured by intervention at inference; value is supervised at training; the two are no longer conflated |
| "chaos" (P7) | no level coordinate, no temperature, no λ: the remaining control is a dimensionless budget, and the verdict is replaced by a frontier |

---

## Q3. Strongest negative results

Five statements that hold under the stated constraints, each with its consequence. Together they define the boundary of what any engineering effort can achieve here.

**NR1 — No label-free value signal can be unbiased (self-assessment bias).**
*For p = p_θ(·|h_out) and p′ = p_θ(·|h_out^{continue}), E_{y~p}[ΔCE] = KL(p‖p′) ≥ 0, while the true value is E_{y~p*}[ΔCE] = KL(p*‖p) − KL(p*‖p′).*
Consequence: any controller, critic, or gate that judges the value of its own computation from its own outputs will *always* prefer to stop, with a bias exactly equal to its own drift. Label-free "causal effect" estimates are not merely noisy; they are signed wrong. The value of computation is necessarily a supervised object.

**NR2 — Direct training of a halting rule starves itself (gradient starvation).**
*∂A/∂z_k = a_k(1−q_k)Adv_k; the relaxed optimum is a vertex; and at every deterministic halting policy all halting gradients are exactly zero. Moreover the depth-discriminative part of the advantage is O(ρ) while the depth-blind part is O(1).*
Consequence: the halting policy is not identified by the objective. What is learned is whatever the early, high-regret phase produces, plus whatever the regularizer and the level coordinate impose. All hyperparameter sensitivity, all "one good point" behavior, and all seed-dependent verdicts follow from this, not from chaos. Any *new* loss term on the halting probability inherits the same fate unless it is depth-contrastive and supervised by a value.

**NR3 — The soft–hard gap is irreducible under a threshold rule.**
*The deployed objective θ ↦ ℓ(h_{l*(θ)}) is piecewise constant in θ, hence has zero gradient almost everywhere; the only admissible differentiable surrogate (no RL, no sampling) is the mixture, whose value differs from the deployed value by exactly the mixing slack Σ_l a_l(ℓ_l − ℓ_{l*}), and the slack vanishes only as the halting entropy vanishes — which removes the gradient.*
Consequence: **"differentiable", "threshold-hard at inference", and "exactly consistent" cannot all three hold**, unless the halting distribution is *derived* from a separately-trained smooth quantity (the Q2 resolution). This is the sharpest statement of why the halting problem has resisted the standard toolbox for a decade: the toolbox's own constraints exclude the object being trained.

**NR4 — Depth adaptation is bounded by the rank of the trajectory.**
*Every halting policy evaluates h_out ∈ conv{h₁,…,h_N} ⊆ h₀ + span(δ₁,…,δ_N); hence the variation attributable to halting lives in an r_eff-dimensional affine set, and a head reading LN(h_l) with ‖W‖ has depth resolution at most ‖W‖·ρ̄.*
Consequence: for a fixed trajectory there is an N beyond which adjacent stages are indistinguishable to any head, and a system whose exit-stage entropy does not exceed log(1 + r_eff) is not adapting over depth, whatever it is doing. Increasing N without increasing the trajectory's rank buys latency, not computation. Corollary of the measured numbers: with ρ̄ ≈ 10⁻³ and a head at the z_a-scale, the depth channel is ~10⁻³ nats against an input channel of ~1 nat, i.e. ~ −50 dB; the observed exit distribution (79/21/0) is the signature of a channel with room for one or two distinguishable levels.

**NR5 — No available instrument can certify progress (metric vacuity).**
*Aggregate soft/hard KL is O(ε log 1/ε) throughout the saturated regime, is blind to which samples exit where, and attains its global minimum at the degenerate identity halter. The completion target's scale depends on run-specific quantities (ℓ₁ − ℓ_N, ε). Correlations between "difficulty" and compute are reproducible by depth-blind routers.*
Consequence: the field cannot currently distinguish a working adaptive system from a decorative one using its own reported statistics. **The only certificate that survives is a compute-matched frontier:** the task metric as a function of average compute, compared against the best *static* depth at the same compute. If the static envelope matches, adaptivity contributed nothing — and no amount of correlation, KL, or ablation-table polish changes that.

### What is impossible, in one paragraph

Under the stated constraints — differentiable only, no sampling, no RL, single phase, no depth labels, same semantics at train and inference — four things are impossible, and they are the four things the seven problems were each reaching for. It is impossible to *train* a threshold halting rule directly and have it remain both differentiable and deployable (NR3), because the deployed rule has no gradient and the differentiable surrogate is a different objective. It is impossible to *estimate the value of computation at inference* from the system's own state or outputs without bias (NR1), because self-assessment is non-positive by construction. It is impossible to *adapt over depth* beyond the rank of the trajectory (NR4), because the output set reachable by halting is confined to the trajectory's affine hull. And it is impossible to *certify* any of this with aggregate halting statistics (NR5), because those statistics are minimized by the mechanism's collapse. What remains possible is narrower, harder, and well-posed: train a *value* of depth by regression on the model's own loss profile (which has a smooth, well-conditioned gradient), *derive* the stopping rule from it (so the deployed rule needs no gradient at all), keep the depth channel alive by measuring contrasts rather than levels (which is what restores identifiability in the first place), and grade the result on a compute-matched frontier.

---

## Q4. Minimal falsification

### The single experiment

**Train the same system with a depth-blind controller.** Same architecture, same objective, same budget, same seeds; the only change is that the halting logit is computed from the *input* alone (or from h₀), with no access to h_l for any l ≥ 1 — so it cannot see the state's evolution along depth at all.

**Then compare:** (i) the difficulty–compute correlation; (ii) accuracy at matched average compute; (iii) the compute–accuracy frontier.

**Why this is the right falsifier.** The paradigm's observable claim is "harder inputs get more computation." A depth-blind controller can produce that correlation trivially, because "hard" is an input property and the correlation is a *marginal* statement about inputs. The paradigm's *mechanistic* claim is stronger: that the computation itself is staged, that intermediate states are informative, and that the halting decision is made *in depth*. Only a depth-blind control separates the two. **If the depth-blind controller reproduces the correlation, the accuracy, and the frontier, then the depth axis contributed nothing in that system, and the adaptive behavior is an input-conditional router with a cost side-effect.**

**A stronger, cheaper companion test:** the **static envelope at matched compute.** Train static models at each fixed depth 1…N, take the upper envelope of accuracy versus compute, and compare it with the adaptive system at the same average compute. The reported numbers make this test unusually sharp: the adaptive system reaches accuracy 0.78–0.79 at 2.50–2.90 units of compute (of 6), while the full-budget system reaches only 0.69–0.71. A static model trained at depth ≈ 2.5–3 with the same recipe is therefore the decisive comparison. If it also reaches 0.78–0.79, then **the entire reported advantage is early-exit regularization** — the deeper stages *hurt* accuracy, and the adaptive system's gain is simply "stop before degrading" — not adaptive computation. (Note that "deeper stages hurt" is itself consistent with the identity-collapse diagnosis: 8 of 12 stages add nothing and can only disturb the readout.)

**A third, one-line kill criterion — the depth-necessity margin.** Measure D\* = E[ℓ(readout(h₁), y)] − E[ℓ(readout(h_N), y)] and the total compute price λ·E[C]. If D\* ≤ λ·E[C], adaptive halting is *vacuous by construction*: a one-stage model attains the objective, and the collapses observed in §1 and §4 are the correct answer, not a failure. This test requires no new architecture and no new loss.

### Measurement-validity caveat to fix before any of the above

Three of the reported statistics are not yet trustworthy as evidence:

1. **The "measured difficulty" label.** If difficulty is computed from the same model's behavior (token counts, confidence, stage losses), then "correlation between difficulty and compute" is partly a statement about the model's own self-consistency. Difficulty must be an *external* quantity: a fixed reference model's loss, a human or procedural annotation, or an oracle budget held constant across systems.
2. **The diagnostics' units (§4, Reason 4, and Appendix A.9).** The reported cosine and the reported per-stage increments cannot both be in a common unit; the arithmetic forces the conclusion that the state's motion is predominantly radial (norm growth), which LN deletes before the halting head sees anything. The increment statistic should be redefined once (ρ_l = ‖LN(h_l) − LN(h_{l−1})‖), and the radial and shape components reported separately.
3. **The verdict's noise floor.** From the reported 5-run fixed configuration: the correlation has mean ≈ 0.502 with seed standard deviation ≈ 0.066, and the compute has mean 2.61 with standard deviation 0.165 (6.3%). Any comparison of correlations differing by less than ≈ 0.12 between configurations trained with 5 seeds each is inside the noise. Several of the sweeps used to establish "each change requires retuning all" may be comparing noise to noise. Report seed-mean verdicts with intervals, and pair configurations by seed.

**Pre-registered prediction (a prior, not a result).** Given the 79/21/0 exit distribution, the ~10³ imbalance between the compute gradient and the depth-discriminative task gradient, and the fact that ρ ≈ 10⁻³ puts the depth channel at roughly −50 dB, I would put **65–75%** on the depth-blind controller reproducing the reported difficulty–compute correlation and accuracy within the seed noise band, and **40–55%** on a static depth-2–3 model matching the adaptive system's accuracy at matched compute. If both happen, the depth-adaptive paradigm as specified is falsified in that system. If the depth-blind controller fails but the static envelope matches, the diagnosis shifts from "the controller is depth-blind" to "the depth axis is decorative," which is a different and equally important conclusion.

---

## Q5. Five-year bet

### Where the money goes

**Bet 1 (≈60%) — Value-based halting.** Fund the construction of Q2: a multi-depth value head trained by regression on the model's own per-stage loss profile, with the stopping rule *derived* from the predicted profile (soft-Bellman/Boltzmann in training, argmin at inference), a budget constraint with a learned dual price instead of a swept λ, and a contrastive, scale-carrying input to the value head.

*Why:* it is the only route in this analysis that converts an unidentified control problem into a well-posed regression problem. It dissolves NR2 and NR3 (the gradient flows through the value, and the decision needs no gradient), removes RC3 (no level coordinate, no temperature to sweep), and yields RC4's certificate (the frontier is a direct output of a value function). It is also *cheap*: a small head plus one extra target per stage, no new data, no new labels.

*Deliverables, in order, each falsifiable:*
1. **Does the state know its own future?** (Test 2A) A critic predicting whether continuing past l reduces the eventual loss, evaluated by AUC and a reliability diagram on held-out data. Kill criterion: AUC ≤ 0.6 — the state carries no information about its own future, and nothing downstream can work.
2. **Does the derived policy beat the static envelope at matched compute?** (Q4.) Kill criterion: the static envelope dominates for all budgets.
3. **Is the frontier monotone in N?** The compute–accuracy frontier must improve with N *at fixed budget* — if it does not, depth adds nothing and the honest design is a shallow model.
4. **Is the soft/hard gap now a controlled quantity?** The entropy of the derived halting distribution must predict the per-sample disagreement between the soft mixture and the hard exit (a testable relation, not a swept constant).

**Bet 2 (≈25%) — Depth-informative representations.** Fund the measurement and repair of the representation problem: normalized contrastive features (the head must see the increment, not the level), step normalization (so stages cannot silently become no-ops), propagated-effect diagnostics (ablate a stage, measure the change in the *output distribution*, not the norm of Δh), and over-depth diagnosis (reduce N until ρ̄ is O(1), before attempting to build a controller at all).

*Why:* Bet 1 cannot work if the depth channel is at −50 dB; this is the precondition, and it is the part of the analysis with the least uncertainty attached. It also produces the instruments (r_eff, the depth-channel SNR, exit-stage entropy) that make every other claim checkable, and it costs almost nothing to run.

**Bet 3 (≈15%) — Instrumentation and reporting.** Fund the frontier as the primary reported object; the response coordinates (utilization, saturation, drift, depth SNR) in place of raw hyperparameter sweeps; the per-sample mismatch metrics (TV at the exit stage, off-mass, trajectory spread, mixing slack, deployment-Δ); and the interventional layer (Ψ_l by ablation at evaluation, calibrated against a learned probe). *Why:* every negative result in this memo was invisible to the field's own metrics. RC4 is not a symptom; it is the reason the other three causes survived for a decade. A subfield that can only report correlations that a depth-blind router reproduces will keep rediscovering the same collapse.

### Explicit non-bets

- **Temperature scheduling and sharpening tricks.** They move the operating point along a curve whose shape is set by a collinearity; they cannot create a depth signal, and they convert a structural quantity into a schedule artifact.
- **Curriculum and depth labels.** Excluded by the constraints, and unnecessary: the value target is the model's own loss profile.
- **RL and sampling.** Excluded, and unnecessary for the same reason.
- **Larger N.** NR4 says increasing depth without increasing the trajectory's rank buys latency. The correct direction is to *find* N\* (the depth at which the trajectory's rank saturates) and then spend the effort on the value.
- **More hyperparameter sweeps.** By Q1's arithmetic, the sweeps are traversing noise and a single curve. They are the least informative experiment per unit of compute available in this field.

### The honest fallback

If Bet 1's first deliverable fails — the state does not predict its own future improvement, AUC ≈ 0.5 — then no halting mechanism, of any design, can be more than an input-conditional router at this scale, and the correct conclusion is that **the depth axis does not carry a learnable value signal in these models.** In that case the rational five-year bet is to stop investing in depth adaptivity and invest in the axis where increments are O(1), externally verifiable, and value-bearing: the token/reasoning axis. That migration has already happened once in the literature, and the analysis here suggests it was not fashion — it was an empirical gradient. The value of the experiments above is that they convert that migration, and its opposite, from a judgement call into a measurement.

---

# Appendix — Derivations and arithmetic

**A.1 Mass identity.** a_l = r_l q_l = r_l − r_{l+1} (since r_{l+1} = r_l(1−q_l)), so Σ_{l=1}^{N} a_l telescopes to r₁ − r_{N+1} = 1 − Π_l(1−q_l). Hence the soft output is a convex combination iff q_N ≡ 1 or the mixture is renormalized; otherwise a free scalar Σ a_l multiplies the output (P1).

**A.2 Halting gradient, in exact form.** With A = Σ_l a_l ℓ_l, a_l = q_l Π_{k<l}(1−q_k):

    ∂a_l/∂q_l = r_l ,        ∂a_l/∂q_k = −a_l/(1−q_k)  (k < l) ,    ∂a_l/∂q_k = 0 (k > l)

    ∂A/∂q_k = r_k(ℓ_k − L_k^+) ,  where L_k^+ = Σ_{l>k} a_l ℓ_l / r_{k+1}  (expected loss of continuing)

    ∂A/∂z_k = q_k(1−q_k)·∂A/∂q_k = a_k(1−q_k)(ℓ_k − L_k^+)  =  a_k(1−q_k)·Adv_k.   ∎

**A.3 Cost gradient and the level direction.** With C = Σ_l r_l and ∂r_l/∂q_k = −r_l/(1−q_k) for k < l:

    ∂C/∂q_k = −(1/(1−q_k)) Σ_{l>k} r_l   ⇒   ∂C/∂z_k = −q_k Σ_{l>k} r_l < 0.

Non-vanishing whenever q_k > 0 and stages remain, i.e. the compute term pushes every logit upward, permanently, with price λ·|∂C/∂z_k| ≈ λ·q_k·(remaining). By contrast the depth-contrastive task gradient at the reported operating point is a_k(1−q_k)‖g‖ρ ≈ 0.14·10⁻³‖g‖ ≈ 10⁻⁴‖g‖. Ratio ≈ 10²–10³ at λ = 0.05. ∎

**A.4 At a deterministic policy, every halting gradient vanishes.** Let a = δ_{l\*}. Then a_k = 0 for k ≠ l\*, and 1 − q_{l\*} = 0. Hence ∂A/∂z_k = a_k(1−q_k)Adv_k = 0 for every k. Since the soft objective is linear in a, its minimizer over the simplex is such a vertex; therefore the halting gradient is proportional to the *regret* and vanishes at the very configuration training is driving toward. ∎ (NR2)

**A.5 Rank/reach bound.** h_out = Σ_l a_l h_l with a ∈ Δ^{N+1}. The convex hull of points in h₀ + span(δ₁,…,δ_N) lies in the affine hull of dimension ≤ r_eff. Therefore all halting-induced variation lives in ≤ r_eff dimensions; a rank-1 trajectory admits only threshold policies along a line, since then ℓ_l = ℓ(⟨u, h_l⟩) is a function of one scalar. The depth-discriminative part of the task gradient is ⟨g, δ_l − δ̄⟩, which is the *only* part that can order stages; it scales as ‖g‖·ρ_l. ∎ (NR4)

**A.6 Soft-Bellman identity.** Define W_{N+1} = +∞ (forced halt), and backward W_l = −τ log(e^{−ℓ_l/τ} + e^{−W_{l+1}/τ}), q_l = e^{−ℓ_l/τ}/(e^{−ℓ_l/τ} + e^{−W_{l+1}/τ}).

*Claim:* a_l = r_l q_l = e^{−ℓ_l/τ}/Z with Z = Σ_{k=1}^{N} e^{−ℓ_k/τ}. *Proof by backward induction.* At l = N: q_N = 1, a_N = r_N, and Z_N := e^{−ℓ_N/τ}, so a_N = e^{−ℓ_N/τ}/Z_N. Assume for l+1: the conditional (given survival to l+1) mass on stage k ≥ l+1 is e^{−ℓ_k/τ}/S_{l+1} with S_{l+1} = Σ_{k≥l+1} e^{−ℓ_k/τ} (note e^{−W_{l+1}/τ} = S_{l+1}). Then q_l = e^{−ℓ_l/τ}/(e^{−ℓ_l/τ} + S_{l+1}) = e^{−ℓ_l/τ}/S_l, and 1 − q_l = S_{l+1}/S_l. So a_l = r_l q_l = r_l e^{−ℓ_l/τ}/S_l, and for k > l, a_k = r_l(1−q_l)·(conditional mass) = r_l (S_{l+1}/S_l)(e^{−ℓ_k/τ}/S_{l+1}) = r_l e^{−ℓ_k/τ}/S_l. Induction closes with r_1 = 1 and S_1 = Z. ∎

Consequences: (i) the halting logit is exactly the *rescaled advantage* z_l = (W_{l+1} − ℓ_l)/τ; (ii) the halting distribution is a Boltzmann distribution over the loss profile; (iii) the soft–hard gap is exactly the entropy of that distribution, and it is controlled by τ relative to the spread of the profile; (iv) a constant added to all ℓ_l cancels — the halting logit is a contrast, so no level coordinate exists to inflate.

**A.7 Self-assessment bias.** For p, p′ fixed distributions and y ~ p:

    E_p[−log p′_y] − E_p[−log p_y] = Σ_y p_y log(p_y/p′_y) = KL(p‖p′) ≥ 0.

    For y ~ p*:  E_{p*}[−log p′_y] − E_{p*}[−log p_y] = [H(p*) + KL(p*‖p′)] − [H(p*) + KL(p*‖p)] = KL(p*‖p) − KL(p*‖p′).

    Therefore  E_p[Δ] − E_{p*}[Δ] = KL(p‖p′) + KL(p*‖p′) − KL(p*‖p).   ∎ (NR1)

The first identity is the reason no self-referential signal can value computation: improvement is invisible from the inside, and the strength of the blindness equals the model's own drift.

**A.8 Mixing slack.** J_soft − J_deployed = Σ_l a_l ℓ_l − ℓ_{l*} = Σ_l a_l (ℓ_l − ℓ_{l*}), first order in the mass off the argmin. If the loss is concave along the trajectory, J_soft > min over vertices, so the training objective is strictly optimistic relative to any hard rule; if convex, pessimistic. Exact consistency ⟺ a is a point mass at the argmin ⟺ zero halting gradient (A.4). ∎ (NR3)

**A.9 Saturation KL and Pinsker.** For p, q ∈ [1−ε, 1]: KL = p log(p/q) + (1−p) log((1−p)/(1−q)) ≤ ε log(1/(1−ε)) + ε log(1/ε′) = O(ε log(1/ε)), *independent of how different the underlying stage distributions are*. Pinsker: |p − q| ≤ √(KL/2), so the reported KL ∈ [0.001, 0.003] bounds the mean difference between the two *halting-mass scalars* by 0.022–0.039 — a statement about saturation, not about agreement. The per-sample quantity that does measure the mismatch is 1 − α_{l*}(x) ≈ 0.17–0.21 on the same runs. ∎ (NR5)

**A.10 The reported diagnostics' units (radial motion and LN blindness).**
arccos(0.9975) = 0.0707 rad = 4.05°; sin(4.05°) = 7.1% of the state norm in perpendicular displacement over eleven stages. LN is invariant to positive scaling, so the head sees only the shape component, of which the angle is a special case: the head's input varies by ~4°, i.e. ~0.006 rad per stage on average. Per-stage *state* increments of 0.07–0.14 (reported) over two or three terminal stages sum to 0.14–0.42 — far more than 7% — which is only consistent if those increments are predominantly along the state's own direction (norm growth), the component LN removes. Conclusion: the state moves; the halting head does not see it; and the cosine, the increment norms and the head's input variation are three different quantities that must be reported separately. ∎ (Problem 4, Reason 4)

**A.11 Noise floor of the reported fixed-configuration runs.**
correlations {+0.433, +0.494, +0.611, +0.482, +0.491}: mean 0.502, seed sd 0.066, se 0.030. compute/token {2.53, 2.60, 2.90, 2.50, 2.53}: mean 2.612, seed sd 0.165 (6.3%). For two configurations with 5 seeds each, the se of the difference in correlations is ≈ 0.042, so differences below ≈ 0.12 are inside the noise band. Several reported sweep-driven verdict changes are of that size. ∎ (Q4, measurement-validity item 3)

---

# Closing statement

The seven problems are one problem seen from seven angles: *a decision along the depth axis is being trained by a signal that barely varies along it, in a parameterization where the one direction the objective prices heavily is invisible to the task, and it is being graded by statistics that attain their optimum when the mechanism stops working.*

The way out is not a better halting loss. It is to stop training the decision and start training the **value**: a supervised predictor of the remaining loss profile, from which the stopping rule is *derived* rather than learned. That move dissolves the gradient starvation (the value's gradient is well-conditioned and does not vanish at the decision), converts the soft–hard mismatch from a hope into the entropy of an explicitly constructed Boltzmann policy, eliminates the level coordinate that the compute term exploits, gives the completion problem a target-free *inference* (with a supervised *training* signal), and replaces the unmeasurable verdict with a frontier that a static baseline can falsify.

And the falsification is cheap, immediate, and available with the numbers already in hand: **train a controller that cannot see depth, and a static model at the adaptive system's average compute. If either matches, the depth axis was decorative — and the measurement will have said so in a way no correlation between difficulty and compute ever could.**
