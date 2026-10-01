# The Missing Invariant

### Seven open problems in adaptive sequential computation, and the measurement that binds them

*A research memo. Pure theory. No code. All empirical figures are quoted from the runs supplied with the inquiry.*

---

## Bottom line up front

1. **The seven problems are not independent.** They are seven views of three missing invariants: (i) a *scale-free parameterization* of the halting decision, (ii) a *depth-informative* state representation, and (iii) a *value* signal for computation rather than a *cost* signal.

2. **The reported numbers already contain the diagnosis.** cos(h₁,h₁₂) = 0.9975 with mid-stage increments of 10⁻⁴–1.5×10⁻³ means the depth-induced motion of the halting head's input is two to three orders of magnitude smaller than its input-induced variation (≈3×10⁻³ nats versus ≳1 nat). A head reading LN(h_l) therefore cannot order the mid stages at all; it can only detect that the state differs from the terminal state, which is a 1-bit decision. The observed exit distribution (79% at L1, 21% at L2, 0% at L3+) is exactly what a *binary input-conditional router* looks like, and that is what the evidence supports — not depth-adaptive computation.

3. **The halting parameterization is singular in probability coordinates.** The relative sensitivity of the halting mass to an earlier stage's halting probability is 1/(1−q): 5.9 at q = 0.83, 100 at q = 0.99. The condition number is unbounded exactly in the regime a decisive halter must occupy. In logit coordinates the same sensitivities are bounded by 1. The temperature fragility is a *coordinate singularity*, not a hyperparameter problem.

4. **The soft/hard metric that reports KL = 0.001–0.003 is measuring the wrong object.** In the saturated regime it reports a number that *any* two near-deterministic halters produce, regardless of whether they halt at the same stage. The per-sample quantity it should report is 0.17–0.21 on the same runs.

5. **There is a training-free upper bound on the entire paradigm.** The halting mass enters the soft objective *linearly*, so the relaxed inner problem is exactly the per-sample oracle exit problem. Hence the maximum gain any halting controller can deliver is the per-sample spread of the fixed-depth loss profile, at matched compute. If a static depth at equal compute matches the adaptive system, adaptivity contributed nothing.

6. **Any label-free causal or value signal is computed in the model's own predictive distribution**, and it is *signed* against continuing: the model's own expected value of more computation is KL(p‖p′) ≥ 0, i.e. always non-positive (§6.4, Prop. 6.1). The bias is largest exactly where the model is changing its mind — which is where the decision would matter. This is the strongest negative result available under the stated constraints.

7. **The three-body instability is not chaos.** It is dimensional analysis plus a variance floor. λ is not dimensionless; N enters the effective price of computation linearly; the verdict metric has a seed-noise band of ±0.07 (one standard deviation over the 5 supplied runs), so between-configuration differences below ≈0.12 are inside the band; and the training target's scale is guarded by an ad hoc ε. The one sweep that worked (λ ∝ N) is the sweep that made the hyperparameter approximately dimensionless. That is confirmatory, not coincidental.

8. **The minimal falsifying experiment is a compute-matched, depth-blind controller:** a gate that sees only h₀ (or only the input), never the depth index. If it reproduces the difficulty–compute correlation, the accuracy, and the loss–compute frontier of the depth-conditioned controller, then the depth axis is decorative in that system, and no adaptive-depth claim survives.

---

## 0. Conventions, interpretation of constraints, and six primitives

### 0.1 Interpretation

The constraints are: differentiable only (no RL, no sampling), single-phase (no curriculum), no depth labels, same semantics at train and inference. I read **"no depth labels" as "no externally annotated exit stages and no per-stage difficulty targets."** If the constraint is stricter and forbids *any* intermediate task loss, then §1.2 shows the halting gradient's depth-discriminative component is suppressed by O(ρ) with ρ the per-stage increment scale, and the supplied ablation row *no stage supervision → Pearson 0.124 (from 0.771)* is the empirical shadow of that suppression. Under that strict reading the paradigm retains no usable learning signal for depth, and the impossibility results collected in §Q3 apply directly.

### 0.2 Notation

| Symbol | Meaning |
|---|---|
| N, d | number of stages, state dimension |
| h_l | state after stage l; h_{l+1} = h_l + f_l(LN(h_l)) |
| δ_l | h_l − h_{l−1}; trajectory increment |
| ρ_l | ‖LN(h_l) − LN(h_{l−1})‖; normalized increment (depth signal) |
| z_l, q_l | halting logit, halting probability: q_l = σ(z_l) |
| r_l, a_l | survival r₁ = 1, r_{l+1} = r_l(1−q_l); halting mass a_l = r_l q_l |
| h̄ | Σ_l a_l h_l / Σ_l a_l (renormalized mixture) |
| g | ∂ℓ/∂h_out at the mixture |
| ℓ_l | ℓ(readout(h_l), y) |
| τ | hard threshold; l*(x) = min{l : q_l ≥ τ} |
| λ | compute price; C = Σ_l r_l = expected depth |
| Δ | matrix of centered increments [δ₁,…,δ_N] |
| r_eff | effective (numerical) rank of Δ |
| n_eff | depth resolution: number of distinguishable depths |

### 0.3 The six primitives

Everything below is a corollary of six elementary facts. They are stated once, proved once (Appendix A), and reused.

**(P1) Mass deficit.**
Σ_{l=1}^{N} a_l = 1 − Π_{l=1}^{N}(1 − q_l).
The soft output is a proper convex combination **iff** a forced halt at N is imposed (q_N = 1) or the mixture is renormalized. Otherwise a free, unregularized scalar — the halting mass Σa_l — multiplies the output, and any downstream linear readout is scaled by it. This is an *uncontrolled temperature*: it acts exactly like the annealed temperature in the sweeps that were tuned by hand.

**(P2) Coordinate singularity of stick-breaking.**
With a_l = q_l Π_{k<l}(1−q_k),

    ∂a_l/∂q_l = a_l / q_l                    (k = l)
    ∂a_l/∂q_k = −a_l / (1 − q_k)             (k < l)
    ∂a_l/∂q_k = 0                            (k > l)

    ∂a_l/∂z_l = a_l (1 − q_l)
    ∂a_l/∂z_k = −a_l q_k                     (k < l)

Relative sensitivities in **q-space**: 1/q_l and 1/(1−q_k) — unbounded as the model becomes decisive. Relative sensitivities in **z-space**: (1−q_l) and q_k, both in [0,1] for all operating points. At q = 0.83 the q-space pair is (1.20, 5.88); at q = 0.99 it is (1.01, 100.0). In z-space both are bounded by 1 everywhere. The halting problem is *uniformly conditioned in logit coordinates and ill-conditioned in probability coordinates.*

**(P3) Depth-dimension theorem.**
Let Δ = [δ₁,…,δ_N] have effective rank r (numerical rank at tolerance ε). Since h_out ∈ conv{h₁,…,h_N} ⊆ h₀ + span(Δ), the set of outputs reachable by *any* halting policy is an affine set of dimension ≤ r. Correspondingly, the depth-discriminative component of the task gradient on the halting mass,

    ∂ℓ/∂a_l = ⟨g, h_l⟩ = ⟨g, h̄⟩ + ⟨g, δ_l⟩,

splits into a depth-blind common mode and a residual that lives entirely in span(Δ). If the trajectory lies near a line (r_eff ≈ 1), the task loss prices depth with **one effective scalar**, so the only depth policies it can express are threshold rules along that line; every finer distinction has to come from the regularizer. This is not a conjecture about optimization: it is a statement about the *rank of the map* from halting mass to task loss.

**(P4) Depth resolution.**
For a linear head z_l = W·LN(h_l), |z_l − z_{l'}| ≤ ‖W‖·ρ_{l,l'}. Define the depth channel range Δz_depth ≈ ‖W‖·ρ̄ and the input channel range (Δz_input)² = Var_x(z). The controller is depth-adaptive only if Δz_depth ≳ Δz_input — otherwise the halting logit is a depth-blind function of the input, and the threshold merely *reveals* an input-side ordering. The dimensionless ratio

    dSNR = Var_l(z) / Var_x(z)

is the missing diagnostic. At the reported operating point (‖W‖-scale set by z_a ≈ 3.05, mid-stage ρ ≈ 10⁻³) the depth channel is ~3×10⁻³ nats against an input-driven swing of ≳1 nat: dSNR ≈ 10⁻⁵ to 10⁻⁶ by variance, i.e. −50 dB. At terminal stages (ρ ≈ 10⁻¹) it is ≈ −20 dB. Both are too small to carry a depth-conditioned decision.

**(P5) Saturation degeneracy of aggregate halting metrics.**
For p, q ∈ [1−ε, 1], KL(Bern(p) ‖ Bern(q)) = O(ε log(1/ε)) *uniformly in the difference of the underlying stage distributions*. Any two near-deterministic halting systems produce a tiny aggregate KL whether or not they halt at the same stage. Pinsker gives the converse reading: KL ≤ 0.003 implies |p − q| ≤ √(0.0015) = 0.039, so the reported 0.001–0.003 is a statement about *saturation*, i.e. q_l(1−q_l) ≲ 0.04, and nothing else. Meanwhile the per-sample disagreement between a soft halting distribution a(x) and the hard decision l*(x) is 1 − a_{l*}(x), a first-order quantity; derived from the reported exit distribution (79% at L1, 21% at L2) it is ≈0.17–0.21, two orders of magnitude larger than the KL that is being reported for the same runs.

**(P6) Oracle relaxation.**
The soft objective is linear in the halting mass: J_soft = Σ_l a_l ℓ_l over the simplex. Its minimum over a is min_l ℓ_l. So the relaxation is *exact* for the inner problem — the soft formulation does not lose expressivity and can in principle attain the per-sample oracle exit. What it loses is the *gradient path*: ∇_θ J_soft = Σ_l a_l ∇_θ ℓ_l + Σ_l ℓ_l ∇_θ a_l. The second (policy) term is the only one that moves the halting distribution toward the argmin, and it is proportional to the *spread* of the depth-loss profile; under depth collapse the spread is O(ρ) and this term vanishes. Also, the mixing slack J_soft − ℓ_{l*} = Σ_l a_l (ℓ_l − ℓ_{l*}) is the exact quantity that the "aggregate KL" should have been reporting; note that its sign is *not* fixed — it is negative whenever the soft mass sits on stages whose loss is below the exit stage's.

---

## 1. Problem 1 — The Halting Problem

### 1.1 Formal statement

Let θ parameterize the trunk, the readout, the halting head, and the last-stage dynamics. Define

    z_l = W_halt · LN(h_l) + b,   q_l = σ(z_l/T),   a_l = r_l q_l,   h_out = Σ_l a_l h_l

    J_train(θ) = E_x[ ℓ(readout(h_out), y) ] + λ E_x[ Σ_l r_l ]

    l*(x) = min{ l : q_l(x) ≥ τ },   ĥ_out(x) = h_{l*(x)}

The *deployed* objective is E_x[ℓ(readout(ĥ_out),y)] and the *trained* objective is J_train. The open problem is: find a training signal for the halting variables {q_l} that (a) does not require unbounded hyperparameter search, (b) scales with N, (c) requires no depth labels. The formal content is not "how do we tune it" but: **under what conditions on the trajectory {h_l} and the parameterization is the halting distribution identifiable from the task loss at all, and how does the conditioning behave as the number of stages grows?**

### 1.2 Why prior approaches fail — mechanistic

Five mechanisms, all visible in the supplied numbers.

**M1 — Coordinate singularity (P2).** Every prior differentiable halter (expected-length costs, geometric-prior KL penalties, soft-exit mixtures) writes its halting loss in **probability space**: KL(Bern(p_l) ‖ Bern(q_l)), log r_l, or products of (1−q). Those losses have relative sensitivity 1/q or 1/(1−q) to the underlying halting probabilities. As the model becomes *decisive* — the regime that produces good accuracy — the Jacobian of the halting mass with respect to earlier stage probabilities diverges like 1/(1−q). This is the mechanistic content of "gradient instability at early stages" in ACT, and it is why sharpening/decisiveness and gradient stability trade off. It also explains the exact shape of the reported temperature table without appeal to luck:

| head_temp | reported KL | reading |
|---|---|---|
| 0.2 | 1.11 | effective logit scale too small; hedge mass spread over stages |
| 0.05 | 0.026 | operating point where the head's logit scale matches the required decision |
| temp_end = 0.3 | 19.05 | sharpening pushes the system across the q ≈ 1 singularity; support of soft and hard rules become disjoint (19 nats ≈ disjointness) |
| temp_end = 0.002 | 0.026 | compensates the head's scale drift, re-entering the window |

The temperature is not a modelling hyperparameter. It is a **slider on the effective confidence of a saturating sigmoid**, and therefore a slider directly on the coordinate singularity. That is why exactly one point passes both correlation and KL: it is the single point with the right distance to a singularity.

**M2 — Unidentifiability of the halting policy (P3, P6).** By P3, the task loss enters the halting mass as ⟨g, h_l⟩ = common mode + ⟨g, δ_l⟩. Under the measured collapse (cos(h₁,h₁₂) = 0.9975; mid-stage increments 10⁻⁴–10⁻³ of the state scale), the centered trajectory is effectively rank-1, so the task term prices depth with a single scalar and its depth-preference is a single bit (early vs late). Everything finer — which stage, for which input — is set by the *regularizer*. That is precisely why the ablation table shows the cost term as load-bearing:

- No cost at all → Pearson 0.103, accuracy 1.000. With no depth price, the halting mass is unconstrained and drifts to whatever the (nearly depth-blind) task gradient dictates; the task is solved, the depth axis is noise.
- No gated cost → Pearson 0.031, accuracy 0.695, KL 1.757. Removing the gate removes the only term that was shaping the depth axis against the collapse direction, and the system loses both the depth structure *and* accuracy.
- No stage supervision → Pearson 0.124, accuracy 0.992. Accuracy survives; the depth *ordering* does not. Consistent with P6: the first-order policy-gradient term is O(ρ) ≈ 10⁻³ and cannot order twelve stages.

This is the correct reading of "cost + gating + stage supervision = all load-bearing." They are not three independent ingredients; they are three patches over one hole: the depth axis carries no task signal of its own.

**M2′ — The exact form of the halting gradient, and why it starves.** Differentiating the soft objective A = Σ_l a_l ℓ_l with respect to the halting logits (Appendix A.2) gives

    ∂A/∂z_k = a_k (1 − q_k) · Adv_k ,      Adv_k = ℓ_k − L_k^+,   L_k^+ = Σ_{l>k} a_l ℓ_l / r_{k+1}

— the halting gradient is *the advantage* times the mass at k times the probability of not having halted there. Three multiplicative factors, each of which training drives toward zero:

1. **The decision margin.** a_k(1−q_k) → 0 as the policy becomes decisive (q → 0 or 1). In the reported regime q ≥ 0.83, so 1−q ≤ 0.17.
2. **The advantage.** At the relaxed optimum the halting mass sits on a single stage (P6: the objective is linear in a, so its minimum is a vertex of the simplex), where the advantage of the chosen stage is ≤ 0 and the mass elsewhere is 0. So the gradient is proportional to the *current regret* and vanishes as the policy approaches optimality.
3. **The depth-discriminative content of the advantage.** Adv_k = ⟨g, δ_k − δ̄_{>k}⟩ = O(ρ) (P3). Under the measured collapse ρ ≈ 10⁻³ in mid stages, and the depth-*blind* component of the same gradient (the input-conditional variation that makes some samples exit early and others late) is O(1) by comparison.

So the signal that trains *depth-adaptive* halting is the product of three small numbers — ≈ 0.17 × 10⁻³ ≈ 10⁻⁴ in the reported regime — while the signal that trains a *depth-blind* input-conditional router is O(1), and the signal that moves the logit level (the compute term, §5.2) is O(10⁻¹). **The training dynamics answers the question it can hear.** That single inequality is the mechanistic root of everything else in this memo.

**M3 — Mass deficit (P1).** With unforced halting, Σa_l < 1 and the output is a shrunken mixture. Two consequences: (i) the effective logit scale of any downstream readout is multiplied by Σa_l, a free parameter that drifts during training; (ii) the amount of shrinkage is exactly the amount of training mass spent on stages whose halting is not resolved. In runs where the halting mass is close to 1 (as in the reported q_l ≥ 0.83 regime) the deficit is invisible; in runs where it is not, it acts as an unintended, time-varying temperature. This is the same degree of freedom that the temperature sweeps were trying to control, so the two knobs are partially redundant and the sweeps interact — which is the observable signature of an over-parameterized control set (several nominal hyperparameters moving one physical quantity; cf. §5).

**M4 — Depth resolution (P4).** A linear head on LN(h_l) has a depth channel of range ‖W‖ρ̄. At the reported scale (head weights at the z_a ≈ 3.05 level; mid-stage ρ ≈ 10⁻³) the depth channel spans ~3×10⁻³ nats, whereas the decision needs ~1 nat of logit movement around τ. The head therefore *cannot* rank the eight mid stages. It can distinguish "the state is at its terminal value" from "not yet," which is a 1-bit decision. The measured exit distribution — 79% at L1, 21% at L2, 0% beyond — is exactly a 1-bit decision, and the reported non-monotonicity of correlation in N is what you get when you add stages to a system whose depth resolution is already saturated at 1 bit.

**M5 — The threshold knife-edge.** The supplied numbers force τ > 0.83 (otherwise exit at L1 would be 100%). Therefore every adaptive decision made by the system is a threshold crossing inside the logit window [logit(0.83), logit(τ)]. For any τ ≲ 0.93 that window is under 1 nat (logit(0.93) − logit(0.83) = 2.59 − 1.59 = 1.00). Meanwhile the halting logit's *input-driven* excursions (z_a reaching 3.05 and "growing further") are several nats wide. The entire adaptive content of the system is the *flux of samples across a hairline window*: a first-order functional of the density of z near τ. This has three observable consequences, all of which are in the data: (i) extreme sensitivity to any global rescaling of the logit (temperature, weight norm, λ); (ii) correlation values that are functions of the *fraction* of samples in the window, i.e., of a single scalar, hence unstable across seeds; (iii) a two-branch exit distribution.
**This is the central diagnosis. The system is not halting in depth; it is voting across a threshold.**

*How this maps to prior art.* ACT: M1 + M3. Universal Transformers: M2 + M4 (depth is constant because the trajectory is rank-deficient and the head cannot resolve it). PonderNet: M1 (probability-space KL against a geometric prior) + a prior that is a *distribution over depths*, not a *value of depth*, so it encodes the same depth-blind information the trajectory lacks. Mixture-of-Depths: it abandons depth-adaptive halting for a fixed budget because it routes with a top-k over depth scores, which converts the unidentifiable halting problem into an identifiable *routing* problem — the honest response to M2, and evidence for it. Chain of Thought: relocates adaptivity to the token axis, where the increments are O(1) and the depth signal is not collapsed. Its success is evidence that the *axis* was the problem, not the idea.

### 1.3 Solvability conditions

**Necessary conditions.**

**(N1) Depth-necessity margin.** The trivial solution family (f_l ≡ 0, q_l ≡ 1, one effective stage) is a feasible point with zero compute cost. Hence adaptive halting is *well-posed* only if

    D* = min_θ E[ℓ(readout(h₁),y)] − min_θ E[ℓ(readout(h_N),y)]  ≫  λ E[C].

If the task is solvable by one stage, depth-adaptive halting is a superfluous structure and will collapse — to constant-early (the observed 79/21/0 regime) — under any pressure to save compute. This is not a tuning issue and no signal can fix it.

**(N2) Depth-identifiability.** The centered trajectory must have effective rank large enough to express the policy you want: if the desired policy is an arbitrary per-input exit stage, r_eff must be ≳ the number of distinguishable exit classes you intend to use. A rank-1 trajectory admits only threshold policies along a line (P3). The measurable quantity is r_eff(Δ) and its spectrum; the diagnostic is whether the *achieved* exit distribution's entropy is bounded by log(1 + r_eff).

**(N3) Depth resolution.** The head must satisfy ‖W_halt‖·ρ̄ ≳ Δz_needed (≈1 nat), or else the head must receive a normalized depth-contrastive feature instead of LN(h_l). At ρ = 10⁻³ this requires ‖W_halt‖ ≳ 10³ under the raw-LN design — three orders of magnitude beyond the observed head scale, and orders of magnitude beyond what weight decay will permit. This condition is the arithmetic that makes "depth ∉ difficulty" structural rather than empirical.

**(N4) Forced halting / renormalization.** Σa_l = 1 either by construction (q_N ≡ 1, "the last stage always halts") or by renormalizing h_out. Without it, the effective output temperature is an unregularized parameter that covaries with everything else being swept.

**(N5) Bounded relative sensitivity.** All halting losses must be expressed in logit coordinates. The cost term must be the *logit-space* expected-depth functional; the halting-distribution losses must be KLs/BCEs over logits, not over probabilities; the temperature, if any, must be applied inside the logit parameterization and — preferably — learned by a proper scoring rule rather than swept.

**Sufficient design (satisfies N1–N5).** Four changes:

1. **Contrastive depth feature.** Feed the halting head ū_l = (LN(h_l) − LN(h_{l−1})) / (‖LN(h_l) − LN(h_{l−1})‖ + ε) instead of LN(h_l). This restores the depth channel to O(1) (P4) and removes the depth-invariant *level* component, which is the coordinate the compute term inflates at a task-side price of order ρ (§5.2). One change fixes M2, M4, and the leverage asymmetry of §5.
2. **Critic instead of policy.** Replace the scalar cost pressure on q (which acts only through the vanishing policy-gradient term) with a *supervised prediction of the model's own future improvement*:

       target:  t_l(x) = σ( (ℓ_l(x) − ℓ_{l+1}(x)) / T_c )
       loss:    L_critic = Σ_l r_l · BCE( q_l , t_l )

   The target uses only the model's own stage-wise loss trajectory — no exit annotations, no depth labels, no sampling, fully differentiable, single phase. It is a proper scoring rule, so q_l is calibrated *by construction* and its reliability diagram becomes a meaningful instrument rather than a diagnostic afterthought. It also makes the halting signal a *value*, not a *rate*, which is the only way an interior (non-endpoint) exit becomes learnable (P6: a rate has no way to represent "stop at stage 5", a value does).
3. **Budget as a constraint, not a price.** Impose E[C] ≤ B with a dual variable μ updated by ascent, instead of a fixed λ:

       μ ← [ μ + η ( E[C] − B ) ]₊ ,   L = E[ℓ] + μ (E[C] − B)

   μ is the adaptive price of a stage, measured in loss units, and it removes the human from the λ sweep entirely. The control becomes the dimensionless utilization u = (B − 1)/(N − 1) ∈ (0,1], which is the *only* quantity that should ever be swept, and which is N-invariant by construction (§7).
4. **Forced halt at N** (q_N ≡ 1) and renormalized mixture.

Under this design the knife-edge (M5) disappears in the following precise sense: the halting logit is bounded by ‖W_halt‖ (its input is unit-norm), its depth variation is O(1), the logit-to-probability map is uniformly conditioned, and the price of computation is a state variable rather than a hyperparameter. Whether the resulting system *beats* a static baseline remains an empirical question — and §Q4 says how to test it — but the fragility is a coordinate artifact of the old parameterization, and the fixes are structural.

### 1.4 Minimal falsifiable test

**Test 1A (depth channel).** Instrument the trained controller and report two numbers per run: dSNR = Var_l(z)/Var_x(z) and the exit-stage entropy H(l*). Prediction of the diagnosis: with a raw-LN head, H(l*) ≤ 1 bit and is *invariant in N* for N ∈ {4, 8, 12, 24, 48} at fixed difficulty distribution, and dSNR ≲ 10⁻³. **Falsifier:** if H(l*) grows like log N while ρ̄ stays fixed, the depth-resolution analysis is wrong and the non-monotone correlation must have a different cause.

**Test 1B (coordinate singularity).** Replace the probability-space cost with the logit-space cost and the fixed λ with the dual variable; force q_N = 1. Then repeat the temperature sweep. **Prediction:** the good region in (T, λ) (or (T, B)) becomes a region rather than a point and the KL no longer explodes at T → 0.3. **Falsifier:** if the KL still explodes at exactly the same T and the good set remains a singleton, the singularity is not the mechanism, and the fragility must be attributed to the head's scale drift alone (a weaker but still actionable diagnosis).

**Test 1C (depth-necessity margin).** Measure D* (the one-stage vs N-stage loss gap) and the compute saving λN. **Prediction:** the regimes where adaptivity "works" and "fails" are separated by the sign of D* − λ·E[C], and the sign of that difference tracks N, seed, and task difficulty distribution. **Falsifier:** if runs with D* ≈ 0 show stable, N-invariant depth adaptation, then the collapse/vacuity theory in P3/P6 is not what governs training.

These three are cheap, label-free at inference, and they discriminate the diagnosis from its competitors. Every one of them should be reported *before* any correlation number, because each is a precondition for the correlation being informative.

---

## 2. Problem 2 — Target-Free Completion

### 2.1 Formal statement

Let ℓ_l(x) = CE(readout(h_l(x)), y) be the per-stage task loss. Define the **oracle remaining improvement**

    m_l(x) = min_{k ≥ l} ℓ_k(x),     V_l(x) = ℓ_l(x) − m_l(x) ≥ 0

The supplied target-aware completion is a *rescaled* version of V:

    C_l = 1 − max(ℓ_l − ℓ_N, 0) / max(ℓ_1 − ℓ_N, ε)

Three defects, before the label question is even raised:

**(D1) Wrong reference.** Completion should be defined against the oracle m_l, not against the last stage ℓ_N. The supplied evidence contains a case where the two disagree: the adaptive system beats the full-budget system (accuracy 0.78–0.79 at 42–48% compute versus 0.69–0.71 at 100%). If ℓ_N > m_1, then the "completed" reference is not the best stage, C_l is not monotone in l, and the target is corrupted by the very phenomenon (depth can hurt) that makes the problem interesting.

**(D2) Unbounded range and ad hoc scale.** C_l ∈ (−∞, 1]; as ℓ_l → ∞ (an atypical sample), C_l → −∞; and when ℓ_1 → ℓ_N the ε-guard swaps a well-scaled target for O(1/ε) noise. This is not a cosmetic issue: the target's *scale* is set by ε and by ℓ_1 − ℓ_N, both of which vary across runs and across N. Any regression target with a run-dependent scale makes cross-run correlation numbers non-comparable. Part of the instability attributed to chaos in §7 is this: **the reported metric is a correlation between a compute statistic and a *target whose units change from run to run*.**

**(D3) Label dependence.** ℓ_l = CE(readout(h_l), y) requires y. So C_l is undefined at inference. Formally: ∂C_l/∂y is not merely undefined, the conditional distribution of ℓ_l given (h_{l−1}, h_l, h_0) is a mixture over y, and C_l is a nonlinear function (max, division) of that mixture. There is no algebraic route from a mixture to C_l.

The actual question — "can completion be defined as f(h_{l−1}, h_l, h_0, dynamics) without y?" — must be split into two questions that have different answers.

### 2.2 Why the naive approaches fail — mechanistic

**Q-A: Is there a *functional* of the trajectory that equals V_l?** No, and this is provable in the strongest available sense: V_l depends on y, and two tasks with identical state trajectories and different labels give different V_l. Constructively: hold the trunk and readout fixed, let x be a fixed input; the trajectory {h_l} is then fixed. Choose two label assignments y₁ ≠ y₂ such that ℓ(readout(h_N), y₁) < ℓ(readout(h_1), y₁) and the reverse for y₂. Every functional f(h₀,…,h_l) gives the same value under both, while V_l differs (even in sign). So no label-free functional is a consistent estimator of V_l for all tasks. Any claim otherwise is a claim about the *model family*, not about the state.

**Q-B: Is there a *decision rule* that achieves the same objective without knowing V_l?** Yes, and this is the right frame. Separate the two roles of y:

- **Evaluation** requires y. The quantity to be evaluated is the *compute-allocation regret*
      Regret(π) = E_x[ ℓ_{l_π(x)}(x) − m_1(x) ] ≥ 0
  which is exactly what the label is for.
- **Decision** does not require y, provided the model carries a *learned* estimate of the value of computation. The policy is a function of x only; the label enters only through training.

So the correct restatement of Problem 2 is: **completion is not a functional; it is a critic.** The question becomes identifiability: what must be true of the state for a critic to be able to predict the future loss profile from h_l?

**Proposition 2.1 (Soundness of geometry — one direction only).**
Assume (i) a single shared readout R applied at every stage, and (ii) exact stationarity h_{l+1} = h_l. Then ℓ_{l+1} = ℓ_l exactly, so V_l = 0 and stopping is optimal. Stationarity is therefore a **sound** stopping certificate: it never discards a strict improvement.
*Proof.* ℓ_{l+1} = CE(R(h_{l+1}), y) = CE(R(h_l), y) = ℓ_l. ∎

**Proposition 2.2 (Incompleteness, both directions).**
(i) *Motion without gain:* for a linear readout r(h) = ⟨u,h⟩, any increment δ ⊥ u leaves the readout, and hence the loss, exactly unchanged while ‖δ‖ is arbitrary. So a geometric rule of the form "‖h_{l+1} − h_l‖ > ε ⇒ continue" never terminates early on task-irrelevant motion.
(ii) *Gain without cheap detection:* the useful part of the motion is only its readout-relevant projection. With η_l = ‖P_R δ_l‖/‖δ_l‖ the readout-relevant fraction, the loss change over a stage is bounded by Lip·η_l‖δ_l‖. For a low-rank readout in a high-dimensional state, η can be O(10⁻²): the state can be 99% "moving in the dark."
*Consequence.* Geometry supplies a *one-sided* certificate: it can certify that nothing more will happen, but it can never certify that something more *will* help. Sign information is exactly what y provides, and the only differentiable way to recover it is to learn it.
∎

**Proposition 2.3 (Information ceiling, and why the *label-free* version is still viable).**
Write I_l = I(h_l; y) for the label information retained in the state. By Fano, the accuracy attainable by any readout of h_l is bounded by a function of I_l; hence the accuracy gain available from continuing past l is bounded by the growth of I_l with l. A label-free surrogate is available: because h_N is downstream of h_l, I(h_l; y) ≤ I(h_N; y) ≤ I(h_l; y) + I(h_N; y | h_l), so the *remaining* information the label can still supply is bounded above by I(h_N; y | h_l). A directly computable, label-free proxy for this is the **predictive drift to the fixed point**:

    Δ_l(x) = D( p_θ(· | h_l) ‖ p_θ(· | h_N) )

(any divergence; forward or symmetrized). Δ_l needs a full-depth forward pass, so it is a *training-time* signal only — which is precisely the right role for it.

*The design that follows.* Train the halting head as a critic of Δ_l (equivalently, of the loss gap ℓ_l − ℓ_N, which is available at training time with labels):

    target  t_l(x) = σ( (ℓ_l(x) − ℓ_{l+1}(x)) / T_c )       (supervised, training only)
    alarm   t̃_l(x) = σ( (Δ_l(x) − Δ_{l+1}(x)) / T_c' )      (label-free, training only)
    loss    Σ_l r_l · BCE(q_l, t_l)   (or BCE(q_l, t̃_l) in the label-free variant)

and evaluate the critic by its reliability diagram. This is what makes Problem 1's "no depth labels" tractable without violating Problem 2's label constraint: the *targets* are constructed from the model's own depth-wise loss trajectory (which is computed anyhow), not from annotated exit stages, and never at inference.

**A warning about the two variants, resolved in §6.4.** The two targets are not interchangeable. The supervised target t_l is a *value*: it is unbiased for the question "would continuing help?" The drift target t̃_l is a *necessity alarm*: it detects that the model's belief is still moving, and by Prop. 6.1 it is *signed wrong as a value* — the model's own expectation of the loss change from continuing is KL(p‖p′) ≥ 0, i.e. always non-positive. The label-free variant may therefore be used as a *screen* (which samples deserve more depth) and never as a calibrated halting probability. Corollary 6.2 states this precisely.

Once a value head exists, the halting distribution need not be trained at all: §Q2 derives it in closed form as the Boltzmann distribution over the loss profile, a_l ∝ e^{−ℓ_l/τ}, with halting logit (W_{l+1} − ℓ_l)/τ. The critic formulation above is the minimal supervised step toward that; the closed form is what makes the soft/hard gap a controlled quantity rather than a hope.

**The invariant that predicts "no further gain."** It is the *fixed point of the predictive distribution*: the halting signal should be a monotone function of the *predicted* predictive drift, Δ̂_l(h_l) = E[Δ_l | h_l]. Its two structural properties:

1. Exact stationarity of the predictive distribution is a sound certificate (Prop 2.1 generalized from h to p_θ).
2. Predictive drift is a *necessary* condition for further gain (Prop 2.2 generalized): if p_θ(·|h_{l'}) = p_θ(·|h_l) for all l' ≥ l, no readout of the state can improve, since all downstream predictions are identical.

Everything between "the predictive distribution has stopped moving" and "the predictive distribution is *right*" requires labels or a critic trained against labels. That gap is not an implementation detail; it is the entire content of the problem.

### 2.3 Solvability conditions

**(N1) Reference is the oracle, not the last stage.** C_l must be defined against m_l = min_{k≥l} ℓ_k, and the target must be a bounded, scale-fixed quantity (a probability, not a ratio of loss gaps). Recommended: the critic predicts the *event* "continuing past l reduces the eventual best loss by more than δ", as a probability, trained with a proper scoring rule.

**(N2) The state must retain label information.** The critic's ceiling is I(h_l; y). If the truncation of depth l destroys label information (through LN, pooling, or a bottleneck), the critic cannot recover it and completion is unidentifiable *for that architecture*. Measurable at training: the label-predictive information curve I_l and the incremental mutual information I(h_N; y | h_l).

**(N3) Monotonicity must be checked, not assumed.** "Completion" presupposes that the loss profile is decreasing in depth, at least in expectation. The supplied evidence contradicts the assumption for the full-budget endpoint (0.69–0.71 at full depth versus 0.78–0.79 at 42–48%). Therefore the correct object is not a completion *fraction* but a *depth profile*, and the correct decision problem is: choose l to minimize ℓ_l, not to reach the end fastest. A completion signal that cannot represent non-monotonicity cannot represent the case the system was built for.

**(N4) Evaluation must be regret, not correlation.** The utility of a completion estimator is not its correlation with C_l over the whole range, but its regret at the decision boundary. A predictor can have high global correlation and still choose the wrong stage, because the decision-relevant region is the narrow band where the profile is nearly flat.

### 2.4 Minimal falsifiable test

**Test 2A (does the state know its own future?).** Train the critic on training data to predict the event "continuing reduces the loss by more than δ" from h_l alone. Evaluate on held-out data by AUC and by reliability diagram. **Decision rule:** AUC must be ≥ 0.65 *and* the reliability curve must be monotone in the predicted-probability bins; the critic is then usable. **Falsifier:** if AUC ≈ 0.5 (equivalently, the reliability diagram is flat), the state carries no information about its own future improvement, target-free completion is impossible for that model, and the halting problem is not merely unsolved but unsolvable within the architecture. This single experiment adjudicates between "we haven't found the right signal" and "there is no signal."

**Test 2B (regret at matched compute).** Compute the realized regret of the learned exit policy against the per-sample oracle m_1. Report regret as a fraction of the oracle gain (E[m_1] − E[ℓ_N]). **Decision rule:** the policy is useful only if its regret is a small fraction (say < 25%) of the oracle gain and the system is strictly above the *static envelope* at the same average compute (§Q4). **Falsifier:** if regret ≈ the full oracle gain, the policy is not using the depth axis at all — the model is simply paying for compute it doesn't need, and the "adaptive" system is a fixed-depth system with an expensive wrapper.

**Test 2C (fixed-point invariant).** Report the function l ↦ E[Δ_l(x)] (mean predictive drift to the fixed point) alongside the loss profile l ↦ E[ℓ_l]. **Prediction:** Δ_l is a decreasing function of l with a knee, and the loss profile's knee coincides with it; the halting decision boundary sits at the knee. **Falsifier:** if Δ_l is flat (or noisy) where the loss profile still falls sharply, then the predictive distribution is not tracking the loss — i.e., the model's *belief* stops changing before its *performance* does, and any label-free completion signal is structurally blind to the remaining gain. That failure mode is exactly the content of §6.

---

## 3. Problem 3 — Soft–Hard Mismatch

### 3.1 Formal statement

Two decision rules are trained and deployed respectively:

- **Soft (training):** halting distribution α(x) ∈ Δ^{N+1}, α_l = a_l for l ≤ N, α_{N+1} = r_{N+1}; output h_out(x) = Σ_l a_l h_l (renormalized or not, per P1).
- **Hard (inference):** exit stage l*(x) = min{l : q_l ≥ τ} (N+1 if no crossing); output h_{l*(x)}.

The mismatch that matters is a *pair of mismatches*:

(M-a) **Decision mismatch:** the soft rule is a randomized first-passage rule; the hard rule is a deterministic threshold rule. Agreement is measured by 1 − α_{l*(x)} per sample, and by P(l*_soft(x) ≠ l*_hard(x)) in aggregate, where l*_soft = argmax_l α_l(x).
(M-b) **Output mismatch:** even when the two rules pick the same stage, the trained output is the mixture and the deployed output is a vertex. The relevant object is the *reachable set* and the propagation of halting uncertainty into the prediction.

The reported metric — KL(Bern(Σ_l a_l) ‖ Bern(q_hard)) = 0.001–0.003 — is neither.

### 3.2 Why the reported metric cannot measure what it claims

Five independent reasons. The first three are definitional; the last two are quantitative and are the ones that matter in the observed regime.

**R1 — Category error.** Σ_l a_l is the *total halting mass*: P(the process halts by stage N). The hard-rule quantity is a *per-stage* halting probability at the exit stage. These are different random variables with different support. Two systems can have identical total halting mass and completely different stage distributions. The correct pair to compare is:

    soft stage distribution:      E_x[α(x)] ∈ Δ^{N+1}
    hard stage distribution:      P_x(l*(x) = l) ∈ Δ^{N+1}

and, more sharply, the *per-sample* disagreement, since marginal equality does not imply per-sample equality (two populations with identical marginals can be assigned in opposite ways).

**R2 — Scalar collapse.** Both quantities are dataset means. The aggregate comparison is invariant to any reshuffling of stages across samples that preserves the mean. It therefore cannot see the entire phenomenon of interest: which inputs exit where.

**R3 — The metric is minimized by degeneracy.** For the identity halter (q_1 ≡ 1), the soft and hard rules agree exactly, and the KL is exactly 0 — while the system has zero adaptive capacity. Any metric whose global optimum is attained by the total collapse of the mechanism is not a consistency metric. The measured 0.001–0.003 is close to that optimum, and it cannot distinguish "consistent and adaptive" from "consistent because there is nothing to be consistent about."

**R4 — Saturation degeneracy (P5).** For p, q ∈ [1−ε, 1], KL(Bern(p)‖Bern(q)) = O(ε log 1/ε) *regardless of the difference in the underlying stage distributions*. With the reported q_l ≥ 0.83 and the exit distribution 79/21/0, the system is deep in this regime: ε ≈ 0.1, and the *first-order* difference between the two rules is not small at all. Pinsker inverts the reading: KL ≤ 0.003 bounds |p − q| ≤ 0.039, i.e. it bounds the *mean stochasticity* of the controller, which is a statement about saturation, not about agreement. The measured value is exactly what a saturated halter produces.

**R5 — Wrong quantity for the deployed objective.** The deployed system's behavior depends on ‖h_out(x) − h_{l*(x)}‖ and on ℓ(h_out) − ℓ(h_{l*}), neither of which is a function of the two halting scalars. The mixture and the vertex differ by exactly the mass placed away from the exit stage, times the trajectory diameter:

    ‖h_out − h_{l*}‖ = ‖Σ_l a_l (h_l − h_{l*})‖ ≤ (1 − α_{l*}) · diam({h_l})

so the *correct* aggregate normalizer is the per-sample off-mass (1 − α_{l*}), which is ≈ 0.21 in the reported runs — two orders of magnitude larger than the number being reported.

### 3.3 The correct metric suite

Any single number will be gameable. The minimum viable suite has five members, three of them label-free.

**(M1) Per-sample decision disagreement (label-free).**
    D_dec = E_x[ 1 − α_{l*(x)}(x) ]   and   P_x( argmax_l α_l(x) ≠ l*(x) ).
The first is the total-variation distance between the soft rule at x and its own hard decision; it is the honest "soft-vs-hard" number. In the reported regime it is ≈ 0.17–0.21, not 0.001.

**(M2) Output gap, normalized (label-free).**
    g(x) = ‖h_out(x) − h_{l*(x)}(x)‖,   V(x) = Σ_l α_l(x)‖h_l(x) − h̄(x)‖²
    γ = E_x[ g(x) ] / ( E_x[ sqrt(V(x)) ] + ε )
Interpretation: γ → 0 can happen for two reasons — genuine agreement, or a dead trajectory (V → 0). Reporting g *without* V is the vacuity trap: in the identity-collapse regime (§4) every consistency metric looks excellent because there is nothing to disagree about. **Report (g, V) jointly, always.**

**(M3) Mixing regret (label-based, evaluation only).**
    Slack = E_x[ Σ_l α_l(x) ( ℓ_l(x) − ℓ_{l*(x)}(x) ) ]
This is the exact amount by which the trained surrogate's value differs from the deployed decision's value, and it is the quantity the aggregate KL was a broken proxy for. **Its sign carries the diagnosis.** If the soft mass sits on stages whose loss is *below* the exit stage's (Slack < 0), the training objective is optimistic relative to deployment: the model is rewarded for a mixture that the hard rule can never realize. If the loss is concave along the trajectory the mixture beats every vertex, so that optimism is systematic and Slack is large in magnitude while every halting-distribution metric remains perfect. In the reported regime the mass away from the exit stage is 0.17–0.21, so Slack is first order in that mass times the local loss differences.

**(M4) Deployment consistency under an update (the operational test).**
The hard system's map θ ↦ l* is piecewise constant, so ∇_θ(hard objective) = 0 almost everywhere: there is no gradient to be consistent *with*. Consistency must therefore be tested by finite differences:

    take a step θ' = θ − η ∇_θ J_soft ;  measure Δdeployed = Acc_hard(θ') − Acc_hard(θ)

Repeat over several steps and several η. **Decision rule:** the soft gradient must be a *useful ascent direction for the deployed system* on average (positive mean Δdeployed, with a confidence interval excluding 0). **Falsifier:** if following the soft gradient systematically degrades the deployed system, the training objective is not a relaxation of the deployment — the two are unrelated, and no amount of halting-distribution tuning can fix it.

**(M5) Per-sample trajectory divergence (label-free), the "does it matter" statistic.**
    E_x KL( p_θ(·|h_out(x)) ‖ p_θ(·|h_{l*(x)})(x) )
This answers the question "how different are the two systems, really, from the model's own point of view?" without labels, and it is bounded by the same calibration caveat as §6.

**A note on the trajectory-level quantity that was flagged as "not yet measured."** The state-space distance ‖h_soft,l − h_hard,l‖ is the natural first instinct and is the least informative of the candidates, for two reasons: it has no intrinsic scale (the same absolute gap means different things in different state norms), and it is measuring the wrong space (the deployment cares about the *output*, not the trajectory). The scale-free, decision-relevant replacements are γ (M2) and Slack (M3). The instinct to move from aggregate to trajectory-level measurement is correct; the fix is the normalizer and the space.

**Why the aggregate KL is insufficient, in one sentence:** it is a scalar function of two saturated means that is *minimized by removing adaptivity entirely*, *blind to which samples exit where*, *blind to the trajectory that determines the actual output*, and *quantitatively consistent with a 21% per-sample disagreement that it reports as 0.3%*.

There is one genuine, defensible use for an aggregate halting metric: as a **saturation diagnostic**. Along with (M1)–(M5), report

    S = E_x[ Σ_l α_l(x)(1 − q_l(x)) ]

the mean halting stochasticity. When S → 0, all soft/hard comparisons become vacuous and the *only* meaningful statement about the system is its exit-stage distribution. Report S next to every consistency number, as the vacuity normalizer.

### 3.4 Minimal falsifiable test

**Test 3A — the identity-halter control (the vacuity certificate).** Evaluate the *trivial* halter (q_1 ≡ 1: never continue, zero adaptive capacity) on every consistency metric in the suite. Its soft and hard rules agree exactly — both are the point mass at stage 1 — so its aggregate KL is exactly 0 and its off-mass is exactly 0. **Decision rule:** if the trained system's consistency metrics are not *strictly better* than the identity halter's on at least one metric that also rises with the exit-stage entropy, then the metric suite certifies nothing beyond degeneracy and must be replaced. This single control separates "consistent because the two rules agree" from "consistent because there is nothing to disagree about."

**Test 3B — per-sample, not marginal.** Report the *histogram* of the per-sample disagreement 1 − α_{l*}(x), not just its mean, together with the realized task-loss difference ℓ(h_out(x)) − ℓ(h_{l*}(x)). **Prediction:** the disagreement mass sits at 0.17–0.21 with a long right tail (the samples in the threshold window), and the task-loss difference is non-zero for a non-trivial fraction of samples — concentrated in that same tail. **Falsifier:** if the task-loss difference is exactly zero wherever the disagreement is large, the two rules are outputs-equivalent on this task and the mismatch question is vacuous *for this model* — which is itself a decisive finding (the trajectory has collapsed, §4, and no consistency claim is meaningful).

**Test 3C — the deployment-Δ test (the operational falsifier).** Take K finite steps θ ← θ − η∇_θJ_soft from a trained checkpoint, for several step sizes η, and measure the change in the *deployed* metric (hard accuracy and hard loss). **Decision rule:** the mean deployed change must be positive with a confidence interval excluding 0 at the step sizes actually used in training. **Falsifier:** if following the soft objective's gradient systematically degrades the deployed system, then the training surrogate is not a relaxation of deployment in any useful sense, and no amount of tuning the halting distribution can repair the discrepancy — the objective itself is the problem (§Q3, NR3).

---

## 4. Problem 4 — Identity Collapse (Ghost Highway)

### 4.1 Formal statement

Residual dynamics h_{l+1} = h_l + f_l(LN(h_l)) with ‖δ_l‖ ≪ ‖h_l‖ for most l. Measured in a 241M-parameter, 12-stage model:

    cos(h₁, h₁₂) = 0.9975
    ‖δ_l‖ ~ 10⁻⁴ – 1.5×10⁻³   (mid stages)
    ‖δ_l‖ ~ 0.07 – 0.14        (terminal stages)

with the downstream consequence z_suff → +3.05 (saturating), resistance bounded at 1.47, q_l ≥ 0.83 uniformly, and an exit distribution of 79% / 21% / 0% at L1 / L2 / L3+.

The formal claim to be tested is a *causal* one: "eight of twelve stages perform no meaningful transformation." This claim is not established by the numbers as reported, for four reasons.

**Reason 1 — Geometry is not causation, and the reported statistics are not commensurate.** cos = 0.9975 corresponds to a total angular deviation of ≈ 4.05°, i.e. a perpendicular displacement of ≈ 7% of the state norm. But per-stage increments of 0.07–0.14 in two or three terminal stages would accumulate to far more than 7% unless they cancel; and mid-stage increments of 10⁻⁴ would contribute under 1% in total. The three statistics — total angle, mid increments, terminal increments — cannot all describe the same trajectory under a single increment definition. **Before any conclusion is drawn, the increment statistic must be fixed to one definition; the right one is the one the controller actually sees:**

    ρ_l = ‖LN(h_l) − LN(h_{l−1})‖

which is invariant to the state's scale and measures the depth signal in the coordinates of the halting head (c.f. P4). Any other normalization makes ρ_l depend on a state norm that drifts with depth, which is exactly the confound under investigation.

**Reason 2 — Local inertness ≠ causal inertness: the propagated effect.**
Define the propagation matrix from stage l to the output and the *propagated* effect of stage l's increment:

    S_l = ∂h_out/∂h_l ,    Ψ_l = ‖ S_l δ_l ‖_metric

A mid stage can have a tiny local increment and a large total effect if the remaining chain amplifies: the residual chain's Jacobian product Π_{k>l}(I + J_k) has spectral gain that grows with the remaining depth. Conversely, a terminal stage with a large local increment has almost no remaining chain to amplify it. Therefore the ordering of |δ_l| across depth carries almost no information about the ordering of |Ψ_l|. **The correct measurement of "does stage l matter" is Ψ_l (with an interventional rather than linearized version — see §6), and it has not been made.** This is the single most important missing measurement in Problem 4.

If Ψ_l is measured and turns out flat across the eight "ghost" stages, the collapse is real. If it is not flat, the "ghost highway" is a measurement artifact of using local norms (Reason 2) and possibly the wrong norm (Reason 3).

**Reason 3 — cos is the wrong metric, and the wrong metric is systematic.** cos(h₁,h₁₂) measures total angular drift including all task-irrelevant components, and it *underweights* small perpendicular displacements quadratically. Two states with cos = 0.9975 can have completely different readouts if the readout direction is nearly orthogonal to the state's common mode — which is the typical situation in a high-dimensional residual stream with a low-rank readout. The readout-relevant metric is the pullback of the loss: ‖J_readout δ‖, or better, the distributional drift KL(p_θ(·|h_{l+1}) ‖ p_θ(·|h_l)). The claim "the state did not change" must be restated as **"the prediction did not change"**, and the latter is measurable exactly and label-free.

**Reason 4 — LayerNorm erases exactly the component that moves.**
This is the most important observation in this section, and it reconciles the otherwise inconsistent diagnostics. LN(h) is invariant to positive rescaling: LN(c·h) = LN(h) for c > 0 (up to the learned affine parameters). So the halting head is *structurally blind* to motion along the ray through h, and sees only changes of shape (the relative-coordinate pattern, of which the angular direction is a special case).

Now the arithmetic of the reported numbers. cos(h₁,h₁₂) = 0.9975 means a total angular change of arccos(0.9975) ≈ 0.0707 rad ≈ 4.05°, i.e. a perpendicular displacement of ≈ 7% of the state norm over eleven stages. Per-stage increments of ‖Δh‖ ≈ 0.07–0.14 at the terminal stages, over two or three such stages, are far larger than that — unless the increments are predominantly *radial* (norm growth), which the cosine cannot see and which LN deletes before the head sees anything at all. So the picture that fits all of the reported numbers simultaneously is:

- the state moves a great deal, mostly by growing along its own direction;
- its *shape* (what LN exposes, and the only thing the halting head receives) changes by ≈ 4° in total, i.e. ≈ 0.006 rad per stage on average, with a concentration at the terminal stages;
- the head therefore receives an input whose per-stage variation is at the 10⁻³–10⁻² level against an input-conditional variation of O(1).

**Consequence:** the "ghost highway" is not (only) a property of the residual stream; it is a property of *what the controller is allowed to see*. A stage whose effect is to scale the state is inert to LN by construction, and a controller reading LN cannot distinguish a stage that doubles the norm from a stage that does nothing. Any fix that does not change the head's input cannot resolve this, which is why the architectural recommendation of §5.4 (feed a contrastive, scale-carrying feature) appears twice in this memo with two different justifications.

### 4.2 The structural consequence: depth is a resource only if the trajectory is rank-rich

By P3, the reachable output set of *any* halting policy is conv{h₁,…,h_N}, whose affine dimension is ≤ r_eff(Δ). This gives the cleanest statement of what identity collapse costs:

**Proposition 4.1 (Reach bound).** For any halting distribution a on the simplex, h_out(a) ∈ conv{h_1,…,h_N} ⊆ h₀ + span(δ₁,…,δ_N). Hence all achievable variation in the output attributable to the halting decision lives in an r_eff-dimensional affine subspace. If r_eff = 1, every halting policy is a *scalar threshold along a line*; the only decisions available are "stop early" or "stop late," and the eight "ghost" stages are interchangeable as far as the output is concerned. The observed exit distribution 79/21/0 is the signature of exactly this: a binary router, i.e. a rank-1 depth policy.

**Proposition 4.2 (Collapse is a stable equilibrium, not an accident).** The task-side reward for using the depth axis scales with the depth signal: the depth-discriminative part of the halting gradient is a_k(1−q_k)·⟨g, δ_k − δ̄⟩ = O(‖g‖ρ), while the compute term charges λ *per stage* — so the marginal price of one more stage is λ, and the total bill for the deepest path is λN. Two conditions follow:

    per-stage identifiability:   ‖g‖ · ρ      ≳   λ · (expected remaining stages at the marginal stage)
    global viability:            D*           ≳   λ · E[C]        (D* as in §1.3, N1)

Below them, the task term cannot outbid the cost term, the halting mass migrates to the first stage, the late stages' gradients are scaled down by their vanishing halting mass a_l (so their increments stop being trained in any effective sense), ρ stays small, and the system re-enters the same regime — a self-reinforcing fixed point. This is the mechanistic reading of the reported evidence: q_l ≥ 0.83 uniformly, 79% exit at L1, mid-stage increments at 10⁻³ while terminal stages at 10⁻¹. **Collapse is the attractor of any objective that prices depth when the task's depth-necessity margin is small.**

**Corollary (the honest answer to "how deep?").** If ρ̄ ≈ 10⁻³ while 12 stages are available, the model is over-deep by construction. Reducing N until ρ̄ = O(1) is a legitimate architectural fix that requires no labels, no new losses, and no controller — and it should be done *before* attempting to fix the controller, because the controller's job is unidentifiable while ρ is small.

### 4.3 On the α_l proposal, and the recursion objection

The proposed fix h_{l+1} = h_l + α_l f_l(LN(h_l)), α_l = g(state_change), is dismissed on the grounds that α_l is "another controller" and therefore recursive. That objection is wrong, and the distinction matters.

An α is a **controller** if it allocates compute (i.e., it changes *how many* stages run, or *which* computation happens). An α is a **normalization** if it only fixes the gain of a stage while the halting head continues to allocate compute. The recursion worry is a worry about *nested allocation*, which is a genuine problem: two allocators with coupled objectives create the identifiability failure of §1.2 applied at a second level. Normalization creates no such problem because it does not have an objective of its own.

The canonical normalization, borrowed from automatic gain control:

    α_l = clip( μ / (‖f_l(LN(h_l))‖ + ε), α_min, 1 )

with μ a fixed set point and α_min a floor (possibly 0, if the model is allowed to express an exact fixed point). Properties:

1. **Parameter-free.** No new learned parameters, no new hyperparameters beyond a set point μ and a floor.
2. **It converts a scale-collapse into a scale-invariant statement.** With normalization, the increment magnitude is invariant to the scale of f_l, so a stage cannot silently shrink to nothing while nominally existing.
3. **It is not recursive.** The loop is scalar and its stability is analyzable: α is a Lipschitz function of ‖f‖ with the composition f ↦ α(f)·f non-expansive in the relevant norm; the small-gain condition for the closed loop is that the product of the trunk's distortion and the normalization's slope is below 1, which holds by construction when the clip is active.
4. **It does not manufacture computation.** The model can still learn f_l → 0 (a genuine fixed point) — but it must then halt, and halting costs compute. The incentive is corrected, not overridden. Compare this to the failure mode it prevents: unnormalized residual stages *and* a cost term that rewards halting jointly, silently, manufacture a "deep model with one working layer."

If a learned α is nevertheless desired, the correct formulation is not a second halting head but a **single objective with a monotone, bounded gate whose input is the measured increment norm**, trained jointly but with the gate's parameters shared across stages (so it cannot become a depth allocator by accident). The test in §4.4 distinguishes the two.

### 4.4 Minimal falsifiable test

**Test 4A (propagated effect, the decisive one).** For each stage l, intervene on the stage mechanism: replace f_l by (i) the zero map, (ii) a norm-matched random map, and measure the change in the output *distribution* (label-free) and in the task loss (label-based, evaluation only):

    Ψ_l^{ablate} = E_x [ D( p_θ(·|h_out^{do(f_l=0)}) ‖ p_θ(·|h_out) ) ]

Plot Ψ_l^{ablate} against l and against ρ_l. **Decision rule:** the "ghost" claim is confirmed only if the ablated effects for the eight mid stages are all below, say, 10% of the terminal stages' effects *and* below the seed-to-seed variability of the same measurement. **Falsifier:** if mid-stage ablations produce non-negligible output changes, then the "8 of 12 stages do nothing" conclusion was an artifact of measuring local increments rather than propagated effects, and the correct diagnosis is different (probably: the mid stages are doing distributed work with a small per-stage footprint and a large downstream gain).

**Test 4B (depth channel, label-free, pre-training).** Report ρ̄ = E_l ρ_l, r_eff(Δ), and the reach R = E_x max_{l,l'} D(p_θ(·|h_l) ‖ p_θ(·|h_{l'})). **Decision rule:** depth is a usable resource only if r_eff ≳ 2 and R is large relative to the accuracy-relevant variation of p_θ. **Falsifier:** if r_eff ≈ 1 and R ≈ 0 for a model that reports a difficulty–compute correlation, then the correlation cannot be coming from the depth axis (P3/P4), and the system is a depth-blind router.

**Test 4C (the normalization loop).** Compare three variants at fixed N, in the same sweep protocol: (i) unnormalized residual, (ii) fixed-μ AGC-normalized residual, (iii) learned α with shared parameters. Report ρ̄, r_eff, the exit-stage entropy, and the per-stage identifiability ratio ‖g‖ρ/(λ·remaining) alongside the global ratio D*/λ·E[C]. **Prediction:** variant (ii) restores ρ̄ to O(1) and *reduces* the degradation of the task loss with depth, without a controller or extra loss terms, and the exit-stage entropy grows only if the task actually has a depth-necessity margin (otherwise the system correctly learns to halt immediately, which is a valid outcome — it says the architecture was over-deep). **Falsifier:** if the AGC destabilizes training at any (μ, α_min) in a wide band, the small-gain analysis is wrong and the normalization route should be abandoned in favor of reducing N.

---

## 5. Problem 5 — Asymmetric Leverage

### 5.1 Formal statement

    z_l = z_a(l) − z_b(l),   z_a = W_a·LN(h_l) + b_a,   z_b = α·tanh(ρ_l)·(1 + persist)

    α ∈ [1,2] learned, tanh ∈ [−1,1], persist ∈ [0,1]  ⇒  z_b ∈ [−4, 4].

Measured: z_a reaches +3.05 by 5,000 steps and is unbounded; z_b is bounded by 4; q_l ≥ 0.83 throughout, with 79%/21%/0% exit at L1/L2/L3+.

The question posed — "is there a normalization invariant to scale?" — has a precise answer, but the framing is off. The problem is not that one branch is bounded and the other is not. It is that **the two branches are not the same kind of coordinate**: z_a sets the *level* of the whole logit field (and hence the sharpness of the halting distribution), while z_b is a *contrast* between positions in that field. Level and contrast are not competing hypotheses about whether to halt; they are different physical quantities, and one of them is a coordinate in which the objective has a strong, one-sided gradient. The right response is to fix the level by construction and let all retained comparisons be contrasts.

### 5.2 Three structural facts

**Fact 1 — There is no gauge here, and that is exactly the problem.**
It is tempting to invoke the shift-invariance of a softmax: adding a constant to all logits changes nothing. That invariance holds for a *softmax over stages*, a_l = e^{z_l}/Σe^{z_k}. It does **not** hold for the sequential parameterization used here. The stick-breaking map

    {q_l} ∈ [0,1]^N  ⟼  {a_l},  a_l = q_l Π_{k<l}(1−q_k),  a_{N+1} = Π_k(1−q_k)

is a bijection onto the simplex: the per-stage conditional halting probabilities *identify* the distribution, with no redundancy. Since q = σ(z) is a bijection, the map z ↦ a has a nonsingular Jacobian and an empty kernel — there is no unidentifiable direction, no gauge to fix.

The consequence is the opposite of what a gauge intuition suggests, and it is worse. Because the level of the logit field is *identified*, changing it changes the policy — and the only thing that controls is the **sharpness** of the halting distribution (its entropy), not its shape. Sharpness is therefore not a nuisance parameter of the loss; it is the operative coordinate of the deployed policy: as the level rises, the halting distribution concentrates toward the earliest stage, and in the limit the system halts at stage 1 with probability 1.

**Fact 2 — The sharpness coordinate is the one the compute term pushes on, and the task term barely prices.**
Two gradients act on each logit. The compute term's is explicit: with C = Σ_l r_l,

    ∂C/∂q_k = −(1/(1−q_k)) Σ_{l>k} r_l      ⇒      ∂C/∂z_k = −q_k · Σ_{l>k} r_l  < 0

always negative, non-vanishing whenever q_k > 0 and stages remain downstream. Raising any logit always buys compute; the price multiplies the *remaining* depth. At q_k ≈ 0.83 with a few stages remaining, λ·|∂C/∂z_k| ≈ 0.05 · (0.83 · O(1)) ≈ 0.04–0.1 in loss units.

The task term's gradient has the standard halting form (derived in the Appendix):

    ∂J_task/∂z_k = a_k (1 − q_k) · Adv_k ,     Adv_k = ℓ_k − L_k^+  (stop now vs continue)

and its *depth-discriminative* content is Adv_k ≈ ⟨g, δ_k − δ̄_{>k}⟩ = O(‖g‖·ρ). With the measured per-stage increments ρ ≈ 10⁻³ and a_k(1−q_k) ≈ 0.83·0.17 ≈ 0.14, the task-side gradient that distinguishes one stage from another is ≈ 1.4×10⁻⁴·‖g‖. **The two gradients differ by three to four orders of magnitude, and only the weaker of the two is depth-discriminative.** The strong one pushes uniformly upward; the weak one is the only force that can hold it back, and it holds back only through its *input-conditional* component (the O(1) variation of z across samples), not through anything related to depth.

That is the asymmetry, stated correctly: not "an unbounded competitor versus a bounded one," but **a gradient of magnitude 10⁻¹ pushing in a direction the task does not care about, versus a gradient of magnitude 10⁻⁴ carrying all the depth information.** The system goes where the strong gradient points.

**Fact 3 — Bounded branches lose their influence as the level rises; the loss is measurable in advance.**
Because z_b ∈ [−4,4] while z_a is unbounded and monotonically pushed upward, the bounded branch's controllable range shrinks monotonically in training:

| z_a level | range of q induced by the full bounded branch (z_b ∈ [−4,4]) |
|---|---|
| 3.05 (measured at 5k steps) | σ(3.05) − σ(3.05 − 4) = 0.955 → 0.279 (range 0.68) |
| 6 | 0.9975 → 0.881 (range 0.12) |
| 10 | 0.99995 → 0.99753 (range 0.0024) |

The reported "z_b is noise" is therefore not a statement about the current model — at z_a = 3.05 the bounded branch still has substantial authority — it is a *forecast*: the branch is being made irrelevant at a predictable rate. And the failure is quiet: because the shrink happens in the same window where the hard decision lives, the model's halting behavior freezes into whatever the input-conditional structure established early, and the exit distribution drifts leftward (toward the observed 79/21/0) with no visible change in any aggregate metric (the soft metrics are saturated, §3).

### 5.3 Why the standard remedies fail

- **Temperature annealing** treats the symptom by keeping the window artificially wide, but the window's width *is* the entropy of the deployed policy; changing it between phases changes the semantics of τ, and the final schedule value is what the deployed system actually uses. It converts a structural quantity into a schedule artifact — which is the reported "one point passes" behavior.
- **Weight clipping** constrains a weight norm, not the level of the logit field, which depends on the product of weight norms, activation norms, and the bias b_a, all of which drift independently. Under an adaptive optimizer, clipping can even *encourage* the drift, since the update magnitude is preconditioned and the model can grow activations instead of weights.
- **Manual balancing (rescaling z_b)** keeps a level-versus-contrast comparison and merely moves the crossover point. As the level drifts, the chosen balance stops being a balance; the hyperparameter is not a constant of the system.

### 5.4 The fix: fix the level, keep only contrasts

    z̃_l = z_l − (1/N) Σ_k z_k                                  (remove the level)
    ẑ_l = √N · z̃_l / (‖z̃‖ + ε)                                 (fix the scale)

Both operations are affine-invariant: any transformation z ↦ c·z + d·1 with c > 0, d ∈ R produces the same ẑ, hence the same soft halting distribution and (for fixed τ) the same hard decision. The deployed policy's sharpness becomes a *design constant* rather than a free coordinate the compute term can inflate, and the local temperature is no longer a hyperparameter at all — the only remaining temperature-like quantity is ε, whose role is numerical.

**Proposition 5.1 (Leverage bound).** If Σ_l ẑ_l = 0 and Σ_l ẑ_l² = N, then for any modulation m_l ∈ [0,1],
    |Σ_l m_l ẑ_l| ≤ (Σ_l m_l²)^{1/2}(Σ_l ẑ_l²)^{1/2} ≤ N,
with equality iff m ≡ 1. Hence no bounded gate can have more than √N leverage on the halting logit per unit of its own magnitude, and the maximum influence of a gate that modulates a fraction f of the depth axis is √(fN)·√N. This is the correct sense in which "neither branch dominates": all branches are measured in a common, bounded frame, and the comparison is an allocation rather than a race between different units.

**A second fix that subsumes the first.** In the reported construction the head reads LN(h_l), whose depth-variation is ρ ≈ 10⁻³ while its depth-invariant, input-driven variation is O(1). Feed the head the normalized increment instead:

    ū_l = (LN(h_l) − LN(h_{l−1})) / (‖LN(h_l) − LN(h_{l−1})‖ + ε)

Now the head's input is unit-norm and carries no level component; its output is bounded by ‖W_halt‖; its per-stage variation is O(1) by construction (it is a *relative* measurement); and the depth channel's signal-to-noise is set by the model's own geometry rather than by the ratio of a 10⁻³ increment to a drifting state norm. Centering the resulting logits (or relying on the approximate telescoping of the increments) then removes the level exactly. This is the single change that simultaneously addresses Problems 1, 4, and 5 — it is the depth-analogue of *measuring contrasts, not levels.*

### 5.5 Solvability conditions

**(N1) No coordinate with a task-irrelevant, one-sided gradient may be left free.** The logit level is such a coordinate: the compute term's derivative in that direction is non-vanishing and always of the same sign, while the task term's derivative in that direction is bounded by the regret. It must be removed by construction.

**(N2) All retained comparisons must be contrasts within one normalized frame.** Bounding a branch does not bound its *influence*: influence depends on the geometry of the other branch, which is drifting. Only a common normalization makes the comparison invariant.

**(N3) The deployed threshold must be defined on a scale-fixed statistic.** A fixed τ on an unnormalized logit selects a different quantile of the sample distribution as the level drifts — moving from "an often-reached stage" to "never reached" without any change of intent. Under the normalization above, τ has a scale-free interpretation as a quantile of each sample's own depth profile.

**(N4) The temperature must be either fixed structurally or learned as a proper scoring rule — never swept.** The temperature is the entropy of the deployed policy, and the entropy is exactly the quantity that governs the soft/hard mismatch (§3). Treating it as a hyperparameter means treating the deployment-time consistency as a hyperparameter.

### 5.6 Minimal falsifiable test

**Test 5A (level audit).** During training, log z̄(x) = mean_l z_l(x) and correlate it with step and with the mean exit stage. **Prediction:** z̄ increases monotonically, and the exit-stage distribution drifts leftward on the same trajectory (through 79/21/0 toward 100/0/0), with the *shape* of the input-conditional structure unchanged. **Falsifier:** if z̄ is stationary while the exit distribution still drifts, the drift is driven by the contrast structure and the level story is insufficient.

**Test 5B (gradient-magnitude audit — the decisive measurement).** During training, log the norms of the two gradient contributions to the halting logits: g_task and g_cost, decomposed into their depth-common and depth-contrast components. **Prediction:** ‖g_cost‖/‖g_task^contrast‖ ≳ 10², and the ratio grows as ρ shrinks; the depth-common component of g_task is near zero. **Falsifier:** if the task gradient's contrast component is comparable to the cost gradient, then the logits are genuinely task-driven and both the collapse and the gauge/level story must be abandoned.

**Test 5C (normalized head).** Replace LN(h_l) with ū_l and normalize the logits per sample; hold everything else fixed including seeds. **Prediction:** (i) z̄ is stationary by construction; (ii) the good region of the (temperature, cost) plane becomes a region rather than a point; (iii) the temperature's effect on the soft/hard mismatch falls below seed dispersion. **Falsifier:** the mismatch still depends sharply on a single temperature value — in which case the fragility is not a coordinate artifact but a property of the coupled training dynamics (§7, C2/C4).

---

## 6. Problem 6 — Causal Evidence Gap

### 6.1 Formal statement

At each stage l the system exposes two different objects:

1. **The controller's decision statistic:** q_l(x) = σ(W·LN(h_l) + b) ∈ [0,1], a function of the state, observable at inference without labels.
2. **The intervention effect:** the change in the deployed output caused by changing the state at stage l.

The two are related by nothing. The correct formalization of the second object must avoid three traps. First, intervening on h_l *at its natural value* h_l(x) has zero effect by definition, so the honest object is a *sensitivity*, not a difference. Second, the sensitivity of the *scores* is not the sensitivity of the *task loss*; the first is label-free, the second is not. Third, causal effects at depth come in two flavors — the *direct* effect of the state on the output, and the *indirect* effect mediated by later stages — and a naive backward pass conflates them.

The right construction:

**Local sensitivity (label-free, per-sample):**
    ω_l(x) = sup_{‖δ‖ ≤ ε} D( p_θ(· | h_out^{h_l → h_l(x)+δ}) ‖ p_θ(· | h_out) )
i.e. how much the model's *output distribution* can be moved by a bounded perturbation of the state at stage l. This is exactly the norm of the Jacobian propagated through the remaining chain into output-distribution space, ‖∂p/∂h_l‖, and it is computable with one gradient per stage.

**Interventional contribution (label-free, distribution-level, the gold standard):**
    Ψ_l(x) = D( p_θ(· | h_out^{do(mechanism_l = 0)}) ‖ p_θ(· | h_out) )
i.e. ablate the stage's mechanism and measure the change in the output distribution. This is a true intervention (it removes the *cause*, not the correlate), it needs no labels, and it is affordable at training time or at evaluation time (N extra forward passes) even if not at deployment.

**Task-weighted causal effect (the thing that actually matters):**
    Ω_l(x) = ω_l(x) · sign_or_magnitude of the *predicted* task change. The first factor is free; the second factor is the entire difficulty, and §6.4 shows that any purely self-referential estimate of it is *biased by a computable amount and in a known direction*.

### 6.2 Why the candidate answers are wrong (mechanistically)

**Mutual information I(h_l; h_out | h_{l−1}) — wrong type of quantity.** Mutual information is association, not causation; it is invariant to which direction the dependence runs and to the mechanism, so it cannot answer "would the output change if this stage changed." It is also a *population* quantity: per-sample, conditioning on x makes h_l and h_out deterministic and the mutual information is 0/undefined. Finally, its expectation over x is dominated by samples the model is already confident about, which are precisely the samples where the stage does not matter. **Verdict: unusable for the stated purpose.**

**Downstream gradient magnitude — right idea, wrong measurement.** ‖∂h_out/∂h_l‖ is the linearization of ω_l, but (i) it conflates direct and indirect paths, (ii) it is a *local* object that misses the nonlinear regime (the model's behavior changes discontinuously when a later stage flips its own halting decision — see the knife-edge in §1), and (iii) it is an upper bound on the effect, not the effect. It is the correct *proxy* if and only if a *calibration study* shows that its ranking agrees with the interventional ranking Ψ_l. Absent that study, it is an untested surrogate, and the reported evidence (q = 0.88 while nothing changes) is evidence that the untested surrogate is what is currently driving decisions.

**Perturbation response — the right family, with one specification needed.** ω_l as defined above is correct in kind but must be specified in output-distribution space (or task-loss space), not in state space: a perturbation that moves the state a lot but the prediction little is exactly the "ghost" phenomenon, so the metric must be the pullback of the output, not the norm of the state change. The evidence in the inquiry is the precise illustration: cos(h₁,h₁₂) = 0.9975 while the controller signals q = 0.88. Under a state-space metric, the stage looks inert; under an output-distribution metric, one measures the truth — and if the distribution also barely moves, the ghost is confirmed rather than assumed.

### 6.3 The causal arithmetic

For a fixed sample with label y:

    true causal task effect:  Δℓ_l(x) = ℓ(readout(h_out^{do(l)}), y) − ℓ(readout(h_out), y)

- Δℓ_l ≈ 0 while q_l is high → the controller is halting because it cannot see a reason to continue, and it is *correct that it cannot see one* only if the intervention also shows nothing. The reported measurements establish the first half (cos ≈ 1, q high) and do not establish the second (the interventional profile Ψ_l was never measured — that is Test 4A). **The causal gap is currently an unmeasured gap, and the burden of proof is on whoever claims the controller is right.**
- Δℓ_l large while q_l is low → the controller is failing to exploit a stage that matters. This is the failure mode in the direction of accuracy, and it is the more expensive one: it is invisible in any aggregate accuracy number because it only affects the samples near the decision boundary.
- The *correlation* r(q_l, Δℓ_l) across samples and stages is the honest summary of "does the controller know what it's doing", and it should be reported *conditional on the sample being near the decision boundary*, because for samples in the bulk the question is uninteresting.

### 6.4 The central negative result: self-assessment is systematically biased toward halting

This is the strongest provable statement in this memo, and it settles the "without labels" question.

**Proposition 6.1 (Self-assessment bias).**
Let p = p_θ(·|h_out) be the model's predictive distribution before continuing past stage l, and p′ = p_θ(·|h_out^{continue}) the predictive distribution after continuing. Consider the model's own expectation of the loss change from continuing:

    E_{y ~ p}[ ΔCE ] = E_{y~p}[ −log p′_y + log p_y ] = Σ_y p_y log(p_y / p′_y) = KL( p ‖ p′ ) ≥ 0.

Meanwhile, under the true conditional label distribution p*,

    E_{y ~ p*}[ ΔCE ] = KL(p* ‖ p) − KL(p* ‖ p′)   (which may be of either sign).

Therefore the error of the self-assessment relative to truth is

    E_p[ΔCE] − E_{p*}[ΔCE] = KL(p ‖ p′) + KL(p* ‖ p′) − KL(p* ‖ p).

*Interpretation.* The model, asked to judge whether continuing its own computation will help it, will always answer "no," and the strength of that "no" is exactly its own drift KL(p‖p′). Improvement is *invisible from the inside*: any change of mind is, from the current mind's point of view, a loss. This is the information-theoretic content of "a system cannot extract information about y from its own output" (data processing), and it is the reason every label-free causal/value signal proposed in the literature is biased in the same direction. It is also the cleanest available explanation of the reported evidence: **a gate trained on any self-referential criterion will sit high (q = 0.88) while the state is unchanged, and the correlation between the gate and the downstream effect will be weak — not as a failure of engineering, but as the exact behavior of the estimator being used.**

**Corollary 6.2.** *The drift Δ_l = KL(p_θ(·|h_l) ‖ p_θ(·|h_N)) proposed in §2 as a label-free alarm is the same quantity that generates the bias.* It is admissible as a *necessity certificate* ("my mind is still changing, so there might be something here") and inadmissible as a *value* ("continuing will help"). Conflating the two is exactly the error that produces gates correlated with their own activation rather than with their effect.

**Corollary 6.3 (bound on what can be computed at inference).** Any inference-time causal statistic that is a function of the model's own outputs has mutual information with the true task effect bounded by the model's calibration error in the region of interest:

    I( self-signal ; Δℓ_task ) ≤ I( p_θ ; y ) − I( p_θ ; y | self-signal ),

and in particular, self-signals are informative only to the extent that the model's predictive distribution is *itself* accurate. They cannot add information. Consequently the causal statistic must be *learned at training* (where y exists) and only *evaluated* at inference. There is no free lunch here: the value of computation is a supervised quantity by necessity, and the constraints in the problem statement ("no labels at inference" is fine; "no labels at training" is not) leave exactly one door open — a critic.

**The one honest positive direction.** The *interventional* statistic Ψ_l (ablate the mechanism, measure output-distribution change) is label-free, unbiased for the question "does this stage change the function the model computes," and computable at training/evaluation. It answers "does the stage matter," not "does continuing help." That is a weaker question, but it is answerable, and it is the correct input to a supervised critic: the critic learns to map Ψ_l (and the state) to the true loss change, and at inference the model needs only to *evaluate* the critic on Ψ_l — which is label-free. **The recipe: causality at inference, value at training.**

### 6.5 Minimal falsifiable test

**Test 6A (the correlation that settles it).** Over a held-out batch, compute for each stage and sample: the controller's q_l, the local sensitivity ω_l (one gradient per stage), and the interventional effect Ψ_l (N forward passes with the stage mechanism ablated; with 12 stages this is 12 forward passes per batch — trivially affordable at evaluation). Report Spearman ρ(q_l, Ψ_l) computed (i) over all (sample, stage) pairs and (ii) restricted to stages where Ψ_l is in the top decile. **Decision rule:** if ρ ≥ 0.5 over the top-decile-restricted set, the controller already tracks the causal effect and the gap is not the binding problem. **Falsifier:** ρ < 0.2 in the restricted set means the controller is blind where it matters — the gate is a function of its own activation, not of the mechanism — and every report containing q_l as evidence must be re-read.

**Test 6B (the bias test, directly targeting Prop 6.1).** For a batch, compute the model's self-assessed ΔCE (which equals KL(p‖p′) for each sample if the model is queried for its own distribution) and the true ΔCE (with labels). **Prediction:** the self-assessed value is ≥ 0 for essentially every sample where an interior exit is chosen (it is ≥ 0 in expectation by the proposition), while the true value is negative for a substantial fraction — the samples that would have benefited from more depth. **Falsifier:** if self-assessed and true ΔCE agree in sign for most samples, the model's predictive distribution is not behaving as a belief (e.g., the training has decoupled the readout from the trunk, or an external temperature breaks the correspondence), and the quantitative form of the bias must be recomputed for that parameterization.

**Test 6C (does a learned critic survive distribution shift?).** Train a probe ψ(h_{l−1}, h_l, ū_l) → Δℓ_l at training time; evaluate its calibration on (i) the training distribution, (ii) an input-shifted region, (iii) the region where the halting decision is in doubt. **Prediction:** the probe is well-calibrated in (i), degrades in (ii), and is worst in (iii) — because (iii) is where the model's own confidence is lowest, and consequently where the causal estimate is least reliable *precisely at the samples that determine the decision*. **Falsifier:** the probe remains calibrated in (iii); then the causal estimate can be deployed as-is and the problem reduces to engineering.

---

## 7. Problem 7 — The 3-Body Problem

### 7.1 Formal statement

The system is a composite map

    (task D, architecture A, objective J, optimizer, RNG)  ⟼  verdict v ∈ R

with the components coupled: changing the objective changes which parameters are learnable (and hence the effective architecture); changing the architecture changes the curvature of the objective (and hence the effective optimizer step sizes); changing the task changes whether the depth axis is identifiable at all (§1, N1). The reported behavior: 10+ one-dimensional sweeps, each changing the verdict; σ ∈ {0.2, 0.1, 0.0} → c₂ ∈ {0.173, 0.291, 0.242}; λ = 0.05 → corr +0.771 at N=12 but +0.266 at N=24; λ ∝ N → +0.512 at N=24.

The formal question — "is this fundamentally chaotic?" — requires a definition, because the answer determines the remedy. *Chaotic* in the technical sense means bounded, sensitive dependence on initial conditions, with a positive Lyapunov exponent: infinitesimally nearby configurations diverge exponentially in the parameter, and the divergence is not explained by stochasticity. That is a property of the *dynamics*. **Practical intractability** means the map from controls to verdict is many-to-one, ill-conditioned, and dominated by a noise floor, so that a finite number of point-evaluations cannot locate the good set. That is a property of the *identification problem*. The evidence supports the second, and the distinction is not academic: chaos needs a different strategy (averaging, smoothing the objective, or abandoning the frame); a poorly identified map needs reparameterization and a better reporting object.

### 7.2 Five candidate mechanisms, ranked by explanatory power

**C1 — Missing dimensionless variables (units).** A cost coefficient λ must absorb two dimensions to be well-defined: a loss dimension (so it is a price in units of loss) and a scale dimension (so it is a price per unit *of what*). In the reported sweeps, λ is compared between runs with different N at fixed value and behaves differently at each N, and the single sweep that works sets λ ∝ N. That is the signature of an interaction term that has not been normalized away. In the parameterization with a dual variable (§1), the primal control is not λ but the *normalized budget utilization* u = (B − 1)/(N − 1) ∈ (0,1], and the price μ becomes state. In these coordinates the λ ∝ N result is not a lucky hyperparameter — it is the statement that the runs were being compared at *different utilizations*, and the correct comparison has no N-dependence left. **This is confirmatory evidence, not a coincidence**, and it is the primary explanation.

**C2 — Knife-edge flux (the verdict is a function of the operating point, not of the mechanism).** According to §1.2 (M5), the observable correlation is largely determined by the fraction of samples whose halting logit lands in the threshold window, which is a single scalar analog of the operating point. Two implications: (i) any sweep that moves the operating point sweeps the verdict, whether or not it changes anything mechanistically; (ii) the *sign* of the change is set by the shape of the logit distribution near the threshold, which is determined early in training and is effectively a random variable across seeds. This produces exactly the observed pattern: conclusions that reverse under changes of a "hyperparameter" that has no mechanism attached to it. **This is the second primary explanation**, and it is *testable*: it predicts that the observed correlation should be a function of a single measurable scalar (the fraction of samples in the threshold window, or the mean exit stage). Plotting the correlation against that scalar across all sweeps should collapse all 10+ sweeps onto one curve. If it does, the "three-body" problem was a one-dimensional curve viewed from the wrong coordinates.

**C3 — Noise floor of the verdict metric.** The reported 5-run data for the fixed configuration gives compute/token ∈ {2.53, 2.60, 2.90, 2.50, 2.53} and correlations ∈ {+0.433, +0.494, +0.611, +0.482, +0.491}. The seed standard deviation of the correlation is ≈ 0.07 on a mean of ≈ 0.50; the seed standard deviation of the compute is ≈ 0.16 on a mean of ≈ 2.6, i.e. 6%. Any comparison that does not exceed this dispersion is uninterpretable, and *any* sequence of one-dimensional sweeps generates apparent "verdict changes" at this amplitude. Since the reported correlations are per-run rather than per-seed-mean, part of the 10+ sweep sensitivity is simply the sampling distribution of a noisy statistic. This candidate is *boring* and therefore usually under-weighted in practice; it should be quantified and subtracted first, before any mechanism is invoked.

**C4 — Multi-stability from the runaway coordinate (the attractor structure, not chaos).** Facts 1–3 of §5 give the training dynamics a trivial attractor (the static one-stage model) and a non-trivial one (the adaptive solution) whose basins depend on initialization and on the early phase of training. Moreover §4.2 shows the collapse is *self-reinforcing*: below a threshold depth signal, the task cannot sustain depth against the cost, so the trajectory collapses further. Two attractors with a shared boundary and a self-reinforcing mechanism produce *history dependence* — the same hyperparameters give different verdicts depending on the seed, and small changes to the objective move the boundary. This is **multi-stability**, not chaos: it has no exponential divergence, it can be identified by hysteresis (sweeping a control up and down returns to a different branch) and eliminated by making the boundary explicit (a curriculum-free initialization near the adaptive branch, or a constraint that prevents the collapse attractor).

**C5 — Genuine chaos.** Possible in principle at large N (N coupled gates, each with its own multiplicative effect on the mixture, trained by a non-convex procedure); the evidence does not distinguish it, and the discriminating test is standard: the Lyapunov exponent along the training trajectory, or comparing the *seed dispersion* of the verdict at a fixed control against the *control sensitivity* of the seed-averaged verdict. Only if the latter substantially exceeds the former across adjacent controls is chaos indicated. Note that the σ-sweep is *unimodal* (0.173 → 0.291 → 0.242), and a unimodal interior optimum in a noise/regularization scale is the classic bias–variance signature — the opposite of chaotic scatter. **The reported data therefore argues against chaos as the primary explanation.**

### 7.3 The invariant that was missed

Two of them, and they are the same invariant seen from two sides.

**Invariant A (dimensional consistency at the primal level).** The objective must be formulated so that its trade-off parameter is dimensionless. The clean form is the constrained form of §1: minimize the task loss subject to average compute ≤ B. The remaining control is u = (B−1)/(N−1), which is *defined* to have no residual N-dependence; the price μ is learned by dual ascent. Under this formulation, the question "what λ for this N?" disappears, and any residual N-dependence in the verdict is evidence of a mechanism (e.g., depth resolution, §4) rather than of units — which is exactly the information one wants from a sweep.

**Invariant B (report the frontier, not the point).** The decision-relevant object is the function

    F(B) = best task metric at average compute B,     G(A) = the analogous curve for the best *static* depth

The verdict "adaptive is good" is the statement F(B) > G(B) for some B < N, and the *entire* reported fragility is an artifact of compressing a function into a scalar (a correlation, a single accuracy) and then comparing scalars across configurations where B, N, and the target's scale (§2, D2) all differ. Under this framing, sweeping λ no longer changes a "verdict" — it moves a point along F; sweeping N changes the domain of B; and the σ-sweep changes the task, hence F itself. Everything that looked like chaos becomes a family of curves, and the comparisons that were being attempted point-to-point become comparisons of curves.

**Answer to the posed question.** It is not fundamentally chaotic. It is a many-to-one, ill-conditioned, multi-stable identification problem whose reported symptom (verdict reversal) is produced by (i) a non-dimensionless cost parameter, (ii) a threshold-window operating point that converts any control change into a verdict change, (iii) a noise floor comparable to the effect sizes, and (iv) two competing attractors generated by a runaway coordinate. All four are structural and fixable. If genuine chaos also exists at large N, it is a smaller effect than these, and none of the remedies for it (ensembling, smoothing) address the reported failures.

### 7.4 If it *were* chaotic: the strategies, and why they are second-best

- **Ensemble:** average the verdict over seeds. This is a remedy for the metric, not for the system: it converts a noisy point estimate into a mean, but leaves the multi-stable structure untouched. It is worth doing regardless (as C3).
- **Meta-optimizer:** optimizes the controls (u, μ, threshold) against the *frontier* rather than against a scalar verdict. This is not a remedy for chaos either; it is a remedy for a badly specified objective, and it is the practical form of Invariant B.
- **"Is 'adaptive' the wrong frame?"** This is the one question in the set with a clean yes/no answer: **yes, as a design goal, and no, as a consequence.** "Adaptive" names an emergent property of a decision problem — spend computation where its marginal value exceeds its marginal cost — not an architectural target. Framing it as a target forces the search for adaptivity even when the task has no depth-necessity margin (§1, N1), which is precisely when the system collapses to a static router and the sweeps begin. Framed decision-theoretically — a value of computation, a price, and a budget — the same architecture either adapts or correctly declines to, and both outcomes are informative. The reported evidence (adaptive 0.78–0.79 versus full-budget 0.69–0.71 at 42–48% of compute) is *ambiguous between the two frames* — an adaptive system and a system that stops before it degrades look identical at the level of the reported numbers. §Q4 gives the test that separates them.

### 7.5 Minimal falsifiable test

**Test 7A (collapse the sweeps).** Replot all existing runs with the x-axis replaced by a single measured response coordinate — the fraction of samples whose halting logit lies within one nat of the threshold, or equivalently the mean exit stage — and the y-axis replaced by the compute–accuracy frontier's distance from the static envelope. **Prediction:** the 10+ sweeps collapse onto one curve, and the "different verdicts" become different points on it. **Falsifier:** if runs with the same response coordinate have widely different frontiers (beyond seed dispersion), then C2 is refuted and a genuine multi-modal or chaotic structure is present, at which point the ensemble and hysteresis tests (7B, 7C) become necessary.

**Test 7B (hysteresis).** Sweep a control (the budget/utilization) upward and then downward over the same range in the same training run family. **Prediction:** if the mapping is a curve (C1/C2), the two directions coincide within seed noise; if there are two attractors (C4), the up-sweep and down-sweep return different branches, with a hysteresis loop whose width measures the basin separation. **Falsifier:** a wide, reproducible hysteresis loop with *no* response-coordinate collapse would indicate genuine multi-stability, and the correct remedy becomes initialization/annealing of the collapse attractor rather than reparameterization.

**Test 7C (chaos, properly).** Compute the seed dispersion of the verdict at fixed controls (sd_seed) and the control sensitivity of the seed-averaged verdict (d⟨v⟩/dc × Δc) over adjacent control values. **Prediction:** d⟨v⟩/dc × Δc ≫ sd_seed only along the response coordinate, and ≈ 0 along all orthogonal directions once the objective is written in constrained form. **Falsifier:** comparable, non-collapsing sensitivity in every direction, with |Δv| growing under infinitesimal control perturbations and no convergence under seed averaging — the signature of a positive Lyapunov exponent. If that is found, no reparameterization will help and the correct response is to abandon point estimates of the verdict entirely in favor of distributional ones (ensembles over seeds and controls reported as distributions, with the frontier as the only stable object).
