# D2-B2: risk minimization under a history-dependent health budget

## Evidence boundary

D2-B2 is a new, independent development screen on the D2 branch. It does not change the frozen D2-B protocol or its corrected result. It does not reuse any D2-B path for policy fitting or evaluation. It does not authorize D2-C or confirmatory paper claims.

The corrected D2-B result motivates this objective change. Its risk-only policy assigns the exact single-step incremental-damage label zero to q=0. This dominated the additive risk-plus-health objective on nearly all observed alarm decisions. D2-B remains preserved as the additive-objective result.

## Constrained finite-action rule

The candidate weights, common H1 alarm mask, mode-dependent common-G gains, M7 graph, risk table construction, path windows, and four-dimensional physical model remain as frozen in D2-B. On alarm, define the legal M7 set `Q_stab(k)` from the policy's preceding six executed actions. Off alarm, all policies use q=1.

Let `H_abs,k(q)` be the policy's nominal finite-horizon health cost from its own propagated reference covariance, before subtracting the q=1 comparator. Let `H_abs,*(q)` be the corresponding cost from the fixed full-acceptance initial covariance. Define relative penalties

`DeltaH_k(q) = H_abs,k(q) - H_abs,k(1)`

`DeltaH_*(q) = H_abs,*(q) - H_abs,*(1)`.

For each structural link and dynamics, reuse the frozen D1-B0 registered value budget

`epsilon_s = 0.05 - 0.04 * max(kappa_s,0)/(1+max(kappa_s,0))`,

with the exact six registered `kappa_L60` values in `results/B0_definition_audit.json`. Watermark variants of the same physical structure share that frozen budget. No budget or scalar coefficient is fitted on D2-B2 paths.

Four policies use the same pathwise H1 mask, M7 graph, H1 threshold, and fitted conditional single-step risk table `U_1(q|cell)`.

1. `R1` minimizes `U_1(q|cell)` over `Q_stab(k)`.
2. `HC-static` minimizes `U_1(q|cell)` over `Q_stab(k)` subject to `DeltaH_*(q) <= epsilon_s * H_abs,*(1)`.
3. `HC-scalar` first fits `a_k = argmin_{a>=0} sum_{q in Q_stab(k)} [DeltaH_k(q)-a*DeltaH_*(q)]^2`. It minimizes `U_1(q|cell)` subject to `a_k*DeltaH_*(q) <= epsilon_s * H_abs,k(1)`.
4. `HC-dynamic` minimizes `U_1(q|cell)` subject to `DeltaH_k(q) <= epsilon_s * H_abs,k(1)`.

The scalar fit is policy-specific and uses only the covariance reference state, model matrices, and current M7 set. If its denominator is at most `1e-24`, set `a_k=0` and count the event. Each constrained policy's health-feasible action set is intersected with `Q_stab(k)`. Health feasibility permits an absolute numerical tolerance of `1e-10`. q=1 has zero relative health penalty, positive registered epsilon, and is always M7-admissible. Thus each optimization is nonempty. Use the existing larger-q tie rule within absolute objective tolerance `1e-10`.

`U_1` is a path-fitted simultaneous upper estimate of the conditional mean single-step incremental attack damage relative to the clean update. It is not an upper bound on closed-loop MSE. `H_abs` is a nominal covariance-propagation proxy under the registered linear model and mode gains. Because H1 gates depend on measured data, the proxy is not a formal health-performance guarantee under the adaptive alarm process. The epsilon constraint is an algorithmic health budget. Independent healthy-path MSE and FWER measurements remain required.

## Mathematical properties claimed in this screen

For every alarm and every policy, q=1 is feasible, so the finite constrained action exists. R1's risk minimizer remains the same risk objective used in D2-B. `HC-dynamic` chooses the smallest fitted risk upper value available among M7 actions meeting the registered dynamic nominal-health budget. It therefore removes the arbitrary relative scaling between `U_1` and `H_k` in an additive objective.

For a fixed history and alarm cell, write the finite legal set as `S`, the risk proxy as `u(q)`, and the policy health excess as `h(q)`. The value function `V(epsilon)=min{u(q):q in S, h(q)<=epsilon*H0}` is well-defined because q=1 belongs to the feasible set. If `epsilon_2 >= epsilon_1`, the feasible sets are nested and `V(epsilon_2) <= V(epsilon_1)`. Thus the chosen fitted-risk value is a nonincreasing, piecewise-constant function of the permitted nominal-health budget. This is a property of the finite constrained decision rule. It does not assert monotonicity of the true closed-loop attack MSE or healthy MSE.

No strict stability theorem is added. M7 retains its existing numerical screening-certificate scope. No claim is made that nominal health feasibility implies the B0 pathwise statistical health budget, or that `U_1` bounds attacked closed-loop MSE.

## Independent development paths

Use seed namespace `[20261125, 7319, 1]`, distinct from B1, B2, D2-A/A-prime, D2-B, and smoke namespaces. Retain 12 four-dimensional scenes and both continuous and burst fixtures. Each scene-fixture uses 512 new attack paths split by path into 256 risk-table fit paths and 256 policy-evaluation paths. Each scene uses 256 new healthy calibration and 256 separate healthy evaluation paths. Health calibration, risk fitting, attack evaluation, and health evaluation remain disjoint.

The conditional risk table retains the frozen D2-B bins, whole-path bootstrap, support threshold, fallback hierarchy, and recorded-source counts. The four policies receive bitwise-identical H1 masks. They keep separate M7 action histories and nominal covariance histories. The full network and receiver 26 missing-block outputs use the same frozen D2-B definitions.

The primary descriptive development comparisons are `HC-dynamic - R1` and `HC-dynamic - HC-scalar`. `HC-dynamic - HC-static` is secondary. Also compare each policy with the q=1 diagnostic. Report attack-window and recovery-window missing-block MSE, network MSE, healthy MSE, target/all-edge FWER, actions and effective weights, risk-table supports/fallbacks, health-infeasible candidate counts, scalar-fit zero denominators, and paired whole-path descriptive bootstrap intervals. Equal-weight aggregation across twelve scenes and two fixtures remains frozen. No selection of parameters is allowed.

## Inputs and reproducibility

Read the frozen D1 checkout `7e32a41b9802438f6645d9048ee46386caa53418`, its mode table and M7 graphs. Record SHA-256 for this protocol, runner and analysis code, D1 B0 audit and registration, B1/B2 manifests and policy lock, physical configuration, mode tables, path/H1/attack generators, every M7 graph, and every output archive. Do not modify D1 or `main`.

The first execution is a deterministic smoke test with disposable seed `[20261125, 7319, 99]`. It checks candidate health-cost calculations, nonempty feasibility (including q=1), risk minimization within the feasible set, shared H1 masks, q=1 agreement with frozen B1 and an independent per-neighbor implementation, M7 action-history replay, and atomic archive writes. Commit this protocol, runner and smoke-passing implementation before generating registered development paths.
