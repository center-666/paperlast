# D2-B2 constrained-health development-screening decision

## Evidence identity

- Frozen D2-B2 protocol: `af279ba71b0edbdc10131f9318338ebd2599a119`.
- Corrected runner execution commit: `d1cfa56095f7e57d9b95356f930c4eb4123c445b`.
- D1 inputs are frozen at `7e32a41b9802438f6645d9048ee46386caa53418`. The D1 and `main` worktrees were not modified.
- The run used the new physical-path namespace `[20261125, 7319, 1]`, 12 scenes, 24 attack conditions, 512 attack paths per scene-fixture, and 512 healthy paths per scene. Attack risk fitting and policy evaluation used disjoint halves. Health calibration and health evaluation used disjoint halves.
- Corrected outputs contain 12 hashed scene archives, 24 attack conditions, 6,144 attack trajectories and 6,144 healthy trajectories. They are development-screening evidence.
- An initial run at `293d5f4` stopped before any scene archive was written because scene-zero result packaging referenced an unbound fixed-health-cost variable. The progress manifest listed no completed scenes. That incomplete attempt is preserved outside the repository. The correction record is `D2_B2_FIRST_ATTEMPT_CORRECTION.md`. The corrected run replays the same registered namespace and is not a new replication.

## Constrained rule and verification

The four policies all minimize the same frozen conditional single-step attack-risk proxy over a nonempty action set. R1 uses M7 only. HC-static intersects M7 with the D1-B0 value-budget constraint based on the fixed full-acceptance covariance. HC-scalar uses a policy-history scalar fit of the dynamic relative health curve to the static curve. HC-dynamic uses its full dynamic relative health curve. All three budgets inherit the six D1-B0 epsilon values without retuning.

An independent replay reimplemented the M7 history update, three health-feasible sets, risk argmin, and larger-q tie rule without importing the D2-B2 runner. It replayed 223,735 target-alarm decisions with zero action mismatches, zero saved-feasible-set mismatches, zero M7 violations, and zero selected-action health-budget violations. The reconstructed attack and health means matched the analysis file.

## Main results

Equal-weighting over the 12 scenes and two attack fixtures gave the following target missing-block attack-window MSE.

| Policy | Attack-window MSE | Recovery-window MSE |
|---|---:|---:|
| R1 | 9.6742 | 1.4235 |
| HC-static | 7.8798 | 0.9778 |
| HC-scalar | 7.8855 | 0.9750 |
| HC-dynamic | 7.8798 | 0.9778 |
| Full-accept diagnostic | 11.2112 | 0.9484 |

HC-dynamic reduced the attack-window MSE by 18.55% relative to R1. Its path-paired descriptive 95% bootstrap interval for the difference was `[−2.037, −1.568]`. The recovery-window MSE fell by 31.31% relative to R1, with a paired difference of `−0.446` and interval `[−0.504, −0.391]`. Relative to full acceptance, HC-dynamic reduced attack-window error by 29.71% and increased recovery error by 3.09%.

The healthy missing-block MSE was 0.930297 for HC-dynamic and 0.930210 for full acceptance, a 0.0094% increase. The descriptive equal-scene target-link FWER was 0.521%, and all-edge FWER was 1.888%. These path counts do not prove a uniform health-risk guarantee.

## What caused the gains

The aggregate risk-plus-health constraint altered decisions relative to R1 on 32.07% of target-alarm decisions. It removed `q=0` from the feasible set on 71,831 of 223,735 alarm decisions, about 32.1%, across this run. These decisions account for the large difference between constrained policies and R1.

The independent audit found **no difference at all between HC-static and HC-dynamic feasible action sets** on all 223,735 decisions. Their selected actions and every estimated trajectory were therefore identical. The observed 18.55% attack-window improvement over R1 comes from the shared health-budget constraint, not from dynamic history.

The scalar constraint did differ from the static set on 8,995 alarm decisions, about 4.0%. HC-dynamic and HC-scalar disagreed on 3.98% of alarm decisions. Their aggregate attack-window MSE difference was only `−0.00565`, or `−0.072%`, in favor of HC-dynamic. The descriptive path-bootstrap interval was `[−0.01080, −0.00092]`. Their recovery-window MSE differed by `+0.00274`, or `+0.281%`, with interval `[+0.00224, +0.00322]`. Thus this small attack-window difference came with a small recovery cost.

The dynamic-versus-scalar action change was confined to stable-U scenes 6 and 7, where the one-step comparator based on a fitted scalar removed some candidate weights that the full dynamic curve allowed. In those two scenes, HC-dynamic improved the attack-window MSE relative to HC-scalar by 0.0468 and 0.0210, while recovery MSE increased by 0.0194 and 0.0135. Across the other ten scenes the policies selected the same actions.

Health-budget constraints were never active at the selected action in this sample. The mean nominal excess-cost utilization was approximately 1.6–1.7% of the q=1 reference, while epsilon ranged from 1.0% to 2.84%. The lower feasible action counts largely reflect the ordered candidate grid and the M7 lower bound. The budget was a meaningful feasibility screen, but the selected actions sat well inside its boundary.

## Research decision

D2-B2 demonstrates a promising development-screening tradeoff for a health-budget-constrained risk-minimizing soft-fusion rule relative to pure risk minimization. The dominant observed contribution is the nominal-health feasibility constraint. This sample does not support an independent benefit for full dynamic history over the static health constraint. It provides only a small, localized incremental difference over the fitted-scalar control, with a recovery cost.

The result does not establish that history-dependent health costs are universally unnecessary. The registered dynamic and static constraints happened to induce identical feasible sets throughout these trajectories. Any follow-up must make that set geometry, and the independent value of the history update, its explicit question. The selected-candidate nominal proxy is not a formal guarantee for adaptive H1 health MSE. D2-C remains unstarted.

## Reproduction outputs

- `b2_results/B2_RUN.json` records run identity, seeds, value budgets, thresholds, input hashes, results, and output hashes.
- `b2_results/B2_INDEPENDENT_AUDIT.json` records the independent action, M7, budget, and archived-feasible-set replay.
- `b2_results/B2_ANALYSIS.json` and `b2_results/B2_ANALYSIS.md` contain paired path-level descriptive summaries.
