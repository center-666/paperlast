# D2-B corrected development-screening decision

## Evidence identity

- Frozen D2-B protocol: `af279ba71b0edbdc10131f9318338ebd2599a119`.
- Corrected runner execution commit: `ac4d58c4ccac69fb101ab40baa7d9a06e3bffdf2`.
- The runner uses the frozen D1 checkout at `7e32a41b9802438f6645d9048ee46386caa53418`. D1 and `main` are read-only inputs.
- The corrected run uses the registered physical-path namespace `[20261124, 4401, 1]`, with 12 scenes, 24 attack conditions, 512 attack paths per scene-fixture split into 256 fit and 256 evaluation paths, and 512 health paths per scene split into 256 calibration and 256 evaluation paths.
- All 12 corrected scene archives passed SHA-256 checks. The run and analysis are development-screening evidence, not confirmatory evidence.
- The initial run at `2144a462` is preserved outside the worktree and superseded for RH-scalar comparisons because its scalar fit mixed an absolute dynamic health curve with a q=1-relative fixed curve. The corrected run replays the same registered paths. It is not an additional independent replication. See `D2_B_FIRST_RUN_CORRECTION.md`.

## Corrected run result

Across 220,526 target-link alarm decisions in the attack windows, R1, RH-static, RH-scalar, and RH-dynamic chose exactly the same action on every decision. Their attack-window, recovery-window, health, and network MSE values are therefore identical path by path in this sample.

The selected alarm weights were `q=0` on 217,579 decisions and `q=0.8` on 2,947 decisions. The M7 graph admitted all seven actions on 217,579 alarm decisions and only `{0.8, 0.9, 1}` on 2,947 decisions. Thus every nonzero action occurred where M7 excluded `q=0`, and the risk-only policy selected the smallest remaining action. No risk-table fallback occurred.

When `q=0` was legal, the smallest nonzero candidate's risk-table upper value exceeded the `q=0` value by at least 0.447 across the evaluated scene events. The equal-scene mean gap was 3.235. The fitted table sets the `q=0` single-step attack-damage label to zero by construction. This made risk-only isolation the action in almost every alarm case, leaving little decision space for the added health costs.

The dynamic health curve contains history-dependent shape variation. Its median relative residual from the M7-restricted RH-scalar fit was 9.05%, and dynamic-versus-static pairwise candidate ranks differed in 1.26% of comparisons. However, the dynamic and static health-only minimizers never differed, and the risk-plus-health actions did not change. In this sample, the dynamic health term did not produce independent policy value beyond the scalar control.

Relative to the full-accept diagnostic, RH-dynamic reduced mean missing-block attack-window MSE from 11.224 to 9.857. Its paired difference was `−1.366` with a descriptive path-bootstrap 95% interval of `[−1.620, −1.118]`, or 12.17% lower. Recovery-window MSE increased from 0.958 to 1.425, a paired increase of 0.468 with a descriptive interval of `[0.421, 0.519]`, or 48.84% higher. This is the observed attack-versus-recovery cost of the shared alarm-driven hard isolation action.

The mean health missing-block MSE was 0.926761 for all four policies and 0.925940 for full acceptance, a paired increase of 0.000821. Equal-scene mean target-link FWER was 0.814%, and equal-scene mean all-edge FWER was 2.441%. These are descriptive estimates under this calibration and holdout split. They do not establish a uniform health-risk guarantee.

All attack evaluation paths had at least one target alarm, while 39.07% of actively polluted ticks were unalarmed. The B1 q=1 regression passed in all 12 scenes. Across the smoke checks, the maximum discrepancy between the optimized and independent per-neighbor update was `1.04e-10`, and the maximum discrepancy against frozen B1 was `1.34e-11`.

## Decision

D2-B does not support a D2-C claim that dynamic history-aware health fusion improves the frozen risk-based policy. The direct RH-dynamic versus RH-scalar difference was exactly zero on these paths. It also does not establish that history-dependent health costs are generally unhelpful. They changed some candidate-curve ranks but did not change the minimizing action under the registered risk table and M7 constraint.

The diagnosed limitation is the current decision objective. The single-step incremental-risk table assigns `q=0` an exact zero attack label. Its upper value strictly dominated nonzero candidates whenever zero was feasible in this sample. The health term was not large enough to change those choices, and M7 forced the few remaining `q=0.8` choices. Any follow-up would need a separately reviewed objective that gives the attack-risk proxy and the healthy-information cost an explicit, decision-relevant tradeoff. The current B outputs must not be reused to tune such a follow-up, and D2-C remains unstarted.

## Reproduction outputs

- `b_results_corrected/B_RUN.json` records run inputs, seeds, environment, thresholds, per-scene statistics, and all scene-archive hashes.
- `b_results_corrected/B_MECHANISM_AUDIT.json` records action replay, curve-shape diagnostics, M7 action-set counts, and the 12-scene q=1 regression.
- `b_results_corrected/B_ANALYSIS.json` and `b_results_corrected/B_ANALYSIS.md` contain path-paired descriptive summaries and bootstrap intervals.
