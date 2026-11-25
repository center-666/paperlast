# D2-B development screening analysis

Execution commit `ac4d58c4ccac69fb101ab40baa7d9a06e3bffdf2` with protocol `af279ba71b0edbdc10131f9318338ebd2599a119`.
The results are developmental and descriptive. Bootstrap intervals are path-level and do not support confirmatory claims.

Attack MSE is averaged equally across twelve scenes and their continuous/burst fixtures.
Negative paired differences indicate lower MSE for the candidate policy.

## Attack window and recovery

| Policy | Missing-block window | Missing-block recovery | Network window | Network post-20 |
|---|---:|---:|---:|---:|
| R1 | 9.85746 | 1.42521 | 1.49156 | 1.39887 |
| RH_static | 9.85746 | 1.42521 | 1.49156 | 1.39887 |
| RH_scalar | 9.85746 | 1.42521 | 1.49156 | 1.39887 |
| RH_dynamic | 9.85746 | 1.42521 | 1.49156 | 1.39887 |
| Full-accept diagnostic | 11.2236 | 0.957576 | 1.5205 | 1.40934 |

## Paired policy differences

| Comparison | Window difference (95% path bootstrap interval) | Recovery difference | Relative window change |
|---|---:|---:|---:|
| RH_dynamic_minus_R1 | 0 [0, 0] | 0 | 0.000% |
| RH_dynamic_minus_RH_static | 0 [0, 0] | 0 | 0.000% |
| RH_dynamic_minus_RH_scalar | 0 [0, 0] | 0 | 0.000% |

RH-dynamic minus full-accept attack-window difference: -1.3661 [-1.62009, -1.11834]. Recovery-window difference: 0.467636.

## Attack-window contrast by fixture

| Fixture | RH-dynamic minus R1 | RH-dynamic minus RH-scalar |
|---|---:|---:|
| continuous | 0 | 0 |
| burst | 0 | 0 |

## Attack detection and action diagnostics

Equal-condition path-miss fraction: 0.000%.
Equal-condition polluted-tick non-alarm fraction: 39.065%.
Risk-table fallback count: 0.

## RH-dynamic minus RH-scalar by scene

| Scene | Attack-window difference | Recovery-window difference |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 3 | 0 | 0 |
| 4 | 0 | 0 |
| 5 | 0 | 0 |
| 6 | 0 | 0 |
| 7 | 0 | 0 |
| 8 | 0 | 0 |
| 9 | 0 | 0 |
| 10 | 0 | 0 |
| 11 | 0 | 0 |

## Healthy evaluation

| Policy | Missing-block post-20 MSE | Difference from full acceptance | Network post-20 MSE |
|---|---:|---:|---:|
| R1 | 0.926761 | 0.000820515 | 1.31427 |
| RH_static | 0.926761 | 0.000820515 | 1.31427 |
| RH_scalar | 0.926761 | 0.000820515 | 1.31427 |
| RH_dynamic | 0.926761 | 0.000820515 | 1.31427 |
| full_accept | 0.92594 | — | 1.31425 |

Equal-scene mean target-link FWER: 0.8138%.
Equal-scene mean all-edge FWER: 2.4414%.

## Decision activity

- RH_dynamic_minus_R1: 0.000% alarm-action disagreement rate.
- RH_dynamic_minus_RH_static: 0.000% alarm-action disagreement rate.
- RH_dynamic_minus_RH_scalar: 0.000% alarm-action disagreement rate.

## Health-cost history diagnostic

Median relative residual from the RH-scalar fit: 9.051%.
Dynamic-versus-static pairwise health-curve rank discordance: 1.263%.
Dynamic-versus-static health-only argmin disagreement: 0.000%.
The health curves sometimes differ in shape, but neither the health-only minimizer nor the risk-plus-health action changed in this development sample.
The q=1 regression passed in all 12 scenes. Maximum literal-update discrepancy was 1.04e-10.

The conditional single-step risk table is an action-ranking proxy. Its values are not closed-loop MSE bounds.
The interval calculations are descriptive development-screening summaries, not multiplicity-adjusted confirmatory inference.
D2-C is not authorized by this analysis.
