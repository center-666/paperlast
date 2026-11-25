# D2-B2 constrained-objective development screen

Execution commit `d1cfa56095f7e57d9b95356f930c4eb4123c445b`.
This is independent development-screening evidence. Nominal-health feasibility is not a formal adaptive-alarm health guarantee.

Attack MSE averages equally over 12 scenes and both attack fixtures. Negative paired differences favor the candidate policy.

## Attack and recovery MSE

| Policy | Missing block, attack | Missing block, recovery | Network, attack | Network, post-20 |
|---|---:|---:|---:|---:|
| R1 | 9.67415 | 1.42349 | 1.48847 | 1.39759 |
| HC_static | 7.87982 | 0.977769 | 1.45374 | 1.37881 |
| HC_scalar | 7.88548 | 0.975032 | 1.45388 | 1.37885 |
| HC_dynamic | 7.87982 | 0.977769 | 1.45374 | 1.37881 |
| Full-accept diagnostic | 11.2112 | 0.948437 | 1.52083 | 1.40959 |

## Paired attack-window contrasts

| Comparison | Difference, 95% path-bootstrap interval | Relative change |
|---|---:|---:|
| HC_dynamic_minus_R1 | -1.79432 [-2.03728, -1.56794] | -18.548% |
| HC_dynamic_minus_HC_static | 0 [0, 0] | 0.000% |
| HC_dynamic_minus_HC_scalar | -0.00565167 [-0.0108007, -0.000920481] | -0.072% |

## Per-scene HC-dynamic minus HC-scalar

| Scene | Attack window | Recovery window |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 3 | 0 | 0 |
| 4 | 0 | 0 |
| 5 | 0 | 0 |
| 6 | -0.0468311 | 0.0193621 |
| 7 | -0.020989 | 0.0134858 |
| 8 | 0 | 0 |
| 9 | 0 | 0 |
| 10 | 0 | 0 |
| 11 | 0 | 0 |

## Attack-window contrast by fixture

| Fixture | HC-dynamic minus R1 | HC-dynamic minus HC-scalar |
|---|---:|---:|
| continuous | -2.9281 | -0.0223389 |
| burst | -0.660546 | 0.0110355 |

## Healthy evaluation

| Policy | Missing-block post-20 MSE | Paired difference vs full acceptance |
|---|---:|---:|
| R1 | 0.930596 | 0.000385641 |
| HC_static | 0.930297 | 8.70456e-05 |
| HC_scalar | 0.930297 | 8.70456e-05 |
| HC_dynamic | 0.930297 | 8.70456e-05 |
| full_accept | 0.93021 | — |

Equal-scene target-link FWER: 0.521%.
Equal-scene all-edge FWER: 1.888%.
Equal-condition path-miss fraction: 0.000%.
Equal-condition polluted-tick non-alarm fraction: 39.726%.
Risk-table fallback count: 0.
Mean health-feasible action counts R1/static/scalar/dynamic: [6.9482901040806215, 5.830010830258641, 5.790012298768287, 5.830010830258641].
q=0 feasible fractions R1/static/scalar/dynamic: [0.9870725260201553, 0.6659920700479102, 0.6659920700479102, 0.6659920700479102].
Mean nominal-health budget utilization static/scalar/dynamic: [0.01594644185697791, 0.017175556452918492, 0.017445104995904855].

Health-budget active decisions for HC-static, HC-scalar and HC-dynamic: [0, 0, 0].
## Action changes

- HC_dynamic_minus_R1: 32.071% alarm-decision disagreement.
- HC_dynamic_minus_HC_static: 0.000% alarm-decision disagreement.
- HC_dynamic_minus_HC_scalar: 3.984% alarm-decision disagreement.

## Independent replay audit

Replayed 223,735 target-alarm decisions. The independent controller replay found 0 decision mismatches, 0 M7 violations, and 0 nominal-budget violations.
HC-static and HC-dynamic feasible action sets differed on 0 target-alarm decisions.

Independently replayed alarm action counts.

| Policy | q=0 | q=.2 | q=.4 | q=.6 | q=.8 | q=.9 | q=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | 220833 | 0 | 0 | 0 | 2902 | 0 | 0 |
| HC_static | 149002 | 0 | 0 | 37089 | 37644 | 0 | 0 |
| HC_scalar | 149002 | 0 | 0 | 28094 | 46639 | 0 | 0 |
| HC_dynamic | 149002 | 0 | 0 | 37089 | 37644 | 0 | 0 |

The risk table is a conditional incremental-damage proxy, not a closed-loop MSE bound. All intervals are descriptive development-screening summaries.
D2-C remains unstarted.
