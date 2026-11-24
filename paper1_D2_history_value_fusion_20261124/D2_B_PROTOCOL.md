# D2-B Independent Development Screening Protocol

## Status

This protocol is frozen after D2-A?. B1, B2, A and A? are existing or posthoc exploratory results. D2-B uses new physical paths and new health calibration paths. It is a development screen and does not authorize D2-C or deployment.

## Objective

Test whether a dynamic history-dependent health cost has independent closed-loop value relative to an attack-risk-only policy and simpler health-cost controls. The primary model is the four-dimensional local common-G estimator at receiver 26. The two-dimensional common-G model remains a historical regression baseline and is not pooled with the four-dimensional results.

## Policies

All four policies share one H1 threshold, one alarm mask, one M7 numerical feasibility graph, one conditional risk table and one tie rule. The target link uses q=1 when it is not alarming.

- R1 uses U1(q).
- RH-static uses U1(q) + H*(q).
- RH-scalar uses U1(q) + a_k H*(q), where a_k is the nonnegative least-squares scalar fit of the current dynamic health vector to H*.
- RH-dynamic uses U1(q) + H_k(q).

The risk table is a frozen A-reference-prior action-ranking proxy. It is not a closed-loop MSE upper bound for any of the four policies. Every policy stores its own action history, dynamic H vector, risk-table fallback and support counts.

The static phi(kappa) term is excluded from the primary B policies. It remains a secondary posthoc correction because A? showed a topology-dependent numerical increment.

## Data splits

Each four-dimensional scenario uses 512 new attack paths. The first 256 paths fit the conditional risk table and the last 256 paths evaluate the closed-loop policies. Continuous and burst fixtures are generated separately and are combined only within the same watermark label during risk fitting.

Each scenario also uses 512 new healthy paths. The first 256 calibrate H1 and the last 256 evaluate health MSE and conditional FWER. Health and attack paths are disjoint.

The attack design contains 12 dynamics-topology-target-watermark scenes and two fixtures, giving 24 attack conditions. The seed namespace is [20261124, 4401, 1]. The runner must reject any seed tuple reused by B1, B2 or D2-A/A?.

## Conditional risk table

Inputs are frozen H1 score, consecutive alarm length, current alarm bit and the A-reference-prior single-step label Y1(q). Scores are normalized by the scene threshold. Fitting boundaries use training alarm-packet tertiles. Consecutive alarms are grouped as 1, 2?3 and 4+.

Each cell requires at least 30 distinct training paths. Fallback order is score-by-run, score-only and pooled-alarm. q=0 has zero current attack input by algebra. Other candidates use path-level means and approximate one-sided 95% upper bounds. The table is an action-ranking proxy and its support, source fixture and fallback level must be saved.

## Health cost

H* starts from the no-alarm full-acceptance P+ mode. H_k starts from the policy's own executed-action covariance reference. The current candidate action is applied once, and the next 59 steps use the no-alarm full-acceptance F1/Omega1 continuation. RH-scalar fits a_k to the current feasible candidate set. A zero denominator sets a_k=0 and increments a counter.

The H1 score must not read estimator state. All four policies must receive bitwise-identical H1 masks on the same path.

## Stability

The M7 graph comes from the frozen D1 mode matrices and seven-action enumeration. A policy can select only actions that extend the current six-action history in the numerical feasibility graph. The graph is a numerical screening certificate. It is not a strict roundoff-certified proof.

## Outputs

Each attack condition stores four policy path-level attack-window MSE, recovery-window MSE, network MSE, effective weight, action counts, action disagreement, polluted-packet non-alarm fraction, risk support and risk fallback counts. Each health scene stores four policy health MSE, target FWER, alarm counts and threshold.

The primary comparison is RH-dynamic minus R1. Secondary comparisons are RH-dynamic minus RH-static and RH-dynamic minus RH-scalar. Results are paired by path and aggregated equally across the 12 physical scenes. B is only a development decision gate for C and does not enter a confirmatory paper claim.
