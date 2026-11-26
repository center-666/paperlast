# Recovery-tail and partial-link identification protocol

## Frozen source and prior experiment

Parent source is commit `f97748fcba7c45981878352aa03bc1b15f78dc65`, based on frozen 50-node common-G physics `8953e1dd9a5e2535ead2698acb6f345a85d4f532`. W1 path archives and detector thresholds are read-only. Reuse the W1 healthy calibration thresholds because the estimators, complete six-link scan family, start k=5, full 150-report horizon, score algorithms and watermark marginal laws are unchanged. The targeted-disturbance tests still score all six registered edges and use the corresponding W1 family maximum threshold.

## W0 replay finding and tail event

Path 184 in `confirm/unstable_2_0.npz` is fixed as a mechanism case study. It is not a new test sample. In the quarter-turn / true-key matched arm its five-receiver attack-window MSE is 69.7663; the iid and negative-Watson values on the same physical path are 0.05491 and 0.05898. The receiver-zero link index is 0, sender 1. At report 109 it accepts a decoded attack shift of norm 1465.14, despite alarmed earlier reports. The common-G dynamics have spectral radius 1.09193. The current W1 archive does not retain full report-by-report gates, so the path replay reconstructs those traces from its recorded seed namespace and frozen algorithm.

Three frozen counterfactuals distinguish the mechanism. Removing only link 0's report-109 shift reduces path MSE to 0.06813. Gating that one report gives 0.06816. Latching link 0 closed after its first alarm also gives 0.06816. In the eight reports 102–109 the quarter-turn key for link 0 is exactly (I_2). Its noiseless parity-residual whitened norm is at numerical zero and the online score at k=109 is 0.0826 times its calibrated threshold. The late accepted packet, rather than a long sequence of post-alarm acceptances, causes the catastrophic state error on this path.

Before drawing any W3 attack path, freeze a deliberately targeted event

[
E_{m tail}={operatorname{MSE}_{Q,mathrm{match}}>1
  land operatorname{MSE}_{Q,mathrm{match}}ge10operatorname{MSE}_{N,mathrm{match}}},
]

where Q is the uniform four-way quarter-turn code and N is the original fixed-negative Watson code. MSE is the path mean over the original five affected receivers and the 60 disturbed reports. The threshold is motivated by the observed W1 outlier, so W1 is discovery evidence. W3 uses completely new physical and key streams and estimates the locked event frequency. Also record (E_{m tail}) under iid and two-coordinate signs descriptively, late high-amplitude accepted reports after a previous alarm, the minimum eight-report noiseless whitened signal energy, and the alarm-to-report timeline. Do not remove high paths or select a replacement metric.

W3 tail discovery has 256 paths per original 12 model-angle-onset cells and namespace ([20261127,52387,1]). Independent targeted-event confirmation has 512 paths per cell and namespace ([20261127,52387,2]). It reuses only the frozen W1 healthy thresholds. A large accepted report is defined as a decoded perturbation norm of at least 100, accepted after that edge has previously alarmed. A state-compatible window has relative noiseless whitened parity energy below 10^-20, using the exact eight-report frozen covariance operator. Report all-path means, medians, 95th and 99th percentiles, maxima and event counts. Give cell-level exact Clopper-Pearson intervals and model/overall intervals from a path bootstrap stratified within the original 12 cells. Use the same path bootstrap for paired MSE and absolute mean contrasts. Event incidence is a descriptive, targeted confirmation of a post hoc event threshold and is identified as such.

## W4 partial-support localization

Test the following eight frozen directed-edge supports in source edge order ((0,ldots,5)):

- Six singleton supports ({0},ldots,{5}).
- ({3,5}), the two links into receiver 17.
- ({0,1}), two links into different receivers 0 and 2.

The score always scans all six links. Only the listed supports receive the frozen 60-report disturbance. Unselected links remain clean and are included in the healthy-edge false-alarm denominator. Compare quarter-turn, fixed-negative Watson and iid Watson. Run both the frozen plugin score and the true-sender-key matched score. Reuse exact W1 thresholds and score start k=5. No threshold is recalibrated on attacked paths.

Use both original models, three registered directions and both onsets, giving 96 support-by-model-by-angle-by-onset cells. Within each cell, one physical path and key stream are shared across all eight supports and all six detector/codec arms. W4 development uses 256 paths per cell and namespace ([20261127,52387,3]). Independent confirmation uses 512 paths per cell and namespace ([20261127,52387,4]). Archive exact source hashes and frozen threshold families before running.

For each path report each attacked edge's first-three-report detection, each un-attacked edge's false-alarm probability, path-level any false alarm among clean edges, and exact support identification (every attacked edge detected at least once and no healthy edge alerted in the same first-three-report window). Also report per-edge delay censored at 60, full-window alarms, pre-onset alarms, targeted-receiver / five-original-receiver / whole-network attack and recovery MSE. The event of exact set identification is a path event; do not infer it by multiplying marginal edge rates.

Primary localization endpoints are exact-support probability and the clean-edge first-three-report false-alarm rate. Report paired differences Q−N and N−iid with 95% whole-path bootstrap intervals stratified by the 96 fixed cells, preserving all arms and supports within path. W1 full-health path FWER is reported separately from attack-conditional clean-edge false alarms. This design covers the six known monitored edges only. It does not claim open-set source localization or arbitrary network attack support.

## Reproduction and status

The complete run identity binds source commit, all Python/C inputs, W1 threshold manifest hash, random namespace and archive hashes. Development and confirmation streams are disjoint. Confirmation uses the fixed development code and W1 thresholds. Any failed numerical or archive check stops interpretation until fixed and documented. The prior radial-binning stopping decision remains closed.
