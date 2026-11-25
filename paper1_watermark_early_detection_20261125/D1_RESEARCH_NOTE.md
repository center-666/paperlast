# D1: frozen-source research audit and pre-run decision

## Frozen baseline and existing evidence

The implementation is isolated in a new worktree and branch based on `8953e1dd9a5e2535ead2698acb6f345a85d4f532`. The existing V3 development selector and final test remain read-only historical exploratory evidence. The historical iid lock remains untouched. No new registered Stage-1 attack or healthy calibration/holdout paths have been generated.

`PHASE0_REPLAY_AUDIT.json` reproduces every checked V3 path-level metric exactly for the unstable/stable original-angle onset-50 scenarios and both iid/fixed-negative schemes. Maximum difference is zero. It independently recomputes scenario-equal-weight summaries from the archived V3 path arrays. The old archives are used only for implementation regression and context.

The original V3 repository result is now correctly summarized at the per-link early endpoint. In the reused onset-50 paths, the probability a particular affected link is detected in its first three attacked reports averaged over the six links was 32.13% iid and 33.56% fixed-negative in the unstable model, and 66.05% versus 68.13% in the stable model. These are historical path descriptions, not new inference. The any-link event is saturated at 100% in the stable case, motivating the registered per-link primary metric.

## Source audit: common-G versus legacy stacked-KF

The experiment uses the frozen steady mode-lookup common-G estimator from `paper1_common_g50_gain_acceptance_20261121/run_common_g_50.py`, not the earlier stacked-Kalman estimator implemented in the original script and exercised by the 50-node time-varying-gate theory archive. The parity residual and plugin score are intentionally the frozen shared detector input. New paths will be run through the common-G estimator.

| Existing function | Actual input | Reused behavior | New experiment code |
|---|---|---|---|
| `run_crn_development.load_backend` | Hash-pinned physical configuration, source model, gain table and source-validation fixtures | Restores the actual 50-node common-G estimator and the six ordered monitored links | Used as the only model and estimator loader |
| `run_common_g_50.physical_paths` | Model, count and isolated seed | Generates the full state and 50-sensor measurement path | One physical path is shared across A-F within each fixed scenario |
| `watson_crn.make_common_draws`, `beta_from_common` | `[path,60,50]` radial variables, per-sender initial axes, transition quantiles and lift bits | Exact signed-Watson chain; `S=0.5 chi-square_2` which is `Exp(1)`; sender-shared transition key | New code generates each candidate schedule from a complete 150-time sender key process |
| `watermark_attack_from_beta` | Raw six-link waveform and beta schedule | Decoded injected report perturbation `delta=M_k a_k` | Reused to inject the same raw attack waveform at onset 50 or 65 |
| `replay.scores` | Physical state, decoded sender packet histories and six-link delta array | Frozen ordered five-report/8-D parity residual, four-residual plugin Gaussian KL, covariance regularizer `1e-6 I` | Used for A/B and the independent V3 regression; no watermark matrix enters this scorer |
| `matched_glrt` (new) | Current eight decoded reports, current/past eight sender M matrices, frozen parity basis and frozen healthy noise model | Nothing; it implements the registered known-model comparator | Builds the full overlapping residual covariance, whitens its rank-14 support, fits shared two-state attack templates for eight candidate onsets, and takes the maximum causal projection score |
| `calibration_rule.path_fwer_cutoff` | One path maximum over six edges and decisions `k=7..149` | Strict `score>threshold` with one-based rank `ceil((n+1)*0.95)` | Calibrates plugin, matched iid, and matched fixed-negative families separately for each model |
| `run_common_g_50.run_common_g` | Physical path, decoded packet perturbation, raw sender packets, explicit gate mask and steady common-G gain table | Full frozen 50-node estimator and source scoring path | Receives arm-specific alarms through `gate_override`; scores are recorded outside the estimator |
| `run_crn_final.measure` | Estimator world, physical paths, affected nodes, onset and threshold | Legacy MSE, full-attack and postattack summaries, eligible detector statistics | Legacy metrics are retained; early endpoints are evaluated on true per-link attack onset windows |

The exact frozen six directed links are `(0,1)`, `(2,3)`, `(15,16)`, `(17,31)`, `(14,49)` and `(17,34)`. Watermark matrices are indexed by sender and time, so repeated outgoing links from one sender share the same key. The decoded raw attack is `M_k a_k`; the GLRT templates use the actual `M_k` directly and never apply it a second time.

## Matched-GLRT support audit

Eight two-dimensional reports create four overlapping residuals in 32 coordinates. Under the frozen V3 healthy noise model and source parity matrices, each monitored sender's residual covariance has rank 14. The new scorer computes the entire covariance from the 8-report raw covariance and the 4×8 overlap map, rather than treating residual blocks as independent. It whitens only the retained 14-dimensional support. All six sender matrices satisfy the registered sender-specific rank 14 check, and the deterministic whitened covariance identity error is under `3.3e-15`.

Independent 8,192-path white-noise checks found empirical whitened covariance errors from about 0.026 to 0.051. These are finite-sample Monte Carlo deviations around the exact identity. The detector is a known-model matched subspace GLRT comparator. This classical comparator is not claimed as a new detector contribution.

All three registered physical perturbation directions are included by one common two-state amplitude template, rather than choosing a template using the test direction. The onset scan has eight choices inside the current buffer, including the pre-active buffer-start nuisance-state case. Checks cover attack angles 0 and ±45 degrees, and onset 50 and 65, for all six links. For onset 65 the first complete eight-report score is at k=72, four reports after onset, as implied by the common first-valid score time k=7.

## Independent radial-bin formula audit

`FORMULA_AUDIT.json` independently solves the mean adjacent-MI budget equation with stable scaled Bessel functions and checks the two-packet expectation using the exponential-radial integral. It obtains `t=-1.0149016255359247`, `2t=-2.0298032510718493`, conditional mutual information 0.061410 and 0.215770 nats, and exactly 0.1 nats average per step. The actual hidden-label random-bin mixture mutual information is 0.090491 nats, below the radial-conditioned design value.

The strong-pair event probability is 0.25, but its radial-product weight is `E[S0*S1*1_strong]=((1+ln2)/2)^2=0.716687`. The analytic one-step packet-difference energy is 1.5 for iid, 1.656114 for fixed-negative and 1.696677 for radial-bin negative, a 2.4493% increase over fixed-negative.

A direct 250,000-pair check with seed 345792 produced 1.50095 (SE 0.00668), 1.65758 (SE 0.00700) and 1.69851 (SE 0.00709), respectively. The radial-bin mean increases relative to fixed-negative by assigning stronger coupling to high-radial pairs. This elementary mean proxy does not guarantee higher detector power, improved lower-tail whitened energy or lower closed-loop MSE. Registered Stage-1 remains the test of useful early information.

The original task-sheet two-bin formula is retained with its correct distinction: radial-conditioned average MI is 0.1 nats, whereas the hidden-label random-bin mixture has 0.09049 nats. Random binning is a diagnostic control, not the strict equal-MI primary comparison.

## Statistical boundary and next execution

The frozen V3 held-out healthy path FWER is 20/512=3.91% unstable and 34/512=6.64% stable, with the recorded exact intervals in the replay audit. New plugin and matched scores are independently calibrated. Development uses 512 path maxima per model/family and a separate 512-path healthy holdout; confirmation, if reached, uses separate 4096-path calibration and holdout sets per model. The calibration rule is marginal over calibration and a fresh path, not a conditional guarantee given one realized threshold.

The primary detection variable is first-three-attacked-report detection per true affected link. Each path's six link indicators are averaged before the path-level paired contrasts. This keeps the six dependent links and full temporal history together in path resampling. A/B/C/D and true-key versus independent same-law-key contrasts are preregistered in `REGISTRATION.json`. Confirmation parameters and score code are committed before development paths. The development report cannot be reused to tune thresholds or algorithms for confirmation.
