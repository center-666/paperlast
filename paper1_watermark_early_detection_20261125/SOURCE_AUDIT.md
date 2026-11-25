# Frozen 50-node common-G source audit

## Baseline and evidence status

This isolated worktree is detached from `8953e1dd9a5e2535ead2698acb6f345a85d4f532`. The active experimental branch was created from that baseline. Phase-0 replay uses only read-only V3 archives, including the exact recorded per-scenario physical and watermark seed words. The replay generated no new physical paths and reproduced every checked archived metric exactly, with maximum absolute difference `0.0` in both stable and unstable `rot=0,onset=50` cases for iid and fixed-negative watermarks.

The V3 summary is historical exploratory evidence. Its development selector and final path set are not used to fit or evaluate any new effect. The old V3 FWER point estimates, 3.91% unstable and 6.64% stable, are retained as the original report and are not silently recast as satisfying a strict 5% realized holdout target.

## Function-to-interface map

| Existing function | Actual input | Reused behavior | New experiment boundary |
|---|---|---|---|
| `run_crn_development.load_backend` | Frozen source/configuration and gain table | Loads source parity model and actual 50-node common-G gains through pinned fixtures | Used as the only model loader. Six monitored directed edges remain in source order. |
| `run_common_g_50.physical_paths` | Model, path count, physical seed | Generates the same state and all 50 sensor trajectories used by V3 | One physical path is shared by A–F within a fixed model-angle-onset scenario. |
| `watson_crn.make_common_draws` and `beta_from_common` | Path count, 150 times, 50 sender keys, radial/initial/transition seeds, theta | Exact signed-Watson radial and direction transitions; common radial, initial direction, inverse-CDF quantile and lift-bit streams | New runs generate a complete `[path,150,sender,2]` beta stream. Every sender key is shared by all outgoing monitored links from that sender. |
| `watermark_attack_from_beta` algebra | Raw six-edge attack template and beta sequence | Decoded packet perturbation is `M_k a_k`, where `M_k=.5 I+beta beta^T` | New schedule forms M for all 150 times. Only the registered 60-tick attack support is injected. |
| `replay.scores` | Physical state reference, decoded sender packets, decoded additive attack perturbation | Frozen five-report parity residual, four-residual history buffer, sample covariance, `1e-6 I` regularization and plugin Gaussian KL | Plugin A/B use the source score and exclude k<7, giving the same eight-report input window used by C–F. The score never reads M. |
| `run_common_g_50.run_common_g` | Frozen model, physical paths, decoded delta, raw sender packets, threshold or explicit gate mask | True 50-node steady mode-lookup common-G estimator and six-edge gate application | Its explicit `gate_override` accepts the independently calculated matched-score alarms without changing the estimator. MSE always comes from this frozen estimator. |
| `calibration_rule.path_fwer_cutoff` | One maximum score per independent healthy path | Strict `score > threshold`, exact one-based `ceil((n+1)*.95)` order-statistic threshold | Applied over all six links and all source decision times k=7…149. Matched iid and negative-key score laws are calibrated separately. The same-law wrong-key arm shares the corresponding matched-score threshold. |
| `run_crn_final.measure` | Detector world, physical paths, receiver IDs, onset and threshold | V3 MSE, attack/postattack windows, path-any-alarm, eligible statistic and censored delay | Legacy metrics are regression-checked. The new primary endpoint is computed directly over each true link in `[onset,onset+3)`, without using the old shifted eligible window. |

## Frozen model facts

- The six monitored directed links are `(0,1)`, `(2,3)`, `(15,16)`, `(17,31)`, `(14,49)` and `(17,34)`, in this order.
- Watermark keys are indexed by sender and time, not by directed link. Repeated links from the same sender would share one key schedule.
- Each monitored sender parity matrix is `8×10`. Four ordered parity residuals from eight overlapping two-dimensional reports form a `32×16` raw-report map with effective covariance rank 14.
- For every monitored sender, the deterministic residual covariance has rank 14 under the registered relative eigenvalue cutoff `1e-10`; whitened covariance identity residuals are around machine precision. Independent 8,192-path white-noise checks produced maximum covariance deviations between 0.026 and 0.051, consistent with sampling variation at dimension 14.
- Clean encode/decode maximum scaled error in smoke paths was at most `8.4e-15` stable and `6.6e-14` unstable. This is floating-point roundtrip error, not a physical decoded-noise effect.
- Independent template projection and direct least-squares residual difference agreed within `9e-15` in smoke checks. Replacing future key matrices, future packets and future attack samples left the score at the current decision time unchanged.

## V3 legacy replay

The recorded V3 seeds, thresholds and archived arrays were reconstructed for iid and fixed-negative keys in one original-angle, onset-50 scenario for each dynamics. All path-level source metrics matched exactly. The plugin score was identical with gating enabled and disabled.

On those reused historical paths, the new correctly onset-aligned early-three-report descriptive rates were:

| Model | Watermark | Mean six-link early3 rate | Mean link-time alarm fraction over first three attacked reports | Any monitored link alarm by early3 |
|---|---|---:|---:|---:|
| Unstable | iid | 32.13% | 17.15% | 89.65% |
| Unstable | fixed negative | 33.56% | 17.75% | 90.63% |
| Stable | iid | 66.05% | 41.69% | 100% |
| Stable | fixed negative | 68.13% | 42.78% | 100% |

The link-time column is the average per-link decision probability over three reports. The any-link event saturates in the stable sample, which is why the registered primary endpoint is the mean per-path fraction of the six truly affected links detected within the first three attacked reports. These old observations are regression summaries only.
