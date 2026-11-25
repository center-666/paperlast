# Frozen-baseline watermark early-detection study

## Worktree and scope

This study uses an isolated worktree at commit `8953e1dd9a5e2535ead2698acb6f345a85d4f532` on branch `watermark-early-detection-20261125`. It uses the frozen two-dimensional, 50-node common-G estimator and the original six monitored directed links. It does not modify the user's `main`, D1, or D2 worktrees.

The archival V3 paths are used only for deterministic replay. New effect estimates use the registered namespaces `[20261125, 8317, 1]` for development and `[20261125, 8317, 2]` for confirmation. Smoke namespace is `[20261125, 8317, 99]`.

## Deterministic audits

Set up Python with NumPy and SciPy, then run from this directory:

```powershell
python -B audit_legacy.py
python -B 01_formula_audit.py
python -B audit_interfaces.py
python -B smoke_stage1.py
```

The old V3 path replay must return exact equality. The formula check re-solves the average-MI constraint and the radial-bin two-packet formula. Interface smoke checks rank-14 covariance support, sender-shared M keys, template spanning of the three registered attack directions at both onset registrations, causal score invariance to future inputs, and agreement between projection and least-squares GLRT scores. The full pipeline smoke uses disposable paths and does not populate a registered Stage-1 split.

Commit the registration, scripts and passing deterministic outputs before starting Stage-1 paths. Formal runs reject a dirty worktree and bind outputs to the run commit and input hashes.

## Development run

```powershell
python -B run_stage1.py --split development
python -B analyze_stage1.py --split development
```

This generates 12 model-angle-onset scenarios, 256 physical paths per attack scenario, and 512 independent calibration and 512 healthy holdout paths per dynamics. Each of A-F uses the same path within a scenario. The plugin score and the two matched-key score laws have separate path-maximum thresholds. Each wrong-key arm has an independently sampled same-law template schedule and its own healthy calibration threshold.

The primary path endpoint averages the six truly affected-link detection indicators over the first three attacked reports. The analysis resamples complete paths within each fixed scenario and weights all 12 scenarios equally. It reports B-A, D-C, their interaction, and the true-key versus same-law wrong-key controls. Development estimates are not confirmatory.

## Independent confirmation

After the development report and a frozen confirmation plan are committed, run:

```powershell
python -B run_stage1.py --split confirmation
python -B analyze_stage1.py --split confirmation
```

The confirmation split uses 512 attack paths per scenario and 4096 independent calibration plus 4096 independent healthy holdout paths per model. Its seed namespace is disjoint from development. The execution identity requires the same committed detector and watermark implementations as the development lock. If the path-paired variance implies that the registered confirmation size cannot attain its stated precision reference, report the expected interval width before running and keep the registered size unless a separate amendment is committed.

## Result status

The matched GLRT is a classical known-model comparator. The study does not claim unknown-support detection or cryptographic secrecy. The radial-bin dependence allocation is a fixed candidate. Any development outcome, including a lower tail or MSE regression, is preserved and reported. Confirmation, if executed, is an independent Stage-1 split; no method is chosen or tuned from it.
