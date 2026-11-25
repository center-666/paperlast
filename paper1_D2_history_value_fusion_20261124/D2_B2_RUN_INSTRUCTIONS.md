# D2-B2 constrained-objective runner

## Scope

D2-B2 is an independent development screen. It reuses no D2-B physical paths. It does not authorize D2-C. The runner reads the frozen D1 worktree at `7e32a41b9802438f6645d9048ee46386caa53418` and does not modify D1 or `main`.

## Environment and smoke checks

Set `D2_D1_DIR` to the absolute D1 directory containing `b1_health.py` and `b1_mode_models.py`. Run from this directory:

```powershell
$env:D2_D1_DIR = 'C:\path\to\paperlast-d1\paper1_D1_link_value_20261123'
python -B d2b2_run.py --smoke
python -B d2b2_run.py --preflight
```

The smoke run uses the separate namespace `[20261125, 7319, 99]`. It checks that q=1 remains feasible, the four-policy recursion matches an independent per-neighbor update, and q=1 matches frozen B1. It is not part of the registered B2 sample.

Commit the protocol, runner, analyzer, and smoke-passing changes before formal sampling. The preflight rejects a dirty worktree and verifies D1/D2 input hashes, B1's posthoc identity, M7 graphs, and disjoint seed prefixes.

## Formal screen

```powershell
python -B d2b2_run.py
python -B audit_b2_budget_margin.py
python -B analyze_b2.py
```

The registered namespace is `[20261125, 7319, 1]`. Health calibration uses stage 0, health evaluation stage 1, and attack inputs stage 2. Every scene has 256 health-calibration and 256 disjoint health-evaluation paths. Each of 24 scene-fixture attack conditions has 256 risk-fit and 256 disjoint evaluation paths. The risk bootstrap uses separate child streams below the registered namespace.

The runner writes `b2_results/scene_XX.npz` atomically after each scene and updates `B2_RUN_PROGRESS.json`. It preserves all evaluated actions, H1 scores and masks, per-policy nominal health-cost curves and feasible sets, path-level losses, reference losses, and calibration maxima. `B2_RUN.json` records the execution identity, environment, input hashes, value budgets, and output hashes.

The analyzer verifies the scene archive hashes and computes path-paired descriptive bootstrap intervals. The output is a development screen, not confirmatory evidence. It is not a formal statistical guarantee for the nominal health constraint.

After analysis, run `python -B audit_b2.py`. This independently reconstructs the M7 history, all three nominal-health feasible sets, and the risk argmin without importing the B2 runner. It checks every alarm action, budget constraint, and archive hash, and compares paired loss summaries against `B2_ANALYSIS.json`. Then run `python -B audit_b2_budget_margin.py`. This checks where the dynamic and scalar health curves change the candidate feasibility sets. Commit `D2_B2_SCREENING_DECISION.md`, both result reports, and the complete machine-readable outputs only after these audits pass.
