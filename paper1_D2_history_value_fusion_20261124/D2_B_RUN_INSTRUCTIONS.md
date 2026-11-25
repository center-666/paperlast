# D2-B runner and analysis instructions

## Evidence scope

D2-B is a development screen under the frozen protocol. The conditional single-step risk table ranks actions and is not a closed-loop MSE bound. B results do not authorize D2-C.

The four-dimensional common-G model, physical configuration, mode gains, and M7 graphs are read from the frozen D1 checkout at `7e32a41b9802438f6645d9048ee46386caa53418`. D1 and `main` are inputs only.

## Environment

Use Python with NumPy and SciPy installed. Set `D2_D1_DIR` to the absolute path of the D1 worktree directory containing `b1_health.py` and `b1_mode_models.py`. Run commands from this directory or invoke the scripts by absolute path.

## Before sampling

Run the deterministic smoke test:

```powershell
$env:D2_D1_DIR = 'C:\path\to\paperlast-d1\paper1_D1_link_value_20261123'
python -B b_run.py --smoke
```

The smoke test checks health-cost recursion, the frozen B1 full-acceptance regression, packet-weighting algebra, independent per-neighbor damage labels and estimator updates, common H1-mask use, and atomic output serialization. It generates only a small disposable smoke batch under a separate seed namespace.

## Formal D2-B development screen

After the runner and analysis source are committed, run:

```powershell
python -B b_run.py
python -B analyze_b.py
```

The runner verifies the D2 protocol commit, D1 commit and archived hashes, the physical configuration, M7 graphs, B1 posthoc status, and disjoint B1/B2/D2-A′ seed namespaces. The registered physical-path namespace is `[20261124, 4401, 1]`. Health calibration uses stage 0, health evaluation uses stage 1, and attack streams use stage 2. Continuous and burst attack conditions each use 256 risk-fit paths and 256 disjoint evaluation paths. Each scene uses 256 health-calibration paths and 256 disjoint health-evaluation paths.

The runner writes and hashes one scene archive at a time. `B_RUN_PROGRESS.json` records completed scenes and hashes so an interrupted run can resume only when all frozen inputs and the run identity still match. Scene archives are written atomically. Existing completed outputs are never overwritten. A full run ends with `B_RUN.json` and twelve `scene_XX.npz` files.

`analyze_b.py` reads only those archived results. It does not generate physical paths. It verifies every scene archive hash, reports equal-scene/equal-fixture means, computes paired whole-path bootstrap intervals for descriptive development comparisons, and writes `B_ANALYSIS.json` and `B_ANALYSIS.md`.

The frozen protocol lists RH-dynamic minus R1 as the primary comparison and RH-dynamic minus RH-static / RH-scalar as secondary comparisons. RH-dynamic versus RH-scalar is the direct diagnostic for whether history changes candidate-relative health costs beyond a scalar penalty. All B comparisons remain developmental.
