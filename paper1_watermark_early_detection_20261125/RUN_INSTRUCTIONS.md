# Reproduce the six-link study

Use the isolated watermark research branch. From this directory, set OPENBLAS_NUM_THREADS=1 and PYTHONUTF8=1. Requirements are Python 3.11, NumPy, SciPy and a C compiler (MSVC Build Tools 2022 on Windows, cc on POSIX). The projection kernel is built into the machine temporary directory. Exact package versions, source hashes, baseline fixture hashes and the Git revision are recorded in RUN.json.

```text
python -B audit_legacy.py
python -B 01_formula_audit.py
python -B audit_interfaces.py
python -B smoke_stage1.py
python -B verify_receiver.py
python -B study.py 1
python -B analyze_stage1.py 1
python -B study.py 3
python -B analyze_stage1.py 3
python -B analyze_stage1.py lock
python -B study.py 2
python -B analyze_stage1.py 2
```

Run study.py 4 and analyze_stage1.py 4 only when CONFIRM_LOCK.json names a stage-two receiver. To reproduce from scratch, use a fresh checkout at the study code commit; completed archives are hash-checked on rerun. study.py is the formal runner. run_stage1.py provides the Phase-0-compatible smoke interfaces and legacy helpers.

Within each split, the file names identify the dynamics, calibration/healthy/evaluation population and scene. Scene index equals twice the angle index (0°, −45°, +45°) plus the onset index (50,65). Within an archive, path_id indexes the shared physical path and all receiver arms. The random arrays are reproducible from the namespace, stage and chunk. Delays are per physical path and directed edge, zero-based from the first affected report, censored at 60. The NPZ files retain all path-level endpoints, including early maximum scores for threshold sensitivity, paired MSE, and radial geometry at 2/3/5 ticks. The early scores are per-link maxima for exactly the first three affected reports.

FWER is the probability of any healthy alarm across six edges and times 5–149. Score and replay seconds are sums of worker wall times. Shared healthy plugin calibration is charged once. Replay time includes the unchanged common-G wrapper's internal plugin pass.
