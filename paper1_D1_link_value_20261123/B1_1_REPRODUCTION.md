# B1-1 reproduction

From `paper1_D1_link_value_20261123`, run

```text
python b1_mode_models.py
python verify_b1_mode_models.py
```

The first program constructs deterministic mode-dependent common-(G) steady-state tables and finite mode-sequence covariance checks. The second independently rebuilds the matrices. Neither program generates healthy paths, H1 scores, alarm sequences, or attacks.

The output directory is `b1_mode_results`. The matrix archive is large because it contains all 42 tables and all 50 receiver rows. The output directory must be absent before running the main program because the entry point refuses to overwrite results.
