# B0 deterministic reproduction

From `paper1_D1_link_value_20261123`, run

```text
python audit_b0_definition.py
```

The program reads the archived D1-A JSON, gain matrices and covariance traces. It does not generate healthy paths, H1 scores, alarm sequences or attack Monte Carlo paths. It deterministically checks the fixed-weight budget sets and enumerates all 60-step schedules with exactly three single-packet alarm positions.

The output is `results/B0_definition_audit.json`. `B0_SHA256.txt` covers the B0 definition, audit script, audit report and this reproduction file. The manifest excludes itself.
