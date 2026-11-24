# B2 reproduction

Run from this directory after B1 and B1-1 archives are present.

```text
python b2_attack.py --smoke
python b2_attack.py
python verify_b2_attack.py
python verify_b2_metric_correction.py
python verify_b2_input_correction.py
python b2_attack_analysis.py
```

B2 uses 24 conditions, each with 2,048 fresh attack paths, for 49,152 total paths. It reads the frozen B1 thresholds and `policy_lock.json`. B1 paths are excluded. `verification.json` independently reconstructs the first four paths in every condition, including residual scores, masks, and path metrics. `verify_b2_metric_correction.py` audits the earlier burst-denominator correction. `verify_b2_input_correction.py` checks that the final weight acts on the complete attacked packet and that the same H1 masks and paths were used. `analysis.json` and `B2_ATTACK_REPORT.md` contain descriptive paired comparisons. `run_manifest.json` records input and output SHA-256 hashes, thresholds, seed namespace, and software versions.

B1 was executed as a posthoc analysis under the B0 definitions and remains pending manual confirmation. B2 results are descriptive and conditional on those thresholds and the locked policy table. The formula correction and its audit are recorded in `B2_EXECUTION_AMENDMENT.md`. Earlier outputs are preserved under their respective `b2_attack_results_pre_*_correction` folders. They do not authorize D1-C.
