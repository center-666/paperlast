# Seven-weight replay reproduction

The analysis is a posthoc deterministic replay of the B2 paths archived in commit `1f4871bb50db2e02897a7fc00760248e4cb5682a`. It requires the B1 mode tables, B1 health archives, final B2 archives, and their original input dependencies. The replay reconstructs the same random streams in batches of 64 and checks all B2 alarm masks against the saved arrays before using them.

Use a fresh working copy with no existing `b2_weight_response_results`, `b2_weight_response_results_tmp`, `b2_shadow_decomposition_results`, or `b2_shadow_decomposition_results_tmp` directories. From this directory run:

```text
python b2_weight_response.py
python verify_b2_weight_response.py
python b2_shadow_decomposition.py
python verify_b2_shadow_decomposition.py
python b2_weight_response_analysis.py
```

The response and shadow runners refuse to overwrite an existing output directory. The response verifier independently updates per-neighbor innovations for four regenerated paths per condition and compares all seven weights. The shadow verifier independently updates clean and attacked target states for four regenerated paths per condition. All paths are checked against the B2 strategy weights and against the pathwise shadow identity in the respective main runners. Input and output hashes are recorded in both manifests and `analysis_sha256.json`.

The B1 health curves reuse the holdout data used during B1 strategy selection. The B2 response and shadow curves reuse the existing attack paths and attack-generated alarm sequences. These analyses are exploratory and do not constitute a new independent validation sample. A q-grid minimum is a descriptive result, and the `q=0` unstable U point fails the registered `P=I` single-mode screen.
