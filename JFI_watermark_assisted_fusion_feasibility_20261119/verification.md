# Replay verification

Baseline `ea62f758c8d179f3aa198c4d706f69442fed765e`.

34 local input SHA-256 digests verified against the two manifests; all normalized bytes match the corresponding blobs in the public baseline tree.
Fresh calls to both scripts reproduced both result JSON files byte for byte. No new paths generated.

```text
python -B check_gaussian_identities.py
Three exact Gaussian comparisons verified. No sampled observations.
```

```text
python -B offline_covariance_screen.py
unstable 0.005 directional pass: False
  V0                             health +0.0000%  early MSE 0.269417  early pre-alarm 5
  V1a                            health -0.0081%  early MSE 0.221636  early pre-alarm 3
  state_saturation               health +0.3411%  early MSE 0.220928  early pre-alarm 3
  innovation_clipping            health +0.4349%  early MSE 0.224082  early pre-alarm 3
  residual_score                 health +0.4621%  early MSE 0.214494  early pre-alarm 2
  posterior_mean_score           health +0.4649%  early MSE 0.224080  early pre-alarm 2
  posterior_second_moment_score  health +0.4333%  early MSE 0.217888  early pre-alarm 0
unstable 0.02 directional pass: False
  V0                             health +0.0000%  early MSE 0.269417  early pre-alarm 5
  V1a                            health -0.0081%  early MSE 0.221636  early pre-alarm 3
  state_saturation               health +1.7021%  early MSE 0.218665  early pre-alarm 3
  innovation_clipping            health +1.7584%  early MSE 0.217726  early pre-alarm 3
  residual_score                 health +1.8711%  early MSE 0.215436  early pre-alarm 0
  posterior_mean_score           health +1.7828%  early MSE 0.222168  early pre-alarm 1
  posterior_second_moment_score  health +1.7404%  early MSE 0.220908  early pre-alarm 0
stable 0.005 directional pass: False
  V0                             health +0.0000%  early MSE 0.568448  early pre-alarm 8
  V1a                            health -0.1191%  early MSE 0.333051  early pre-alarm 8
  state_saturation               health +0.4433%  early MSE 0.324190  early pre-alarm 4
  innovation_clipping            health +0.4442%  early MSE 0.330336  early pre-alarm 6
  residual_score                 health +0.4775%  early MSE 0.302916  early pre-alarm 4
  posterior_mean_score           health +0.7621%  early MSE 0.318056  early pre-alarm 6
  posterior_second_moment_score  health +0.6360%  early MSE 0.303577  early pre-alarm 4
stable 0.02 directional pass: False
  V0                             health +0.0000%  early MSE 0.568448  early pre-alarm 8
  V1a                            health -0.1191%  early MSE 0.333051  early pre-alarm 8
  state_saturation               health +1.9913%  early MSE 0.316265  early pre-alarm 4
  innovation_clipping            health +1.9210%  early MSE 0.315853  early pre-alarm 4
  residual_score                 health +1.9639%  early MSE 0.298730  early pre-alarm 3
  posterior_mean_score           health +2.6487%  early MSE 0.296928  early pre-alarm 6
  posterior_second_moment_score  health +2.2174%  early MSE 0.294274  early pre-alarm 2
```

Result SHA-256:
- `b15311497f59547a3b631b8ffbc3a1c9d8a1b38bc670010faf7844c23d16ca3e`  `exact_gaussian_results.json`
- `fb93e1dab07e5b678ad1084f1ae43c8a7d7bc4140e582bda4cbd009108369f40`  `replay_results.json`
