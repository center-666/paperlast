# D2-B2 first-attempt interruption record

The first execution attempt used commit `293d5f495df2ae4666a660e1f011f93e03cab42d` and the registered seed namespace `[20261125, 7319, 1]`. It completed the in-memory computation for scene 0, then raised a `NameError` while assembling the scene result because the fixed-reference covariance cost had not been bound in `run_scene`.

No scene archive or result report was written. The progress record showed `completed_scenes=[]`. The progress file is preserved outside the repository at `C:\Users\zhongxin\shixidoc\d2b2_correction_archive\first_failed_attempt_293d5f4` and has SHA-256 `fb7161d455a8af9a7edd366a173073a919bcfceb5b1d437bc0415041ae11ac67`.

The runner now binds the frozen initial covariance health curve before simulation, passes the same precomputed curves to all policy replays, and exercises the complete scene-packaging path in the deterministic smoke test. The corrected registered run replays the same seed namespace. It is the continuation of the same development draw, not an independent replication and not a parameter-tuning run.
