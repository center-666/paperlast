# D2-B first-run scalar-fit correction

## First-pass status

The first D2-B run used execution commit `2144a46213d6bf886e8296520dd16c2000b4deb7`, protocol commit `af279ba71b0edbdc10131f9318338ebd2599a119`, and the registered D2-B path seed namespace `[20261124, 4401, 1]`.

During post-run code review, the RH-scalar fit was found to compare an absolute dynamic health cost `H_k(q)` with the fixed health excess curve `H_*(q)-H_*(1)`. The scalar control is intended to fit comparable candidate-relative health penalties. The correct dynamic input is `H_k(q)-H_k(1)`, matching the fixed curve's q=1 reference. RH-dynamic's candidate ranking is invariant to subtracting the q-independent `H_k(1)`, but RH-scalar's fitted coefficient and action ranking are not.

The first-pass archives and reports are therefore retained as an audit record and must not be used for the RH-dynamic versus RH-scalar mechanism comparison. They are preserved outside the Git worktree at `C:\Users\zhongxin\shixidoc\d2b_correction_archive\first_pass_2144a46`.

The archived first-pass `B_RUN.json` SHA-256 is `5150db2bf767c3b9d0f6b9b5bc0483a5a2475aca2c967306c29cb61ab5688876`. Its `B_ANALYSIS.json` SHA-256 is `bdeae608e02323b1638b60d7b73e9858bfec20f419ffe7c526978a8b6b61c9c6`. The archive manifest includes hashes for all 15 first-pass files.

## Corrected replay

The corrected runner uses `H_k(q)-H_k(1)` for RH-dynamic's health penalty and for the nonnegative least-squares RH-scalar fit. It also archives both the relative curve and the absolute curve. The q=1 relative penalty is checked in the smoke test.

The corrected replay regenerates the same registered D2-B physical and attack streams from the same seed namespace. It is a code-corrected replay of the original development sample, not an additional independent replication. The result directory is `b_results_corrected`. The frozen D2-B protocol and all model configurations, H1 rules, seeds, path counts, and attack inputs are unchanged. D2-B remains a development screen, and D2-C remains out of scope.
