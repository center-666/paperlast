# D2-B first-run scalar-fit correction

## First-pass status

The first D2-B run used execution commit `2144a46213d6bf886e8296520dd16c2000b4deb7`, protocol commit `af279ba71b0edbdc10131f9318338ebd2599a119`, and the registered D2-B path seed namespace `[20261124, 4401, 1]`.

During post-run code review, the RH-scalar fit was found to compare an absolute dynamic health cost `H_k(q)` with the fixed health excess curve `H_*(q)-H_*(1)`. The scalar control is intended to fit comparable candidate-relative health penalties. The correct dynamic input is `H_k(q)-H_k(1)`, matching the fixed curve's q=1 reference. RH-dynamic's candidate ranking is invariant to subtracting the q-independent `H_k(1)`, but RH-scalar's fitted coefficient and action ranking are not.

The first-pass archives and reports are therefore retained as an audit record and must not be used for the RH-dynamic versus RH-scalar mechanism comparison. They are preserved outside the Git worktree at `C:\Users\zhongxin\shixidoc\d2b_correction_archive\first_pass_2144a46`.

## Corrected replay

The corrected runner uses `H_k(q)-H_k(1)` for RH-dynamic's health penalty and for the nonnegative least-squares RH-scalar fit. It also archives both the relative curve and the absolute curve. The q=1 relative penalty is checked in the smoke test.

The corrected replay regenerates the same registered D2-B physical and attack streams from the same seed namespace. It is a code-corrected replay of the original development sample, not an additional independent replication. The result directory is `b_results_corrected`. The frozen D2-B protocol and all model configurations, H1 rules, seeds, path counts, and attack inputs are unchanged. D2-B remains a development screen, and D2-C remains out of scope.
