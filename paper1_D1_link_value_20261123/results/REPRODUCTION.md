# D1-A deterministic reproduction

Run `python link_value_design.py` followed by `python verify_link_value.py`. The programs create deterministic matrix, topology, gain and covariance artifacts only. They do not generate Monte Carlo attack paths. The output directory is never overwritten.

`sha256.txt` records all pinned inputs and generated artifacts except itself, whose digest cannot be included in its own manifest. The independent verifier checks the listed inputs and result artifacts.
