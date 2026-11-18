"""Joint-Gaussian prediction from trusted packets alone."""
from dataclasses import dataclass

import numpy as np

from v14_inputs import V13  # noqa: F401


@dataclass(frozen=True)
class TrustedPrediction:
    mean_y: np.ndarray
    cov_y: np.ndarray
    mean_x: np.ndarray
    cov_x: np.ndarray


class TrustedConditioning:
    def __init__(self, prepared):
        self.m = prepared.m
        self.prefixes = prepared.prefixes
        self.cov_y = []
        self.cov_x = []
        self.lower_y = []
        for k, prefix in enumerate(self.prefixes):
            latent_dim = prepared.state_maps.shape[-1]
            trusted = prepared.trusted_maps[:k+1].reshape(-1, latent_dim)
            residual_y = prepared.target_maps[k]-prefix.target_from_trusted[-self.m:] @ trusted
            residual_x = prepared.state_maps[k]-prefix.state_from_trusted @ trusted
            y_cov, x_cov = residual_y @ residual_y.T, residual_x @ residual_x.T
            # The current row block of the joint conditional factor includes its cross-time terms.
            last = prefix.target_cholesky[-self.m:]
            np.testing.assert_allclose(y_cov, last @ last.T, atol=1e-12, rtol=1e-11)
            self.cov_y.append(y_cov)
            self.cov_x.append(x_cov)
            self.lower_y.append(np.linalg.cholesky(y_cov))
        self.cov_y = np.stack(self.cov_y)
        self.cov_x = np.stack(self.cov_x)
        self.lower_y = np.stack(self.lower_y)

    def at(self, trusted, k):
        prefix = self.prefixes[k]
        u = trusted[:, :k+1].reshape(len(trusted), -1)
        return TrustedPrediction(u @ prefix.target_from_trusted[-self.m:].T, self.cov_y[k],
                                 u @ prefix.state_from_trusted.T, self.cov_x[k])

    def score(self, deviation, k):
        compressed = np.linalg.solve(self.lower_y[k], deviation.T).T
        return np.sum(compressed*compressed, axis=-1)
