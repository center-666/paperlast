"""Batched two-column regression, equivalent to the frozen V1 observation model."""
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from inputs import Streams  # noqa: F401
from nominal_audit import regression_cache


@dataclass(frozen=True)
class Frame:
    delta: np.ndarray
    mean_null: np.ndarray
    mean_working: np.ndarray
    probability_active: np.ndarray
    log_bf: np.ndarray
    posterior_weights: np.ndarray
    posterior_coef: np.ndarray
    inverse_precision: np.ndarray | None
    transfer: np.ndarray
    shift_sensitivity: np.ndarray | None


class BatchedObserver:
    def __init__(self, prepared):
        self.prepared = prepared
        self.whiteners = regression_cache(prepared)
        self.design = prepared.template @ prepared.parameter_prior_cholesky

    def at(self, streams, k, risk=False):
        p = self.prepared
        prefix = p.prefixes[k]
        s = p.settings
        n = k+1
        size = len(streams)
        dim = n*p.m
        trusted = streams.trusted[:, :n].reshape(size, -1)
        target = streams.target[:, :n].reshape(size, -1)
        whitener = self.whiteners[k]
        white = (target-trusted @ prefix.target_from_trusted.T) @ whitener.T
        mean0 = trusted @ prefix.state_from_trusted.T + white @ prefix.state_from_white.T
        design = np.zeros((size, n, n, p.m, p.n))
        for onset in range(n):
            end = min(n, onset+s.template_duration)
            design[:, onset, onset:end] = streams.decoder[:, onset:end] @ self.design[:end-onset]
        last = design[:, :, k].copy()
        x = whitener @ design.reshape(size, n, dim, p.n)
        xt = x.swapaxes(-2, -1)
        precision = xt @ x + np.eye(p.n)
        info = (xt @ white[:, None, :, None])[..., 0]
        inv = np.linalg.inv(precision) if risk else None
        coef = (inv @ info[..., None] if risk else np.linalg.solve(precision, info[..., None]))
        lbfs = .5*(np.sum(info*coef[..., 0], axis=-1) - np.linalg.slogdet(precision)[1])
        future = s.horizon-n
        partial = logsumexp(lbfs, axis=1)
        event_bf = np.logaddexp(partial, np.log(max(1, future)) if future else -np.inf)-np.log(s.horizon)
        blank = s.horizon*s.no_event_prior/(1-s.no_event_prior)+future
        weights = np.exp(lbfs-np.logaddexp(np.log(blank), partial)[:, None])
        means = (last @ coef)[..., 0]
        delta = np.sum(weights[..., None]*means, axis=1)
        white_fitted = (x @ coef)[..., 0]
        projected = np.sum(weights[..., None]*white_fitted, axis=1)
        mean_working = mean0-projected @ prefix.state_from_white.T
        active = weights[:, max(0,n-s.template_duration):].sum(axis=1)
        sensitivity = None
        if risk:
            matrix = last @ inv @ xt
            weight_derivative = (means-delta[:, None, :])[..., None]*white_fitted[..., None, :]
            jacobian = np.sum(weights[..., None, None]*(matrix+weight_derivative), axis=1)
            sensitivity = jacobian[:, None] @ x
        return Frame(delta, mean0, mean_working, active, event_bf, weights, coef[..., 0],
                     inv, last, sensitivity)
