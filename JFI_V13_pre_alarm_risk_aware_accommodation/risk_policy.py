"""Conditional displacement proxy and a causal pre-alarm scalar decision."""
from dataclasses import dataclass

import numpy as np

from batched_observer import BatchedObserver
from replay import Observations
from analytical_weight import suffix_gramians

GRID = np.array([1., .75, .5, .25, 0.])


def circle_grid(count):
    t = 2*np.pi*(np.arange(count)+.5)/count
    return np.stack((np.cos(t), np.sin(t)))


def radial_tail(mean, direction, danger_squared):
    """Average exact chi-squared radial probabilities over fixed angles."""
    a = np.sum(direction*direction, axis=-2)
    b = np.einsum("...i,...ij->...j", mean, direction)
    c = np.sum(mean*mean, axis=-1)[..., None]-danger_squared
    discr = b*b-a*c
    root = np.sqrt(np.maximum(0., discr))
    denom = np.where(a>1e-24, a, 1.)
    lo, hi = (-b-root)/denom, (-b+root)/denom
    stable_hi = np.divide(c, -b-root, out=hi.copy(), where=(b>0)&(root>0))
    stable_lo = np.divide(c, -b+root, out=lo.copy(), where=(b<0)&(root>0))
    lo = np.clip(stable_lo, 0, 45)
    hi = np.clip(stable_hi, 0, 45)
    inside = np.where((discr>=0)&(hi>lo), np.exp(-.5*lo**2)-np.exp(-.5*hi**2), 0.)
    inside = np.where(a>1e-24, inside, c<0.)
    return np.clip(1-inside.mean(axis=-1), 0, 1)


class RiskDecision:
    def __init__(self, prepared, methods, paths, lambdas, danger_squared, angles):
        p, s = prepared, prepared.settings
        if p.n != 2:
            raise ValueError("The angular Gaussian integration requires two latent coordinates.")
        self.f = p.model.F(s.receiver)
        self.g = p.model.G[s.receiver]
        self.nominal = p.nominal_mse()
        self.threshold = danger_squared
        self.angular = circle_grid(angles)
        self.lambdas = lambdas
        self.names = [(i, m) for i, m in enumerate(methods) if m in lambdas or m == "V1a"]
        shape = (paths, s.horizon, p.n, p.n)
        self.uncompensated = np.zeros(shape)
        self.corrected = {m: np.zeros(shape) for _, m in self.names}

    def choose(self, frame, k, working_error, d, available, prealarm, suffix):
        n = k+1
        active = slice(0, n)
        raw = self.f @ self.uncompensated[:, active] + self.g @ frame.transfer
        self.uncompensated[:, active] = raw
        gain = self.g @ frame.shift_sensitivity
        score = np.full_like(available, np.nan)
        keep = np.flatnonzero(prealarm)
        if keep.size:
            posterior_mean = frame.posterior_coef[keep, ..., None]
            rays = np.linalg.cholesky(frame.inverse_precision[keep]) @ self.angular
            prior_weights = frame.posterior_weights[keep]
            direction = d[keep]
            update_mean = (gain[keep] @ posterior_mean)[..., 0]
            update_rays = gain[keep] @ rays
        for j, name in self.names:
            history = self.f @ self.corrected[name][:, active]
            if keep.size:
                b = raw[keep]-history[keep]
                mean, offsets = (b @ posterior_mean)[..., 0], b @ rays
                grid = GRID if name in self.lambdas else GRID[:1]
                candidates = []
                for choice in grid:
                    per_model = radial_tail(mean-choice*update_mean, offsets-choice*update_rays,
                                            self.threshold)
                    candidates.append(np.sum(prior_weights*per_model, axis=1))
                forecasts = np.stack(candidates, axis=1)
                if name in self.lambdas:
                    proposed_error = working_error[keep, j, None, :]-grid[None, :, None]*direction[:, None]
                    risk_j = np.sum((proposed_error @ suffix)*proposed_error, axis=-1)/self.nominal[k]
                    chosen = np.argmin(forecasts+self.lambdas[name]*risk_j, axis=1)
                    available[keep, j] = grid[chosen]
                    score[keep, j] = forecasts[np.arange(len(chosen)), chosen]
                else:
                    score[keep, j] = forecasts[:, 0]
            self.corrected[name][:, active] = history+available[:, j, None, None, None]*gain
        return available, score


def make_risk(lambdas, danger_squared, angles):
    def factory(prepared, methods, paths):
        return RiskDecision(prepared, methods, paths, lambdas, danger_squared, angles)
    return factory


@dataclass(frozen=True)
class Update:
    estimate: np.ndarray
    weight: float
    risk_score: float | None
    alarm: bool


class OnlineRiskReceiver:
    """One receiver run in packet order. No truth, event metadata or clean input."""

    def __init__(self, prepared, common_step, penalty, danger_squared=.1, angles=16):
        self.prepared = prepared
        self.common_step = common_step
        self.observer = BatchedObserver(prepared)
        self.risk = RiskDecision(prepared, ("V1.3",), 1, {"V1.3": penalty}, danger_squared, angles)
        s, m = prepared.settings, prepared.m
        self.observations = Observations(np.zeros((1, s.horizon, 2*m)),
                                         np.zeros((1, s.horizon, m)),
                                         np.zeros((1, s.horizon, m, m)))
        self.posterior = np.zeros(prepared.n)
        self.suffix = suffix_gramians(prepared.model.F(s.receiver), s.horizon)
        self.k = 0
        self.alarmed = False

    def step(self, local, healthy, target, decoder):
        p = self.prepared
        model, s, k = p.model, p.settings, self.k
        obs = self.observations
        if k >= s.horizon:
            raise ValueError("The configured episode ended.")
        obs.trusted[0, k] = np.r_[local, healthy]
        obs.target[0, k] = target
        obs.decoder[0, k] = decoder
        frame = self.observer.at(obs, k, risk=True)
        alarm = bool(frame.log_bf[0] >= -np.log(s.event_false_alarm))
        self.alarmed |= alarm
        h = (model.H[s.healthy], model.H[s.target])
        gain = model.G[s.receiver]
        args = (model.A, model.H[s.receiver], h, model.K[s.receiver], gain)
        _, preview = self.common_step(self.posterior, local, (healthy, target), *args)
        decision, score = self.risk.choose(frame, k, (preview-frame.mean_working)[:, None],
                                           frame.delta @ gain.T, np.ones((1, 1)),
                                           np.array([not self.alarmed]), self.suffix[k])
        weight = float(decision[0, 0])
        _, self.posterior = self.common_step(self.posterior, local,
                                              (healthy, target-weight*frame.delta[0]), *args)
        self.k += 1
        return Update(self.posterior.copy(), weight,
                      float(score[0, 0]) if not self.alarmed else None, alarm)
