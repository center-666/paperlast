"""Receiver-only two-direction accommodation with fixed common-G dynamics."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from v14_inputs import HERE, load_json
from accommodation import OnlineEvidence
from analytical_weight import suffix_gramians
from trusted_prediction import TrustedConditioning


DESIGN = load_json(HERE / "registration.json")
RAYS = np.array(DESIGN["rays_S_T"])
METHODS = ("V0", "V1a", "V1.2a", "trusted-replacement", "GLS",
           "trusted-MMSE", "innovation-clipping", "trusted-budget", "V1.4")


def projection_scale(vector, error):
    length = np.sum(vector*vector, axis=-1)
    return np.divide(np.sum(vector*error, axis=-1), length,
                     out=np.zeros_like(length), where=length>1e-24).clip(0., 1.)


def ledger_cap(curvature, linear, available):
    """Upper feasible length on a unit ray, including the zero correction."""
    result = np.ones_like(curvature)
    constrained = curvature-2*linear > available
    c, h, a = curvature[constrained], linear[constrained], available[constrained]
    discriminant = np.sqrt(h*h+c*a)
    inside = np.empty_like(h)
    positive = h>=0
    inside[positive] = (h[positive]+discriminant[positive])/c[positive]
    inside[~positive] = a[~positive]/(discriminant[~positive]-h[~positive])
    result[constrained] = inside
    return result.clip(0., 1.)


@dataclass(frozen=True)
class BudgetAction:
    weights: np.ndarray
    proposed_scale: np.ndarray
    cap: np.ndarray
    available: np.ndarray
    charge: np.ndarray
    remainder: np.ndarray


class DirectionLedger:
    def __init__(self, prepared, count, rays):
        s = prepared.settings
        self.rays = np.asarray(rays)
        self.suffixes = suffix_gramians(prepared.model.F(s.receiver), s.horizon)
        self.allowance = s.nominal_one_step_allowance*prepared.nominal_mse()
        self.credit = np.zeros(count)
        self.k = 0

    def decide(self, k, preview, case_center, nominal_center, directions, allowed=None):
        if k!=self.k:
            raise ValueError("Expected the next registered measurement.")
        nominal_error = preview-nominal_center
        vectors = self.rays @ directions
        metric = self.suffixes[k]
        transformed = vectors @ metric
        curvature = np.sum(vectors*transformed, axis=-1)
        linear = np.sum(transformed*nominal_error[:,None], axis=-1)
        available = self.credit+self.allowance[k]
        maximum = ledger_cap(curvature, linear, np.broadcast_to(available[:,None], curvature.shape))
        desired = projection_scale(vectors, (preview-case_center)[:,None])
        step = np.minimum(desired, maximum)
        losses = np.sum(((preview-case_center)[:,None]-step[...,None]*vectors)**2, axis=-1)
        if allowed is not None:
            losses = np.where(allowed, losses, np.inf)
        if not np.isfinite(losses).any(axis=1).all():
            raise ValueError("There must be an allowed action for each observation.")
        winner = np.argmin(losses, axis=1)
        rows = np.arange(len(preview))
        scale = step[rows,winner]
        charge = scale*(scale*curvature[rows,winner]-2*linear[rows,winner])
        remainder = available-charge
        if np.any(remainder < -2e-12*np.maximum(1.,available)):
            raise ArithmeticError("The nominal ledger was exceeded.")
        self.credit = np.maximum(0., remainder)
        self.k += 1
        return BudgetAction(scale[:,None]*self.rays[winner], desired[rows,winner],
                            maximum[rows,winner], available, charge, self.credit.copy())


def trusted_center(score, working_state, trusted_state, cutoff):
    trust = np.maximum(0., 1.-cutoff/np.maximum(cutoff,score))
    center = working_state+trust[:,None]*(trusted_state-working_state)
    return center, trust


def correction(directions, weights):
    return np.sum(directions*weights[...,None], axis=-2)


@dataclass(frozen=True)
class ReceiverUpdate:
    posterior: np.ndarray
    w_S_T: np.ndarray
    correction: np.ndarray
    trusted_target: np.ndarray
    trusted_covariance: np.ndarray
    target_deviation: np.ndarray
    corrected_consistency: float
    log_event_bf: float
    budget: BudgetAction


class OnlineDualReceiver:
    """One synchronized receiver. Each call reads only its actual packet triple."""
    def __init__(self, prepared, common_step):
        self.prepared = prepared
        s = prepared.settings
        self.step_common = common_step
        self.trusted = TrustedConditioning(prepared)
        self.observer = OnlineEvidence(prepared,common_step)
        self.ledger = DirectionLedger(prepared,1,RAYS)
        self.post = np.zeros(prepared.n)
        self.cutoff = chi2.ppf(DESIGN["trusted_score_level"],prepared.m)
        self.gain = prepared.model.G[s.receiver]

    def step(self, local, healthy, target, decoder):
        p = self.prepared
        model, s = p.model, p.settings
        ev = self.observer.observe(local,healthy,target,decoder)
        k = ev.timestamp
        forecast = self.trusted.at(self.observer.trusted[None],k)
        deviation = np.asarray(target)-forecast.mean_y[0]
        score = self.trusted.score((deviation-ev.delta)[None],k)
        center, trust = trusted_center(score,ev.mixture_state_mean[None],forecast.mean_x,self.cutoff)
        allowed = (RAYS[:,1]==0)[None] | (trust>0)[:,None]
        _, u = self.step_common(self.post,local,[healthy,target],model.A,model.H[s.receiver],
                                [model.H[s.healthy],model.H[s.target]],model.K[s.receiver],self.gain)
        directions = (np.stack((ev.delta,deviation)) @ self.gain.T)[None]
        action = self.ledger.decide(k,u[None],center,ev.nominal_state_mean[None],directions,allowed)
        change = correction(directions,action.weights)[0]
        self.post = u-change
        return ReceiverUpdate(self.post.copy(),action.weights[0].copy(),change,forecast.mean_y[0],
                              forecast.cov_y.copy(),deviation,float(score[0]),ev.log_event_bf,action)
