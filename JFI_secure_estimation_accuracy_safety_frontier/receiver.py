"""Receive-only comparison with independent common-G recursions for each method."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from frontier import ball_projection
from frozen_inputs import BUDGETS, TrustedConditioning
from batched_observer import BatchedObserver
from dual_receiver import DESIGN, RAYS, DirectionLedger, correction, trusted_center
from replay import Observations


@dataclass(frozen=True)
class Method:
    name: str
    operation: str
    parameter: float = 0.


def label(operation, budget):
    return f"{operation}@{100*budget:g}%"


def method_table(p, strict_radius, anchor_radii, calibrated_thresholds):
    fraction = p.settings.nominal_one_step_allowance
    chi95 = np.sqrt(chi2.ppf(DESIGN["trusted_score_level"],p.m))
    table = [Method("V0","raw"),Method("V1a","structure"),
             Method("V1.2a","structure-ledger",fraction),Method("V1.4","dual",fraction),
             Method("clip95","clip",chi95), Method("trusted-MMSE","anchor",0.),
             Method("anchor-safe","anchor",strict_radius)]
    for budget,radius,threshold in zip(BUDGETS,anchor_radii,calibrated_thresholds,strict=True):
        table.extend((Method(label("anchor",budget),"anchor",radius),
                      Method(label("trusted-ledger",budget),"trusted-ledger",budget),
                      Method(label("innovation-clip",budget),"clip",threshold)))
    return tuple(table)


def clipping_removal(norm, threshold):
    clipped = norm>threshold
    retention = np.divide(threshold,norm,out=np.ones_like(norm),where=clipped)
    return 1.-retention


@dataclass(frozen=True)
class ReplayResult:
    post: np.ndarray
    change: np.ndarray
    null_mean: np.ndarray
    log_bf: np.ndarray
    ball_violation: float
    ledger_min_remainder: float


def run_receiver(p, obs, table):
    """Consume only trusted packets, actual target packets and decoder history."""
    model,s = p.model,p.settings
    count,T = len(obs),s.horizon
    F,G,K = model.F(s.receiver),model.G[s.receiver],model.K[s.receiver]
    if np.linalg.matrix_rank(G)!=p.n:
        raise ValueError("See reachable_ball for the rank-deficient target gain case.")
    right_inverse = np.linalg.pinv(G)
    trusted,observer = TrustedConditioning(p),BatchedObserver(p)
    cutoff = chi2.ppf(DESIGN["trusted_score_level"],p.m)
    ledgers = {}
    for a in table:
        rays = {"structure-ledger":[[1.,0.]],"trusted-ledger":[[0.,1.]],"dual":RAYS}.get(a.operation)
        if rays is not None:
            ledger = DirectionLedger(p,count,rays)
            ledger.allowance = a.parameter*p.nominal_mse()
            ledgers[a.name] = ledger
    output = np.zeros((count,len(table),T,p.n))
    change = np.zeros_like(output)
    now = np.zeros((count,len(table),p.n))
    mean0 = np.empty((count,T,p.n))
    logbf = np.empty((count,T))
    max_ball_error,min_remainder = 0.,0.
    for k in range(T):
        frame = observer.at(obs,k,risk=False)
        pred = trusted.at(obs.trusted,k)
        mean0[:,k] = frame.mean_null
        logbf[:,k] = frame.log_bf
        dv = obs.target[:,k]-pred.mean_y
        norm = np.sqrt(trusted.score(dv,k))
        score = trusted.score(dv-frame.delta,k)
        center,trust = trusted_center(score,frame.mean_working,pred.mean_x,cutoff)
        active_rays = (RAYS[:,1]==0)[None,:] | (trust>0)[:,None]
        directions = np.stack((frame.delta,dv),axis=1) @ G.T
        raw = obs.trusted[:,k,:p.m] @ K.T+(obs.trusted[:,k,p.m:]+obs.target[:,k]) @ G.T
        u = now @ F.T+raw[:,None]
        for j,a in enumerate(table):
            step = np.zeros((count,p.n))
            if a.name in ledgers:
                loss_center = {"structure-ledger":frame.mean_working,
                               "trusted-ledger":pred.mean_x,"dual":center}[a.operation]
                act = ledgers[a.name].decide(k,u[:,j],loss_center,frame.mean_null,directions,
                                             active_rays if a.operation=="dual" else None)
                step = correction(directions,act.weights)
                min_remainder = min(min_remainder,float(act.remainder.min()))
            elif a.operation=="structure":
                step = directions[:,0]
            elif a.operation=="clip":
                step = clipping_removal(norm,a.parameter)[:,None]*directions[:,1]
            elif a.operation=="anchor":
                projected = ball_projection(frame.mean_null,pred.mean_x,a.parameter)
                # Change the target measurement inside the original common-G update.
                packet_shift = (projected-u[:,j]) @ right_inverse.T
                step = -(packet_shift @ G.T)
                np.testing.assert_allclose(u[:,j]-step,projected,atol=2e-12,rtol=2e-12)
                error = np.linalg.norm(u[:,j]-step-pred.mean_x,axis=1)-a.parameter
                max_ball_error = max(max_ball_error,float(error.max()))
            elif a.operation!="raw":
                raise ValueError(f"Unknown method operation {a.operation}")
            now[:,j] = u[:,j]-step
            change[:,j,k] = step
        output[:,:,k] = now
    return ReplayResult(output,change,mean0,logbf,max_ball_error,min_remainder)


def collect(p, stream, table, batch=64):
    batches = [run_receiver(p,Observations(stream.trusted[i:i+batch],stream.target[i:i+batch],
                                         stream.decoder[i:i+batch]),table) for i in range(0,len(stream),batch)]
    return ReplayResult(*(np.concatenate([getattr(b,k) for b in batches]) for k in
                          ("post","change","null_mean","log_bf")),
                        max(b.ball_violation for b in batches),
                        min(b.ledger_min_remainder for b in batches))
