"""Receiver-side operator moments. No physical trajectory, perturbation or policy is generated."""
import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import qmc

from energy_inputs import (BASE_COMMIT, DEFAULT_BASELINE, HERE, MODELS, MODEL_PATH,
                           digest, load_prepared, setup, validate_spec, write_json)


def coefficients(prepared,k):
    p = prepared
    F,G = p.model.F(p.settings.receiver),p.model.G[p.settings.receiver]
    common = np.stack([np.linalg.matrix_power(F,k-t) @ G for t in range(k+1)])
    prefix = p.prefixes[k]
    rows = solve_triangular(prefix.target_cholesky.T,prefix.state_from_white.T,lower=False).T
    full = rows.reshape(p.n,k+1,p.m).transpose(1,0,2)
    return {"common_G":common,"full_nominal_conditional":full}


def decoded_square_moment(lam,q,m):
    return lam*lam+2*lam*q+(m+2)*q*q


def projected_moment(lam,q,S):
    return (lam*lam+2*lam*q+2*q*q)*S+q*q*np.trace(S)*np.eye(len(S))


def deterministic_open_loop_bound(B,lam,q):
    """Largest fixed-waveform expected gain, returned as a scalar with no maximizing data."""
    count,_,m = B.shape
    row = B.transpose(1,0,2).reshape(B.shape[1],count*m)
    matrix = (lam+q)**2 * (row.T @ row)
    for t,block in enumerate(B):
        T = slice(t*m,(t+1)*m)
        S = block.T @ block
        matrix[T,T] = projected_moment(lam,q,S)
    return float(np.linalg.eigvalsh(matrix)[-1])


def spectral_interval(B,lam,q):
    """Analytic bounds for E spectral norm squared in a two-dimensional output."""
    count,dim,m = B.shape
    if dim!=2:
        raise ValueError("The trace-difference certificate uses two state coordinates.")
    mean = decoded_square_moment(lam,q,m)*np.einsum("tij,tkj->ik",B,B)
    rad = q**np.arange(1,5)*np.cumprod(m+2*np.arange(4))
    eta_mean = 2*lam*rad[0]+rad[1]
    eta2 = 4*lam*lam*rad[1]+4*lam*rad[2]+rad[3]
    trace = np.trace(mean)
    axis = np.array([[[1.,0.],[0.,-1.]],[[0.,1.],[1.,0.]]])
    spreads = []
    for D in axis:
        A = B.transpose(0,2,1) @ D @ B
        tr = np.trace(A,axis1=1,axis2=2)
        sq = np.sum(A*A,axis=(1,2))
        variance = eta2*(tr*tr+2*sq)/(m*(m+2))-(eta_mean*tr/m)**2
        spreads.append(np.sum(variance)+np.sum(mean*D)**2)
    lower = float(np.linalg.eigvalsh(mean)[-1])
    upper = float(min(trace,(trace+np.sqrt(sum(spreads)))/2))
    return lower,upper


def decoder_quadrature(n_steps,q,m,power,seed):
    if m!=2:
        raise ValueError("This receiver uses a two-component Gaussian watermark.")
    sequence = qmc.Sobol(d=2*n_steps,scramble=True,seed=seed).random_base2(power).reshape(-1,n_steps,2)
    radius2 = -2*q*np.log1p(-sequence[:,:,0])
    angle = np.pi*sequence[:,:,1]
    u = np.stack((np.cos(angle),np.sin(angle)),axis=-1)
    return radius2,u


def spectral_quadrature(B,lam,rule):
    t = len(B)
    R,u = rule
    w = np.einsum("tij,btj->bti",B,u[:,:t],optimize=True)
    eta = (2*lam+R[:,:t])*R[:,:t]
    S = lam*lam*np.einsum("tij,tkj->ik",B,B)+np.einsum("bt,bti,btj->bij",eta,w,w,optimize=True)
    eigen = .5*(S[:,0,0]+S[:,1,1]+np.hypot(S[:,0,0]-S[:,1,1],2*S[:,0,1]))
    return float(eigen.mean())


def audit(base):
    common,settings,spec,_ = setup(base)
    validate_spec(settings,spec)
    h = settings.horizon
    lam,q = (spec["watermark"][k] for k in ("lambda","q"))
    rule_args = [(13,617),(14,809)]
    rules = [decoder_quadrature(h,q,2,*arg) for arg in rule_args]
    models = {}
    for name in MODELS:
        p = load_prepared(base,name,common,settings)
        stages = []
        for k in range(h):
            entries = {}
            for policy,B in coefficients(p,k).items():
                lower,upper = spectral_interval(B,lam,q)
                ol = deterministic_open_loop_bound(B,lam,q)
                e = [spectral_quadrature(B,lam,rule) for rule in rules]
                # Compare numeric quadrature with the deterministic moment sandwich.
                assert ol<=upper+1e-11 and lower<=upper+1e-11
                assert lower-.018<=e[-1]<=upper+.018
                entries[policy] = {"expected_operator_sq_analytic_lower":lower,
                                   "expected_operator_sq_analytic_upper":upper,
                                   "expected_operator_sq_watermark_qmc":e,
                                   "best_deterministic_open_loop_expected_sq_gain":ol}
            stages.append({"time":k,**entries})
        models[name] = stages
        for k in (0,h-1):
            print(name,k,{m:row["expected_operator_sq_watermark_qmc"][-1]
                  for m,row in stages[k].items() if isinstance(row,dict)},flush=True)
    return {"baseline":BASE_COMMIT,"model_sha256":digest(base/MODEL_PATH),
            "quadrature_power_and_seed":rule_args,"physical_episodes_generated":0,
            "models":models}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    args = parser.parse_args()
    record = audit(args.baseline)
    (HERE/"results").mkdir(exist_ok=True)
    write_json(HERE/"results/linear_influence.json",record)
