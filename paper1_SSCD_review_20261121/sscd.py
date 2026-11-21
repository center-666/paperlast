"""Reviewed executable SSCD prototype, derived from source commit 548a031.

The unchanged original is under source_snapshot/. This module does not load
or invent the original 50-node network. The signed kernel is implemented on
the explicit, validated compact theta domain [-20,20]. Finite-template
screening is NOT a continuum worst-case guarantee; see continuous_design.py
for analytical discretization bounds when effective parameter dimension=2.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import roots_jacobi, logsumexp
from functools import lru_cache

THETA_CAP = 20.0


@lru_cache(maxsize=64)
def _beta_quadrature(m: int, order: int = 256):
    if int(m) != m or m < 2:
        raise ValueError("measurement dimension m must be >= 2")
    x, w = roots_jacobi(order, (m - 3) / 2, -0.5)
    return (x + 1.0) / 2.0, w / np.sum(w)


@lru_cache(maxsize=4096)
def watson_stats(m: int, theta: float, order: int = 256):
    """Log-scaled Beta quadrature on an explicit compact numerical domain.

    Exact-arithmetic identities are in review.md. Order comparisons are
    numerical diagnostics, not rigorous quadrature error certificates.
    """
    if not np.isfinite(theta) or abs(theta) > THETA_CAP + 1e-12:
        raise ValueError(f"numerical theta domain is [-{THETA_CAP},{THETA_CAP}], not all R")
    t, w = _beta_quadrature(m, order)
    log_weights = np.log(w) + theta*t
    log_z = float(logsumexp(log_weights))
    tilted = np.exp(log_weights-log_z)
    c = float(tilted @ t)
    variance = float(tilted @ (t-c)**2)
    zeta = (m * c - 1.0) / (m - 1.0)
    mi = max(0.0, float(theta*c-log_z))
    if theta == 0:
        c,zeta,mi,log_z=1/m,0.0,0.0,0.0
    return {"normalizer": float(np.exp(log_z)), "log_normalizer": log_z,
            "c": c, "variance_t":variance, "zeta": zeta,"mi_nats": mi,
            "density_min":float(np.exp(min(0.0,theta)-log_z))}


def theta_from_zeta(m: int, zeta: float, tol: float = 1e-11):
    lower=watson_stats(m,-THETA_CAP)['zeta']
    upper=watson_stats(m,THETA_CAP)['zeta']
    if not np.isfinite(zeta) or zeta < lower-1e-12 or zeta > upper+1e-12:
        raise ValueError(f"zeta outside validated compact image [{lower},{upper}]; endpoint asymptotics are not implemented")
    if zeta==0:
        return 0.0
    if abs(zeta-lower)<1e-12:return -THETA_CAP
    if abs(zeta-upper)<1e-12:return THETA_CAP
    def residual(theta):
        return watson_stats(m, theta)["zeta"] - zeta
    return float(brentq(residual, -THETA_CAP, THETA_CAP, xtol=tol))


def dependence_interval(m: int, L: int, budget: float):
    """Feasible signed-zeta interval for total correlation <= budget.

    If the information budget reaches the numerical theta cap, the returned
    domain is explicitly marked truncated; no whole-real-line optimum claim.
    """
    if L<2 or budget<0 or not np.isfinite(budget):
        raise ValueError('L>=2 and finite nonnegative dependence budget required')
    if budget==0:return {'zeta_interval':[0.,0.],'theta_interval':[0.,0.],'numerically_truncated':False}
    target=budget/(L-1)
    ends=[];clipped=False
    for sign in (-1.,1.):
        edge=sign*THETA_CAP
        if watson_stats(m,edge)['mi_nats']<=target:
            theta=edge;clipped=True
        else:
            theta=brentq(lambda th:watson_stats(m,th)['mi_nats']-target,
                         min(edge,0.),max(edge,0.),xtol=1e-12)
        ends.append(float(theta))
    return {'zeta_interval':[watson_stats(m,t)['zeta'] for t in ends],
            'theta_interval':ends,'numerically_truncated':clipped}


def sample_signed_watermark_pairs(rng, count: int, length: int, m: int,
                                  theta: float, q: float = 0.5):
    """Sample signed-Watson directions by rejection and independent radii.

    Each direction proposal is uniform on S^(m-1). Acceptance uses the
    envelope exp(max(theta,0)); initialize the first direction uniformly.
    The resulting single-time beta marginal is N(0,q I_m).
    """
    if count<1 or length<1 or m<2 or q<=0:
        raise ValueError('count,length>=1, m>=2 and q>0 required')
    if abs(theta)>THETA_CAP:
        raise ValueError('sampler uses the validated compact theta domain')
    u=rng.normal(size=(count,m));u/=np.linalg.norm(u,axis=1,keepdims=True)
    result=np.empty((count,length,m))
    for k in range(length):
        if k:
            nxt=rng.normal(size=(count,m));nxt/=np.linalg.norm(nxt,axis=1,keepdims=True)
            pending=np.arange(count)
            while pending.size:
                t=np.sum(nxt[pending]*u[pending],axis=1)**2
                probability=np.exp(theta*t-max(theta,0.0))
                accepted=rng.random(pending.size)<probability
                u[pending[accepted]]=nxt[pending[accepted]]
                pending=pending[~accepted]
                if pending.size:
                    proposal=rng.normal(size=(pending.size,m))
                    proposal/=np.linalg.norm(proposal,axis=1,keepdims=True)
                    nxt[pending]=proposal
        radii=np.sqrt(q*rng.chisquare(m,size=count))
        result[:,k]=radii[:,None]*u
    return result


def mutual_information_for_zeta(m: int, zeta: float):
    theta = theta_from_zeta(m, zeta)
    return watson_stats(m, theta)["mi_nats"]


def full_watermark_covariance(d: np.ndarray, zeta: float, q: float):
    """Covariance of stack X_l=(beta_l beta_l^T)d_l, d shape [L,m]."""
    d = np.asarray(d, dtype=float)
    if d.ndim != 2 or d.shape[1]<2 or not np.isfinite(q) or q < 0 or not np.all(np.isfinite(d)):
        raise ValueError("d must be [L,m] and q nonnegative")
    L, m = d.shape
    lower, upper = -1.0 / (m - 1.0), 1.0
    if not lower - 1e-10 <= zeta <= upper + 1e-10:
        raise ValueError("zeta outside attainable range")
    C = np.zeros((L*m, L*m))
    for ell in range(L):
        dl = d[ell]
        Cll = q*q * (np.dot(dl, dl)*np.eye(m) + np.outer(dl, dl))
        sl = slice(ell*m, (ell+1)*m)
        C[sl, sl] = Cll
        for r in range(ell+1, L):
            dr = d[r]
            h = r - ell
            eta = q*q*m/(m+2.0) * zeta**h
            Clr = eta * (
                np.dot(dl, dr)*np.eye(m)
                + np.outer(dr, dl)
                - (2.0/m)*np.outer(dl, dr)
            )
            sr = slice(r*m, (r+1)*m)
            C[sl, sr] = Clr
            C[sr, sl] = Clr.T
    # Numerical PSD is a useful implementation check, not a theorem certificate.
    C = (C + C.T) / 2.0
    if np.linalg.eigvalsh(C)[0] < -1e-8 * max(1.0, np.linalg.norm(C, 2)):
        raise ArithmeticError("constructed watermark covariance is not PSD")
    return C


def parity_covariance(d: np.ndarray, Z: np.ndarray, zeta: float, q: float):
    d = np.asarray(d, float)
    Z = np.asarray(Z, float)
    L, m = d.shape
    if Z.shape[1] != L*m:
        raise ValueError("parity matrix width must equal L*m")
    return Z @ full_watermark_covariance(d, zeta, q) @ Z.T


def gaussian_covariance_kl_proxy(theta0: np.ndarray, C: np.ndarray):
    """Covariance Gaussian KL proxy; not the actual non-Gaussian KL/power."""
    theta0 = np.asarray(theta0, float)
    C = np.asarray(C, float)
    if not np.allclose(theta0,theta0.T) or not np.allclose(C,C.T):
        raise ValueError('symmetric covariance matrices required')
    chol=np.linalg.cholesky(theta0)
    whitened=np.linalg.solve(chol,np.linalg.solve(chol,C).T).T
    ev=np.linalg.eigvalsh((whitened+whitened.T)/2)
    if ev[0]<-1e-10:raise ValueError('attacked covariance increment must be PSD')
    ev=np.maximum(ev,0)
    return float(0.5*np.sum(ev-np.log1p(ev)))


def common_g_risk_matrix(T_blocks: Sequence[np.ndarray],
                         K_blocks: Sequence[np.ndarray],
                         W: np.ndarray, zeta: float, q: float,
                         lam: float):
    """Exact Q_Gamma for delta_l=T_l xi and output sum K_l M_l delta_l.

    Returns Q such that E[output.T W output] = xi.T Q xi.
    T_l: [m,p], K_l: [dout,m], W: [dout,dout].
    """
    if len(T_blocks) != len(K_blocks) or not T_blocks:
        raise ValueError("T_blocks and K_blocks must have equal nonzero length")
    p = np.asarray(T_blocks[0]).shape[1]
    dout = np.asarray(K_blocks[0]).shape[0]
    m = np.asarray(T_blocks[0]).shape[0]
    L = len(T_blocks)
    if any(np.asarray(T).shape != (m, p) for T in T_blocks):
        raise ValueError("all T blocks must have shape [m,p]")
    if any(np.asarray(K).shape != (dout, m) for K in K_blocks):
        raise ValueError("all K blocks must have shape [dout,m]")
    W = np.asarray(W, float)
    if W.shape != (dout, dout):
        raise ValueError("W dimension mismatch")
    Kmap = np.concatenate([np.asarray(K, float) for K in K_blocks], axis=1)
    mu = lam + q

    def risk(xi):
        d = np.stack([np.asarray(T) @ xi for T in T_blocks])
        mean = mu * sum((np.asarray(K) @ d[i] for i, K in enumerate(K_blocks)),
                        np.zeros(dout))
        Cx = full_watermark_covariance(d, zeta, q)
        cov_out = Kmap @ Cx @ Kmap.T
        return float(mean @ W @ mean + np.trace(W @ cov_out))

    Q = np.zeros((p, p))
    eye = np.eye(p)
    for i in range(p):
        Q[i, i] = risk(eye[i])
        for j in range(i):
            val = 0.5 * (risk(eye[i] + eye[j]) - Q[i, i] - Q[j, j])
            Q[i, j] = Q[j, i] = val
    return (Q + Q.T) / 2.0


@dataclass
class Candidate:
    zeta: float
    theta: float
    mutual_information: float
    worst_detection_proxy: float
    worst_common_g_risk: float
    feasible: bool


def _worst_energy_risk(Q: np.ndarray, E: np.ndarray, energy: float):
    E = np.asarray(E, float)
    Q = (np.asarray(Q, float) + np.asarray(Q, float).T) / 2.0
    if Q.shape!=E.shape or E.ndim!=2 or E.shape[0]!=E.shape[1] or not np.isfinite(energy) or energy<0:
        raise ValueError('matching square Q,E and nonnegative energy required')
    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(E)):
        raise ValueError('finite risk and energy matrices required')
    if not np.allclose(E,E.T):raise ValueError('symmetric coded-energy Gram required')
    if np.linalg.eigvalsh(Q)[0] < -1e-9*max(1.,np.linalg.norm(Q,2)):
        raise ValueError('risk Gram Q must be PSD')
    vals, vecs = np.linalg.eigh(E)
    if vals[0] <= 0:
        raise ValueError("coded-energy Gram E must be positive definite")
    Einv = (vecs / np.sqrt(vals)) @ vecs.T
    return float(energy * np.linalg.eigvalsh(Einv @ Q @ Einv)[-1])


def optimize_signed_zeta(
    m: int,
    attack_templates: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    risk_callback: Callable[[float], np.ndarray],
    energy_gram: np.ndarray,
    attack_energy: float,
    dependence_budget: float,
    state_risk_budget: float,
    watermark_variance: float,
    grid_size: int = 257,
):
    """Finite-template signed screen, never a continuous-attack certificate.

    Each attack template is (d[L,m], Z[r,L*m], Theta[r,r]).
    risk_callback(zeta) returns Q_Gamma in attack-parameter coordinates.
    The dependence budget is total correlation over the declared window:
    (L-1) I_m(theta). Zero attack is excluded by the caller's templates.
    """
    if not attack_templates:
        raise ValueError("at least one nonzero, parity-effective attack template is required")
    L = attack_templates[0][0].shape[0]
    if any(np.asarray(d).shape[0] != L for d, _, _ in attack_templates):
        raise ValueError('all attack templates must use the same window length')
    if watermark_variance <= 0:
        raise ValueError('watermark_variance must be positive')
    if not np.isfinite(attack_energy) or attack_energy<=0:
        raise ValueError('strictly positive coded attack energy required')
    if grid_size<3:raise ValueError('grid_size must be >=3')
    domain=dependence_interval(m,L,dependence_budget)
    zlo,zhi=domain['zeta_interval']
    grid = np.unique(np.r_[np.linspace(zlo,zhi,grid_size),0.])

    def evaluate(zeta):
        theta = theta_from_zeta(m, zeta)
        info = (L-1) * watson_stats(m, theta)["mi_nats"]
        Q = risk_callback(zeta)
        risk = _worst_energy_risk(Q, energy_gram, attack_energy)
        detections = []
        for d, Z, Theta in attack_templates:
            if not np.any(np.asarray(d)) or np.linalg.norm(np.asarray(Z)@np.asarray(d).reshape(-1))>1e-7:
                raise ValueError('templates must be nonzero and parity-stealth in the declared window')
            C = parity_covariance(d, Z, zeta, q=watermark_variance)
            detections.append(gaussian_covariance_kl_proxy(Theta, C))
        det = float(min(detections))
        feasible = info <= dependence_budget+1e-11 and risk <= state_risk_budget+1e-12
        return Candidate(zeta, theta, info, det, risk, feasible)

    iid=evaluate(0.)
    if not iid.feasible:
        raise ValueError('IID baseline is infeasible under these budgets; do not silently call it a safe fallback')
    rows = [evaluate(float(z)) for z in grid]
    feasible = [r for r in rows if r.feasible]
    if not feasible:
        return {"status": "no feasible sampled candidate",
                "candidates": rows, "best": None}
    # Refine every grid-local maximum that is feasible at its neighborhood.
    score = np.array([r.worst_detection_proxy if r.feasible else -np.inf for r in rows])
    local = [i for i in range(1, len(grid)-1)
             if score[i] >= score[i-1] and score[i] >= score[i+1]
             and np.isfinite(score[i])]
    refined = []
    for i in local:
        a, b = grid[i-1], grid[i+1]
        def objective(z):
            r = evaluate(float(z))
            return -r.worst_detection_proxy if r.feasible else 1e12
        opt = minimize_scalar(objective, bounds=(a, b), method="bounded",
                              options={"xatol": 1e-9})
        refined.append(evaluate(float(opt.x)))
    pool = feasible + [r for r in refined if r.feasible]
    best = max(pool, key=lambda r: r.worst_detection_proxy)
    # Include iid explicitly; it may be feasible and should never be silently omitted.
    if iid.feasible and iid.worst_detection_proxy >= best.worst_detection_proxy:
        best = iid
    if best.worst_detection_proxy <= iid.worst_detection_proxy+1e-10:
        best=iid
    return {"status": "finite-template screened candidate; not a continuous-attack or global optimum guarantee",
            "best": best.__dict__, "iid": iid.__dict__,
            "grid_feasible_count": len(feasible),
            "grid_size": grid_size,
            "domain":domain,
            "warning": "D_G is a covariance proxy. Validate true detector power and full common-G trajectory risk on independent held-out runs."}
