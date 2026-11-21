"""System-aware signed angular-coupled watermark design for Paper 1.

This is a generic numerical core. It does NOT load or invent the original
50-node network. Supply source-verified matrices and fixed attack templates.
Dependencies: numpy, scipy.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import hyp1f1, roots_jacobi


def _beta_quadrature(m: int, order: int = 192):
    if m < 2:
        raise ValueError("measurement dimension m must be >= 2")
    x, w = roots_jacobi(order, (m - 3) / 2, -0.5)
    return (x + 1.0) / 2.0, w / np.sum(w)


def watson_stats(m: int, theta: float, order: int = 192):
    """Return c(theta), signed zeta, adjacent MI and density minimum."""
    z = float(hyp1f1(0.5, m / 2.0, theta))
    c = float(hyp1f1(1.5, m / 2.0 + 1.0, theta) / (m * z))
    zeta = (m * c - 1.0) / (m - 1.0)
    t, w = _beta_quadrature(m, order)
    log_g = theta * t - np.log(z)
    g = np.exp(log_g)
    mi = float(np.sum(w * g * log_g))
    return {"normalizer": z, "c": c, "zeta": zeta,
            "mi_nats": mi, "density_min": float(np.exp(np.min(log_g)))}


def theta_from_zeta(m: int, zeta: float, tol: float = 1e-11):
    lo, hi = -1.0 / (m - 1.0) + 1e-9, 1.0 - 1e-9
    if not lo < zeta < hi:
        raise ValueError("zeta must be strictly inside its attainable interval")
    def residual(theta):
        return watson_stats(m, theta, order=96)["zeta"] - zeta
    return float(brentq(residual, -80.0, 80.0, xtol=tol))


def mutual_information_for_zeta(m: int, zeta: float):
    theta = theta_from_zeta(m, zeta)
    return watson_stats(m, theta)["mi_nats"]


def full_watermark_covariance(d: np.ndarray, zeta: float, q: float):
    """Covariance of stack X_l=(beta_l beta_l^T)d_l, d shape [L,m]."""
    d = np.asarray(d, dtype=float)
    if d.ndim != 2 or q < 0:
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
    sign0, ld0 = np.linalg.slogdet(theta0)
    sign1, ld1 = np.linalg.slogdet(theta0 + C)
    if sign0 <= 0 or sign1 <= 0:
        raise ValueError("nominal and attacked covariance must be positive definite")
    return float(0.5 * (
        np.trace(np.linalg.solve(theta0, C)) + ld0 - ld1
    ))


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
    grid_size: int = 2001,
):
    """Robust one-dimensional screen.

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
    candidates = []
    zlo = -1.0/(m-1.0) + 1e-5
    zhi = 1.0 - 1e-5
    grid = np.linspace(zlo, zhi, grid_size)

    def evaluate(zeta):
        theta = theta_from_zeta(m, zeta)
        info = (L-1) * watson_stats(m, theta)["mi_nats"]
        Q = risk_callback(zeta)
        risk = _worst_energy_risk(Q, energy_gram, attack_energy)
        detections = []
        for d, Z, Theta in attack_templates:
            C = parity_covariance(d, Z, zeta, q=watermark_variance)
            detections.append(gaussian_covariance_kl_proxy(Theta, C))
        det = float(min(detections))
        feasible = info <= dependence_budget and risk <= state_risk_budget
        return Candidate(zeta, theta, info, det, risk, feasible)

    rows = [evaluate(float(z)) for z in grid]
    feasible = [r for r in rows if r.feasible]
    if not feasible:
        return {"status": "no feasible coupling; retain iid baseline or revisit budgets",
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
    iid = evaluate(0.0)
    if iid.feasible and iid.worst_detection_proxy >= best.worst_detection_proxy:
        best = iid
    return {"status": "screened feasible candidate; not a certified global optimum",
            "best": best.__dict__, "iid": iid.__dict__,
            "grid_feasible_count": len(feasible),
            "grid_size": grid_size,
            "warning": "D_G is a covariance proxy. Validate true detector power and full common-G trajectory risk on independent held-out runs."}
