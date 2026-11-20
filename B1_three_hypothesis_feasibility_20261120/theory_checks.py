"""Independent exact coefficient and finite-signal Gaussian mixture checks."""
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("prior_fraction_checks", BASE / "research_discussions/2026-11-20_B1_new_theory_review/verify_algebra.py")
alg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alg)


def exact_coefficient():
    z0 = alg.add(alg.scale(alg.ONE, F(1, 2)), {(2, 0, 1): F(1)})
    z1 = {(1, 1, 1): F(1)}
    task = alg.add(alg.scale(alg.mul(z0, z0), F(2)), alg.mul(z1, z1))
    precision = alg.add(alg.add(alg.mul(z0, z0), alg.mul(z1, z1)), alg.mul(z0, z1))
    info = alg.scale(alg.mul(task, precision), F(4, 9))
    gI = [alg.moment(alg.mul(info, f), F(1, 2)) for f in (alg.C4, alg.S4)]
    gA = [alg.moment(alg.mul(alg.scale(task, F(2, 3)), f), F(1, 2)) for f in (alg.C4, alg.S4)]
    assert gA == [F(1, 12), F(0)]
    assert gI[1] == F(11, 6)
    # Removing the cross precision removes all sine harmonics exactly.
    aligned = alg.scale(alg.mul(task, alg.add(alg.mul(z0, z0), alg.mul(z1, z1))), F(4, 9))
    assert alg.moment(alg.mul(aligned, alg.S4), F(1, 2)) == 0
    return dict(g_A_unit_amplitude=list(map(str, gA)), g_I_fourth_order=list(map(str, gI)),
                orthogonal_information_coefficient=str(gI[1]), aligned_sine_coefficient="0")


def finite_signal(radial, angles, normal):
    x, wx = np.polynomial.laguerre.laggauss(radial)
    theta = np.arange(angles) * np.pi / angles
    u = np.stack([np.cos(theta), np.sin(theta)], -1)
    M = (.5 * np.eye(2) + x[:, None, None, None] * (u[:, :, None] * u[:, None, :])[None]).reshape(-1, 2, 2)
    w = np.repeat(wx, angles) / angles
    f = np.tile(np.stack([np.cos(4 * theta), np.sin(4 * theta)], -1), (radial, 1))
    z = M[:, :, 0]
    Q = np.diag([2., 1.])
    R = np.array([[1., .5], [.5, 1.]])
    task = np.einsum("bi,ij,bj->b", z, Q, z)
    precision = np.einsum("bi,ij,bj->b", z, R, z)
    hn, hw = hermgauss(normal)
    hn, hw = hn * np.sqrt(2), hw / np.sqrt(np.pi)
    values = np.arange(3)
    results = []
    for eps in (.01, .02, .04, .08, .16, .32, .64, 1.):
        distance = eps * np.sqrt(precision)
        y = distance[:, None, None] * values[None, :, None] + hn[None, None]
        logp = -.5 * (y[..., None] - distance[:, None, None, None] * values) ** 2
        posterior = np.exp(logp - logsumexp(logp, axis=-1, keepdims=True))
        mean = posterior @ values
        I = eps ** 2 * task * np.mean(np.sum((mean - 1) ** 2 * hw, axis=-1), axis=-1)
        A = eps ** 2 * task * F(2, 3)
        A = np.asarray(A, dtype=float)
        gI = (w * I) @ f
        gA = (w * A) @ f
        results.append(dict(eps=eps, gI=gI.tolist(), gA=gA.tolist(), gain_rho_point6=float(.6 * gI[1]),
                            gain_coefficient=float(gI[1] / eps ** 4), target_coefficient=11 / 6,
                            A=float(w @ A), I=float(w @ I)))
    return results


def main():
    result = dict(exact=exact_coefficient(), finite_signal=finite_signal(32, 96, 51),
                  refinement=finite_signal(48, 128, 71),
                  limitation="Specified snapshot Q/R, known amplitude candidates, no actual A0 history. Exact algebra establishes no system-wide performance advantage.")
    (HERE / "results/theory_checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
