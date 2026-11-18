"""Finite-confidence audit of expected healthy MSE with an analytic tail bound."""
import argparse
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from shared import (DEFAULT_BASELINE, HERE, REFERENCE, load_json, load_prepared,
                    setup, verify_registration, write_json)
from analytical_weight import suffix_gramians

NAMES = ("V0", "V1a", "V1b", "V1.1")


def inv2(p):
    a, b, c, d = map(Q, p.ravel())
    return np.array([[d, -b], [-c, a]], dtype=object) / (a*d-b*c)


def rational_envelope(path, name, spec):
    """Check simple deterministic constants for the frozen decimal model."""
    import json
    obj = json.loads(path.read_text(encoding="utf-8"), parse_float=Q)
    model = obj["models"][name]
    i, j, l = (spec[key] for key in ("receiver", "target", "healthy"))
    nodes = (i, j, l)
    A, noise = (np.array(a, dtype=object) for a in (model["A"], obj["Q"]))
    H, R = (np.array([obj[key][s] for s in nodes], dtype=object) for key in ("H", "R"))
    K, G = (np.array(model["nodes"][i][key], dtype=object) for key in ("K", "G"))
    eye = np.eye(2, dtype=int).astype(object)
    J = eye - K @ H[0] - G @ (H[1]+H[2])
    F = J @ A

    def squared_norm_bound(a):
        z = np.abs(a)
        return max(z.sum(axis=0)) * max(z.sum(axis=1))

    assert squared_norm_bound(F) < Q("0.12")**2
    assert squared_norm_bound(G) < Q("0.271")**2
    info = [h.T @ inv2(r) @ h for h, r in zip(H, R)]
    trusted = R[1] + H[1] @ inv2(info[0]+info[2]) @ H[1].T
    gap = Q("0.69") * eye - trusted
    assert gap[0, 0] > 0 and gap[0, 0]*gap[1, 1] > gap[0, 1]*gap[1, 0]
    v0, all_info = eye.copy(), eye.copy()
    v0_trace, base_trace = [], []
    v_gain = K @ R[0] @ K.T + G @ (R[1]+R[2]) @ G.T
    for _ in range(spec["horizon"]):
        v0_post = J @ v0 @ J.T + v_gain
        all_post = inv2(inv2(all_info)+sum(info))
        assert 0 <= np.trace(v0_post-all_post) < Q("0.003")
        v0_trace.append(float(np.trace(v0_post)))
        base_trace.append(float(np.trace(all_post)))
        v0 = A @ v0_post @ A.T + noise
        all_info = A @ all_post @ A.T + noise
    # Use rational upper bounds on both square roots.
    assert (Q("0.271")*Q("0.831")/Q("0.88")+Q("0.055"))**2 < Q("0.1")
    assert Q("0.831")**2 > Q("0.69") and Q("0.055")**2 > Q("0.003")
    return np.array(v0_trace), np.array(base_trace)


def cap(curvature, linear, budget):
    root = np.sqrt(linear**2+curvature*budget)
    proposed = np.ones_like(linear)
    np.divide(root+linear, curvature, out=proposed, where=curvature > 1e-24)
    np.divide(budget, root-linear, out=proposed, where=(linear < 0))
    return np.clip(proposed, 0, 1)


def regression_cache(prepared):
    return [solve_triangular(p.target_cholesky, np.eye(len(p.target_cholesky)), lower=True)
            for p in prepared.prefixes]


def fit_prefix(prepared, k, white, decoder, whitener):
    """Batched version of the two-column Gaussian regression in V1."""
    p, s = prepared, prepared.settings
    count = k + 1
    prior = p.template @ p.parameter_prior_cholesky
    design = np.zeros((len(white), count, count, p.m, p.n))
    for onset in range(count):
        stop = min(count, onset+s.template_duration)
        design[:, onset, onset:stop] = decoder[:, onset:stop] @ prior[:stop-onset]
    last = design[:, :, k].copy()
    scale = whitener @ design.reshape(len(white), count, count*p.m, p.n)
    precision = scale.swapaxes(-2, -1) @ scale + np.eye(p.n)
    information = (scale.swapaxes(-2, -1) @ white[:, None, :, None])[..., 0]
    coef = np.linalg.solve(precision, information[..., None])
    half = .5 * (np.sum(information * coef[..., 0], axis=-1) - np.linalg.slogdet(precision)[1])
    blank = s.horizon*s.no_event_prior/(1-s.no_event_prior) + s.horizon-count
    probability = np.exp(half - np.logaddexp(np.log(blank), logsumexp(half, axis=1))[:, None])
    correction = np.sum(probability[..., None] * (last @ coef)[..., 0], axis=1)
    response = p.prefixes[k].state_from_white @ scale
    state = (response @ coef)[..., 0]
    shift = np.sum(probability[..., None] * state, axis=1)
    return correction, shift, probability, response, precision, state


def null_batch(prepared, trusted, target, decoder, whiteners):
    """Healthy receiver-only quadrature integrands and ordinary reference outputs."""
    p, s, model = prepared, prepared.settings, prepared.model
    b = len(trusted)
    f, g, gain = model.F(s.receiver), model.G[s.receiver], model.K[s.receiver]
    posterior = np.zeros((b, len(NAMES), p.n))
    error_mean = np.empty((b, len(NAMES), s.horizon, p.n))
    posts = np.empty_like(error_mean)
    suffixes = suffix_gramians(f, s.horizon)
    budgets = s.nominal_one_step_allowance * p.nominal_mse()

    for k, prefix in enumerate(p.prefixes):
        u = trusted[:, :k+1].reshape(b, -1)
        y = target[:, :k+1].reshape(b, -1)
        white = (y-u @ prefix.target_from_trusted.T) @ whiteners[k].T
        mu0 = u @ prefix.state_from_trusted.T + white @ prefix.state_from_white.T
        correction, shift, w, response, precision, state = fit_prefix(p, k, white, decoder, whiteners[k])
        d = correction @ g.T
        x_raw = (trusted[:, k, :p.m] @ gain.T +
                 (trusted[:, k, p.m:] + target[:, k]) @ g.T)
        before = posterior @ f.T + x_raw[:, None, :]
        nominal = before-mu0[:, None, :]
        mixture = nominal+shift[:, None, :]
        s2 = np.sum(d*d, axis=-1)
        packed = (response.swapaxes(-2, -1) @ d[:, None, :, None])
        within = np.sum(packed * np.linalg.solve(precision, packed), axis=(-2, -1))
        between = np.sum(w*np.sum(state*d[:, None, :], axis=-1)**2, axis=1) - np.sum(shift*d, axis=1)**2
        sigma = np.sqrt(np.maximum(0., np.sum(w*within, axis=1)+between))
        lin_b = np.sum(d*mixture[:, 2], axis=1)-s.uncertainty_standard_deviations*sigma
        lin_11 = np.sum(d*mixture[:, 3], axis=1)
        valid = s2 > 1e-24
        a_b = np.divide(lin_b, s2, out=np.zeros_like(s2), where=valid).clip(0, 1)
        a_11 = np.divide(lin_11, s2, out=np.zeros_like(s2), where=valid).clip(0, 1)
        cap_b = cap(s2, np.sum(d*nominal[:, 2], axis=1),
                    s.nominal_one_step_allowance*(np.sum(nominal[:, 2]**2, axis=1) +
                                                  np.trace(prefix.conditional_state_cov)))
        sf = d @ suffixes[k]
        cap_11 = cap(np.sum(sf*d, axis=1), np.sum(sf*nominal[:, 3], axis=1), budgets[k])
        alpha = np.stack((np.zeros(b), np.ones(b), np.minimum(a_b, cap_b),
                          np.minimum(a_11, cap_11)), axis=1)
        posterior = before-alpha[..., None]*d[:, None, :]
        posts[:, :, k] = posterior
        error_mean[:, :, k] = posterior-mu0[:, None, :]
    return posts, np.sum(error_mean**2, axis=-1)


def bounded_mean_upper(values, clip, error_probability):
    """One-sided e-value inversion for bounded iid means."""
    t = .95 / clip
    log_product = np.log1p(-t*np.minimum(values, clip)).sum(axis=0)
    return -np.expm1((np.log(error_probability)+log_product)/len(values))/t


def evaluate(prepared, registered, spec, exact_stage, exact_floor):
    count = registered["nominal_certificate"]["paths_per_model"]
    batch = registered["nominal_certificate"]["batch"]
    clip = registered["nominal_certificate"]["clip"]
    base_seed = registered["samples"][prepared.model.name]["nominal_certificate_seed"]
    seed_x, seed_m = np.random.SeedSequence(base_seed).spawn(2)
    rng_x, rng_m = np.random.default_rng(seed_x), np.random.default_rng(seed_m)
    s = prepared.settings
    u, y = prepared.trusted_maps, prepared.target_maps
    dims = u.shape[-1]
    whiteners = regression_cache(prepared)
    values = []
    for first in range(0, count, batch):
        m = min(batch, count-first)
        noise = rng_x.standard_normal((m, dims))
        beta = rng_m.normal(0, np.sqrt(spec["watermark"]["q"]), (m, s.horizon, prepared.m))
        decoder = spec["watermark"]["lambda"]*np.eye(prepared.m) + beta[..., None]*beta[..., None, :]
        trusted = (noise @ u.reshape(-1, dims).T).reshape((m,)+u.shape[:-1])
        target = (noise @ y.reshape(-1, dims).T).reshape((m,)+y.shape[:-1])
        _, integrands = null_batch(prepared, trusted, target, decoder, whiteners)
        values.append(integrands)
    score = np.concatenate(values)
    tail = .2 * s.horizon*(s.template_duration+1) * np.exp(-clip/.2)
    means = score.mean(axis=-1)
    fp = registered["nominal_certificate"]["family_error"] / (len(registered["samples"])*3)
    upper = bounded_mean_upper(means[:, 1:], clip, fp)+tail+1e-9
    output = {}
    for j, label in enumerate(NAMES):
        per_step = exact_floor+score[:, j].mean(axis=0) if j else exact_stage
        bound = min(exact_stage.mean()*1.02, exact_floor.mean()+upper[j-1]) if label == "V1.1" else (
            exact_floor.mean()+upper[j-1] if j else exact_stage.mean())
        output[label] = {"stage_mse_model_estimate": per_step.tolist(),
                         "average_mse_model_estimate": float(per_step.mean()),
                         "relative_model_estimate": float(per_step.mean()/exact_stage.mean()-1),
                         "upper_average": float(bound),
                         "upper_relative": float(bound/exact_stage.mean()-1),
                         "max_integrand": float(means[:, j].max()),
                         "clipped_count": int(np.sum(means[:, j] > clip))}
    return {"n": count, "analytic_tail_allowance": float(tail), "pointwise_error_probability": fp,
            "V0_exact_stage_trace": exact_stage.tolist(), "conditional_cov_trace": exact_floor.tolist(),
            "policies": output}, means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()
    reg = load_json(HERE / "registration.json")
    verify_registration(reg, args.baseline)
    common, settings, spec, _ = setup(args.baseline)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    audits = {}
    for name in reg["samples"]:
        p = load_prepared(args.baseline, name, common, settings)
        model_path = args.baseline / REFERENCE / "inputs" / "five_node_aggregate_model.json"
        stage, base = rational_envelope(model_path, name, spec)
        np.testing.assert_allclose(stage, p.nominal_mse(), atol=1e-12)
        np.testing.assert_allclose(base, [np.trace(m.conditional_state_cov) for m in p.prefixes], atol=1e-12)
        result, scores = evaluate(p, reg, spec, stage, base)
        audits[name] = result
        np.save(out / f"{name}_bounded_health_scores.npy", scores, allow_pickle=False)
        print(name, "V1a expected relative MSE upper", result["policies"]["V1a"]["upper_relative"], flush=True)
    write_json(out / "nominal_audit.json", audits)


if __name__ == "__main__":
    main()
