"""Read-only A0 early correction audit. No sender, new watermark, or disturbance generator."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OLD = Path("JFI_watermark_assisted_fusion_feasibility_20261119")
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE / "JFI_defensive_offline_validation"))

from replay_evaluator import (MODEL, OBSERVATIONS, V1, V11, load_receiver_modules,
                              read_json, saved_sensor_inputs, verify_inputs)

TERMS = ("activity_gate", "onset_mixture", "known_onset_fit")
GAPS = ("known_onset_mismatch", "mixture_squared", "mixture_fit_cross",
        "posterior_trace_excess")


def norm2(a):
    return np.sum(a * a, axis=-1)


def protected_inputs(base, settings):
    paths = settings["previous_inputs"].copy()
    old = read_json(base / OLD / "replay_helpers_sha256.json")
    assert old["text_crlf_normalized"]
    paths.update(old["files"])
    for path, digest in paths.items():
        assert sha256((base / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest, path
    inherited = verify_inputs(base)
    return len(paths.keys() | inherited["files"].keys())


def blocks(prepared, saved_decoder, k, whiten):
    """Build each finite-prefix response exactly as the frozen receive-only model."""
    t = k + 1
    m, n = prepared.m, prepared.n
    code = prepared.template
    designs = np.zeros((t, t, m, n))
    for s in range(t):
        end = min(t, s + len(code))
        designs[s, s:end] = saved_decoder[s:end] @ code[:end-s]
    prior_root = prepared.parameter_prior_cholesky
    last = designs[:, -1]
    x = whiten @ designs.reshape(t, t*m, n)
    return x @ prior_root, last @ prior_root, x, last


class Conditional:
    """Fixed model, prefix and stored watermarks; scores are batched over Gaussian nodes."""

    def __init__(self, p, decoder, k, start, shift, whiten, rank_tol):
        self.start = start
        self.n = p.n
        self.active = np.arange(k+1) >= k+1-p.settings.template_duration
        self.total_onsets = p.settings.horizon
        self.future = self.total_onsets-k-1
        self.prior_null_scaled = self.total_onsets * p.settings.no_event_prior / (1-p.settings.no_event_prior)
        self.G = p.model.G[p.settings.receiver]
        self.x, self.last, self.phys_x, self.phys_last = blocks(p, decoder, k, whiten)
        self.true_white_shift = whiten @ shift[:k+1].ravel()
        self.target = self.G @ shift[k]
        self.info = self.x.swapaxes(-1, -2) @ self.x
        self.precision = np.eye(p.n) + self.info
        self.post = np.linalg.solve(self.precision, np.broadcast_to(np.eye(p.n), self.precision.shape))
        self.logdet = np.linalg.slogdet(self.precision)[1]
        self.b = self.G @ self.last
        self.component_cov = self.b @ self.post @ self.b.swapaxes(-1, -2)
        actual = self.info.shape[0]
        scores = self.x.transpose(1, 0, 2).reshape(self.x.shape[1], actual*p.n)
        evals, eigvec = np.linalg.eigh(scores @ scores.T)
        order = np.argsort(evals)[::-1]
        keep = order[evals[order] > rank_tol * evals[order[0]]]
        axes = eigvec[:, keep]
        axes *= np.sign(axes[np.argmax(np.abs(axes), axis=0), np.arange(len(keep))])
        self.rank = len(keep)
        self.transformed_scores = axes.T @ scores
        self.noncentral_scores = self.true_white_shift @ scores
        residual = scores - axes @ self.transformed_scores
        np.testing.assert_allclose(residual, 0, rtol=0, atol=5e-10)
        self.score_projection_residual = float(np.linalg.norm(residual))
        self.B = self.b[start]
        self.P = self.post[start]
        self.H = self.info[start]

    def from_score(self, raw):
        z = raw.reshape(len(raw), -1, self.n)
        c = np.einsum("sij,lsj->lsi", self.post, z)
        logb = 0.5 * (np.sum(z*c, axis=-1) - self.logdet)
        logpartial = logsumexp(logb, axis=1)
        w = np.exp(logb-np.logaddexp(np.log(self.prior_null_scaled+self.future), logpartial)[:, None])
        logbf = np.logaddexp(logpartial, np.log(self.future) if self.future else -np.inf) - np.log(self.total_onsets)
        b = np.einsum("sij,lsj->lsi", self.b, c)
        wm = w[..., None] * b
        mean = wm.sum(axis=1)
        p_active = w[:, self.active].sum(axis=1)
        active_mean = wm[:, self.active].sum(axis=1) / p_active[:, None]
        label_mean = b[:, self.start]
        m_end = self.target
        v = np.stack((mean-active_mean, active_mean-label_mean, label_mean-m_end), axis=1)
        cov_same_start = np.diagonal(self.component_cov, axis1=-2, axis2=-1).sum(axis=1)
        cov_within = w @ cov_same_start
        cov_activity = p_active*(1-p_active)*norm2(active_mean)
        cov_onset = np.sum(w[:, self.active]*norm2(b[:, self.active]-active_mean[:, None]), axis=1)
        predicted = cov_within + cov_activity + cov_onset
        # Ended hypotheses, when present, have zero current transfer and trace.
        u_star = cov_same_start[self.start]
        inactive_trace = (1-p_active)*u_star
        active_trace = (w[:, self.active]*(u_star-cov_same_start[self.active])).sum(axis=1)
        np.testing.assert_allclose(predicted, w @ cov_same_start + np.sum(w*norm2(b), axis=1) - norm2(mean),
                                   rtol=6e-10, atol=2e-12)
        np.testing.assert_allclose(u_star-predicted,
                                   inactive_trace+active_trace-cov_activity-cov_onset,
                                   rtol=2e-11, atol=2e-12)
        return {
            "ordered_terms": v,
            "means": mean,
            "known_onset_mean": label_mean,
            "weighted_same_start_trace": cov_within,
            "between_activity_trace": cov_activity,
            "between_onset_active_trace": cov_onset,
            "inactive_probability_trace_discount": inactive_trace,
            "other_active_components_trace_shift": active_trace,
            "predicted": predicted,
            "p_active": p_active,
            "p_true_onset": w[:, self.start],
            "p_true_given_active": w[:, self.start] / p_active,
            "p_no_event": self.prior_null_scaled*np.exp(
                -np.logaddexp(np.log(self.prior_null_scaled+self.future), logpartial)),
            "p_future": self.future*np.exp(
                -np.logaddexp(np.log(self.prior_null_scaled+self.future), logpartial)),
            "weights_started": w,
            "logbf": logbf,
        }

    def for_white(self, z):
        score = np.einsum("srj,lr->lsj", self.x, np.atleast_2d(z))
        return self.from_score(score)

    def for_nodes(self, standard):
        score = standard[:, :self.rank] @ self.transformed_scores + self.noncentral_scores
        return self.from_score(score)

    def values(self, frame):
        v = frame["ordered_terms"]
        e = v.sum(axis=1)
        terms = np.column_stack((norm2(v), 2*np.sum(v[:, 0]*v[:, 1], axis=1),
                                 2*np.sum(v[:, 0]*v[:, 2], axis=1),
                                 2*np.sum(v[:, 1]*v[:, 2], axis=1)))
        err = norm2(e)
        mixture = norm2(v[:, 0]+v[:, 1])
        cross = 2*np.sum((v[:, 0]+v[:, 1])*v[:, 2], axis=1)
        fit = norm2(v[:, 2])
        p_known = np.trace(self.component_cov[self.start])
        gaps = np.column_stack((fit-p_known, mixture, cross, p_known-frame["predicted"]))
        np.testing.assert_allclose(err, terms.sum(axis=1), atol=2e-12, rtol=2e-11)
        np.testing.assert_allclose(err-frame["predicted"], gaps.sum(axis=1), atol=2e-12, rtol=2e-11)
        return np.column_stack((err, frame["predicted"], terms, gaps, frame["p_active"],
                                frame["p_true_given_active"], frame["p_true_onset"],
                                frame["p_no_event"], frame["p_future"],
                                frame["weighted_same_start_trace"], frame["between_activity_trace"],
                                frame["between_onset_active_trace"],
                                frame["inactive_probability_trace_discount"],
                                frame["other_active_components_trace_shift"],
                                norm2(v[:, 1]+v[:, 2]), p_known+np.zeros(len(v))))

    def exact_fit(self, physical_truth, prior_root):
        # Direction and gain at the observed record only; no new template or receiver.
        u = np.linalg.solve(prior_root, physical_truth)
        np.testing.assert_allclose(self.b[self.start] @ u, self.target, atol=2e-12)
        BP = self.B @ self.P
        v = BP @ u
        t = BP @ BP.T
        q = np.trace(t)
        var = np.trace(BP @ self.H @ BP.T)
        second = var + np.dot(v, v)
        prior = np.trace(self.component_cov[self.start])
        np.testing.assert_allclose(second-prior, v @ v-q, atol=2e-12)
        u2 = u @ u
        alignment = float(v @ v/q) if q > 0 else 0.
        assert 0 <= alignment <= u2 + 2e-12
        return {
            "known_start_exact_mse": float(second),
            "known_start_noise_variance": float(var),
            "known_start_squared_shrinkage_bias": float(v @ v),
            "known_start_working_trace": float(prior),
            "known_start_mismatch": float(v @ v-q),
            "mean_prior_direction_mismatch_matrix": (v[:, None] @ v[None, :] - t).tolist(),
            "sensitivity_trace": float(q),
            "direction_alignment_sq_over_trace": alignment,
            "prior_whitened_energy": float(u2),
            "known_start_working_impact": float(np.trace(self.B @ self.B.T)),
            "known_start_information_trace": float(np.trace(self.H)),
            "known_start_working_bayes_information_gain": float(np.trace(self.B @ self.B.T)-prior),
            "physical_information_eigenvalues": np.linalg.eigvalsh(
                self.phys_x[self.start].T@self.phys_x[self.start]).tolist(),
            "raw_G_shift_squared": float(self.target @ self.target),
        }


LABELS = ("actual_G_error", "working_trace", *TERMS,
          "activity_x_onset", "activity_x_fit", "onset_x_fit", *GAPS,
          "active_mass", "true_onset_given_active", "true_onset_mass", "no_event_mass",
          "future_mass", "weighted_same_start_trace", "between_activity_trace",
          "between_onset_active_trace", "inactive_probability_trace_discount",
          "other_active_components_trace_shift", "activity_reference_mse",
          "known_onset_working_trace")


def row(values):
    d = dict(zip(LABELS, map(float, values), strict=True))
    d["actual_minus_working"] = d["actual_G_error"]-d["working_trace"]
    d["known_start_fit_this_law"] = d["known_onset_mismatch"]+d["known_onset_working_trace"]
    return d


def lattice(power, dimension):
    # Positive offset keeps inverse Gaussian CDF away from the unit-cube boundary.
    raw = qmc.Sobol(d=dimension, scramble=False).random_base2(power)
    return ndtri(raw+0.5/(2**power))


def integrate(problem, nodes, chunk):
    total = np.zeros(len(LABELS))
    for j in range(0, len(nodes), chunk):
        total += problem.values(problem.for_nodes(nodes[j:j+chunk])).sum(axis=0)
    return row(total/len(nodes))


def averaged(rows):
    return {name: float(np.mean([r[name] for r in rows])) for name in rows[0]}


def coding_resource(p, watermarks):
    """Normal coding resource; no trial encoder is evaluated."""
    code_dim = p.m
    if code_dim != 2:
        raise ValueError("The frozen diagnostic uses two link coordinates.")
    lam, q = watermarks["lambda"], watermarks["q"]
    # One-dimensional exponential mixing for chi-square with two degrees of freedom.
    from scipy.integrate import quad
    ratio = q/lam
    h = quad(lambda x: np.exp(-x) / (1 + 2*ratio*x)**2, 0, np.inf, epsabs=2e-13)[0]
    return {
        "lambda": lam, "q": q,
        "normal_coded_energy_multiplier": (1+h)/(2*lam**2),
        "encoding_operator_bound": 1/lam,
        "decoder_bound_is_finite": False,
    }


def compare_model(base, name, spec, settings, receiver, common, evidence, prior_work, nodes):
    s = evidence.Settings.from_registration(spec)
    p = evidence.PreparedModel(common.load_model(name, base / MODEL), s)
    assert s.template_duration == settings["evaluation"]["window_length"] == 10
    assert (s.receiver, s.target, s.healthy) == tuple(
        settings["evaluation"][key] for key in ("receiver", "target", "healthy"))
    indices = spec["samples"]["physical_indices"]
    a, b = settings["evaluation"]["physical_indices_inclusive"]
    assert indices == list(range(a,b+1))
    start = settings["evaluation"]["onset"]
    assert start == spec["evaluation_onset"]
    assert settings["evaluation"]["bf"] == 1/s.event_false_alarm
    trusted, target, decoder = saved_sensor_inputs(base, name, s, indices)
    with np.load(base / OBSERVATIONS, allow_pickle=False) as bank:
        coded = bank[f"{name}__original_Gaussian__ordinary__attack"]
    with np.load(base / V11 / f"phase1_results/{name}_step_logs.npz", allow_pickle=False) as saved:
        shift = saved["A0__decoded_difference"]
        np.testing.assert_allclose((decoder @ coded[None, :, :, None])[..., 0], shift, atol=2e-12, rtol=2e-12)
        np.testing.assert_array_equal(shift[:, :start], 0.)
        np.testing.assert_array_equal(shift[:, start+10:], 0.)
        energy = float(norm2(coded).sum())
        np.testing.assert_allclose(energy, s.coded_energy_reference, atol=2e-11, rtol=0)
        design = p.template.reshape(-1,p.n)
        theta = np.linalg.lstsq(design, coded[start:start+10].ravel(), rcond=None)[0]
        np.testing.assert_allclose(design @ theta, coded[start:start+10].ravel(), atol=2e-12, rtol=0)
        obs = receiver.Observations(trusted, target+shift, decoder)
        observer = receiver.BatchedObserver(p)
        results = []
        gain = p.model.G[s.receiver]
        for j, age in enumerate(settings["evaluation"]["packet_ages"]):
            k = start + age - 1
            frame = observer.at(obs, k, risk=True)
            prefix = p.prefixes[k]
            whiten = solve_triangular(prefix.target_cholesky, np.eye((k+1)*p.m), lower=True)
            np.testing.assert_allclose(whiten, observer.whiteners[k], atol=2e-12, rtol=2e-12)
            whites = (obs.target[:, :k+1].reshape(len(obs), -1) -
                      obs.trusted[:, :k+1].reshape(len(obs),-1) @ prefix.target_from_trusted.T) @ whiten.T
            old = prior_work["early_pollution_diagnostics"][j]
            observed = []
            fits = []
            integrals = {power:[] for power in nodes}
            r = []
            for i in range(len(indices)):
                q = Conditional(p, decoder[i], k, start, shift[i], whiten,
                                settings["conditional_integral"]["rank_eigen_rtol"])
                w = q.for_white(whites[i])
                np.testing.assert_allclose(w["means"][0], gain @ saved["A0__delta_hat"][i,k], atol=2e-11, rtol=2e-11)
                np.testing.assert_allclose(w["weights_started"][0], frame.posterior_weights[i], atol=2e-11, rtol=2e-11)
                np.testing.assert_allclose(w["logbf"][0], saved["A0__log_bf"][i,k], atol=2e-11, rtol=2e-11)
                np.testing.assert_allclose(w["predicted"][0], np.trace(
                    gain@saved["A0__delta_cov"][i,k]@gain.T), atol=2e-11, rtol=2e-11)
                values = row(q.values(w)[0])
                v = w["ordered_terms"][0]
                values["terms_in_update_coordinates"] = v.tolist()
                observed.append(values)
                tag = q.exact_fit(theta, p.parameter_prior_cholesky)
                fits.append(tag)
                r.append((q.rank,q.score_projection_residual))
                for power, lattice_nodes in nodes.items():
                    values = integrate(q, lattice_nodes, settings["conditional_integral"]["batch"])
                    expected_fit = tag["known_start_exact_mse"]
                    values["exact_known_start_fit"] = expected_fit
                    values["known_start_fit_error"] = values["known_start_fit_this_law"]-expected_fit
                    # Integrate only nonlinear mixture terms; substitute the exact known-onset term.
                    values["hybrid_exact_fit_error"] = values["actual_G_error"]-values["known_start_fit_error"]
                    values["hybrid_actual_minus_working"] = values["hybrid_exact_fit_error"]-values["working_trace"]
                    values["hybrid_known_start_mismatch"] = tag["known_start_mismatch"]
                    integrals[power].append(values)
            scalar_observed = [row_without_vectors(x) for x in observed]
            empirical = averaged(scalar_observed)
            np.testing.assert_allclose(empirical["actual_G_error"], old["state_update_squared_error_from_distortion"],
                                       atol=2e-12, rtol=2e-11)
            np.testing.assert_allclose(empirical["working_trace"], old["working_posterior_trace_state_update"],
                                       atol=2e-12, rtol=2e-11)
            exact = {n:float(np.mean([x[n] for x in fits])) for n in fits[0]
                     if n not in {"mean_prior_direction_mismatch_matrix","physical_information_eigenvalues"}}
            integrated = {power:averaged(data) for power,data in integrals.items()}
            high, mid = sorted(nodes)[-1], sorted(nodes)[-2]
            for values in integrated.values():
                np.testing.assert_allclose(values["hybrid_actual_minus_working"],
                    sum(values[x] for x in GAPS[1:])+values["hybrid_known_start_mismatch"],
                    atol=2e-12, rtol=2e-11)
            results.append({
                "packet_age": age, "absolute_tick": k, "path_count":len(indices),
                "empirical":empirical, "empirical_rows":observed, "known_onset_exact":exact,
                "known_onset_rows":fits, "conditional_quadrature":integrated,
                "maximum_refinement_per_key": {
                    f:max(abs(v[f]-u[f]) for u,v in zip(integrals[mid], integrals[high], strict=True))
                    for f in ("hybrid_exact_fit_error","working_trace","hybrid_actual_minus_working")},
                "noise_score_dimension": sorted({x[0] for x in r}),
                "largest_score_projection_residual": max(x[1] for x in r),
            })
    return {"archived_total_coded_energy":energy, "normal_coding":coding_resource(p,spec["watermark"]),
            "packet_results":results}


def row_without_vectors(data):
    return {name:value for name,value in data.items() if name != "terms_in_update_coordinates"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small quadrature check, not the reference result.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    settings = read_json(HERE / "registration.json")
    count = protected_inputs(BASE,settings)
    prior = read_json(BASE / OLD / "replay_results.json")
    spec = read_json(BASE / V1 / "registration.json")
    receiver, common, evidence = load_receiver_modules(BASE)
    powers = (7, 9) if args.quick else settings["conditional_integral"]["powers"]
    dim = 2*(settings["evaluation"]["onset"]+max(settings["evaluation"]["packet_ages"]))
    nodes = {power:lattice(power,dim) for power in powers}
    models = {}
    for name in settings["models"]:
        models[name] = compare_model(BASE,name,spec,settings,receiver,common,evidence,
                                     prior["models"][name],nodes)
        for data in models[name]["packet_results"]:
            emp, ref = data["empirical"],data["conditional_quadrature"][max(nodes)]
            print(f"{name:8s} age={data['packet_age']}  saved {emp['actual_G_error']:.6f} / {emp['working_trace']:.6f}"
                  f"  conditional {ref['hybrid_exact_fit_error']:.6f} / {ref['working_trace']:.6f}",
                  flush=True)
    assert protected_inputs(BASE,settings)==count
    payload = {"baseline":settings["baseline"],"verified_dependencies":count,"new_physical_paths":0,
               "quad_powers":powers,"certified_numerics":False,"models":models}
    path = args.out or HERE / ("quick_results.json" if args.quick else "early_calibration_results.json")
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")


if __name__ == "__main__":
    main()
