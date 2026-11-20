"""Compare covariance-informed and simpler output fusion rules on frozen records."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE / "JFI_defensive_offline_validation"))

from replay_evaluator import (FRONTIER, MODEL, OBSERVATIONS, V1, V11, load_receiver_modules,
                              read_json, saved_sensor_inputs, verify_inputs)
from risk_metrics import score_estimates


SCORES = ("residual_score", "posterior_mean_score", "posterior_second_moment_score")


def squared(v):
    return np.sum(v**2, axis=-1)


def verify_local_replay_helpers(base):
    """Protect the four helper modules and the inherited manifest itself."""
    lock = read_json(HERE / "replay_helpers_sha256.json")
    for path, expected in lock["files"].items():
        data = (base / path).read_bytes().replace(b"\r\n", b"\n")
        if sha256(data).hexdigest() != expected:
            raise ValueError(f"Frozen replay helper changed: {path}")
    return lock


def mixture_moments(matrix, frame):
    """Displacement under all started hypotheses, with remaining mass at zero."""
    means = (matrix @ frame.posterior_coef[..., None])[..., 0]
    within = matrix @ frame.inverse_precision @ matrix.swapaxes(-1, -2)
    w = frame.posterior_weights
    first = np.sum(w[..., None] * means, axis=1)
    outer = means[..., None] * means[..., None, :]
    second = np.sum(w[..., None, None] * (within + outer), axis=1)
    return first, second - first[..., None] * first[..., None, :]


def prefix_state_effect(p, observer, obs, k):
    """Map the existing whitened-prior coefficients to the current state shift."""
    prefix = p.prefixes[k]
    phys = (prefix.state_from_white @ observer.whiteners[k]).reshape(p.n, k+1, p.m)
    blocks = []
    for start in range(k+1):
        end = min(k+1, start+p.settings.template_duration)
        delta = obs.decoder[:, start:end] @ observer.design[:end-start]
        blocks.append(np.einsum("itm,btmj->bij", phys[:, start:end], delta))
    return np.stack(blocks, axis=1)


@dataclass
class Features:
    v0: np.ndarray
    v1a: np.ndarray
    trusted: np.ndarray
    full_nominal: np.ndarray
    log_bf: np.ndarray
    innovation_norm: np.ndarray
    innovation_step: np.ndarray
    scores: dict
    delta: np.ndarray
    delta_cov: np.ndarray
    active_mass: np.ndarray


def features(receiver, p, obs):
    """Use only the frozen receiver's legitimate inputs, with no truth labels."""
    table = (receiver.Method("V0", "raw"), receiver.Method("V1a", "structure"))
    run = receiver.run_receiver(p, obs, table)
    trust = receiver.TrustedConditioning(p)
    observer = receiver.BatchedObserver(p)
    gain = p.model.G[p.settings.receiver]
    shape = (len(obs), p.settings.horizon)
    center = np.empty(shape + (p.n,))
    normalized = np.empty(shape)
    innovation = np.empty_like(center)
    delta = np.empty(shape + (p.m,))
    delta_cov = np.empty(shape + (p.m, p.m))
    active = np.empty(shape)
    scores = {name: np.empty(shape) for name in SCORES}
    for k in range(p.settings.horizon):
        frame = observer.at(obs, k, risk=True)
        pred = trust.at(obs.trusted, k)
        deviation = obs.target[:, k] - pred.mean_y
        center[:, k] = pred.mean_x
        normalized[:, k] = np.sqrt(trust.score(deviation, k))
        innovation[:, k] = deviation @ gain.T
        delta[:, k], delta_cov[:, k] = mixture_moments(frame.transfer, frame)
        active[:, k] = frame.probability_active
        state_mean, state_cov = mixture_moments(prefix_state_effect(p, observer, obs, k), frame)
        np.testing.assert_allclose(frame.mean_null - state_mean, frame.mean_working,
                                   atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(delta[:, k], frame.delta, atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(frame.log_bf, run.log_bf[:, k], atol=2e-11, rtol=2e-11)
        assert np.min(np.linalg.eigvalsh(state_cov)) > -2e-11
        u = run.post[:, 1, k] - pred.mean_x
        length = np.linalg.norm(u, axis=-1, keepdims=True)
        unit = np.divide(u, length, out=np.zeros_like(u), where=length > 0)
        scores[SCORES[0]][:, k] = np.sum(unit * innovation[:, k], axis=-1)**2
        scores[SCORES[1]][:, k] = np.sum(unit * state_mean, axis=-1)**2
        spread = (unit[:, None] @ state_cov @ unit[..., None])[:, 0, 0]
        scores[SCORES[2]][:, k] = scores[SCORES[1]][:, k] + np.maximum(0., spread)
    return Features(run.post[:, 0], run.post[:, 1], center, run.null_mean, run.log_bf,
                    normalized, innovation, scores, delta, delta_cov, active)


def output(f, method, scale, transition, receiver):
    if method == "V0":
        return f.v0
    if method == "V1a":
        return f.v1a
    if method == "state_saturation":
        return receiver.ball_projection(f.full_nominal, f.trusted, scale)
    if method == "innovation_clipping":
        obs_change = receiver.clipping_removal(f.innovation_norm, scale)[..., None] * f.innovation_step
        correction = np.zeros_like(obs_change[:, 0])
        out = np.empty_like(f.v0)
        for k in range(out.shape[1]):
            correction = correction @ transition.T + obs_change[:, k]
            out[:, k] = f.v0[:, k] - correction
        return out
    u = f.v1a - f.trusted
    length = squared(u)
    denominator = length + scale * f.scores[method]
    gamma = np.divide(length, denominator, out=np.ones_like(length), where=denominator > 0)
    return f.trusted + gamma[..., None] * u


def nominal_cost(f, estimate, floor):
    return float(np.mean(floor + squared(estimate - f.full_nominal)))


def fit(f, method, cost_target, floor, transition, receiver):
    def diff(scale):
        return nominal_cost(f, output(f, method, scale, transition, receiver), floor) - cost_target

    if method in ("V0", "V1a"):
        return 0., "fixed"
    if method in ("state_saturation", "innovation_clipping"):
        lengths = (np.linalg.norm(f.full_nominal-f.trusted, axis=-1)
                   if method == "state_saturation" else f.innovation_norm)
        high = float(np.max(lengths))
        if diff(high) > 0:
            return high, "cost_above_target"
        if diff(0.) < 0:
            return 0., "target_unreachable"
        return float(brentq(diff, 0., high, xtol=1e-13)), "matched_on_calibration"
    if diff(0.) > 0:
        return 0., "base_cost_above_target"
    a = 0.
    for exponent in range(-24, 61):
        b = 2.**exponent
        if diff(b) >= 0:
            return float(brentq(diff, a, b, xtol=1e-13)), "matched_on_calibration"
        a = b
    return a, "target_unreachable"


def ids(config, key):
    return np.array([i for a, b in config[key] for i in range(a, b+1)])


def assemble(base, receiver, name, settings, indices):
    trusted, target, decoder = saved_sensor_inputs(base, name, settings, indices)
    return receiver.Observations(trusted, target, decoder)


def compare_model(base, receiver, common, evidence, spec, name, config):
    s = evidence.Settings.from_registration(spec)
    p = evidence.PreparedModel(common.load_model(name, base / MODEL), s)
    calib = ids(config, "healthy_calibration_inclusive_ranges")
    check = ids(config, "healthy_check_inclusive_ranges")
    pairs = ids(config, "paired_a0_inclusive_ranges")
    assert len(np.unique(np.concatenate((calib, check, pairs)))) == 128
    transition = p.model.F(s.receiver)
    gain = p.model.G[s.receiver]
    floor = np.array([np.trace(x.conditional_state_cov) for x in p.prefixes])
    f_cal = features(receiver, p, assemble(base, receiver, name, s, calib))
    test_obs = assemble(base, receiver, name, s, check)
    f_val = features(receiver, p, test_obs)
    c0 = nominal_cost(f_cal, f_cal.v0, floor)
    v0 = nominal_cost(f_val, f_val.v0, floor)
    # Healthy-only calibration is finished before opening the A0 anomaly records.
    plans = {
        budget: {m: fit(f_cal, m, c0*(1+budget), floor, transition, receiver)
                 for m in config["methods"]}
        for budget in config["healthy_cost_increase_limits"]
    }
    # Direct regression check against the old clipping receiver.
    clip_scale = plans[config["healthy_cost_increase_limits"][0]]["innovation_clipping"][0]
    old_clip = receiver.run_receiver(p, test_obs, (receiver.Method("validation", "clip", clip_scale),))
    np.testing.assert_allclose(old_clip.post[:, 0], output(f_val, "innovation_clipping", clip_scale,
                                                         transition, receiver), atol=2e-11, rtol=2e-11)
    pair_obs = assemble(base, receiver, name, s, pairs)
    f_pair = features(receiver, p, pair_obs)
    start, stop = config["event"]["onset"], config["event"]["stop_exclusive"]
    with np.load(base / OBSERVATIONS, allow_pickle=False) as bank:
        truth = bank[f"{name}__physical__x"]
    with np.load(base / V11 / f"phase1_results/{name}_step_logs.npz", allow_pickle=False) as saved:
        shift = saved["A0__decoded_difference"]
        np.testing.assert_array_equal(shift[:, :start], 0.)
        np.testing.assert_array_equal(shift[:, stop:], 0.)
        affected = receiver.Observations(pair_obs.trusted, pair_obs.target+shift, pair_obs.decoder)
        f_event = features(receiver, p, affected)
        for tag, feature in (("healthy", f_pair), ("A0", f_event)):
            np.testing.assert_allclose(feature.v0, saved[f"{tag}__post"][:, 0], atol=2e-11, rtol=2e-11)
            np.testing.assert_allclose(feature.v1a, saved[f"{tag}__post"][:, 1], atol=2e-11, rtol=2e-11)
            np.testing.assert_allclose(feature.delta_cov, saved[f"{tag}__delta_cov"], atol=2e-11, rtol=2e-11)
            np.testing.assert_allclose(feature.delta, saved[f"{tag}__delta_hat"], atol=2e-11, rtol=2e-11)
            np.testing.assert_allclose(feature.log_bf, saved[f"{tag}__log_bf"], atol=2e-11, rtol=2e-11)
    early = []
    observer = receiver.BatchedObserver(p)
    for k in range(start, start+3):
        variance = f_event.delta_cov[:, k]
        delta_error = f_event.delta[:, k] - shift[:, k]
        applied_error = delta_error @ gain.T
        observed_second_moment = (applied_error.T @ applied_error) / len(pairs)
        predicted_covariance = np.mean(gain @ variance @ gain.T, axis=0)
        # Known onset is used only to index an offline information diagnostic,
        # never to form a feature, choose a weight, or update the receiver.
        age = k-start+1
        block = (affected.decoder[:, start:k+1] @ p.template[:age]).reshape(len(pairs), age*p.m, p.n)
        whitened = observer.whiteners[k][:, start*p.m:(k+1)*p.m] @ block
        conditional_information = whitened.swapaxes(-1, -2) @ whitened
        early.append({
            "offline_known_onset_J_xi_eigenvalues_mean":np.mean(
                np.linalg.eigvalsh(conditional_information), axis=0).tolist(),
            "age":k-start+1, "average_active_posterior":float(np.mean(f_event.active_mass[:, k])),
            "decoded_distortion_squared_error":float(np.mean(squared(delta_error))),
            "working_posterior_trace_decoded":float(np.mean(np.trace(variance, axis1=-2, axis2=-1))),
            "observed_immediate_G_effect_squared":float(np.mean(squared(shift[:, k] @ gain.T))),
            "state_update_error_mean":np.mean(applied_error, axis=0).tolist(),
            "state_update_error_second_moment":observed_second_moment.tolist(),
            "average_working_posterior_covariance_state_update":predicted_covariance.tolist(),
            "state_update_squared_error_from_distortion":float(np.trace(observed_second_moment)),
            "working_posterior_trace_state_update":float(np.trace(predicted_covariance)),
        })
    results = {}
    for budget, plan in plans.items():
        rows, events, stage_losses = {}, {}, {}
        for method, (scale, status) in plan.items():
            x_cal, x_val, x_pair, x_event = (
                output(f, method, scale, transition, receiver) for f in (f_cal, f_val, f_pair, f_event)
            )
            metrics = score_estimates(x_event, x_pair, truth[pairs], f_event.log_bf,
                                      np.full(len(pairs), start), np.full(len(pairs), stop))
            early_mse = squared(x_event[:, start:start+3] - truth[pairs][:, start:start+3]).mean(axis=1)
            first3 = metrics.events.before & (metrics.events.danger < start+3)
            events[method], stage_losses[method] = first3, early_mse
            rows[method] = {
                "parameter":scale, "calibration_status":status,
                "calibration_pct_over_v0":100*(nominal_cost(f_cal, x_cal, floor)/c0-1),
                "healthy_check_pct_over_v0":100*(nominal_cost(f_val, x_val, floor)/v0-1),
                "healthy_check_truth_mse":float(np.mean(squared(x_val-truth[check]))),
                "first_three_true_mse":float(early_mse.mean()),
                "first_three_pre_alarm_count":int(np.sum(first3)),
                "first_packet_pre_alarm_count":metrics.events.counts()["first_packet_before_alarm"],
                "full_pre_alarm_count":metrics.events.counts()["before_alarm"],
                "affected_interval_truth_mse":float(metrics.support_mse.mean()),
            }
        chosen = SCORES[2]
        contrasts = {}
        tol = config["cost_matching_tolerance_percentage_points_on_healthy_check"]
        for rival in SCORES[:2] + ("state_saturation", "innovation_clipping", "V1a"):
            a, b = events[chosen], events[rival]
            health_diff = rows[chosen]["healthy_check_pct_over_v0"] - rows[rival]["healthy_check_pct_over_v0"]
            contrasts[rival] = {
                "early_mse_paired_difference":float(np.mean(stage_losses[chosen]-stage_losses[rival])),
                "early_mse_paths_lower":int(np.sum(stage_losses[chosen]<stage_losses[rival])),
                "first3_strict_pre_alarm_only_cov_and_only_rival":[int((a & ~b).sum()), int((b & ~a).sum())],
                "healthy_check_cost_difference_percentage_points":health_diff,
                "within_healthy_check_budget_and_cost_tolerance":
                    abs(health_diff) <= tol and max(rows[x]["healthy_check_pct_over_v0"]
                                                   for x in (chosen, rival)) <= 100*budget+tol,
            }
        passed = all(contrasts[x]["within_healthy_check_budget_and_cost_tolerance"]
                     and contrasts[x]["early_mse_paired_difference"] < 0
                     and int(events[chosen].sum()) <= int(events[x].sum()) for x in SCORES[:2])
        results[str(budget)] = {"methods":rows, "covariance_contrasts":contrasts, "directional_pass":passed}
    hit = f_event.log_bf >= np.log(config["event"]["bf_threshold"])
    alarm = [int(np.flatnonzero(flag)[0]) if np.any(flag) else None for flag in hit]
    return {"healthy_calibration_paths":len(calib), "healthy_check_paths":len(check),
            "affected_check_paths":len(pairs), "affected_first_alarm_by_path":dict(zip(map(str, pairs), alarm)),
            "early_pollution_diagnostics":early, "results":results}


def archived_context(base, name):
    """Report the sizes and stored outcomes for the already-existing combined rule."""
    folder = base / FRONTIER / "results"
    summary = read_json(folder / "comparison.json")["models"][name]
    index = {record["name"]:j for j, record in enumerate(summary["method_table"])}
    methods = ("V0", "V1a", "V1.4", "trusted-ledger@2%", "anchor@2%")
    with np.load(folder / f"{name}_paired.npz", allow_pickle=False) as a:
        block = (a["unknown_start__onset"] == 7) & (a["unknown_start__stop"] == 17)
        before = a["unknown_start__pre_alarm"][block]
        first = a["unknown_start__first_packet_pre_alarm"][block]
        danger = a["unknown_start__first_danger"][block]
        early = before & (danger < 10)
        active = a["unknown_start__active_mean_mse"][block]
        costs = a["healthy__health_mean_mse"]
        stats = {}
        for m in methods:
            i = index[m]
            stats[m] = {"full_pre_alarm_count":int(np.sum(before[:, i])),
                        "first_packet_pre_alarm_count":int(np.sum(first[:, i])),
                        "first_three_pre_alarm_count":int(np.sum(early[:, i])),
                        "active_mse":float(np.mean(active[:, i])),
                        "healthy_sample_increase_pct":float(100*(np.mean(costs[:, i])/np.mean(costs[:, 0])-1))}
    return {"stored_fixed_onset_affected_paths":int(block.sum()), "stored_healthy_paths":len(costs),
            "methods":stats}


def main():
    config = read_json(HERE / "registration.json")
    assert config["event"]["bf_threshold"] == 20 and config["event"]["danger_squared"] == .1
    frozen = verify_inputs(BASE)
    helper_lock = verify_local_replay_helpers(BASE)
    receiver, common, evidence = load_receiver_modules(BASE)
    spec = read_json(BASE / V1 / "registration.json")
    models = {name: compare_model(BASE, receiver, common, evidence, spec, name, config)
              for name in config["models"]}
    historical = {name: archived_context(BASE, name) for name in config["models"]}
    assert frozen == verify_inputs(BASE)
    assert helper_lock == verify_local_replay_helpers(BASE)
    registry_sha = sha256((HERE / "registration.json").read_bytes()).hexdigest()
    result = {"registration_sha256":registry_sha, "protected_input_files":len(set(frozen["files"]) | set(helper_lock["files"])),
              "new_synthetic_paths":0, "models":models, "frozen_existing_fusion_archives":historical,
              "any_common_budget_directional_pass":
                  any(all(m["results"][str(b)]["directional_pass"] for m in models.values())
                      for b in config["healthy_cost_increase_limits"])}
    (HERE / "replay_results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                            encoding="utf-8", newline="\n")
    for name, runs in models.items():
        for budget, row in runs["results"].items():
            print(name, budget, "directional pass:", row["directional_pass"])
            for method, q in row["methods"].items():
                print(f"  {method:30s} health {q['healthy_check_pct_over_v0']:+.4f}%  "
                      f"early MSE {q['first_three_true_mse']:.6f}  early pre-alarm {q['first_three_pre_alarm_count']}")


if __name__ == "__main__":
    main()
