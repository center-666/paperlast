"""Run the weight-only receiver comparison on already archived test inputs.

The original V1 likelihoods and common-G update are imported unmodified.
"""
import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import beta, chi2, t as student

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "JFI_reconstruction_attack_aware_soft_accommodation_V1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(V1))

from accommodation import (CommonGPath, OnlineEvidence, PreparedModel, Settings,  # noqa: E402
                           choose_soft_weight)
from run_validation import DEFAULT_BASELINE, REFERENCE, archived_cases, load_json, open_baseline  # noqa: E402

import json

BASE_POLICIES = ("V0", "V1a", "V1b-original", "V1b-no-margin", "V1b-no-cap", "V1b-unconstrained")


def write_json(path, result):
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def setup(baseline):
    phase1 = load_json(HERE / "phase1_registration.json")
    for name, expected in phase1["protected_V1"].items():
        if hashlib.sha256((V1 / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen V1 source differs: {name}")
    spec = load_json(V1 / "registration.json")
    if spec["samples"]["physical_indices"] != phase1["archive_source_indices"]:
        raise ValueError("Unexpected source sample indices.")
    common = open_baseline(baseline, spec)
    settings = Settings.from_registration(spec)
    return common, settings, spec, phase1


@dataclass(frozen=True)
class Weight:
    alpha: float
    cap: float
    margin_binds: bool
    cap_binds: bool
    tail_nominal_increment: float = 0.0
    tail_nominal_budget: float = 0.0


def original_ablation(name, d):
    if name == "V0":
        return Weight(0.0, 1.0, False, False)
    if name == "V1a":
        return Weight(1.0, 1.0, False, False)
    has_margin = name in ("V1b-original", "V1b-no-cap")
    has_cap = name in ("V1b-original", "V1b-no-margin")
    proposed = d.uncertainty_limited_weight if has_margin else d.posterior_mean_weight
    cap = d.nominal_weight_limit if has_cap else 1.0
    return Weight(min(proposed, cap), cap, has_margin and proposed < min(cap, d.posterior_mean_weight) - 1e-12,
                  has_cap and cap < proposed - 1e-12)


def make_parity(common, prepared, spec):
    s, model = prepared.settings, prepared.model
    order = spec["parity_order"]
    _, z, _, theta = common.parity_model(model.A, model.H[s.target],
                                         model.Q, model.R[s.target], order)
    threshold = chi2.isf(s.event_false_alarm / (s.horizon - order), len(theta))
    return order, z, np.linalg.cholesky(theta), threshold


def simulate(prepared, common, episode, parity, policies=BASE_POLICIES, extra_policy=None):
    s, model = prepared.settings, prepared.model
    t, state_dim, m = s.horizon, prepared.n, prepared.m
    f, g = model.F(s.receiver), model.G[s.receiver]
    observer = OnlineEvidence(prepared, common.common_step)
    paths = [CommonGPath(prepared, common.common_step) for _ in policies]
    dims = (len(paths), t)
    scalar_fields = (
        "alpha", "posterior_raw_weight", "posterior_clipped_weight", "margin_weight",
        "original_cap", "applied_cap", "margin_binds", "cap_binds",
        "nominal_local_increment", "nominal_local_base", "tail_nominal_increment",
        "tail_nominal_budget", "alpha_same_packet_on_V0_history"
    )
    out = {name: np.zeros(dims) for name in scalar_fields}
    for name in ("post", "preview", "history_feedback"):
        out[name] = np.zeros((*dims, state_dim))
    out["correction"] = np.zeros((*dims, m))
    for name in ("common_alarm", "native_V0_alarm", "log_bf", "p_active", "p_null", "p_started",
                 "sigma_margin", "between_model_margin_variance_fraction", "nominal_cov_trace"):
        out[name] = np.zeros(t)
    out["onset_weights"] = np.zeros((t, t + 1))
    out["delta_hat"] = np.zeros((t, m))
    out["delta_cov"] = np.zeros((t, m, m))
    # Truth is used below only to form the saved evaluator diagnostics.
    out["state_truth"] = episode.states
    out["decoded_difference"] = episode.decoded_difference
    order, z, chol, threshold = parity

    for k in range(t):
        local, healthy, target = episode.local[k], episode.healthy[k], episode.attacked_target[k]
        ev = observer.observe(local, healthy, target, episode.decoders[k])
        out["onset_weights"][k] = ev.weights
        start_active = max(0, k + 1 - s.template_duration)
        out["p_active"][k] = ev.weights[start_active + 1:k + 2].sum()
        out["p_started"][k] = ev.weights[1:k + 2].sum()
        out["p_null"][k] = ev.weights[0]
        out["delta_hat"][k] = ev.delta
        out["delta_cov"][k] = ev.delta_cov
        out["nominal_cov_trace"][k] = np.trace(ev.nominal_state_cov)
        out["common_alarm"][k] = ev.mixture_alarm
        out["log_bf"][k] = ev.log_event_bf
        if k >= order:
            white = solve_triangular(chol, z @ episode.attacked_target[k - order:k + 1].reshape(-1), lower=True)
            out["native_V0_alarm"][k] = white @ white > threshold

        direction = g @ ev.delta
        squared = float(direction @ direction)
        variance = max(0.0, float(direction @ ev.state_mean_uncertainty @ direction))
        within = sum(w * (direction @ fit.state_shift_cov @ direction)
                     for w, fit in zip(ev.weights[1:k + 2], ev.fits))
        out["sigma_margin"][k] = np.sqrt(variance)
        if variance > 1e-24:
            out["between_model_margin_variance_fraction"][k] = np.clip(1 - within / variance, 0, 1)
        v0_preview = paths[0].preview(local, healthy, target)
        v0_decision = choose_soft_weight(v0_preview, ev, g, s)
        v0_previous = paths[0].posterior.copy()
        for j, (name, path) in enumerate(zip(policies, paths)):
            preview = path.preview(local, healthy, target)
            orig = choose_soft_weight(preview, ev, g, s)
            if name in BASE_POLICIES:
                choice = original_ablation(name, orig)
                same_packet = original_ablation(name, v0_decision)
            else:
                choice = extra_policy(k, preview, ev)
                same_packet = extra_policy(k, v0_preview, ev)
            a = choice.alpha
            if not (0 <= a <= 1):
                raise ValueError(f"Invalid receiver weight at frame {k}.")
            out["alpha"][j, k] = a
            out["posterior_raw_weight"][j, k] = direction @ (preview - ev.mixture_state_mean) / squared if squared > 1e-24 else 0
            out["posterior_clipped_weight"][j, k] = orig.posterior_mean_weight
            out["margin_weight"][j, k] = orig.uncertainty_limited_weight
            out["original_cap"][j, k] = orig.nominal_weight_limit
            out["applied_cap"][j, k] = choice.cap
            out["margin_binds"][j, k] = choice.margin_binds
            out["cap_binds"][j, k] = choice.cap_binds
            out["alpha_same_packet_on_V0_history"][j, k] = same_packet.alpha
            out["tail_nominal_increment"][j, k] = choice.tail_nominal_increment
            out["tail_nominal_budget"][j, k] = choice.tail_nominal_budget
            nominal_increment = a*a*squared - 2*a*(direction @ (preview - ev.nominal_state_mean))
            out["nominal_local_increment"][j, k] = nominal_increment
            out["nominal_local_base"][j, k] = orig.risk_without_correction
            if name in ("V1b-original", "V1b-no-margin"):
                assert nominal_increment <= s.nominal_one_step_allowance * orig.risk_without_correction + 1e-11
            out["preview"][j, k] = preview
            out["history_feedback"][j, k] = preview - v0_preview
            # V0 was already updated in this loop, so retain its entering state above.
            entering = v0_previous if j == 0 else path.posterior
            np.testing.assert_allclose(out["history_feedback"][j, k], f @ (entering-v0_previous),
                                       rtol=1e-10, atol=8e-12)
            out["correction"][j, k] = a * ev.delta
            out["post"][j, k] = path.update(local, healthy, target - out["correction"][j, k])
            np.testing.assert_allclose(out["post"][j, k], preview-a*direction, rtol=1e-11, atol=5e-12)
    return out


def stack_episodes(episodes, prepared, common, parity, policies, extra_policy=None):
    records = [simulate(prepared, common, ep, parity, policies, extra_policy) for ep in episodes]
    return {field: np.stack([row[field] for row in records]) for field in records[0]}


def first_flags(flags):
    t = flags.shape[-1]
    return np.where(flags, np.arange(t), t).min(axis=-1)


def paired_mean(diff, independent):
    diff = np.asarray(diff)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    return {
        "mean": mean, "standard_error": se, "per_path": diff.tolist(),
        "one_sided_95_upper_t": float(mean + student.ppf(.95, len(diff)-1)*se) if independent else None
    }


def paired_risk(test, reference, independent):
    n = len(test)
    worse = int(np.sum(test & ~reference))
    better = int(np.sum(~test & reference))
    upper = None
    if independent:
        hi = float(beta.ppf(.975, worse+1, n-worse)) if worse < n else 1.0
        lo = float(beta.ppf(.025, better, n-better+1)) if better else 0.0
        upper = hi - lo
    return {"worse": worse, "better": better, "difference": (worse-better)/n, "one_sided_95_upper_exact": upper}


def summarize(traces, clean, spec, policies, independent=False, attacked=True):
    start = spec["evaluation_onset"]
    stop = start + spec["template_duration"]
    last = stop - 1
    horizon = spec["horizon"]
    post, normal = traces["post"], clean["post"]
    state = traces["state_truth"][:, None, :, :]
    sq = np.sum((post - state)**2, axis=-1)
    healthy_sq = np.sum((normal - state)**2, axis=-1)
    drift = post - normal
    power = np.sum(drift**2, axis=-1)
    cross = 2 * np.sum((normal-state)*drift, axis=-1)
    np.testing.assert_allclose(sq, healthy_sq + power + cross, rtol=1e-11, atol=1e-11)
    alarm = first_flags(traces["common_alarm"].astype(bool))
    native = first_flags(traces["native_V0_alarm"].astype(bool))
    when_active = (alarm >= start) & (alarm < stop)
    danger = first_flags(power >= spec["danger_squared"])
    prealarm = danger < alarm[:, None]
    active_slice = slice(start, stop)
    mse = sq[:, :, active_slice].mean(axis=2)
    quiet = healthy_sq.mean(axis=2)
    after = sq[:, :, stop:].mean(axis=2)
    reference = policies.index("V1a")
    span = active_slice if attacked else slice(None)
    error = traces["delta_hat"] - traces["decoded_difference"]
    posterior_variance = np.trace(traces["delta_cov"], axis1=-2, axis2=-1)
    outputs = {}
    for j, name in enumerate(policies):
        us = traces["alpha"][:, j, span]
        final_residual = traces["decoded_difference"][:, span] - traces["correction"][:, j, span]
        age = np.arange(horizon)[None, :] < alarm[:, None]
        pre = age[:, active_slice]
        pre_sq = sq[:, j, active_slice]
        summary = {
            "healthy_mse": float(quiet[:, j].mean()),
            "healthy_mse_rel_V0": float(quiet[:, j].mean()/quiet[:, 0].mean()-1),
            "healthy_limit_contrast_to_1p02_V0": paired_mean(quiet[:, j] - 1.02*quiet[:, 0], independent),
            "support_mse": float(mse[:, j].mean()),
            "before_stat_alarm_support_mse": float(pre_sq[pre].mean()) if pre.any() else None,
            "at_or_after_stat_alarm_support_mse": float(pre_sq[~pre].mean()) if (~pre).any() else None,
            "post_stop_mse": float(after[:, j].mean()),
            "support_mse_vs_V1a": paired_mean(mse[:, j]-mse[:, reference], independent),
            "post_stop_mse_vs_V1a": paired_mean(after[:, j]-after[:, reference], independent),
            "first_packet_danger": int(np.sum(power[:, j, start] >= spec["danger_squared"])),
            "danger_before_common_stat_alarm": int(prealarm[:, j].sum()),
            "danger_before_common_stat_alarm_vs_V1a": paired_risk(prealarm[:, j], prealarm[:, reference], independent),
            "cumulative_paired_displacement_squared": float(power[:, j, start:].sum(axis=1).mean()),
            "mean_cross_on_support": float(cross[:, j, active_slice].mean()),
            "compensation_residual_mse": float(np.sum(final_residual**2, axis=2).mean()),
            "compensation_residual_signed_mean": final_residual.mean(axis=(0, 1)).tolist(),
            "weight_mean": float(us.mean()),
            "weight_quantiles_10_50_90": np.quantile(us, [.1, .5, .9]).tolist(),
            "weight_zero_fraction": float(np.mean(us <= 1e-12)),
            "weight_one_fraction": float(np.mean(us >= 1-1e-12)),
            "margin_binding_fraction": float(traces["margin_binds"][:, j, span].mean()),
            "cap_binding_fraction": float(traces["cap_binds"][:, j, span].mean()),
            "preview_history_feedback_rms": float(np.sqrt(np.sum(traces["history_feedback"][:, j, span]**2, axis=-1).mean())),
            "weight_difference_from_same_packet_on_V0_history_mean_abs": float(np.mean(np.abs(
                us-traces["alpha_same_packet_on_V0_history"][:, j, span]))),
            "danger_first_times": [int(v) if v < horizon else None for v in danger[:, j]]
        }
        outputs[name] = summary
    rows = traces["onset_weights"][:, last, 1:]
    first_active = max(0, last+1-spec["template_duration"])
    active = rows[:, first_active:last+1]
    largest = active.argmax(axis=1) + first_active
    no_active_decision = 1-active.sum(axis=1) >= active.max(axis=1)
    return {
        "n": len(alarm),
        "cohort_onset_evaluation": {
            "active_only_map_wrong_at_last": int((largest != start).sum()) if attacked else None,
            "map_with_no_active_wrong_at_last": int(((largest != start) | no_active_decision).sum()) if attacked else None,
            "no_active_selected_at_last": int(no_active_decision.sum()),
            "mean_true_onset_mass_at_last": float(rows[:, start].mean()) if attacked else None,
            "mean_active_probability_on_span": float(traces["p_active"][:, span].mean())
        },
        "posterior": {
            "decoded_disturbance_estimation_mse": float(np.sum(error[:, span]**2, axis=2).mean()),
            "decoded_disturbance_estimation_signed_mean": error[:, span].mean(axis=(0, 1)).tolist(),
            "mean_posterior_decoded_disturbance_variance_trace": float(posterior_variance[:, span].mean()),
            "mean_margin_std": float(traces["sigma_margin"][:, span].mean()),
            "mean_between_model_fraction_among_positive_variance": float(
                traces["between_model_margin_variance_fraction"][:, span][traces["sigma_margin"][:, span] > 1e-12].mean()
            ) if np.any(traces["sigma_margin"][:, span] > 1e-12) else 0.0,
        },
        "common_stat_alarm_first_times": [int(v) if v < horizon else None for v in alarm],
        "common_stat_alarms_inside_support": int(when_active.sum()) if attacked else None,
        "common_nominal_alarms": int((alarm < horizon).sum()) if not attacked else None,
        "V0_native_stat_alarm_first_times": [int(v) if v < horizon else None for v in native],
        "V0_native_danger_before_stat_alarm": int((danger[:, 0] < native).sum()),
        "policies": outputs
    }


def save_traces(out, name, cases):
    np.savez_compressed(out / f"{name}_step_logs.npz",
                        **{f"{profile}__{field}": value for profile, row in cases.items() for field, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=HERE / "phase1_results")
    args = parser.parse_args()
    common, settings, spec, phase1 = setup(args.baseline)
    if tuple(phase1["policies"]) != BASE_POLICIES:
        raise ValueError("Ablations differ from their registration.")
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    replay_checks = 0
    with np.load(V1 / "results" / "paired_paths.npz", allow_pickle=False) as saved:
        for name in phase1["models"]:
            model = common.load_model(name, args.baseline / REFERENCE / "inputs" / "five_node_aggregate_model.json")
            prepared = PreparedModel(model, settings)
            parity = make_parity(common, prepared, spec)
            episodes, _ = archived_cases(args.baseline, common, prepared, spec)
            traces = {}
            reports = {}
            for profile in phase1["phase1_profiles"]:
                paths = stack_episodes(episodes[profile], prepared, common, parity, BASE_POLICIES)
                traces[profile] = paths
                base = f"{name}__{profile}"
                for j, old in ((0, "V0"), (1, "V1a"), (2, "V1b")):
                    np.testing.assert_allclose(paths["post"][:, j], saved[f"{base}__{old}__posterior"],
                                               rtol=2e-11, atol=2e-11)
                    old_alarm = paths["native_V0_alarm"] if j == 0 else paths["common_alarm"]
                    np.testing.assert_array_equal(old_alarm, saved[f"{base}__{old}__alarms"])
                    if j:
                        np.testing.assert_allclose(paths["alpha"][:, j], saved[f"{base}__{old}__alpha"],
                                                   rtol=2e-11, atol=2e-11)
                    replay_checks += 1
                np.testing.assert_allclose(paths["onset_weights"], saved[f"{base}__probabilities"], atol=3e-13)
                np.testing.assert_allclose(paths["delta_hat"], saved[f"{base}__V1a__estimated"], atol=3e-13)
                reports[profile] = summarize(paths, traces["healthy"], spec, BASE_POLICIES,
                                             attacked=profile != "healthy")
                row = reports[profile]["policies"]
                print(name, profile, "  ".join(f"{key}={value['support_mse']:.6f}" for key, value in row.items()), flush=True)
            outputs[name] = reports
            save_traces(args.out, name, traces)
    write_json(args.out / "ablation.json", {
        "phase": "archived closed-loop ablation",
        "policies": BASE_POLICIES,
        "baseline_regression_mode_case_comparisons": replay_checks,
        "new_physical_seeds": 0,
        "models": outputs,
    })


if __name__ == "__main__":
    main()
