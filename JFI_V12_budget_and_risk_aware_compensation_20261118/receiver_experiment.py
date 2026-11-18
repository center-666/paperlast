"""Closed-loop receiver evaluation on immutable disturbance replay records."""
import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import binomtest, t as student

from budget_policy import CarryBudgetPolicy
from shared import (DEFAULT_BASELINE, HERE, V11, digest, load_json, load_prepared,
                    setup, verify_registration, write_json)
from analytical_weight import TailBudgetPolicy
from run_validation import Episode, gain_on_nominal_grid
from validate_v11 import paired_packets, replay_bank
from weight_experiment import (CommonGPath, OnlineEvidence, choose_soft_weight, first_flags,
                               make_parity, paired_risk)

METHODS = ("V0", "V1a", "V1b", "V1.1", "V1.2a", "V1.2a-no-cap",
           "GLS-known", "GLS-scan", "GLRT-reject", "shrink-G")
FIXED = METHODS[:6]
DEV_NAMES = ("V0", "V1a", "V1b-original", "V1.1", "V1b-unconstrained")
WEIGHT_FIELDS = ("alpha", "proposal", "cap", "cap_binding", "opening", "allocation",
                 "available", "nominal_charge", "balance", "accepted", "full_alpha_optimal")


def run_episode(prepared, common, spec, episode, onset, tail_policy, scale, parity):
    p, s = prepared, prepared.settings
    paths = [CommonGPath(p, common.common_step, scale if name == "shrink-G" else 1)
             for name in METHODS]
    evidence_path = OnlineEvidence(p, common.common_step)
    carry = CarryBudgetPolicy(p)
    shape = (len(paths), s.horizon)
    data = {key: np.zeros(shape) for key in WEIGHT_FIELDS}
    for key, dim in (("post", p.n), ("preview", p.n), ("correction", p.m)):
        data[key] = np.empty(shape + (dim,))
    data["native_alarm"] = np.zeros(shape, dtype=bool)
    data["common_alarm"] = np.zeros(s.horizon, dtype=bool)
    data["log_bf"] = np.empty(s.horizon)
    data["p_active"] = np.empty(s.horizon)
    data["p_onset"] = np.empty((s.horizon, s.horizon+1))
    data["posterior_delta"] = np.empty((s.horizon, p.m))
    data["conditional_null_mse"] = np.empty(shape)
    data["state"] = episode.states
    order, z, chol, threshold = parity
    hold = 0
    gain = p.model.G[s.receiver]
    observables = zip(episode.local, episode.healthy, episode.attacked_target, episode.decoders)
    for k, (local, healthy, target, decoder) in enumerate(observables):
        ev = evidence_path.observe(local, healthy, target, decoder)
        hint = ev.delta
        d = gain @ hint
        scale2 = float(d @ d)
        data["common_alarm"][k] = ev.mixture_alarm
        data["log_bf"][k] = ev.log_event_bf
        data["p_onset"][k] = ev.weights
        data["p_active"][k] = ev.weights[max(0, k+1-s.template_duration)+1:k+2].sum()
        data["posterior_delta"][k] = hint
        parity_alarm = False
        if k >= order:
            residual = z @ episode.attacked_target[k-order:k+1].reshape(-1)
            white = solve_triangular(chol, residual, lower=True)
            parity_alarm = white @ white > threshold
        if ev.matched_alarm:
            hold = spec["reject_hold_updates"]
        blocked = hold > 0
        hold = max(0, hold-1)

        for j, (name, path) in enumerate(zip(METHODS, paths)):
            preview = path.preview(local, healthy, target)
            scalar = choose_soft_weight(preview, ev, gain, s)
            correction = np.zeros(p.m)
            alpha, cap, accepted = 0., 1., True
            clipped = False
            alarm = ev.mixture_alarm
            proposal = scalar.posterior_mean_weight
            if name == "V1a":
                alpha = 1.
            elif name == "V1b":
                alpha, cap = scalar.alpha, scalar.nominal_weight_limit
                proposal = scalar.uncertainty_limited_weight
                clipped = cap < proposal - 1e-12
            elif name == "V1.1":
                decision = tail_policy(k, preview, ev)
                alpha, cap, clipped = decision.alpha, decision.cap, decision.cap_binds
                data["allocation"][j, k] = data["available"][j, k] = decision.tail_nominal_budget
            elif name == "V1.2a":
                budget = carry.step(k, preview, ev)
                alpha, cap, clipped = budget.alpha, budget.cap, budget.clipped_by_budget
                for key, value in (("opening", budget.credit_before), ("allocation", budget.allocation),
                                   ("available", budget.available), ("balance", budget.credit_after)):
                    data[key][j, k] = value
            elif name == "V1.2a-no-cap":
                alpha = proposal
            elif name in ("V0", "shrink-G"):
                alarm = parity_alarm
            else:
                alarm = ev.matched_alarm
                if name == "GLRT-reject":
                    accepted = not blocked
                if name == "GLS-scan" and ev.matched_alarm:
                    alpha = 1.
                    correction = ev.fits[ev.matched_start].gls_delta
                if name == "GLS-known" and onset <= k < onset+s.template_duration:
                    alpha = 1.
                    correction = ev.fits[onset].gls_delta
            if name in FIXED:
                correction = alpha*hint
                nominal = preview-ev.nominal_state_mean
                sd = tail_policy.suffix[k] @ d
                data["nominal_charge"][j, k] = alpha*(alpha*(sd @ d)-2*(sd @ nominal))
            data["full_alpha_optimal"][j, k] = (
                scale2 <= 1e-24 or d @ (preview-ev.mixture_state_mean) >= scale2
            )
            for field, value in (("alpha", alpha), ("proposal", proposal), ("cap", cap),
                                 ("cap_binding", clipped), ("accepted", accepted)):
                data[field][j, k] = value
            data["native_alarm"][j, k] = alarm
            data["preview"][j, k] = preview
            data["correction"][j, k] = correction
            updated = path.update(local, healthy, target-correction, accept_target=accepted)
            data["post"][j, k] = updated
            error_mean = updated-ev.nominal_state_mean
            data["conditional_null_mse"][j, k] = error_mean @ error_mean + np.trace(ev.nominal_state_cov)
            if name in FIXED:
                np.testing.assert_allclose(updated, preview-alpha*d, atol=5e-12)
    return data


def stack(episodes, onsets, *args):
    p, common, spec, scale, parity = args
    policy = TailBudgetPolicy(p)
    seq = [run_episode(p, common, spec, ep, onset, policy, scale, parity)
           for ep, onset in zip(episodes, onsets)]
    return {key: np.stack([row[key] for row in seq]) for key in seq[0]}


def exhaust_ledger(data, p):
    index = METHODS.index("V1.2a")
    f = p.model.F(p.settings.receiver)
    cost = data["nominal_charge"][:, index]
    quota = data["allocation"][:, index]
    bal = data["balance"][:, index]
    opening = data["opening"][:, index]
    np.testing.assert_allclose(opening[:, 1:], bal[:, :-1], atol=1e-13)
    np.testing.assert_allclose(opening[:, 0], 0., atol=0.)
    np.testing.assert_allclose(bal, np.cumsum(quota-cost, axis=-1), atol=2e-12)
    assert np.all(bal >= 0) and np.max(cost-data["available"][:, index]) < 2e-12
    post = data["post"]
    error0 = post[:, 0]-data["state"]
    applied = post[:, index]-data["preview"][:, index]
    prior_effect = data["preview"][:, index]-post[:, 0]
    v0 = np.sum(error0**2, axis=(1, 2))
    actual = np.sum((post[:, index]-data["state"])**2, axis=(1, 2))-v0
    check = np.zeros(len(post))
    for k in range(p.settings.horizon):
        own, old = applied[:, k].copy(), prior_effect[:, k].copy()
        for h in range(k, p.settings.horizon):
            check += np.sum(own**2 + 2*own*(error0[:, h]+old), axis=-1)
            own, old = own @ f.T, old @ f.T
    np.testing.assert_allclose(actual, check, atol=5e-11)
    return {"maximum_conditional_charge_excess": float(np.max(cost-data["available"][:, index])),
            "maximum_path_telescope_error": float(np.max(np.abs(actual-check))),
            "balance_before_event_mean": None,
            "ending_credit_quantiles": np.quantile(bal[:, -1], [.1, .5, .9]).tolist()}


def mean_ci(v, independent):
    mean = float(np.mean(v))
    if not independent:
        return {"mean": mean, "interval95_t": None}
    radius = float(student.ppf(.975, len(v)-1)*np.std(v, ddof=1)/np.sqrt(len(v)))
    return {"mean": mean, "interval95_t": [mean-radius, mean+radius]}


def paired_event(new, old, independent):
    out = paired_risk(new, old, independent)
    worse, better = out["worse"], out["better"]
    out["conditional_exact_McNemar_two_sided"] = float(binomtest(worse, worse+better, .5).pvalue) if worse+better else 1.
    out["conditional_exact_McNemar_greater"] = float(binomtest(worse, worse+better, .5, alternative="greater").pvalue) if worse+better else 1.
    return out


def assess(data, clean, onsets, spec, independent, attacked):
    n, _, horizon = data["alpha"].shape
    rows = np.arange(n)
    onsets = np.asarray(onsets)
    stop = onsets+spec["template_duration"]
    time = np.arange(horizon)
    active = (time >= onsets[:, None]) & (time < stop[:, None])
    tail = time >= stop[:, None]
    error = data["post"]-data["state"][:, None]
    healthy = clean["post"]-data["state"][:, None]
    b = data["post"]-clean["post"]
    np.testing.assert_allclose(b*(time[None, None, :] < onsets[:, None, None])[..., None],
                               0., atol=1e-12)
    power = np.sum(b*b, axis=-1)
    risk = power >= spec["danger_squared"]
    danger = first_flags(risk)
    alarm = first_flags(data["common_alarm"])
    native = first_flags(data["native_alarm"])
    pre = danger < alarm[:, None]
    first_packet = risk[rows, :, onsets]
    pred = data["p_onset"][rows, stop-1, 1:]
    still_live = ((time >= onsets[:, None]) & (time < stop[:, None]))
    pred_active = np.where(still_live, pred, 0.)
    start_wrong = ((np.argmax(pred_active, axis=-1) != onsets) |
                   (1-pred_active.sum(axis=-1) >= pred_active.max(axis=-1)))
    loss = np.sum(error**2, axis=-1)
    clean_loss = np.sum(healthy**2, axis=-1).mean(axis=-1)
    support_mse = np.sum(loss*active[:, None, :], axis=-1)/active.sum(axis=-1)[:, None]
    tail_mse = np.sum(loss*tail[:, None, :], axis=-1)/tail.sum(axis=-1)[:, None]
    np.testing.assert_allclose(loss, np.sum(healthy**2, axis=-1)+power+2*np.sum(healthy*b, axis=-1), atol=2e-11)
    data.update(first_danger=danger, first_alarm_common=alarm, first_alarm_native=native,
                peak_paired_displacement_squared=power.max(axis=-1), displacement_squared=power,
                support_mse=support_mse, tail_mse=tail_mse, healthy_mse=clean_loss,
                onset=onsets, before_alarm=pre, first_packet_danger=first_packet)
    summary = {}
    for j, name in enumerate(METHODS):
        span = active if attacked else np.ones_like(active)
        summary[name] = {
            "healthy_mse": mean_ci(clean_loss[:, j], independent),
            "healthy_relative_vs_V0": float(clean_loss[:, j].mean()/clean_loss[:, 0].mean()-1),
            "healthy_limit_gap_vs_1p02_V0": mean_ci(clean_loss[:, j]-1.02*clean_loss[:, 0], independent),
            "support_mse": mean_ci(support_mse[:, j], independent),
            "tail_mse": mean_ci(tail_mse[:, j], independent),
            "peak_displacement_squared": mean_ci(power[:, j].max(axis=-1), independent),
            "danger_before_common": int(pre[:, j].sum()),
            "danger_before_native": int(np.sum(danger[:, j] < native[:, j])),
            "nominal_native_alarm_count": int(np.sum(native[:, j] < horizon)) if not attacked else None,
            "first_packet_danger": int(first_packet[:, j].sum()),
            "first_packet_alpha": float(data["alpha"][rows, j, onsets].mean()),
            "first_packet_applied_correction_norm": float(np.mean(np.linalg.norm(
                data["correction"][rows, j, onsets], axis=-1))),
            "mean_alpha": float(data["alpha"][:, j][span].mean()),
            "weight_quantiles": np.quantile(data["alpha"][:, j][span], [.1, .5, .9]).tolist(),
            "cap_binding_fraction": float(data["cap_binding"][:, j][span].mean()),
            "full_alpha_optimal_in_working_conditional_mse": float(data["full_alpha_optimal"][:, j][span].mean()),
            "vs": {
                peer: {"paired_support_difference": mean_ci(support_mse[:, j]-support_mse[:, METHODS.index(peer)], independent),
                       "paired_tail_difference": mean_ci(tail_mse[:, j]-tail_mse[:, METHODS.index(peer)], independent),
                       "paired_before_common": paired_event(pre[:, j], pre[:, METHODS.index(peer)], independent),
                       "paired_first_packet": paired_event(first_packet[:, j], first_packet[:, METHODS.index(peer)], independent)}
                for peer in ("V1a", "V1.1")}
        }
    return {"n": n, "common_nominal_alarm_count": int(np.sum(alarm<horizon)) if not attacked else None,
            "first_alarm_in_support": int(np.sum((alarm>=onsets) & (alarm<stop))) if attacked else None,
            "first_packet_active_probability_mean": float(data["p_active"][rows, onsets].mean()),
            "wrong_onset_or_no_active_at_event_end": int(start_wrong.sum()) if attacked else None,
            "policies": summary}


def fresh_cases(baseline, p, spec, common, registered):
    """Reuse packet replay, adding fixed timing and measurement-tolerance cases."""
    name, s = p.model.name, p.settings
    sample = registered["samples"][name]
    shim = {
        "profiles": registered["profiles"][:5], "archive_pool": registered["archive_pool"],
        "archived_profiles": registered["archived_profiles"], "paths_per_model": registered["paths_per_model"],
        "samples": {name: dict(sample, archive_draw_seed=sample["profile_seed"])}}
    ordinary, bank = replay_bank(baseline, name, shim)
    cases, archive_indices = paired_packets(spec, shim, common, p.model, ordinary, bank)
    start = spec["evaluation_onset"]
    lengths = s.template_duration
    onsets = {case: np.full(len(rows), start, dtype=int) for case, rows in cases.items()}
    rng = np.random.default_rng(np.random.SeedSequence([sample["profile_seed"], 1]))
    drawn = rng.choice(registered["possible_event_starts"], size=len(cases["healthy"]))
    onsets["A0_onset_replay"] = drawn
    onsets["A0_replay_tolerance"] = onsets["A0"]
    cases["A0_onset_replay"] = []
    cases["A0_replay_tolerance"] = []
    for clean, variant, onset in zip(cases["healthy"], cases["A0"], drawn):
        fixed = np.zeros_like(ordinary)
        fixed[onset:onset+lengths] = ordinary[start:start+lengths]
        replay = (clean.decoders @ fixed[..., None])[..., 0]
        cases["A0_onset_replay"].append(Episode(clean.states, clean.local, clean.healthy, clean.clean_target,
                                               clean.clean_target+replay, clean.decoders, replay, variant.energy))
        tolerance = variant.decoded_difference.copy()
        tolerance[start+registered["tolerance_index"]] *= registered["tolerance_factor"]
        cases["A0_replay_tolerance"].append(Episode(clean.states, clean.local, clean.healthy, clean.clean_target,
                                                   clean.clean_target+tolerance, clean.decoders, tolerance,
                                                   variant.energy))
    return cases, onsets, archive_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--phase", choices=("development", "validation"), required=True)
    args = parser.parse_args()
    registration = load_json(HERE / "registration.json")
    verify_registration(registration, args.baseline)
    common, settings, spec, _ = setup(args.baseline)
    previous = load_json(V11 / "phase2_registration.json")
    independent = args.phase == "validation"
    out = HERE / "results" / args.phase
    out.mkdir(parents=True, exist_ok=True)
    if independent:
        marker = HERE / "results/development/summary.json"
        if not marker.is_file() or load_json(marker)["registration_sha256"] != digest(HERE / "registration.json"):
            raise ValueError("Complete the archived 48-path regression under the current registration first.")
    summary = {}
    for model in registration["samples"]:
        prepared = load_prepared(args.baseline, model, common, settings)
        if independent:
            cases, onsets, archive = fresh_cases(args.baseline, prepared, spec, common, registration)
        else:
            ordinary, saved = replay_bank(args.baseline, model, previous)
            cases, archive = paired_packets(spec, previous, common, prepared.model, ordinary, saved)
            onsets = {p: [spec["evaluation_onset"]]*len(rows) for p, rows in cases.items()}
        scale = gain_on_nominal_grid(prepared, spec)
        parity = make_parity(common, prepared, spec)
        trial = {}
        trace = {}
        old = np.load(V11 / f"phase2_results/{model}_step_logs.npz", allow_pickle=False) if not independent else None
        for name, episodes in cases.items():
            current = stack(episodes, onsets[name], prepared, common, spec, scale, parity)
            trace[name] = current
            if old is not None:
                for j, key in zip((0, 1, 2, 3, 5), DEV_NAMES):
                    index = previous["policies"].index(key)
                    np.testing.assert_allclose(current["post"][:, j], old[f"{name}__post"][:, index], atol=2e-11, rtol=2e-11)
                    np.testing.assert_allclose(current["alpha"][:, j], old[f"{name}__alpha"][:, index], atol=2e-11)
                np.testing.assert_array_equal(current["common_alarm"], old[f"{name}__common_alarm"])
                np.testing.assert_allclose(current["posterior_delta"], old[f"{name}__delta_hat"], atol=2e-12)
            paired_clean = trace["healthy"]
            if not np.array_equal(onsets[name], onsets["healthy"]):
                paired_clean = stack(cases["healthy"], onsets[name], prepared, common, spec, scale, parity)
                oracle = METHODS.index("GLS-known")
                current["paired_healthy_oracle_post"] = paired_clean["post"][:, oracle]
                other = np.arange(len(METHODS)) != oracle
                np.testing.assert_array_equal(paired_clean["post"][:, other], trace["healthy"]["post"][:, other])
            trial[name] = assess(current, paired_clean, onsets[name], spec, independent, name != "healthy")
            ledger = exhaust_ledger(current, prepared)
            ledger["balance_before_event_mean"] = float(current["opening"][
                np.arange(len(episodes)), METHODS.index("V1.2a"), onsets[name]].mean())
            trial[name]["V12a_ledger"] = ledger
            trial[name]["archive_indices"] = archive.get(name)
            trial[name]["coded_replay_energy_upper"] = max(e.energy for e in episodes)
            print(args.phase, model, name, "support means:",
                  [round(trial[name]["policies"][m]["support_mse"]["mean"], 5) for m in METHODS[:6]], flush=True)
        if old is not None:
            old.close()
        summary[model] = {"profiles": trial, "fixed_neighbor_gain_scale": scale}
        np.savez_compressed(out / f"{model}_paths.npz",
                            **{f"{case}__{key}": value for case, data in trace.items() for key, value in data.items()})
    write_json(out / "summary.json", {"phase": args.phase, "methods": METHODS,
               "registration_sha256": digest(HERE / "registration.json"), "models": summary})


if __name__ == "__main__":
    main()
