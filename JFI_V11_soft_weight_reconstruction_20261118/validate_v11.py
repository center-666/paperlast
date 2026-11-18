"""Frozen V1.1 evaluation on fresh paired receiver inputs plus archived distortion replay."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy

from analytical_weight import TailBudgetPolicy
from weight_experiment import (BASE_POLICIES, DEFAULT_BASELINE, HERE, REFERENCE, PreparedModel,
                               archived_cases, load_json, make_parity, save_traces, setup, stack_episodes,
                               summarize, write_json)
from run_validation import A1, SWITCH, Episode

POLICIES = BASE_POLICIES + ("V1.1",)


def replay_bank(baseline, name, registered):
    archive = load_json(baseline / A1 / "registration.json")
    rows = [archive["validation_indices"].index(i) for i in registered["archive_pool"]]
    selection = load_json(baseline / A1 / "selection_before_validation.json")["selection"][name]
    with np.load(baseline / A1 / "validation_arrays.npz", allow_pickle=False) as bank:
        perturbations = {}
        for profile, selector in registered["archived_profiles"].items():
            key = selection[selector].replace("/", "_")
            perturbations[profile] = bank[f"{name}__{key}__a"][rows]
    with np.load(baseline / SWITCH / "screen_arrays.npz", allow_pickle=False) as bank:
        ordinary = bank[f"{name}__original_Gaussian__ordinary__attack"]
    return ordinary, perturbations


def paired_packets(spec, registered, common, model, ordinary, archived):
    seeds = registered["samples"][model.name]
    horizon = spec["horizon"]
    start = spec["evaluation_onset"]
    stop = start + spec["template_duration"]
    receiver, target, healthy = (spec[name] for name in ("receiver", "target", "healthy"))
    pool = registered["archive_pool"]
    rng = np.random.default_rng(seeds["archive_draw_seed"])
    n = registered["paths_per_model"]
    draws = {name: rng.integers(len(pool), size=n) for name in archived}
    episodes = {name: [] for name in registered["profiles"]}
    encoded_zero = np.zeros_like(ordinary)
    for i, (physical_seed, watermark_seed) in enumerate(zip(seeds["physical"], seeds["watermark"])):
        realization = common.draw_physical_path(model, horizon, physical_seed)
        y = realization["y"]
        matrix, _ = common.original_watermarks(model, horizon, watermark_seed,
                                                lam=spec["watermark"]["lambda"], q=spec["watermark"]["q"])
        decoder = matrix[:, target]
        clean = y[:, target]
        for profile in episodes:
            if profile == "healthy":
                encoded = encoded_zero
            elif profile == "A0":
                encoded = ordinary
            else:
                encoded = archived[profile][draws[profile][i]]
            energy = float(np.sum(encoded**2))
            if energy > spec["coded_energy_reference"] + 1e-9:
                raise ValueError("Receiver replay record out of scope for the frozen budget.")
            if np.max(np.abs(encoded[:start])) > 1e-12 or np.max(np.abs(encoded[stop:])) > 1e-12:
                raise ValueError("Receiver replay record out of scope for the frozen support.")
            decoded_difference = (decoder @ encoded[..., None])[..., 0]
            # Normal packets decode identically to y; perturbations are fixed replay inputs.
            # No input selection uses new observations, receiver states, or detector results.
            episodes[profile].append(Episode(realization["x"], y[:, receiver], y[:, healthy],
                                             clean, clean + decoded_difference, decoder,
                                             decoded_difference, energy))
    archive_indices = {key: [pool[i] for i in indices] for key, indices in draws.items()}
    return episodes, archive_indices


def check_tail_budget(traces, prepared):
    j = POLICIES.index("V1.1")
    increment = traces["tail_nominal_increment"][:, j]
    budget = traces["tail_nominal_budget"][:, j]
    if np.max(increment - budget) > 2e-12:
        raise AssertionError("A registered conditional nominal budget was exceeded.")
    post, preview = traces["post"][:, j], traces["preview"][:, j]
    v0_error = traces["post"][:, 0] - traces["state_truth"]
    actual = np.sum((post - traces["state_truth"])**2 - v0_error**2, axis=(1, 2))
    evaluated = np.zeros(len(post))
    f = prepared.model.F(prepared.settings.receiver)
    for k in range(prepared.settings.horizon):
        already_applied = preview[:, k] - traces["post"][:, 0, k]
        applied_now = post[:, k] - preview[:, k]
        for t in range(k, prepared.settings.horizon):
            error_without_current = v0_error[:, t] + already_applied
            evaluated += np.sum(applied_now**2 + 2*applied_now*error_without_current, axis=1)
            already_applied = already_applied @ f.T
            applied_now = applied_now @ f.T
    np.testing.assert_allclose(actual, evaluated, rtol=1e-10, atol=1e-10)
    return {
        "largest_conditional_excess": float(np.max(increment - budget)),
        "maximum_realized_telescoping_error": float(np.max(np.abs(actual - evaluated))),
        "mean_conditional_increment_per_update": float(increment.mean()),
        "mean_realized_increment_per_update_vs_V0": float(actual.mean()/prepared.settings.horizon),
        "mean_allocated_budget_per_update": float(budget.mean()),
    }


def verify_protocol(registered, baseline, phase1):
    for name, expected in registered["inputs"].items():
        if sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Registered input changed: {name}")
    if sha256((baseline / SWITCH / "sampling_seed_map.json").read_bytes()).hexdigest() != registered["old_seed_map_sha256"]:
        raise ValueError("The comparison seed map changed.")
    if tuple(registered["policies"]) != POLICIES:
        raise ValueError("Frozen policy ordering changed.")
    if set(registered["archive_pool"]) & set(phase1["archive_source_indices"]):
        raise ValueError("Phase 1 and held-out distortion archives overlap.")
    old = load_json(baseline / SWITCH / "sampling_seed_map.json")
    old_seeds = {seed for group in old.values() for key in ("physical", "watermark") for seed in group[key]}
    seeds = []
    for group in registered["samples"].values():
        for key in ("physical", "watermark"):
            if len(group[key]) != registered["paths_per_model"]:
                raise ValueError("Unexpected number of seeds.")
            seeds.extend(group[key])
        seeds.append(group["archive_draw_seed"])
    if len(set(seeds)) != len(seeds) or set(seeds) & old_seeds:
        raise ValueError("Validation seeds are not held out.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=HERE / "phase2_results")
    args = parser.parse_args()
    registered = load_json(HERE / "phase2_registration.json")
    common, settings, spec, phase1 = setup(args.baseline)
    verify_protocol(registered, args.baseline, phase1)
    args.out.mkdir(parents=True, exist_ok=True)
    reports, matched_reports = {}, {}
    for name in registered["samples"]:
        model = common.load_model(name, args.baseline / REFERENCE / "inputs" / "five_node_aggregate_model.json")
        prepared = PreparedModel(model, settings)
        policy = TailBudgetPolicy(prepared)
        parity = make_parity(common, prepared, spec)
        ordinary, archived = replay_bank(args.baseline, name, registered)
        cases, indices = paired_packets(spec, registered, common, model, ordinary, archived)
        data, model_report = {}, {}
        for profile, episodes in cases.items():
            traces = stack_episodes(episodes, prepared, common, parity, POLICIES, policy)
            data[profile] = traces
            row = summarize(traces, data["healthy"], spec, POLICIES,
                            independent=True, attacked=profile != "healthy")
            row["V1.1_budget_check"] = check_tail_budget(traces, prepared)
            row["archive_source_indices"] = indices.get(profile)
            row["coded_energy_range"] = [min(ep.energy for ep in episodes), max(ep.energy for ep in episodes)]
            model_report[profile] = row
            result = row["policies"]["V1.1"]
            print(name, profile, f"V1.1 MSE {result['support_mse']:.6f}, "
                  f"danger before alarm {result['danger_before_common_stat_alarm']}/{len(episodes)}", flush=True)
        save_traces(args.out, name, data)
        reports[name] = model_report

        # Historical joint records have their original physical/decoder pairing.
        archived_spec = dict(spec, samples={"physical_indices": registered["archive_pool"]})
        historical, _ = archived_cases(args.baseline, common, prepared, archived_spec)
        matched_data, matched_row = {}, {}
        for profile in registered["matched_archive_profiles"]:
            traces = stack_episodes(historical[profile], prepared, common, parity, POLICIES, policy)
            matched_data[profile] = traces
            row = summarize(traces, matched_data["healthy"], spec, POLICIES,
                            attacked=profile != "healthy")
            row["V1.1_budget_check"] = check_tail_budget(traces, prepared)
            matched_row[profile] = row
        save_traces(args.out, f"{name}_matched_archive", matched_data)
        matched_reports[name] = matched_row
    write_json(args.out / "heldout.json", {
        "phase": "independent physical and watermark inputs, conditional on the frozen replay bank",
        "policies": POLICIES,
        "registration_sha256": sha256((HERE / "phase2_registration.json").read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "models": reports,
        "matched_historical_A1_models": matched_reports,
    })


if __name__ == "__main__":
    main()
