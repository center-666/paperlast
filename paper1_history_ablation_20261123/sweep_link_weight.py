"""Registered D0 weight sweep for the critical 35->26 complementary link."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import block_diag, solve_discrete_are
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CRITICAL = ROOT / "paper1_critical_complement_20261123"
PHYSICAL = CRITICAL / "derived_configuration.json"
sys.path.insert(0, str(CRITICAL))
sys.path.insert(0, str(HERE))

from ablation import residuals, scores_all, summary
from run_critical import critical_model, digest, make_common_draws, beta_from_common, physical


T = 150
START, STOP = 50, 110
RECEIVER, SENDER, EDGE = 26, 35, 6
THETAS = {"iid": 0.0, "negative": -1.3152369768257723}
WEIGHTS = np.round(np.linspace(0.0, 1.0, 11), 1)
DETECTORS = ("H4", "H1")
ROOT_SEED = 2026122501
CAL_N = HOLD_N = ATTACK_N = 512
TARGET_FALSE_ALARM = 0.05


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def empirical_threshold(maxima: np.ndarray) -> tuple[float, int]:
    """Split-conformal upper order statistic for a nominal 5% path FWER."""
    n = len(maxima)
    rank = int(math.ceil((n + 1) * (1.0 - TARGET_FALSE_ALARM)))
    if rank > n:
        raise ValueError("Calibration set is too small for the registered target.")
    return float(np.partition(maxima, rank - 1)[rank - 1]), rank


def exact_interval(successes: int, total: int) -> list[float]:
    low = float(beta_dist.ppf(0.025, successes, total - successes + 1)) if successes else 0.0
    high = float(beta_dist.ppf(0.975, successes + 1, total - successes)) if successes < total else 1.0
    return [low, high]


def detector_calibration(model, model_index: int) -> tuple[dict, dict, dict]:
    base = ROOT_SEED + model_index * 100_000
    calibration = physical(model, CAL_N, base + 1)
    cal_scores = scores_all(residuals(model, calibration), model.Theta)
    thresholds, maxima = {}, {}
    ranks = {}
    for name in DETECTORS:
        maxima[name] = cal_scores[name][:, 4:, :].max(axis=(1, 2))
        thresholds[name], ranks[name] = empirical_threshold(maxima[name])
    holdout = physical(model, HOLD_N, base + 2)
    hold_scores = scores_all(residuals(model, holdout), model.Theta)
    health = {}
    for name in DETECTORS:
        gates = hold_scores[name] > thresholds[name]
        alarms = gates[:, 4:, :].any(axis=(1, 2))
        health[name] = {
            "path_alarms": int(alarms.sum()), "paths": int(len(alarms)),
            "fwer": float(alarms.mean()),
            "exact_binomial_ci95": exact_interval(int(alarms.sum()), int(len(alarms))),
            "critical_link_alarm_fraction": float(gates[:, 4:, EDGE].any(axis=1).mean()),
            "threshold": thresholds[name], "calibration_rank_one_based": ranks[name],
            "calibration_maxima": maxima[name],
            "holdout_gates": gates,
            "holdout_paths": holdout,
            "holdout_scores": hold_scores[name],
        }
    return thresholds, health, {name: maxima[name] for name in DETECTORS}


def attack_paths(model, model_index: int, fixture_index: int, fixture: str):
    base = ROOT_SEED + model_index * 100_000 + 100 + fixture_index * 100
    paths = physical(model, ATTACK_N, base)
    active = (np.arange(T) >= START) & (np.arange(T) < STOP)
    if fixture == "burst":
        active &= ((np.arange(T) - START) % 10) < 6
    elif fixture != "continuous":
        raise ValueError(f"Unknown fixture {fixture!r}.")

    draws = make_common_draws(ATTACK_N, STOP - START, 7,
                              base + 1, base + 2, base + 3)
    delta_by_watermark = {}
    for name, theta in THETAS.items():
        beta = beta_from_common(draws, theta)
        delta = np.zeros((ATTACK_N, T, 7, 2), dtype=float)
        fault = np.asarray([6.0, 0.0])
        encoded = 0.5 * fault + beta[:, :, EDGE] * np.sum(beta[:, :, EDGE] * fault, axis=-1, keepdims=True)
        local_active = active[START:STOP]
        delta[:, START:STOP, EDGE] = encoded * local_active[None, :, None]
        delta_by_watermark[name] = delta
    return paths, active, delta_by_watermark, base


def design_weighted_gains(model, weight: float):
    """Build all 128 mode rows, recalculating the affected weighted DARE.

    Only node 26 receives the swept edge. Its keep mode gets a fresh weighted
    aggregate DARE. Its drop mode uses the already registered observable-block
    fallback. The other 49 receivers' exact mode designs are independent of
    this weight and are reused from the validated 4D common-G table.
    """
    K = model.K.copy()
    G = model.G.copy()
    F = model.F.copy()
    if weight == 0.0:
        for mask in range(64):
            fallback = mask | (1 << EDGE)
            K[mask, RECEIVER] = model.K[fallback, RECEIVER]
            G[mask, RECEIVER] = model.G[fallback, RECEIVER]
            F[mask, RECEIVER] = model.F[fallback, RECEIVER]
    else:
        neighbors = model.cfg["graph"]["neighbors"][RECEIVER]
        if neighbors != [16, 28, 35]:
            raise ValueError(f"Critical-link neighborhood changed: {neighbors}.")
        coefficients = np.asarray([weight if j == SENDER else 1.0 for j in neighbors])
        Hsum = sum((coeff * model.H[j] for j, coeff in zip(neighbors, coefficients)),
                   start=np.zeros((2, model.n)))
        Rsum = sum((coeff * coeff * model.R[j] for j, coeff in zip(neighbors, coefficients)),
                   start=np.zeros((2, 2)))
        C = np.vstack((model.H[RECEIVER], Hsum))
        V = block_diag(model.R[RECEIVER], Rsum)
        A, Q = model.A, model.Q
        P = solve_discrete_are(A.T, C.T, Q, V)
        L = np.linalg.solve(C @ P @ C.T + V, C @ P).T
        K_keep, G_keep = L[:, :2], L[:, 2:]
        F_keep = (np.eye(model.n) - K_keep @ model.H[RECEIVER] - G_keep @ Hsum) @ A
        if not np.all(np.isfinite(np.r_[K_keep.ravel(), G_keep.ravel(), F_keep.ravel()])):
            raise FloatingPointError(f"Non-finite weighted gain at omega={weight}.")
        for mask in range(128):
            if mask & (1 << EDGE):
                continue
            K[mask, RECEIVER] = K_keep
            G[mask, RECEIVER] = G_keep
            F[mask, RECEIVER] = F_keep
    return K, G, F


def policy_coefficients(gates: np.ndarray, weights: np.ndarray, policy: str,
                        active_fault: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return [W,B,T,7] packet weights and [W,B,T] discrete mode masks."""
    count = len(gates)
    alpha = np.ones((len(weights), count, T, 7), dtype=float)
    effective_gates = np.zeros((len(weights), count, T, 7), dtype=bool)
    if policy in ("H4_weighted", "H1_weighted"):
        effective_gates[:] = gates[None]
        alpha[:, :, :, :6] = (~gates[:, :, :6])[None]
        alpha[:, :, :, EDGE] = weights[:, None, None] * (~gates[:, :, EDGE])[None]
    elif policy == "fixed_weight":
        alpha[:, :, :, EDGE] = weights[:, None, None]
    elif policy == "known_timing":
        alpha[:, :, :, EDGE] = (~active_fault)[None, None, :]
    elif policy == "permanent_reject":
        alpha[:, :, :, EDGE] = 0.0
    else:
        raise ValueError(f"Unknown policy {policy!r}.")

    modes = np.sum(effective_gates * (1 << np.arange(7)), axis=-1).astype(np.int64)
    if policy in ("fixed_weight", "H4_weighted", "H1_weighted"):
        modes[weights == 0.0, :, :] |= 1 << EDGE
    elif policy == "known_timing":
        modes[:, :, active_fault] |= 1 << EDGE
    elif policy == "permanent_reject":
        modes |= 1 << EDGE
    return alpha, modes


def _window_contraction(F_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pathwise maximum spectral norm of all seven-step products."""
    weights, paths, ticks = F_sequence.shape[:3]
    maxima = np.zeros((weights, paths), dtype=float)
    all_contractive = np.ones((weights, paths), dtype=bool)
    eye = np.broadcast_to(np.eye(4), (weights, paths, 4, 4))
    for start in range(1, ticks - 6):
        product = eye.copy()
        for offset in range(7):
            product = F_sequence[:, :, start + offset] @ product
        norm = np.linalg.svd(product, compute_uv=False)[..., 0]
        maxima = np.maximum(maxima, norm)
        all_contractive &= norm < 1.0
    return maxima, all_contractive


def run_policy(model, paths, delta, gates, weights, policy, active_fault, gains):
    """Run the full 50-node weighted aggregate update for a weight grid."""
    K_tables, G_tables, F_tables = gains
    alpha, masks = policy_coefficients(gates, weights, policy, active_fault)
    W, B = len(weights), len(paths.x)
    adjacency = model.adjacency
    neighbor_Hsum = model.Hsum
    previous = np.zeros((W, B, 50, 4), dtype=float)
    fault_missing_sum = np.zeros((W, B), dtype=float)
    tail_missing_sum = np.zeros((W, B), dtype=float)
    network_fault_sum = np.zeros((W, B), dtype=float)
    network_post20_sum = np.zeros((W, B), dtype=float)
    forcing_sum = np.zeros((W, B), dtype=float)
    low_run = np.zeros((W, B), dtype=int)
    low_run_max = np.zeros((W, B), dtype=int)
    accepted_fraction = np.zeros((W, B), dtype=float)
    weighted_exposure = np.zeros((W, B), dtype=float)
    normal_gap_count = int(np.sum(active_fault[START:STOP] == 0))
    normal_gap_accept = np.zeros((W, B), dtype=float)
    normal_gap_weight = np.zeros((W, B), dtype=float)
    F_sequence = np.empty((W, B, T, 4, 4), dtype=float)

    for k in range(1, T):
        predicted = previous @ model.A.T
        mode = masks[:, :, k]
        K = np.stack([K_tables[w, mode[w]] for w in range(W)], axis=0)
        G = np.stack([G_tables[w, mode[w]] for w in range(W)], axis=0)
        local_prediction = np.einsum("imn,wbin->wbim", model.H, predicted, optimize=True)
        local_innovation = paths.y[:, k][None] - local_prediction

        aggregate_y = np.einsum("ij,bjm->bim", adjacency, paths.y[:, k], optimize=True)
        aggregate_H_prediction = np.einsum("imn,wbin->wbim", neighbor_Hsum, predicted, optimize=True)
        innovation = np.broadcast_to(aggregate_y[None], (W, B, 50, 2)).copy() - aggregate_H_prediction
        for edge, (receiver, sender) in enumerate(model.links):
            neighbor_prediction = np.einsum("mn,wbn->wbm", model.H[sender], predicted[:, :, receiver], optimize=True)
            clean_innovation = paths.y[:, k, sender][None] - neighbor_prediction
            coefficient = alpha[:, :, k, edge]
            innovation[:, :, receiver] += ((coefficient - 1.0)[..., None] * clean_innovation
                                           + coefficient[..., None] * delta[:, k, edge][None])

        posterior = (predicted
                     + np.einsum("wbinm,wbim->wbin", K, local_innovation, optimize=True)
                     + np.einsum("wbinm,wbim->wbin", G, innovation, optimize=True))
        truth = paths.x[:, k][None, :, None, :]
        square = (posterior - truth) ** 2
        missing_error = square[:, :, RECEIVER, 2:].sum(axis=-1)
        if START <= k < STOP:
            fault_missing_sum += missing_error
            network_fault_sum += square.sum(axis=-1).mean(axis=-1)
        if k >= STOP:
            tail_missing_sum += missing_error
        if k >= 20:
            network_post20_sum += square.sum(axis=-1).mean(axis=-1)

        critical_alpha = alpha[:, :, k, EDGE]
        critical_gate = critical_alpha <= 0.1
        low_run = np.where(critical_gate, low_run + 1, 0)
        low_run_max = np.maximum(low_run_max, low_run)
        accepted_fraction += (critical_alpha > 0.0)
        weighted_exposure += critical_alpha
        if START <= k < STOP and not active_fault[k]:
            normal_gap_accept += (critical_alpha > 0.0)
            normal_gap_weight += critical_alpha

        Gcritical = G[:, :, RECEIVER, 2:, :]
        attack_input = critical_alpha[..., None] * delta[:, k, EDGE][None]
        forcing = np.einsum("wbnm,wbm->wbn", Gcritical, attack_input, optimize=True)
        if START <= k < STOP:
            forcing_sum += (forcing ** 2).sum(axis=-1)

        F_sequence[:, :, k] = np.stack([F_tables[w, mode[w], RECEIVER] for w in range(W)])
        previous = posterior

    contraction_max, all_contract = _window_contraction(F_sequence)
    denominator_fault = STOP - START
    denominator_tail = T - STOP
    denominator_post20 = T - 20
    output = {
        "node26_missing_fault_mse": fault_missing_sum / denominator_fault,
        "node26_missing_tail_mse": tail_missing_sum / denominator_tail,
        "network_fault_mse": network_fault_sum / denominator_fault,
        "network_post20_mse": network_post20_sum / denominator_post20,
        "accepted_attack_forcing_energy": forcing_sum / denominator_fault,
        "low_weight_or_rejection_longest_run": low_run_max,
        "node26_max_seven_step_norm": contraction_max,
        "node26_all_seven_step_windows_contract": all_contract.astype(float),
        "critical_link_any_accept_fraction": accepted_fraction / (T - 1),
        "critical_link_mean_weight": weighted_exposure / (T - 1),
        "critical_link_attack_accept_fraction": np.stack([
            ((alpha[:, :, k, EDGE] > 0.0).astype(float) * bool(active_fault[k]))
            for k in range(START, STOP)
        ], axis=-1).sum(axis=-1) / max(1, int(active_fault[START:STOP].sum())),
    }
    if normal_gap_count:
        output["normal_gap_information_accept_fraction"] = normal_gap_accept / normal_gap_count
        output["normal_gap_mean_weight"] = normal_gap_weight / normal_gap_count
    return output


def run_policy_grid(model, paths, delta, gates, weights, policy, active_fault, gains):
    """Run weights one at a time to keep peak memory independent of grid size."""
    rows = []
    for weight in weights:
        weight = float(weight)
        one = np.asarray([weight], dtype=float)
        gain_index = int(np.flatnonzero(np.isclose(WEIGHTS, weight))[0])
        one_gains = tuple(table[gain_index:gain_index + 1] for table in gains)
        single = run_policy(model, paths, delta, gates, one, policy, active_fault, one_gains)
        rows.append(single)
    return {metric: np.concatenate([row[metric] for row in rows], axis=0)
            for metric in rows[0]}


def paired_summary(values: np.ndarray) -> dict:
    return summary(np.asarray(values, dtype=float))


def record_metrics(array_store: dict, key: str, values: dict) -> dict:
    report = {}
    for metric, paths in values.items():
        array_store[f"{key}__{metric}"] = paths
        report[metric] = [paired_summary(row) for row in paths]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run the registered D0 Monte Carlo.")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "D0_20261123")
    args = parser.parse_args()
    registration_path = HERE / "D0_registration.md"
    expected_config = ROOT / "paper1_common_g50_gain_acceptance_20261121" / "physical_configuration.json"
    expected_hash = "c25437077cb73ac2332160e55f6183550bb8fc5318d93bb32c183a11d661b50d"
    if digest(expected_config) != expected_hash:
        raise ValueError("Frozen common-G physical source hash mismatch.")
    cfg = json.loads(expected_config.read_text(encoding="utf-8"))
    if not args.execute:
        for label in ("unstable", "stable"):
            model, geometry, _ = critical_model(label, cfg)
            for weight in WEIGHTS:
                K, G, F = design_weighted_gains(model, float(weight))
                if weight == 1.0:
                    np.testing.assert_allclose(K, model.K, rtol=1e-8, atol=1e-9)
                    np.testing.assert_allclose(G, model.G, rtol=1e-8, atol=1e-9)
                    np.testing.assert_allclose(F, model.F, rtol=1e-8, atol=1e-9)
            print(label, "critical-link geometry", {
                "neighbors": geometry["neighbors"],
                "only_complementary_neighbor": geometry["only_complementary_neighbor"],
                "missing_block_open_loop_radius": geometry["missing_block_open_loop_radius"],
                "omega_grid": WEIGHTS.tolist(),
            })
        print("D0 dry-run passed weighted-gain construction.")
        return

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing D0 results: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary result directory already exists: {temporary}")
    temporary.mkdir(parents=True)

    source_paths = [
        Path(__file__), registration_path,
        ROOT / "paper1_history_ablation_20261123" / "ablation.py",
        CRITICAL / "run_critical.py", CRITICAL / "registration.json",
        ROOT / "paper1_partial_observability_20261123" / "run_partial50.py",
        ROOT / "paper1_sscd_commonrandom_20261122" / "watson_crn.py",
        expected_config,
    ]
    result = {
        "experiment": "D0 continuous critical-link weight feasibility sweep",
        "status": "complete",
        "root_seed": ROOT_SEED,
        "weights": WEIGHTS.tolist(),
        "watermarks": THETAS,
        "fixtures": ["continuous", "burst"],
        "paths_per_model": {"calibration": CAL_N, "healthy_holdout": HOLD_N,
                             "continuous": ATTACK_N, "burst": ATTACK_N},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "source_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in source_paths},
        "models": {},
    }
    arrays = {}

    for model_index, label in enumerate(("unstable", "stable")):
        model, geometry, _ = critical_model(label, cfg)
        gains = [design_weighted_gains(model, float(weight)) for weight in WEIGHTS]
        K_tables = np.stack([row[0] for row in gains])
        G_tables = np.stack([row[1] for row in gains])
        F_tables = np.stack([row[2] for row in gains])
        del gains
        model_result = {"geometry": {k: v for k, v in geometry.items()
                                      if k in ("neighbors", "only_complementary_neighbor",
                                               "missing_block_open_loop_radius",
                                               "keep_norm", "drop_norm")},
                        "health": {}, "attacks": {}}
        thresholds, health, calibration_arrays = detector_calibration(model, model_index)
        model_result["detector_thresholds"] = thresholds
        for detector in DETECTORS:
            arrays[f"{label}__calibration_maxima__{detector}"] = calibration_arrays[detector]
            arrays[f"{label}__healthy_gates__{detector}"] = health[detector]["holdout_gates"]
            model_result["health"][detector] = {
                key: value for key, value in health[detector].items()
                if key not in ("calibration_maxima", "holdout_gates", "holdout_paths", "holdout_scores")
            }
        healthy_paths = health["H4"]["holdout_paths"]
        healthy_h4 = health["H4"]["holdout_gates"]
        healthy_h1 = health["H1"]["holdout_gates"]
        for detector_name, detector_gates in (("H4", healthy_h4), ("H1", healthy_h1)):
            measured = run_policy_grid(model, healthy_paths, np.zeros((HOLD_N, T, 7, 2)),
                                  detector_gates, WEIGHTS, f"{detector_name}_weighted",
                                  np.zeros(T, dtype=bool), (K_tables, G_tables, F_tables))
            model_result["health"][detector_name]["weight_sweep"] = record_metrics(
                arrays, f"{label}__healthy__{detector_name}", measured)
        healthy_fixed = run_policy_grid(model, healthy_paths, np.zeros((HOLD_N, T, 7, 2)),
                                   np.zeros((HOLD_N, T, 7), bool), WEIGHTS, "fixed_weight",
                                   np.zeros(T, dtype=bool), (K_tables, G_tables, F_tables))
        model_result["health"]["all_accept_reference"] = record_metrics(
            arrays, f"{label}__healthy__all_accept_reference", healthy_fixed)
        for detector_name in DETECTORS:
            paired = []
            for weight_index in range(len(WEIGHTS)):
                gate_paths = arrays[f"{label}__healthy__{detector_name}__network_post20_mse"][weight_index]
                ref_paths = arrays[f"{label}__healthy__all_accept_reference__network_post20_mse"][len(WEIGHTS)-1]
                paired.append(paired_summary(gate_paths - ref_paths))
            model_result["health"][detector_name]["paired_healthy_mse_increment_vs_all_accept"] = paired
        del healthy_paths
        for detector_name in DETECTORS:
            health[detector_name]["holdout_paths"] = None
            health[detector_name]["holdout_scores"] = None
            health[detector_name]["holdout_gates"] = None
        del healthy_h4, healthy_h1, detector_gates, detector_name, measured, healthy_fixed
        import gc
        gc.collect()

        for fixture_index, fixture in enumerate(("continuous", "burst")):
            paths, active_fault, deltas, attack_seed = attack_paths(
                model, model_index, fixture_index, fixture)
            fixture_result = {}
            for watermark, delta in deltas.items():
                score = scores_all(residuals(model, paths, delta), model.Theta)
                gates_by_detector = {name: score[name] > thresholds[name] for name in DETECTORS}
                watermark_result = {"seed": attack_seed, "policies": {}}
                watermark_result["baseline_aliases"] = {
                    "all_accept": "fixed_weight at omega=1.0",
                    "permanent_reject": "fixed_weight at omega=0.0",
                    "H4_hard_gate": "H4_weighted at omega=1.0",
                    "H1_hard_gate": "H1_weighted at omega=1.0",
                    "known_timing_gate": "known_timing at omega=1.0",
                }
                arrays[f"{label}__{fixture}__{watermark}__attack_gates__H4"] = gates_by_detector["H4"]
                arrays[f"{label}__{fixture}__{watermark}__attack_gates__H1"] = gates_by_detector["H1"]

                fixed = run_policy_grid(model, paths, delta, np.zeros((ATTACK_N, T, 7), bool),
                                   WEIGHTS, "fixed_weight", active_fault,
                                   (K_tables, G_tables, F_tables))
                watermark_result["policies"]["fixed_weight"] = record_metrics(
                    arrays, f"{label}__{fixture}__{watermark}__fixed_weight", fixed)
                for detector_name in DETECTORS:
                    weighted = run_policy_grid(model, paths, delta, gates_by_detector[detector_name],
                                          WEIGHTS, f"{detector_name}_weighted", active_fault,
                                          (K_tables, G_tables, F_tables))
                    watermark_result["policies"][f"{detector_name}_weighted"] = record_metrics(
                        arrays, f"{label}__{fixture}__{watermark}__{detector_name}_weighted", weighted)
                    weighted_key = f"{label}__{fixture}__{watermark}__{detector_name}_weighted"
                    fixed_key = f"{label}__{fixture}__{watermark}__fixed_weight"
                    comparison = {}
                    for metric in ("node26_missing_fault_mse", "node26_missing_tail_mse",
                                   "network_fault_mse", "accepted_attack_forcing_energy"):
                        weighted_paths = arrays[f"{weighted_key}__{metric}"]
                        fixed_paths = arrays[f"{fixed_key}__{metric}"][len(WEIGHTS)-1]
                        hard_paths = weighted_paths[len(WEIGHTS)-1]
                        comparison[f"{metric}_minus_all_accept"] = [
                            paired_summary(weighted_paths[i] - fixed_paths) for i in range(len(WEIGHTS))]
                        comparison[f"{metric}_minus_hard_gate"] = [
                            paired_summary(weighted_paths[i] - hard_paths) for i in range(len(WEIGHTS))]
                    watermark_result["policies"][f"{detector_name}_weighted"]["paired_comparisons"] = comparison

                oracle = run_policy_grid(model, paths, delta, np.zeros((ATTACK_N, T, 7), bool),
                                    np.asarray([1.0]), "known_timing", active_fault,
                                    (K_tables, G_tables, F_tables))
                watermark_result["policies"]["known_timing"] = record_metrics(
                    arrays, f"{label}__{fixture}__{watermark}__known_timing", oracle)
                fixture_result[watermark] = watermark_result
                print(label, fixture, watermark, "finished", flush=True)
            model_result["attacks"][fixture] = fixture_result
        result["models"][label] = model_result

    archive = temporary / "d0_paths.npz"
    np.savez_compressed(archive, **arrays)
    result["path_archive_sha256"] = digest(archive)
    save_json(temporary / "d0_results.json", result)
    (temporary / "README.md").write_text(
        "# D0 continuous critical-link weight sweep\n\n"
        "This directory contains path-level metric arrays and the corresponding summaries. "
        "The scan is an exploratory feasibility result under the frozen registration.\n\n"
        "- `d0_results.json` records thresholds, seeds, source hashes, paired path summaries, and archive hash.\n"
        "- `d0_paths.npz` stores all path-level metric arrays and detector gate arrays.\n",
        encoding="utf-8", newline="\n")
    temporary.rename(args.output)
    print(f"D0 complete: {args.output}")


if __name__ == "__main__":
    main()
