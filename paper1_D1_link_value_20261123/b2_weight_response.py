"""Deterministic seven-weight replay on the already-registered B2 paths."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

import b1_health as health
import b1_mode_models as modes
import b2_attack as b2


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAN = HERE / "B2_WEIGHT_RESPONSE_PLAN.md"
OUTPUT = HERE / "b2_weight_response_results"
FINAL_B2 = HERE / "b2_attack_results"
POLICY = HERE / "b1_health_results" / "policy_lock.json"
Q_GRID = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
N = 2048
BATCH = 64
T = 150
START, STOP = 50, 110
BITS = 1 << np.arange(7)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q_position(q: float) -> int:
    matches = np.flatnonzero(np.isclose(Q_GRID, q, atol=1e-12, rtol=0))
    if len(matches) != 1:
        raise ValueError(f"Weight {q} is outside the frozen response grid.")
    return int(matches[0])


def curve_metrics(model, paths, delta, masks, active: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized response recursion for every candidate q on one B2 batch."""
    count = len(paths.x)
    q_count = len(Q_GRID)
    base_state = np.zeros((count, 50, 4), dtype=float)
    target_state = np.zeros((count, q_count, 4), dtype=float)
    metrics = {
        name: np.zeros((count, q_count), dtype=float)
        for name in (
            "missing_window_mse", "missing_active_mse", "missing_gap_mse",
            "missing_post20_mse", "missing_tail_mse",
            "missing_tail_110_120_mse", "missing_tail_120_130_mse",
            "missing_tail_130_150_mse", "full_window_mse",
            "network_window_mse", "network_post20_mse",
            "attack_forcing_window", "attack_forcing_active",
        )
    }
    effective_attack_weight = np.zeros((count, q_count), dtype=float)
    alarm_bits = ((masks[:, :, None] & BITS[None, None, :]) != 0)
    target_alarm = alarm_bits[:, :, 6]
    target_sender = int(model.links[6, 1])
    raw = (paths.y.swapaxes(-1, -2) @ model.adjacency.T).swapaxes(-1, -2)

    for tick in range(1, T):
        mask = masks[:, tick]
        bits = alarm_bits[:, tick]
        y = paths.y[:, tick]
        received = raw[:, tick].copy()
        for edge in range(6):
            received[:, model.links[edge, 0]] -= (
                bits[:, edge, None] * y[:, model.links[edge, 1]])
        packet_delta = delta[:, tick, 6]
        received[:, 26] += packet_delta

        base_post = (
            np.einsum("bnij,bnj->bni", model.base["F"][mask], base_state)
            + np.einsum("bnij,bnj->bni", model.base["K"][mask], y)
            + np.einsum("bnij,bnj->bni", model.base["G"][mask], received)
        )
        alarm = target_alarm[:, tick]
        alarm_index = alarm.astype(int)
        effective_q = np.where(alarm[:, None], Q_GRID[None, :], 1.0)
        packet = received[:, 26, None, :] + (effective_q - 1.0)[..., None] * (
            y[:, target_sender, None, :] + packet_delta[:, None, :])
        f_gain = model.target["F"][alarm_index]
        k_gain = model.target["K"][alarm_index]
        g_gain = model.target["G"][alarm_index]
        target_post = (
            np.einsum("bqij,bqj->bqi", f_gain, target_state)
            + np.einsum("bqij,bj->bqi", k_gain, y[:, 26])
            + np.einsum("bqij,bqj->bqi", g_gain, packet)
        )

        base_error = (base_post - paths.x[:, tick, None, :]) ** 2
        target_error = (target_post - paths.x[:, tick, None, :]) ** 2
        missing = target_error[:, :, 2:].sum(axis=-1)
        full = target_error.sum(axis=-1)
        base_network = base_error.sum(axis=(1, 2)) - base_error[:, 26].sum(axis=-1)
        network = (base_network[:, None] + full) / 50.0
        attack_input = np.einsum(
            "bqij,bqj->bqi", g_gain,
            effective_q[..., None] * packet_delta[:, None, :])
        forcing = np.sum(attack_input ** 2, axis=-1)

        if START <= tick < STOP:
            metrics["missing_window_mse"] += missing
            metrics["full_window_mse"] += full
            metrics["network_window_mse"] += network
            metrics["attack_forcing_window"] += forcing
            if active[tick - START]:
                metrics["missing_active_mse"] += missing
                metrics["attack_forcing_active"] += forcing
                effective_attack_weight += effective_q
            else:
                metrics["missing_gap_mse"] += missing
        if 20 <= tick < T:
            metrics["missing_post20_mse"] += missing
            metrics["network_post20_mse"] += network
        if STOP <= tick < T:
            metrics["missing_tail_mse"] += missing
            if tick < 120:
                metrics["missing_tail_110_120_mse"] += missing
            elif tick < 130:
                metrics["missing_tail_120_130_mse"] += missing
            else:
                metrics["missing_tail_130_150_mse"] += missing

        base_state = base_post
        target_state = target_post

    active_count = int(active.sum())
    gap_count = int((~active).sum())
    metrics["missing_window_mse"] /= STOP - START
    metrics["missing_active_mse"] /= active_count
    if gap_count:
        metrics["missing_gap_mse"] /= gap_count
    else:
        metrics["missing_gap_mse"][:] = np.nan
    metrics["missing_post20_mse"] /= T - 20
    metrics["missing_tail_mse"] /= T - STOP
    metrics["missing_tail_110_120_mse"] /= 10
    metrics["missing_tail_120_130_mse"] /= 10
    metrics["missing_tail_130_150_mse"] /= 20
    metrics["full_window_mse"] /= STOP - START
    metrics["network_window_mse"] /= STOP - START
    metrics["network_post20_mse"] /= T - 20
    metrics["attack_forcing_window"] /= STOP - START
    metrics["attack_forcing_active"] /= active_count
    effective_attack_weight /= active_count
    metrics["effective_attack_weight"] = effective_attack_weight
    metrics["target_attack_unalarmed_fraction"] = np.broadcast_to(
        ((~target_alarm[:, START:STOP][:, active]).mean(axis=1))[:, None],
        (count, q_count)).copy()
    metrics["target_path_missed_fraction"] = np.broadcast_to(
        (~target_alarm[:, START:STOP][:, active].any(axis=1))[:, None],
        (count, q_count)).astype(float).copy()
    return metrics


def compare_registered_strategies(curves: dict[str, np.ndarray], archive,
                                  policy_row: dict) -> dict:
    comparisons = {}
    for strategy, policy_key in (("A", "A"), ("BD", "B"), ("C", "C")):
        q = float(policy_row[policy_key])
        pos = q_position(q)
        for metric, old_key in (
            ("missing_window_mse", "missing_window_mse"),
            ("missing_post20_mse", "missing_post20_mse"),
            ("missing_tail_mse", "missing_tail_mse"),
            ("full_window_mse", "full_window_mse"),
            ("network_window_mse", "network_window_mse"),
            ("network_post20_mse", "network_post20_mse"),
            ("attack_forcing_window", "accepted_target_forcing"),
        ):
            current = curves[metric][:, pos]
            registered = archive[f"{strategy}__{old_key}"]
            np.testing.assert_allclose(current, registered, rtol=2e-11,
                                       atol=2e-9,
                                       err_msg=f"{strategy}/{metric}, q={q}")
            comparisons[f"{strategy}.{metric}.q{q:g}"] = float(
                np.max(np.abs(current - registered)))
    return comparisons


def run() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    plan_sha = digest(PLAN)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    temp = HERE / "b2_weight_response_results_tmp"
    temp.mkdir()
    mask_checks = []
    strategy_checks = []
    with np.load(modes.OUTPUT / "mode_tables.npz", allow_pickle=False) as gains:
        for scene_index, spec in enumerate(health.scenarios()):
            dynamics, topology, target, watermark = spec
            model = health.setup(spec, gains)
            row = policy["rows"][scene_index]
            watermark_index = 0 if watermark == "negative" else 1
            for fixture_index, fixture in enumerate(("continuous", "burst")):
                filename = f"scene_{scene_index:02}_{fixture}_{watermark}.npz"
                with np.load(FINAL_B2 / filename, allow_pickle=False) as old:
                    frozen_masks = old["masks"].copy()
                    threshold = float(old["threshold"])
                    old_strategy_arrays = {
                        key: old[key].copy() for key in old.files
                        if key.startswith(("A__", "BD__", "C__"))
                    }
                active = b2.active_attack_ticks(fixture)
                values = {
                    key: [] for key in (
                        "missing_window_mse", "missing_active_mse", "missing_gap_mse",
                        "missing_post20_mse", "missing_tail_mse",
                        "missing_tail_110_120_mse", "missing_tail_120_130_mse",
                        "missing_tail_130_150_mse", "full_window_mse",
                        "network_window_mse", "network_post20_mse",
                        "attack_forcing_window", "attack_forcing_active",
                        "effective_attack_weight", "target_attack_unalarmed_fraction",
                        "target_path_missed_fraction",
                    )
                }
                mask_hash = hashlib.sha256()
                for batch in range(N // BATCH):
                    paths = health.physical(
                        model, BATCH,
                        b2.seed(scene_index, fixture_index, watermark_index, 2, batch))
                    delta = b2.attack_delta(
                        BATCH, model, watermark, fixture, scene_index,
                        fixture_index, watermark_index, batch)
                    replay_scores = b2.attack_scores(model, paths, delta)
                    replay_masks = ((replay_scores > threshold) @ BITS).astype(np.uint8)
                    first = batch * BATCH
                    last = first + BATCH
                    archive_masks = frozen_masks[first:last]
                    np.testing.assert_array_equal(replay_masks, archive_masks)
                    mask_hash.update(archive_masks.tobytes())
                    curves = curve_metrics(model, paths, delta, archive_masks, active)
                    for key in values:
                        values[key].append(curves[key])
                arrays = {key: np.concatenate(parts, axis=0)
                          for key, parts in values.items()}
                condition_strategy_errors = compare_registered_strategies(
                    arrays, old_strategy_arrays, row)
                np.savez_compressed(
                    temp / filename,
                    q_values=Q_GRID,
                    masks=frozen_masks,
                    threshold=np.asarray(threshold),
                    **arrays,
                )
                mask_checks.append({
                    "file": filename,
                    "paths": N,
                    "mask_sha256": mask_hash.hexdigest(),
                    "bitwise_equal_to_B2": True,
                })
                strategy_checks.append({
                    "file": filename,
                    "paths": N,
                    "max_abs_difference": condition_strategy_errors,
                })
                print(scene_index, fixture, watermark, "complete", flush=True)

    output_manifest = {
        "status": "complete_exploratory_B2_seven_weight_replay",
        "analysis_identity": "posthoc deterministic replay on B2 registered random streams; no new independent attack paths",
        "B2_commit": "1f4871bb50db2e02897a7fc00760248e4cb5682a",
        "plan_sha256": plan_sha,
        "q_values": Q_GRID.tolist(),
        "conditions": 24,
        "paths_per_condition": N,
        "total_reconstructed_paths": 24 * N,
        "seed_namespace": [20261124, 2207],
        "seed_rule": "SeedSequence([20261124, 2207, scene_index, fixture_index, watermark_index, 2, batch_index])",
        "frozen_alarm_masks": "All 24 masks reconstructed from frozen B2 thresholds and compared bitwise against archived masks. The archived masks, not candidate-specific masks, drive all q values.",
        "healthy_curves": "Use existing B1 holdout path MSE arrays for seven q values. These same holdout paths contributed to B1 policy selection, so health curves are descriptive and selection-reused.",
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "input_sha256": {},
        "output_sha256": {},
        "mask_checks": mask_checks,
        "registered_strategy_replay_checks_all_paths": strategy_checks,
    }
    inputs = [
        PLAN, HERE / "B2_REGISTRATION.json", HERE / "B2_REGISTRATION_SHA256.txt",
        HERE / "B2_REPRODUCTION.md", HERE / "B2_EXECUTION_AMENDMENT.md",
        HERE / "B1_HEALTH_RUN.json", POLICY, HERE / "b1_health.py",
        HERE / "b1_mode_models.py", HERE / "b2_attack.py",
        HERE / "b2_weight_response.py",
        modes.OUTPUT / "mode_tables.npz",
        HERE / "results" / "D1A_link_value.json",
        ROOT / "paper1_common_g50_gain_acceptance_20261121" / "physical_configuration.json",
    ]
    for scene_index in range(12):
        inputs.extend([
            HERE / "b1_health_results" / f"scene_{scene_index:02}.npz",
            FINAL_B2 / f"scene_{scene_index:02}_continuous_"
            f"{'negative' if scene_index % 2 == 0 else 'iid'}.npz",
            FINAL_B2 / f"scene_{scene_index:02}_burst_"
            f"{'negative' if scene_index % 2 == 0 else 'iid'}.npz",
        ])
    for path in inputs:
        key = str(path.relative_to(ROOT))
        output_manifest["input_sha256"][key] = digest(path)
    for path in sorted(temp.iterdir()):
        if path.is_file():
            output_manifest["output_sha256"][path.name] = digest(path)
    (temp / "response_manifest.json").write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    OUTPUT.mkdir()
    for path in temp.iterdir():
        path.replace(OUTPUT / path.name)
    temp.rmdir()
    print(json.dumps({"status": output_manifest["status"],
                      "conditions": len(mask_checks),
                      "q_values": Q_GRID.tolist(),
                      "output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
