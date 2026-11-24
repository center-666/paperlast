"""B2 fresh attack replay for the three distinct locked B1 policies."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

import b1_health as health
import b1_mode_models as modes
sys.path.insert(0, str(ROOT / "paper1_sscd_commonrandom_20261122"))
sys.path.insert(0, str(ROOT / "paper1_history_ablation_20261123"))
from watson_crn import (
    beta_from_common,
    make_common_draws,
    watermark_attack_from_beta,
)
from ablation import residuals, scores_all


REGISTRATION = HERE / "B2_REGISTRATION.json"
POLICY = HERE / "b1_health_results" / "policy_lock.json"
OUT = HERE / "b2_attack_results"
Q_GRID = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
BITS = 1 << np.arange(7)
N = 2048
BATCH = 64
T = 150
START, STOP = 50, 110
FIXTURE_VECTOR = np.asarray([6.0, 0.0])


def scenarios() -> list[tuple[str, str, int, str]]:
    return health.scenarios()


def seed(scene_index: int, fixture_index: int, watermark_index: int,
         stage: int, batch: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([20261124, 2207, scene_index,
                                   fixture_index, watermark_index, stage, batch])


def attack_template(fixture: str) -> np.ndarray:
    active = active_attack_ticks(fixture)
    template = np.zeros((60, 7, 2), dtype=float)
    template[:, 6] = active[:, None] * FIXTURE_VECTOR
    return template


def active_attack_ticks(fixture: str) -> np.ndarray:
    active = np.zeros(60, dtype=bool)
    if fixture == "continuous":
        active[:] = True
    elif fixture == "burst":
        active[:] = (np.arange(60) % 10) < 6
    else:
        raise ValueError(fixture)
    return active


def attack_delta(count: int, model, watermark: str, fixture: str,
                 scene_index: int, fixture_index: int, watermark_index: int,
                 batch: int) -> np.ndarray:
    theta = -1.3152369768257723 if watermark == "negative" else 0.0
    streams = seed(scene_index, fixture_index, watermark_index, 2, batch)
    child = streams.spawn(3)
    draws = make_common_draws(count, 60, 7,
                              int(child[0].generate_state(1)[0]),
                              int(child[1].generate_state(1)[0]),
                              int(child[2].generate_state(1)[0]))
    beta = beta_from_common(draws, theta)
    return watermark_attack_from_beta(attack_template(fixture), beta, T, START)


def attack_scores(model, paths, delta) -> np.ndarray:
    residual = residuals(model, paths, delta, np.arange(7))
    return scores_all(residual, model.Theta)["H1"]


def strategy_weights(scene_index: int, policy: dict) -> dict[str, float]:
    row = policy["rows"][scene_index]
    return {"A": float(row["A"]), "BD": float(row["B"]), "C": float(row["C"])}


def q_index(q: float) -> int:
    matches = np.flatnonzero(np.isclose(Q_GRID, q, atol=1e-12, rtol=0))
    if len(matches) != 1:
        raise ValueError(f"Weight {q} is not in the registered grid.")
    return int(matches[0])


def matrix_vector(matrix, vector):
    return np.einsum("...ij,...j->...i", matrix, vector)


def evaluate(model, paths, delta, masks, weights: dict[str, float],
             active_attack: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    count = len(paths.x)
    states = {name: np.zeros((count, 50, 4), dtype=float) for name in weights}
    target_states = {name: np.zeros((count, 4), dtype=float) for name in weights}
    missing_window = {name: np.zeros(count) for name in weights}
    missing_post20 = {name: np.zeros(count) for name in weights}
    missing_tail = {name: np.zeros(count) for name in weights}
    full_window = {name: np.zeros(count) for name in weights}
    network_window = {name: np.zeros(count) for name in weights}
    network_post20 = {name: np.zeros(count) for name in weights}
    forcing = {name: np.zeros(count) for name in weights}
    target_alarm = ((masks[:, :, None] & BITS[None, None, :]) != 0)[:, :, 6]
    target_delay = np.where(target_alarm[:, START:STOP].any(axis=1),
                            np.argmax(target_alarm[:, START:STOP], axis=1), -1)
    target_run = np.zeros(count, dtype=int)
    max_target_run = np.zeros(count, dtype=int)
    target_window = target_alarm[:, START:STOP]
    target_attack_alarm = target_window[:, active_attack]
    target_attack_accept_fraction = (~target_attack_alarm).mean(axis=1)
    target_window_accept_fraction = (~target_window).mean(axis=1)
    for column in target_alarm[:, 1:].T:
        target_run = (target_run + 1) * column
        max_target_run = np.maximum(max_target_run, target_run)
    target_sender = int(model.links[6, 1])
    raw = (paths.y.swapaxes(-1, -2) @ model.adjacency.T).swapaxes(-1, -2)
    base_f = model.base["F"]
    base_k = model.base["K"]
    base_g = model.base["G"]
    for tick in range(1, T):
        mask = masks[:, tick]
        bits = (mask[:, None] & BITS) != 0
        y = paths.y[:, tick]
        received = raw[:, tick].copy()
        for edge in range(6):
            received[:, model.links[edge, 0]] -= bits[:, edge, None] * y[:, model.links[edge, 1]]
        target_alarm_now = bits[:, 6]
        packet_delta = delta[:, tick, 6]
        received[:, 26] += packet_delta
        base_post = {
            name: matrix_vector(base_f[mask], states[name])
            + matrix_vector(base_k[mask], y)
            + matrix_vector(base_g[mask], received)
            for name in weights
        }
        # The aggregate includes the target packet before target-specific scaling.
        # On an alarm, q scales the complete received packet, including any
        # watermarked additive attack perturbation. Without an alarm, q_eff=1.
        for name, q in weights.items():
            qpos = q_index(q)
            effective_q = np.where(target_alarm_now, q, 1.0)
            packet = received[:, 26] + (effective_q - 1.0)[:, None] * (
                y[:, target_sender] + packet_delta)
            alarm_index = target_alarm_now.astype(int)
            target_f = model.target["F"][alarm_index, qpos]
            target_k = model.target["K"][alarm_index, qpos]
            target_g = model.target["G"][alarm_index, qpos]
            target_state = (matrix_vector(target_f, target_states[name])
                            + matrix_vector(target_k, y[:, 26])
                            + matrix_vector(target_g, packet))
            states[name] = base_post[name]
            states[name][:, 26] = target_state
            target_states[name] = target_state
            error = (states[name] - paths.x[:, tick, None, :]) ** 2
            missing = error[:, 26, 2:].sum(axis=-1)
            full = error[:, 26].sum(axis=-1)
            network = error.sum(axis=(1, 2)) / 50.0
            if START <= tick < STOP:
                missing_window[name] += missing
                full_window[name] += full
                network_window[name] += network
                effective_delta = effective_q[:, None] * packet_delta
                forcing[name] += np.sum(
                    matrix_vector(target_g, effective_delta) ** 2, axis=-1)
            if 20 <= tick < T:
                missing_post20[name] += missing
                network_post20[name] += network
            if STOP <= tick < T:
                missing_tail[name] += missing
    result = {}
    for name in weights:
        result[name] = {
            "missing_window_mse": missing_window[name] / 60.0,
            "missing_post20_mse": missing_post20[name] / 130.0,
            "missing_tail_mse": missing_tail[name] / 40.0,
            "full_window_mse": full_window[name] / 60.0,
            "network_window_mse": network_window[name] / 60.0,
            "network_post20_mse": network_post20[name] / 130.0,
            "target_alarm_delay": target_delay.astype(float),
            "target_alarm_run_max": max_target_run.astype(float),
            "target_attack_accept_fraction": target_attack_accept_fraction,
            "target_window_accept_fraction": target_window_accept_fraction,
            "accepted_target_forcing": forcing[name] / 60.0,
        }
    return result


def smoke() -> dict:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    with np.load(modes.OUTPUT / "mode_tables.npz") as gains:
        checked = []
        for scene_index, spec in enumerate(scenarios()[::4]):
            model = health.setup(spec, gains)
            paths = health.physical(model, 4, seed(scene_index, 0, 0, 2, 99))
            delta = attack_delta(4, model, spec[3], "burst", scene_index * 4, 0,
                                 0 if spec[3] == "negative" else 1, 99)
            scores = attack_scores(model, paths, delta)
            scene = np.load(health.OUT / f"scene_{scene_index * 4:02}.npz")
            masks = (scores > float(scene["threshold"])) @ BITS
            result = evaluate(model, paths, delta, masks.astype(np.uint8),
                              strategy_weights(scene_index * 4, policy),
                              active_attack_ticks("burst"))
            checked.append({"scene": scene_index * 4,
                            "score_finite": bool(np.isfinite(scores).all()),
                            "masks_shape": list(masks.shape),
                            "target_mse_finite": bool(np.isfinite(
                                result["A"]["missing_window_mse"]).all())})
    return {"status": "PASS", "checked": checked}


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    output_tmp = HERE / "b2_attack_results_tmp"
    output_tmp.mkdir()
    with np.load(modes.OUTPUT / "mode_tables.npz") as gains:
        for scene_index, spec in enumerate(scenarios()):
            dynamics, topology, target, watermark = spec
            model = health.setup(spec, gains)
            scene_health = np.load(health.OUT / f"scene_{scene_index:02}.npz")
            threshold = float(scene_health["threshold"])
            scene_policy = strategy_weights(scene_index, policy)
            for fixture_index, fixture in enumerate(("continuous", "burst")):
                metrics = {name: {key: [] for key in (
                    "missing_window_mse", "missing_post20_mse", "missing_tail_mse",
                    "full_window_mse", "network_window_mse", "network_post20_mse",
                    "target_alarm_delay", "target_alarm_run_max",
                    "target_attack_accept_fraction", "target_window_accept_fraction",
                    "accepted_target_forcing")}
                           for name in scene_policy}
                masks_archive = []
                for batch in range(N // BATCH):
                    paths = health.physical(model, BATCH,
                                            seed(scene_index, fixture_index,
                                                 0 if watermark == "negative" else 1,
                                                 2, batch))
                    delta = attack_delta(BATCH, model, watermark, fixture,
                                         scene_index, fixture_index,
                                         0 if watermark == "negative" else 1,
                                         batch)
                    score = attack_scores(model, paths, delta)
                    masks = ((score > threshold) @ BITS).astype(np.uint8)
                    masks_archive.append(masks)
                    results = evaluate(model, paths, delta, masks, scene_policy,
                                       active_attack_ticks(fixture))
                    for strategy in scene_policy:
                        for key, values in results[strategy].items():
                            metrics[strategy][key].append(values)
                archive = {strategy: {key: np.concatenate(values)
                                      for key, values in metric.items()}
                           for strategy, metric in metrics.items()}
                archive["masks"] = np.concatenate(masks_archive)
                archive["threshold"] = np.asarray(threshold)
                filename = f"scene_{scene_index:02}_{fixture}_{watermark}.npz"
                np.savez_compressed(output_tmp / filename, masks=archive["masks"],
                                    threshold=archive["threshold"], **{
                                        f"{strategy}__{key}": value
                                        for strategy, metric in archive.items()
                                        if strategy in scene_policy
                                        for key, value in metric.items()})
            print(scene_index, spec, "finished", flush=True)
    output_tmp.rename(OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2))
    else:
        run()
