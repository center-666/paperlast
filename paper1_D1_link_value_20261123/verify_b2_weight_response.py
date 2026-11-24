"""Independently check B2 response curves using per-neighbor innovations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import b1_health as health
import b1_mode_models as modes
import b2_attack as b2
import verify_b2_attack as b2_verify


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "b2_weight_response_results"
Q = np.array([0., .2, .4, .6, .8, .9, 1.])
BITS = 1 << np.arange(7)
PREFIX = 4


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def innovations(model, paths, delta, masks, active):
    count = len(paths.x)
    state = np.zeros((count, len(Q), 50, 4))
    sums = {name: np.zeros((count, len(Q))) for name in (
        "missing_window_mse", "missing_active_mse", "missing_gap_mse",
        "missing_post20_mse", "missing_tail_mse", "missing_tail_110_120_mse",
        "missing_tail_120_130_mse", "missing_tail_130_150_mse",
        "full_window_mse", "network_window_mse", "network_post20_mse",
        "attack_forcing_window", "attack_forcing_active", "effective_attack_weight")}
    link_lookup = {tuple(map(int, edge)): e for e, edge in enumerate(model.links)}
    for tick in range(1, 150):
        mask = masks[:, tick]
        alarm = (mask[:, None] & BITS) != 0
        prediction = state @ model.A.T
        observations = paths.y[:, tick]
        next_state = np.empty_like(state)
        for node, neighbors in enumerate(model.neighbors):
            local_prediction = prediction[:, :, node]
            local_residual = observations[:, None, node] - local_prediction @ model.H[node].T
            neighbor_residual = np.zeros((count, len(Q), 2))
            for sender in neighbors:
                residual = observations[:, None, sender] - local_prediction @ model.H[sender].T
                edge = link_lookup.get((node, sender))
                if edge == 6:
                    residual = residual + delta[:, tick, 6, None, :]
                    weight = np.where(alarm[:, 6, None], Q[None, :], 1.)
                    residual = residual * weight[..., None]
                elif edge is not None:
                    residual = residual * (~alarm[:, edge])[:, None, None]
                neighbor_residual += residual
            if node == 26:
                index = alarm[:, 6].astype(int)
                k_gain = model.target["K"][index]
                g_gain = model.target["G"][index]
            else:
                k_gain = model.base["K"][mask, node, None]
                g_gain = model.base["G"][mask, node, None]
            next_state[:, :, node] = (
                local_prediction + np.einsum("bqij,bqj->bqi", k_gain, local_residual)
                + np.einsum("bqij,bqj->bqi", g_gain, neighbor_residual))
        state = next_state
        error = (state - paths.x[:, tick, None, None, :]) ** 2
        missing = error[:, :, 26, 2:].sum(axis=-1)
        full = error[:, :, 26].sum(axis=-1)
        network = error.sum(axis=-1).mean(axis=-1)
        if 50 <= tick < 110:
            weight = np.where(alarm[:, 6, None], Q[None, :], 1.)
            gain = model.target["G"][alarm[:, 6].astype(int)]
            injection = np.einsum("bqij,bqj->bqi", gain,
                                  weight[..., None] * delta[:, tick, 6, None, :])
            forcing = (injection ** 2).sum(axis=-1)
            sums["missing_window_mse"] += missing
            sums["full_window_mse"] += full
            sums["network_window_mse"] += network
            sums["attack_forcing_window"] += forcing
            if active[tick - 50]:
                sums["missing_active_mse"] += missing
                sums["attack_forcing_active"] += forcing
                sums["effective_attack_weight"] += weight
            else:
                sums["missing_gap_mse"] += missing
        if tick >= 20:
            sums["missing_post20_mse"] += missing
            sums["network_post20_mse"] += network
        if tick >= 110:
            sums["missing_tail_mse"] += missing
            if tick < 120:
                sums["missing_tail_110_120_mse"] += missing
            elif tick < 130:
                sums["missing_tail_120_130_mse"] += missing
            else:
                sums["missing_tail_130_150_mse"] += missing
    divisors = {
        "missing_window_mse": 60, "missing_active_mse": int(active.sum()),
        "missing_post20_mse": 130, "missing_tail_mse": 40,
        "missing_tail_110_120_mse": 10, "missing_tail_120_130_mse": 10,
        "missing_tail_130_150_mse": 20, "full_window_mse": 60,
        "network_window_mse": 60, "network_post20_mse": 130,
        "attack_forcing_window": 60, "attack_forcing_active": int(active.sum()),
        "effective_attack_weight": int(active.sum()),
    }
    for key, divisor in divisors.items():
        sums[key] /= divisor
    if (~active).any():
        sums["missing_gap_mse"] /= (~active).sum()
    else:
        sums["missing_gap_mse"][:] = np.nan
    return sums


def main():
    manifest = json.loads((OUTPUT / "response_manifest.json").read_text(encoding="utf-8"))
    root = HERE.parent
    for rel, expected in manifest["input_sha256"].items():
        actual = checksum(root / Path(rel))
        if actual != expected:
            raise AssertionError(f"input hash mismatch {rel}")
    for name, expected in manifest["output_sha256"].items():
        if checksum(OUTPUT / name) != expected:
            raise AssertionError(f"output hash mismatch {name}")
    checks = []
    max_abs = 0.
    max_relative = 0.
    max_decomposition = 0.
    with np.load(modes.OUTPUT / "mode_tables.npz", allow_pickle=False) as gains:
        for scene, spec in enumerate(health.scenarios()):
            model = health.setup(spec, gains)
            watermark = spec[3]
            watermark_index = 0 if watermark == "negative" else 1
            for fixture_index, fixture in enumerate(("continuous", "burst")):
                filename = f"scene_{scene:02}_{fixture}_{watermark}.npz"
                with np.load(OUTPUT / filename, allow_pickle=False) as saved, \
                        np.load(b2.OUT / filename, allow_pickle=False) as original:
                    np.testing.assert_array_equal(saved["masks"], original["masks"])
                    for strategy in ("A", "BD", "C"):
                        np.testing.assert_allclose(
                            saved["target_attack_unalarmed_fraction"][:, 0],
                            original[f"{strategy}__target_attack_accept_fraction"],
                            rtol=0, atol=0)
                    if fixture == "burst":
                        residual = saved["missing_window_mse"] - (
                            36 * saved["missing_active_mse"]
                            + 24 * saved["missing_gap_mse"]) / 60
                    else:
                        residual = saved["missing_window_mse"] - saved["missing_active_mse"]
                        if not np.isnan(saved["missing_gap_mse"]).all():
                            raise AssertionError("continuous attack has no gap")
                    max_decomposition = max(max_decomposition, float(np.max(abs(residual))))
                    np.testing.assert_allclose(residual, 0, atol=2e-7, rtol=0)
                    paths_all = health.physical(model, 64, b2.seed(
                        scene, fixture_index, watermark_index, 2, 0))
                    paths = type(paths_all)(x=paths_all.x[:PREFIX], y=paths_all.y[:PREFIX])
                    delta = b2.attack_delta(64, model, watermark, fixture, scene,
                                            fixture_index, watermark_index, 0)[:PREFIX]
                    masks = saved["masks"][:PREFIX]
                    scores = b2_verify.direct_scores(model, paths, delta)
                    calculated = ((scores > float(saved["threshold"])) @ BITS).astype(np.uint8)
                    np.testing.assert_array_equal(calculated, masks)
                    active = np.ones(60, dtype=bool)
                    if fixture == "burst":
                        active = np.arange(60) % 10 < 6
                    direct = innovations(model, paths, delta, masks, active)
                    for metric, values in direct.items():
                        actual = saved[metric][:PREFIX]
                        np.testing.assert_allclose(values, actual, rtol=2e-8, atol=1e-7,
                                                   equal_nan=True, err_msg=f"{filename}/{metric}")
                        finite = np.isfinite(values) & np.isfinite(actual)
                        if finite.any():
                            difference = abs(values[finite] - actual[finite])
                            max_abs = max(max_abs, float(difference.max()))
                            max_relative = max(max_relative, float(
                                (difference / np.maximum(abs(actual[finite]), 1.)).max()))
                    checks.append({"file": filename, "paths": PREFIX,
                                   "weights": Q.tolist()})
    result = {"status": "PASS", "conditions": len(checks),
              "independently_reconstructed_paths_per_condition": PREFIX,
              "independent_method": "per-neighbor innovations with independently expanded observation update",
              "max_absolute_path_metric_difference": max_abs,
              "max_scaled_path_metric_difference": max_relative,
              "max_window_decomposition_residual": max_decomposition,
              "checks": checks}
    (OUTPUT / "independent_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
