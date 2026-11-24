"""Verify shadow components by independently updating two full target states."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import b1_health as health
import b1_mode_models as modes
import b2_attack as b2


HERE = Path(__file__).resolve().parent
CURVE = HERE / "b2_weight_response_results"
OUT = HERE / "b2_shadow_decomposition_results"
Q = np.array([0., .2, .4, .6, .8, .9, 1.])
PREFIX = 4


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def two_worlds(model, paths, perturbation, masks, active):
    n = len(paths.x)
    clean_state = np.zeros((n, 7, 4))
    attacked_state = np.zeros_like(clean_state)
    sums = {period: np.zeros((n, 7, 3)) for period in ("window", "active", "gap", "post")}
    target_sender = int(model.links[6, 1])
    for tick in range(1, 150):
        alarm = (masks[:, tick] & 64) != 0
        q = np.where(alarm[:, None], Q[None, :], 1.)
        observations = paths.y[:, tick]
        k_gain = model.target["K"][alarm.astype(int)]
        g_gain = model.target["G"][alarm.astype(int)]
        updated = []
        for previous, attacked in ((clean_state, False), (attacked_state, True)):
            prediction = previous @ model.A.T
            local_residual = observations[:, None, 26] - prediction @ model.H[26].T
            neighbor_residual = np.zeros((n, 7, 2))
            for sender in model.neighbors[26]:
                residual = observations[:, None, sender] - prediction @ model.H[sender].T
                if sender == target_sender:
                    if attacked:
                        residual += perturbation[:, tick, 6, None, :]
                    residual *= q[..., None]
                neighbor_residual += residual
            updated.append(prediction + np.einsum("bqij,bqj->bqi", k_gain, local_residual)
                           + np.einsum("bqij,bqj->bqi", g_gain, neighbor_residual))
        clean_state, attacked_state = updated
        e0 = clean_state[:, :, 2:] - paths.x[:, tick, None, 2:]
        difference = attacked_state[:, :, 2:] - clean_state[:, :, 2:]
        components = np.stack([(e0 * e0).sum(axis=-1),
                               2 * (e0 * difference).sum(axis=-1),
                               (difference * difference).sum(axis=-1)], axis=-1)
        if 50 <= tick < 110:
            sums["window"] += components
            sums["active" if active[tick - 50] else "gap"] += components
        if tick >= 110:
            sums["post"] += components
    sums["window"] /= 60
    sums["active"] /= active.sum()
    sums["post"] /= 40
    if (~active).any():
        sums["gap"] /= (~active).sum()
    else:
        sums["gap"][:] = np.nan
    return sums


def main():
    manifest = json.loads((OUT / "shadow_manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["input_sha256"].items():
        if checksum(HERE.parent / Path(rel)) != expected:
            raise AssertionError(f"input hash mismatch {rel}")
    for name, expected in manifest["curve_input_sha256"].items():
        if checksum(CURVE / name) != expected:
            raise AssertionError(f"curve hash mismatch {name}")
    for name, expected in manifest["output_sha256"].items():
        if checksum(OUT / name) != expected:
            raise AssertionError(f"shadow output hash mismatch {name}")
    max_abs = 0.
    max_scaled = 0.
    checks = []
    with np.load(modes.OUTPUT / "mode_tables.npz", allow_pickle=False) as gains:
        for scene, spec in enumerate(health.scenarios()):
            model = health.setup(spec, gains)
            watermark = spec[3]
            wi = 0 if watermark == "negative" else 1
            for fi, fixture in enumerate(("continuous", "burst")):
                filename = f"scene_{scene:02}_{fixture}_{watermark}.npz"
                all_paths = health.physical(model, 64, b2.seed(scene, fi, wi, 2, 0))
                paths = type(all_paths)(x=all_paths.x[:PREFIX], y=all_paths.y[:PREFIX])
                delta = b2.attack_delta(64, model, watermark, fixture, scene, fi, wi, 0)[:PREFIX]
                active = np.ones(60, dtype=bool)
                if fixture == "burst":
                    active = np.arange(60) % 10 < 6
                with np.load(CURVE / filename, allow_pickle=False) as curve, \
                        np.load(OUT / filename, allow_pickle=False) as saved:
                    direct = two_worlds(model, paths, delta, curve["masks"][:PREFIX], active)
                    for period, values in direct.items():
                        actual = saved[period][:PREFIX]
                        np.testing.assert_allclose(values, actual, rtol=2e-8, atol=2e-7,
                                                   equal_nan=True, err_msg=f"{filename}/{period}")
                        finite = np.isfinite(values) & np.isfinite(actual)
                        if finite.any():
                            diff = abs(values[finite] - actual[finite])
                            max_abs = max(max_abs, float(diff.max()))
                            max_scaled = max(max_scaled, float(
                                (diff / np.maximum(abs(actual[finite]), 1.)).max()))
                    checks.append({"file": filename, "independent_paths": PREFIX})
    report = {"status": "PASS", "conditions": len(checks),
              "method": "independent separate clean and attacked state updates from per-neighbor innovations",
              "independent_paths_per_condition": PREFIX,
              "max_absolute_term_difference": max_abs,
              "max_scaled_term_difference": max_scaled,
              "checks": checks}
    (OUT / "independent_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
