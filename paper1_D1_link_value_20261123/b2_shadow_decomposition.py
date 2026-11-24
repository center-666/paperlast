"""Decompose B2 response under the same attack-generated alarm sequence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import b1_health as health
import b1_mode_models as modes
import b2_attack as b2


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CURVE = HERE / "b2_weight_response_results"
OUTPUT = HERE / "b2_shadow_decomposition_results"
Q = np.asarray([0., .2, .4, .6, .8, .9, 1.])
N = 2048
BATCH = 64


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shadow(model, paths, delta, masks, active):
    count = len(paths.x)
    clean = np.zeros((count, len(Q), 4))
    response = np.zeros_like(clean)
    terms = {period: np.zeros((count, len(Q), 3))
             for period in ("window", "active", "gap", "post")}
    sender = int(model.links[6, 1])
    receiver_neighbors = model.neighbors[26]
    for tick in range(1, 150):
        alarm = ((masks[:, tick] & 64) != 0).astype(int)
        effective = np.where(alarm[:, None] == 1, Q[None, :], 1.)
        y = paths.y[:, tick]
        aggregate = y[:, receiver_neighbors].sum(axis=1)
        aggregate = aggregate[:, None, :] + (effective - 1.)[..., None] * y[:, None, sender, :]
        f = model.target["F"][alarm]
        k = model.target["K"][alarm]
        g = model.target["G"][alarm]
        clean = (np.einsum("bqij,bqj->bqi", f, clean)
                 + np.einsum("bqij,bj->bqi", k, y[:, 26])
                 + np.einsum("bqij,bqj->bqi", g, aggregate))
        response = (np.einsum("bqij,bqj->bqi", f, response)
                    + np.einsum("bqij,bqj->bqi", g,
                                effective[..., None] * delta[:, tick, 6, None, :]))
        clean_error = clean[:, :, 2:] - paths.x[:, tick, None, 2:]
        attack_response = response[:, :, 2:]
        loss_terms = np.stack([
            (clean_error ** 2).sum(axis=-1),
            2 * (clean_error * attack_response).sum(axis=-1),
            (attack_response ** 2).sum(axis=-1),
        ], axis=-1)
        if 50 <= tick < 110:
            terms["window"] += loss_terms
            if active[tick - 50]:
                terms["active"] += loss_terms
            else:
                terms["gap"] += loss_terms
        if tick >= 110:
            terms["post"] += loss_terms
    terms["window"] /= 60
    terms["active"] /= active.sum()
    terms["post"] /= 40
    if (~active).any():
        terms["gap"] /= (~active).sum()
    else:
        terms["gap"][:] = np.nan
    return terms


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    temp = HERE / "b2_shadow_decomposition_results_tmp"
    temp.mkdir()
    checks = []
    with np.load(modes.OUTPUT / "mode_tables.npz", allow_pickle=False) as gains:
        for scene, spec in enumerate(health.scenarios()):
            model = health.setup(spec, gains)
            watermark = spec[3]
            watermark_index = 0 if watermark == "negative" else 1
            for fixture_index, fixture in enumerate(("continuous", "burst")):
                filename = f"scene_{scene:02}_{fixture}_{watermark}.npz"
                active = np.ones(60, dtype=bool)
                if fixture == "burst":
                    active = np.arange(60) % 10 < 6
                with np.load(CURVE / filename, allow_pickle=False) as curve:
                    masks = curve["masks"].copy()
                    saved = {period: curve[f"missing_{key}_mse"].copy()
                             for period, key in (("window", "window"),
                                                 ("active", "active"),
                                                 ("gap", "gap"),
                                                 ("post", "tail"))}
                items = {period: [] for period in saved}
                for batch in range(N // BATCH):
                    paths = health.physical(model, BATCH, b2.seed(
                        scene, fixture_index, watermark_index, 2, batch))
                    delta = b2.attack_delta(BATCH, model, watermark, fixture, scene,
                                            fixture_index, watermark_index, batch)
                    terms = shadow(model, paths, delta,
                                   masks[batch * BATCH:(batch + 1) * BATCH], active)
                    for period in items:
                        items[period].append(terms[period])
                data = {period: np.concatenate(values) for period, values in items.items()}
                errors = {}
                for period, terms in data.items():
                    reconstructed = terms.sum(axis=-1)
                    np.testing.assert_allclose(reconstructed, saved[period],
                                               rtol=2e-8, atol=2e-8, equal_nan=True,
                                               err_msg=f"{filename}/{period}")
                    finite = np.isfinite(reconstructed)
                    errors[period] = (float(np.max(abs(reconstructed[finite] - saved[period][finite])))
                                      if finite.any() else None)
                np.savez_compressed(temp / filename, q_values=Q, **data)
                checks.append({"file": filename, "max_abs_reconstructed_mse_error": errors})
                print(scene, fixture, watermark, "shadow complete", flush=True)
    manifest = {
        "status": "complete_posthoc_fixed_attack_alarm_decomposition",
        "B2_commit": "1f4871bb50db2e02897a7fc00760248e4cb5682a",
        "terms": ["clean_measurement_shadow_loss", "twice_shadow_attack_cross_term",
                  "attack_response_squared"],
        "schedule": "The attacked B2 alarm mask is held fixed for both worlds. Clean shadow uses the same q, gain modes, physical noise, and local measurement, with target attack perturbation set to zero.",
        "interpretation": "Conditional on an attack-generated schedule. The clean shadow is generally selection-dependent and is not an independent healthy risk. Terms sum pathwise to the observed attacked MSE.",
        "paths_per_condition": N,
        "conditions": len(checks),
        "input_sha256": {
            str(path.relative_to(ROOT)): sha(path) for path in (
                HERE / "b2_shadow_decomposition.py",
                HERE / "b2_attack.py",
                HERE / "b1_health.py",
                HERE / "b1_mode_models.py",
                CURVE / "response_manifest.json",
                modes.OUTPUT / "mode_tables.npz",
            )
        },
        "curve_input_sha256": {path.name: sha(path) for path in sorted(CURVE.glob("scene_*.npz"))},
        "output_sha256": {path.name: sha(path) for path in sorted(temp.glob("scene_*.npz"))},
        "pathwise_reconstruction_checks": checks,
    }
    (temp / "shadow_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")
    temp.rename(OUTPUT)
    print(json.dumps({"status": manifest["status"], "conditions": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
