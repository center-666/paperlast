"""Frozen-support transfer and equal-information four-state rotation experiments."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
COMPARATORS = HERE.parent
sys.path.insert(0, str(COMPARATORS))
import experiment as exp

MODELS = ("unstable", "stable")
SCHEMES = ("iid", "no_repeat", "no_opposite", "negative")
SCORERS = ("plugin", "exact")
ARMS = tuple(s + "_" + r for s in SCHEMES for r in SCORERS)
SUPPORTS = tuple((f"single_{e}", (e,)) for e in range(6)) + (
    ("same_receiver", (3, 5)), ("different_receivers", (0, 1)),
    ("all_six", tuple(range(6))),
)
METRICS = (
    "attacked_early", "clean_edge_early", "clean_any_early", "exact_early",
    "attacked_full", "clean_edge_full", "exact_full",
    "attack_target_mse", "attack_five_mse", "attack_network_mse",
    "recovery_target_mse", "recovery_five_mse", "recovery_network_mse",
    "compatible_path", "large_late_accept_path", "tail_event", "mse_gt1",
)
CHUNK = 32
PARENT = "46535058416adea9150a73ba84d81af3bfc75036"
SPLITS = {"dev": (21, 256, 512), "confirm": (22, 1024, 4096)}
DEPENDENCIES = (
    HERE / "PROTOCOL.md", HERE / "study_transfer.py", HERE / "analyze_transfer.py",
    HERE / "verify_transfer.py",
    COMPARATORS / "experiment.py", COMPARATORS / "projection.c",
    exp.BASE / "study.py", exp.BASE / "run_stage1.py", exp.BASE / "online_scores.py",
    exp.BASE / "matched_glrt.py", exp.BASE / "watermark_keys.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def atomic_npz(path, arrays):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def roots(split, model, scene, population, batch):
    return [20261126, 61783, split, model, scene, population, batch]


def rotation_codes(uniforms, scheme):
    codes = np.floor(4 * uniforms).astype(np.uint8)
    if scheme == "iid":
        return codes
    offsets = np.array((1, 2, 3) if scheme == "no_repeat" else (0, 1, 3), dtype=np.uint8)
    for tick in range(1, uniforms.shape[1]):
        codes[:, tick] = (codes[:, tick-1] + offsets[np.floor(3 * uniforms[:, tick]).astype(int)]) % 4
    return codes


def inputs(split, mi, scene, population, batch, count):
    root = roots(split, mi, scene, population, batch)
    source, model, generator, estimator, operators = exp.old.backend(MODELS[mi])
    physical = generator(model, count, np.random.SeedSequence([*root, 0]))
    uniforms = np.random.default_rng(np.random.SeedSequence([*root, 1])).random((count, 150, 50))
    seeds = [np.random.SeedSequence([*root, 2, component]) for component in range(3)]
    draws = exp.old.make_key_draws(count, *seeds)
    negative = exp.old.matrices(draws, "negative_fixed")[:, :, model.links[:, 1]]
    return source, model, estimator, operators, physical, uniforms, negative


def cutoff(maxima, alpha=.05):
    rank = int(np.ceil((len(maxima)+1)*(1-alpha)))
    return {"threshold": float(np.sort(maxima)[rank-1]),
            "one_based_rank": rank, "paths": len(maxima), "target_fwer": alpha,
            "alarm": "score > threshold"}


def evaluate(split, mi, scene, population, batch, count, thresholds=None):
    source, model, estimator, operators, physical, uniforms, negative = inputs(
        split, mi, scene, population, batch, count)
    raw = physical.y[:, :, model.links[:, 1]]
    codes = {s: rotation_codes(uniforms, s) for s in SCHEMES[:3]}
    keys = {s: exp.ROTATION[c][:, :, model.links[:, 1]] for s, c in codes.items()}
    keys["negative"] = negative
    transitions = np.stack([
        np.bincount((c[:, :-1].ravel()*4+c[:, 1:].ravel()), minlength=16).reshape(4, 4)
        for c in codes.values()
    ])
    states = np.stack([np.bincount(c.ravel(), minlength=4) for c in codes.values()])
    record = {
        "path_id": np.arange(batch*CHUNK, batch*CHUNK+count),
        "physical_probe": physical.y[:, 1, 0], "key_probe": uniforms[:, 1, 0],
        "transitions": transitions[None], "states": states[None],
    }
    clean_scores = {}
    for s, key in keys.items():
        clean_scores[s + "_exact"] = exp.receiver.matched(raw, key, operators)
    plugin = exp.old.base.plugin_scores(model, physical, np.zeros_like(raw))
    clean_scores.update({s + "_plugin": plugin for s in SCHEMES})
    if population != 2:
        record["maxima"] = np.stack([clean_scores[a][:, 5:].max(axis=(1, 2)) for a in ARMS], axis=1)
        if population == 1:
            health_mse = []
            for arm in ARMS:
                gate = clean_scores[arm] > thresholds[arm]["threshold"]
                world = estimator(model, physical, np.zeros_like(raw),
                                  sender_packets=physical.y, gate_override=gate)
                error = np.sum((world.post-physical.x[:, :, None, :])**2, axis=-1)
                health_mse.append(np.column_stack((
                    error[:, 20:, model.receiver_ids].mean(axis=(1, 2)),
                    error[:, 20:].mean(axis=(1, 2)),
                )))
            record["health_mse"] = np.stack(health_mse, axis=1)
        return record

    onset = (50, 65)[scene % 2]
    stop = onset + 60
    fixture = source.rotated_raw_attack(model, (0., -45., 45.)[scene // 2])[0][50:110]
    shape = (count, len(SUPPORTS), len(ARMS))
    values = np.zeros((*shape, len(METRICS)))
    early_saved = np.zeros((*shape, 6), dtype=np.uint8)
    full_saved = np.zeros_like(early_saved)
    early_max = np.zeros((*shape, 6))
    delay_saved = np.full((*shape, 6), 60, dtype=np.uint8)
    compatible = np.zeros((count, len(SCHEMES), 6), dtype=np.uint8)
    late_counts = np.zeros((count, len(ARMS), 6), dtype=np.uint16)
    for si, scheme in enumerate(SCHEMES):
        key = keys[scheme]
        delta = np.zeros_like(raw)
        delta[:, onset:stop] = (key[:, onset:stop] @ fixture[None, ..., None])[..., 0]
        blocks = operators[8, False][0]
        for tick in range(onset+7, stop):
            white = np.einsum("bewm,ewrm->ber",
                              delta[:, tick-7:tick+1].transpose(0, 2, 1, 3), blocks)
            relative = np.sum(white**2, axis=-1) / np.maximum(
                1., np.sum(delta[:, tick]**2, axis=-1))
            compatible[:, si] += (relative < 1e-20).astype(np.uint8)
        for scorer in SCORERS:
            arm = scheme + "_" + scorer
            ai = ARMS.index(arm)
            all_score = (exp.old.base.plugin_scores(model, physical, delta)
                         if scorer == "plugin" else
                         exp.receiver.matched(raw+delta, key, operators))
            all_gate = all_score > thresholds[arm]["threshold"]
            attack_gate = all_gate[:, onset:stop]
            earlier_alarm = np.cumsum(attack_gate, axis=1) - attack_gate > 0
            large = np.linalg.norm(delta[:, onset:stop], axis=-1) >= 100
            late_counts[:, ai] = (earlier_alarm & ~attack_gate & large).sum(axis=1)
            for support_index, (_, edges) in enumerate(SUPPORTS):
                attacked = np.array(edges)
                clean = np.setdiff1d(np.arange(6), attacked)
                mask = np.zeros(6, dtype=bool)
                mask[attacked] = True
                score = np.where(mask[None, None], all_score, clean_scores[arm])
                gate = score > thresholds[arm]["threshold"]
                affected = delta * mask[None, None, :, None]
                early = gate[:, onset:onset+3].any(axis=1)
                full = gate[:, onset:stop].any(axis=1)
                early_saved[:, support_index, ai] = early
                full_saved[:, support_index, ai] = full
                early_max[:, support_index, ai] = score[:, onset:onset+3].max(axis=1)
                delay = np.where(full, gate[:, onset:stop].argmax(axis=1), 60)
                delay_saved[:, support_index, ai] = delay
                world = estimator(model, physical, affected,
                                  sender_packets=physical.y, gate_override=gate)
                error = np.sum((world.post-physical.x[:, :, None, :])**2, axis=-1)
                receivers = np.unique(model.links[attacked, 0])
                clean_early = early[:, clean]
                clean_full = full[:, clean]
                row = values[:, support_index, ai]
                row[:, 0] = early[:, attacked].mean(axis=1)
                row[:, 1] = clean_early.mean(axis=1) if len(clean) else 0.
                row[:, 2] = clean_early.any(axis=1)
                row[:, 3] = early[:, attacked].all(axis=1) & ~clean_early.any(axis=1)
                row[:, 4] = full[:, attacked].mean(axis=1)
                row[:, 5] = clean_full.mean(axis=1) if len(clean) else 0.
                row[:, 6] = full[:, attacked].all(axis=1) & ~clean_full.any(axis=1)
                row[:, 7:10] = np.column_stack((
                    error[:, onset:stop, receivers].mean(axis=(1, 2)),
                    error[:, onset:stop, model.receiver_ids].mean(axis=(1, 2)),
                    error[:, onset:stop].mean(axis=(1, 2)),
                ))
                row[:, 10:13] = np.column_stack((
                    error[:, stop:, receivers].mean(axis=(1, 2)),
                    error[:, stop:, model.receiver_ids].mean(axis=(1, 2)),
                    error[:, stop:].mean(axis=(1, 2)),
                ))
                row[:, 13] = compatible[:, si, attacked].any(axis=1)
                row[:, 14] = late_counts[:, ai, attacked].any(axis=1)
                row[:, 16] = row[:, 8] > 1
    negative_mse = values[:, :, ARMS.index("negative_exact"), 8]
    values[..., 15] = (values[..., 8] > 1) & (values[..., 8] >= 10*negative_mse[..., None])
    record.update(values=values, early=early_saved, full=full_saved, early_max=early_max,
                  delay=delay_saved, compatible_windows=compatible, large_late_count=late_counts)
    return record


def worker(arguments):
    return evaluate(*arguments)


def identity(split_name):
    split, count, healthy = SPLITS[split_name]
    return {
        "parent": PARENT, "frozen_physics": exp.old.base.BASELINE_COMMIT,
        "split": split, "namespace": [20261126, 61783, split],
        "paths_per_scene": count, "calibration_per_model": healthy,
        "health_holdout_per_model": healthy, "chunk": CHUNK,
        "population": {"calibration": 0, "health": 1, "attack": 2},
        "arms": list(ARMS), "supports": [[s, list(e)] for s, e in SUPPORTS],
        "metrics": list(METRICS),
        "scenes": [[m, angle, onset] for m in MODELS
                   for angle in (0, -45, 45) for onset in (50, 65)],
        "sources": {os.path.relpath(p, HERE).replace("\\", "/"): sha(p) for p in DEPENDENCIES},
        "physical_fixture_hashes": exp.old.backend("stable")[0].validate_fixtures(),
    }


def run(split_name, workers):
    run_identity = identity(split_name)
    split, count, healthy = SPLITS[split_name]
    folder = HERE / split_name
    folder.mkdir(exist_ok=True)
    if split_name == "confirm":
        lock = json.loads((HERE / "CONFIRM_LOCK.json").read_text(encoding="utf-8"))
        assert lock["sources"] == run_identity["sources"]
        assert lock["development_run_sha256"] == sha(HERE / "dev/RUN.json")
        assert lock["paths_per_scene"] == count
    completed = folder / "RUN.json"
    progress = folder / "PROGRESS.json"
    if completed.exists():
        state = json.loads(completed.read_text(encoding="utf-8"))
        assert state["identity"] == run_identity
        for name, digest in state["archives"].items():
            assert sha(folder / name) == digest
        return
    state = {
        "identity": run_identity, "archives": {}, "thresholds": {},
        "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "host": platform.platform(), "workers": workers,
    }
    if progress.exists():
        state = json.loads(progress.read_text(encoding="utf-8"))
        assert state["identity"] == run_identity
        for name, digest in state["archives"].items():
            assert sha(folder / name) == digest
    else:
        atomic_json(progress, state)
    exp.receiver.kernel()
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for mi, model in enumerate(MODELS):
            for population, size, scenes in ((0, healthy, (0,)), (1, healthy, (0,)),
                                              (2, count, range(6))):
                for scene in scenes:
                    filename = f"{model}_p{population}_s{scene}.npz"
                    if filename in state["archives"]:
                        continue
                    thresholds = state["thresholds"].get(model)
                    jobs = [(split, mi, scene, population, b, min(CHUNK, size-i), thresholds)
                            for b, i in enumerate(range(0, size, CHUNK))]
                    rows = list(pool.map(worker, jobs))
                    arrays = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
                    np.testing.assert_array_equal(arrays["path_id"], np.arange(size))
                    atomic_npz(folder / filename, arrays)
                    state["archives"][filename] = sha(folder / filename)
                    if population == 0:
                        state["thresholds"][model] = {
                            arm: cutoff(arrays["maxima"][:, a]) for a, arm in enumerate(ARMS)}
                    atomic_json(progress, state)
                    print(split_name, filename, flush=True)
    state["wall_seconds"] = time.perf_counter()-started
    atomic_json(completed, state)
    progress.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=tuple(SPLITS))
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    run(arguments.split, arguments.workers)
