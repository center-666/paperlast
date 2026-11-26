"""Frozen D2 policy transfer against model-specific and topology-only fixed weights."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import d2b2_run as frozen

common = frozen.common
Q = common.Q
POLICIES = ("R1", "HC_static", "Class_model", "Class_topology")
NAMESPACE = (20261126, 88403, 1)
OUT = HERE / "results"
OLD_RUN = HERE.parent / "b2_results/B2_RUN.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_old():
    return json.loads(OLD_RUN.read_text(encoding="utf-8"))


def class_indices(spec):
    model_index = 4 if spec[0] == "unstable" else 3
    return (model_index, 4) if spec[1] == "U" else (0, 0)


def static_mask(result):
    cost = np.array(result["static_nominal_health_cost_relative_to_q1"])
    reference = result["static_nominal_health_cost_absolute"][-1]
    return cost <= result["epsilon"]*reference+1e-10


def risk_select(risk, feasible):
    objective = np.where(feasible, risk, np.inf)
    tie = feasible & (objective <= objective.min(axis=1, keepdims=True)+1e-10)
    assert np.all(tie.any(axis=1))
    return 6-np.argmax(tie[:, ::-1], axis=1)


def simulate(model, graph, physical, delta, scores, masks, result):
    n = len(physical.x)
    state = np.zeros((n, 4, 50, 4))
    history = np.full((n, 4), 7**6-1, dtype=np.int32)
    actions = np.full((n, 4, 150), 6, dtype=np.int8)
    losses = np.zeros((n, 4, 5))
    absolute = model.target
    table = result["risk_table"]
    bounds = np.array(table["boundaries"])
    lookup = np.array(table["fallback_cell"])
    upper = np.array(table["simultaneous_upper"])
    alarm_run = np.zeros(n, dtype=int)
    raw = (physical.y.swapaxes(-1, -2) @ model.adjacency.T).swapaxes(-1, -2)
    requested = class_indices(result["spec"])
    health_legal = static_mask(result)
    sender = model.links[6, 1]
    fallback_counts = np.zeros(2, dtype=np.int64)
    for tick in range(1, 150):
        bits = masks[:, tick]
        y = physical.y[:, tick]
        received = raw[:, tick].copy()
        for edge in range(6):
            received[:, model.links[edge, 0]] -= bits[:, edge, None]*y[:, model.links[edge, 1]]
        alarm = bits[:, 6]
        alarm_run = np.where(alarm, alarm_run+1, 0)
        cells = lookup[common.cell_index(scores[:, tick, 6]/result["threshold"], alarm_run, bounds)]
        risk = upper[cells]
        legal = graph[history]
        chosen = np.full((n, 4), 6, dtype=np.int8)
        chosen[alarm, 0] = risk_select(risk[alarm], legal[alarm, 0])
        chosen[alarm, 1] = risk_select(risk[alarm], legal[alarm, 1] & health_legal)
        for p in (2, 3):
            target = requested[p-2]
            allowed = legal[:, p] & (np.arange(7)[None] >= target)
            assert allowed.any(axis=1).all()
            chosen[alarm, p] = allowed[alarm].argmax(axis=1)
            fallback_counts[p-2] += int(np.count_nonzero(alarm & (chosen[:, p] != target)))
        assert np.take_along_axis(legal, chosen[..., None], axis=-1).all()
        actions[:, :, tick] = chosen
        history = (history*7+chosen) % (7**6)
        next_state = np.empty_like(state)
        mode = alarm.astype(int)
        for p in range(4):
            index = chosen[:, p]
            f, k, g = (absolute[name][mode, index] for name in ("F", "K", "G"))
            effective = np.where(alarm, Q[index], 1.)
            packet = received[:, 26] + delta[:, tick, 6] + \
                (effective-1)[:, None]*(y[:, sender]+delta[:, tick, 6])
            next_state[:, p, 26] = (
                np.einsum("bij,bj->bi", f, state[:, p, 26])
                + np.einsum("bij,bj->bi", k, y[:, 26])
                + np.einsum("bij,bj->bi", g, packet)
            )
        global_mode = bits.astype(np.int64) @ common.BITS
        other = (
            np.einsum("bnij,bnj->bni", model.base["F"][global_mode], state[:, 0])
            + np.einsum("bnij,bnj->bni", model.base["K"][global_mode], y)
            + np.einsum("bnij,bnj->bni", model.base["G"][global_mode], received)
        )
        next_state[:, :, :26] = other[:, None, :26]
        next_state[:, :, 27:] = other[:, None, 27:]
        state = next_state
        squared = (state-physical.x[:, tick, None, None])**2
        missing = squared[:, :, 26, 2:].sum(axis=-1)
        network = squared.sum(axis=(2, 3))/50
        if 50 <= tick < 110:
            losses[:, :, 0] += missing/60
            losses[:, :, 3] += network/60
        if tick >= 110:
            losses[:, :, 1] += missing/40
        if tick >= 20:
            losses[:, :, 2] += missing/130
            losses[:, :, 4] += network/130
    return losses, actions, fallback_counts


def deterministic_audit():
    old = load_old()
    rows = []
    with np.load(frozen.modes.OUTPUT / "mode_tables.npz", allow_pickle=False) as gains:
        for scene in range(12):
            result = old["results"][str(scene)]
            model, graph, _ = common.configure(result["spec"], gains)
            index, _ = class_indices(result["spec"])
            mask = static_mask(result)
            assert np.flatnonzero(mask)[0] == index
            risk = np.array(result["risk_table"]["simultaneous_upper"])
            mismatches = 0
            legality_failures = 0
            for word in itertools.product((index, 6), repeat=6):
                history = int(np.array(word) @ 7**np.arange(5, -1, -1))
                legal = graph[history]
                legality_failures += int(not legal[index])
                for cell in range(len(risk)):
                    feasible = legal & mask
                    best = risk[cell, feasible].min()
                    chosen = np.flatnonzero(feasible & (risk[cell] <= best+1e-10))[-1]
                    mismatches += int(chosen != index)
            rows.append({"scene": scene, "spec": result["spec"],
                         "class_model_q": float(Q[index]), "histories": 64,
                         "risk_cells": len(risk), "m7_failures": legality_failures,
                         "risk_decision_mismatches": mismatches})
            # Separate-path regression against the frozen and literal implementations.
            if scene in (0, 6):
                namespace = (20261126, 88403, 99)
                physical, delta, score, masks = common.generate_paths(
                    model, result["spec"], scene, "burst", 64, 2,
                    threshold=result["threshold"], namespace=namespace)
                losses, actions, _ = simulate(model, graph, physical, delta, score, masks, result)
                literal = common.literal_policy_replay(model, physical, delta, masks, actions)
                np.testing.assert_allclose(losses, literal, rtol=2e-9, atol=2e-8)
                inherited = frozen.simulate_constrained(
                    model, graph, physical, score, masks, delta, result["risk_table"],
                    result["threshold"], result["epsilon"])
                np.testing.assert_array_equal(actions[:, :2], inherited[1][:, :2])
                np.testing.assert_allclose(losses[:, :2], inherited[0][:, :2], rtol=2e-12, atol=2e-12)
    output = {"status": "PASS", "source_sha256": sha(Path(__file__)),
              "frozen_B2_run_sha256": sha(OLD_RUN),
              "rows": rows, "universal_action_equivalence_with_frozen_tables":
              all(r["m7_failures"] == r["risk_decision_mismatches"] == 0 for r in rows)}
    common.atomic_json(HERE / "DETERMINISTIC_AUDIT.json", output)
    print(json.dumps(output, indent=2))


def run():
    old = load_old()
    OUT.mkdir(exist_ok=True)
    identity = {
        "namespace": NAMESPACE, "D2_frozen": "d6d958348ec3bbc73fe42df0a9e9cf8a337a8d78",
        "D1_frozen": "7e32a41b9802438f6645d9048ee46386caa53418",
        "attack_paths_per_scene_fixture": 512, "health_paths_per_scene": 1024,
        "policies": POLICIES, "source_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(HERE / "PROTOCOL.md"),
        "old_run_sha256": sha(OLD_RUN), "old_input_hashes": frozen.current_input_hashes(),
    }
    identity = json.loads(json.dumps(identity))
    progress = OUT / "PROGRESS.json"
    final = OUT / "RUN.json"
    state = {"identity": identity, "archives": {}, "results": {},
             "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
             "python": sys.version, "numpy": np.__version__, "host": platform.platform()}
    if final.exists():
        previous = json.loads(final.read_text(encoding="utf-8"))
        assert previous["identity"] == identity
        for name, digest in previous["archives"].items():
            assert sha(OUT / name) == digest
        return
    if progress.exists():
        state = json.loads(progress.read_text(encoding="utf-8"))
        assert state["identity"] == identity
    else:
        common.atomic_json(progress, state)
    with np.load(frozen.modes.OUTPUT / "mode_tables.npz", allow_pickle=False) as gains:
        for scene in range(12):
            filename = f"scene_{scene:02}.npz"
            if filename in state["archives"]:
                assert sha(OUT / filename) == state["archives"][filename]
                continue
            result = old["results"][str(scene)]
            model, graph, _ = common.configure(result["spec"], gains)
            arrays = {}
            summary = {"spec": result["spec"]}
            for fixture in ("continuous", "burst", "health"):
                if fixture == "health":
                    physical, scores, masks = common.generate_health(
                        model, scene, result["threshold"], 1, 1024, NAMESPACE)
                    delta = np.zeros((1024, 150, 7, 2))
                else:
                    physical, delta, scores, masks = common.generate_paths(
                        model, result["spec"], scene, fixture, 512, 2,
                        threshold=result["threshold"], namespace=NAMESPACE)
                loss, actions, fallback = simulate(
                    model, graph, physical, delta, scores, masks, result)
                arrays[fixture+"_loss"] = loss
                arrays[fixture+"_actions"] = actions
                arrays[fixture+"_masks"] = masks.astype(np.uint8)
                arrays[fixture+"_physical_probe"] = physical.y[:, 1, 0]
                summary[fixture] = {
                    "paths": len(loss), "class_fallbacks": fallback.tolist(),
                    "HC_class_action_mismatches": int(np.count_nonzero(actions[:, 1] != actions[:, 2])),
                    "HC_class_max_loss_difference": float(np.max(np.abs(loss[:, 1]-loss[:, 2]))),
                    "loss_means": loss.mean(axis=0).tolist(),
                }
            common.atomic_npz(OUT / filename, arrays)
            state["archives"][filename] = sha(OUT / filename)
            state["results"][str(scene)] = summary
            common.atomic_json(progress, state)
            print("D2 transfer", filename, flush=True)
    common.atomic_json(final, state)
    progress.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        deterministic_audit()
    else:
        run()
