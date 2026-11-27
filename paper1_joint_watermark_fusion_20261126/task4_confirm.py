"""Independent confirmatory 2x2 for frozen four-state watermarks and fusion."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

import frontend
import models_quality as quality
import task3

HERE = Path(__file__).resolve().parent
OUT = HERE / "task4_confirm"
ROOT_SEED = (20261126, 94017, 85)
BITS = quality.frozen.BITS
N = 512
HEALTH_N = 512
BATCH = 32
WATERMARKS = ("iid", "no_repeat")
POLICIES = ("A_q0_raw", "A_M7_min", "S_fixed_table_T", "q1_full")
FIXTURES = ("continuous", "burst")
QUALITY_ORDER = task3.QUALITY_ORDER


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    names = (
        "TASK4_CONFIRM_PROTOCOL.md", "TASK4_SMOKE.json",
        "task4_confirm.py", "verify_task4.py", "analyze_task4.py",
        "prepare_task4_lock.py",
        "task3.py", "task3_dev/RUN.json", "task3_dev/ANALYSIS.json",
        "task3_dev/VERIFY.json", "TASK3_RESULTS_ZH.md",
        "models_quality.py", "quality_models/MODELS.json",
        "frontend.py", "projection.c", "frontend_health/RUN.json",
        "FRONTEND_AUDIT.json", "TASK3_DEVELOPMENT_PROTOCOL.md",
    )
    return {name: sha(HERE/name) for name in names}


def atomic_json(path, value):
    quality.frozen.atomic_json(path, value)


def atomic_npz(path, arrays):
    quality.frozen.atomic_npz(path, arrays)


def target_request(row):
    if row["topology"] == "R":
        return 0.
    return .8 if row["dynamics"] == "unstable" else .6


def thresholds_for(config_index, health_run):
    entries = health_run["results"][config_index]["rows"]
    return {
        "iid": float(next(r["threshold"] for r in entries
                          if r["family"] == "iid_matched")),
        "no_repeat": float(next(r["threshold"] for r in entries
                                if r["family"] == "no_repeat_matched")),
    }


def make_batch(config_index, fixture_index, purpose, batch, count,
               model, operators, thresholds):
    seed = [*ROOT_SEED, config_index, fixture_index, purpose, batch]
    paths = quality.physical(model, count, [*seed, 0])
    uniforms = np.random.default_rng(np.random.SeedSequence([*seed, 1])).random(
        (count, 150, 50))
    raw = paths.y[:, :, model.links[:, 1]]
    active = np.ones(60) if fixture_index == 0 else (np.arange(60) % 10 < 6).astype(float)
    keys, deltas, scores, gates = {}, {}, {}, {}
    for watermark in WATERMARKS:
        all_keys = frontend.keys(uniforms, watermark == "no_repeat")
        key = all_keys[:, :, model.links[:, 1]]
        delta = np.zeros_like(raw)
        if purpose == 1:
            delta[:, 50:110, 6] = (
                key[:, 50:110, 6] @
                (active[:, None]*np.array([6., 0.]))[None, ..., None]
            )[..., 0]
        score = frontend.scores(raw+delta, key, operators)
        keys[watermark], deltas[watermark] = key, delta
        scores[watermark] = score
        gates[watermark] = score > thresholds[watermark]
    h1_delta = deltas["no_repeat"] if purpose == 1 else np.zeros_like(raw)
    h1_scores = quality.frozen.common.h1_score(model, paths, h1_delta)
    return {
        "paths": paths, "raw": raw, "keys": keys, "deltas": deltas,
        "scores": scores, "gates": gates, "h1_scores": h1_scores,
        "active": active, "uniform_probe": uniforms[:, 1, 0],
        "physical_probe": paths.y[:, 1, 0],
    }


def simulate_policy(model, graph, data, watermark, policy, requested):
    paths = data["paths"]
    delta = data["deltas"][watermark]
    gates = data["gates"][watermark]
    n = len(paths.x)
    state = np.zeros((n, 50, 4))
    history = np.full(n, 7**6-1, dtype=np.int64)
    actions = np.full((n, 150), 6, dtype=np.int8)
    losses = np.zeros((n, 8), dtype=float)
    active = data["active"]
    raw_network = (paths.y.swapaxes(-1, -2) @ model.adjacency.T).swapaxes(-1, -2)
    target_sender = int(model.links[6, 1])
    m7_violations = 0
    m7_fallbacks = 0
    request_index = int(np.argmin(abs(quality.Q-requested)))
    for tick in range(1, 150):
        bits = gates[:, tick]
        alarm = bits[:, 6]
        y = paths.y[:, tick]
        received = raw_network[:, tick].copy()
        for edge in range(6):
            received[:, model.links[edge, 0]] -= bits[:, edge, None]*y[:, model.links[edge, 1]]
        legal = graph[history]
        if policy == "A_q0_raw":
            chosen = np.where(alarm, 0, 6).astype(np.int8)
        elif policy == "q1_full":
            chosen = np.full(n, 6, dtype=np.int8)
        else:
            chosen = np.full(n, 6, dtype=np.int8)
            minimum_index = 0 if policy == "A_M7_min" else request_index
            for path in np.flatnonzero(alarm):
                candidates = np.flatnonzero(legal[path] & (np.arange(7) >= minimum_index))
                if not candidates.size:
                    raise AssertionError("The M7 graph must keep q=1 available.")
                chosen[path] = int(candidates[0])
                m7_fallbacks += int(chosen[path] != minimum_index)
        m7_violations += int(np.count_nonzero(
            alarm & ~legal[np.arange(n), chosen]))
        actions[:, tick] = chosen
        history = (history*7+chosen) % (7**6)
        effective_q = np.where(alarm, quality.Q[chosen], 1.)
        mode = alarm.astype(int)
        f = model.target["F"][mode, chosen]
        k = model.target["K"][mode, chosen]
        g = model.target["G"][mode, chosen]
        packet = (received[:, 26]+delta[:, tick, 6]
                  +(effective_q-1)[:, None]*(y[:, target_sender]+delta[:, tick, 6]))
        updated = (
            np.einsum("bij,bj->bi", f, state[:, 26])
            + np.einsum("bij,bj->bi", k, y[:, 26])
            + np.einsum("bij,bj->bi", g, packet)
        )
        all_mode = bits.astype(np.int64) @ BITS
        next_all = (
            np.einsum("bnij,bnj->bni", model.base["F"][all_mode], state)
            + np.einsum("bnij,bnj->bni", model.base["K"][all_mode], y)
            + np.einsum("bnij,bnj->bni", model.base["G"][all_mode], received)
        )
        next_state = next_all
        next_state[:, 26] = updated
        state = next_state
        squared = (state-paths.x[:, tick, None, :])**2
        missing = squared[:, 26, 2:].sum(axis=-1)
        network = squared.sum(axis=(1, 2))/50
        if 50 <= tick < 110:
            losses[:, 0] += missing/60
            losses[:, 3] += network/60
            if active[tick-50]:
                losses[:, 1] += missing/max(float(active.sum()), 1.)
            else:
                losses[:, 2] += missing/max(float(60-active.sum()), 1.)
            losses[:, 7] = np.maximum(losses[:, 7], squared[:, 26, 2:].max(axis=-1))
        elif tick >= 110:
            losses[:, 4] += missing/40
        if tick >= 20:
            losses[:, 5] += missing/130
            losses[:, 6] += network/130
    return {"actions": actions, "losses": losses,
            "M7_violations": m7_violations, "M7_fallbacks": m7_fallbacks}


def simulate_cells(model, graph, data, requested):
    cells = {}
    for watermark in WATERMARKS:
        cells[watermark] = {}
        for policy in POLICIES:
            cells[watermark][policy] = simulate_policy(
                model, graph, data, watermark, policy, requested)
    return cells


def smoke():
    health_run = json.loads((HERE/"frontend_health/RUN.json").read_text(encoding="utf-8"))
    rows = []
    for config_index in (0, 2, 3, 11):
        model = task3.audit_frontend.load_model(config_index)
        operators = frontend.operators(model)
        threshold = thresholds_for(config_index, health_run)
        model_record = task3.read_quality_model(config_index)
        graph = task3.configure(config_index)[1]
        sample = make_batch(config_index, 0, 99, 0, 16, model, operators, threshold)
        row = {wm: sample["gates"][wm] for wm in WATERMARKS}
        assert sample["paths"].x.shape == (16, 150, 4)
        for watermark in WATERMARKS:
            for policy in POLICIES:
                output = simulate_policy(model, graph, sample, watermark, policy,
                                         target_request(model_record))
                assert np.isfinite(output["losses"]).all()
                assert output["M7_violations"] == 0 if policy != "A_q0_raw" else True
        for policy in ("A_q0_raw", "S_fixed_table_T"):
            # Paired fusion actions on a single watermark row share the identical alarm array.
            assert np.array_equal(row["iid"], sample["gates"]["iid"])
        rows.append({"configuration_index": config_index, "name": model_record["name"],
                     "paths": 16, "policies": len(WATERMARKS)*len(POLICIES),
                     "finite_outputs": True, "formal_paths_generated": False})
        print("task4 smoke", model_record["name"], "passed", flush=True)
    atomic_json(HERE/"TASK4_SMOKE.json", {
        "status": "PASS", "source_sha256": sha(Path(__file__)),
        "namespace": [*ROOT_SEED, 99], "configurations": rows,
        "formal_confirmation_paths_generated": False,
    })


def run():
    health_run = json.loads((HERE/"frontend_health/RUN.json").read_text(encoding="utf-8"))
    lock = json.loads((HERE/"TASK4_CONFIRM_LOCK.json").read_text(encoding="utf-8"))
    frozen_sources = source_hashes()
    assert lock["selection"] == "S_fixed_table_T"
    assert lock["source_hashes"] == frozen_sources
    assert lock["task3_RUN_sha256"] == sha(HERE/"task3_dev/RUN.json")
    assert lock["task3_ANALYSIS_sha256"] == sha(HERE/"task3_dev/ANALYSIS.json")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", lock["lock_creation_parent_commit"],
         subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()],
        cwd=HERE, check=True)
    run_identity = {
        "namespace": list(ROOT_SEED), "policies": list(POLICIES), "watermarks": list(WATERMARKS),
        "paths_per_config_fixture": N, "healthy_paths_per_config": HEALTH_N,
        "thresholds_run_sha256": sha(HERE/"frontend_health/RUN.json"),
        "task3_run_sha256": sha(HERE/"task3_dev/RUN.json"),
        "confirm_lock_sha256": sha(HERE/"TASK4_CONFIRM_LOCK.json"),
        "selection": "S_fixed_table_T", "source_hashes": frozen_sources,
        "quality_configuration_order": QUALITY_ORDER,
    }
    out = OUT
    out.mkdir(exist_ok=True)
    progress = out/"PROGRESS.json"
    final = out/"RUN.json"
    state = {
        "identity": run_identity, "archives": {}, "configs": {},
        "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    cwd=HERE, text=True).strip(),
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "start_time": time.time(),
    }
    if final.exists():
        previous = json.loads(final.read_text(encoding="utf-8"))
        assert previous["identity"] == run_identity
        for filename, expected in previous["archives"].items():
            assert sha(out/filename) == expected
        return
    if progress.exists():
        state = json.loads(progress.read_text(encoding="utf-8"))
        assert state["identity"] == run_identity
        for filename, expected in state["archives"].items():
            assert sha(out/filename) == expected
    else:
        atomic_json(progress, state)
    with np.load(quality.frozen.modes.OUTPUT/"mode_tables.npz", allow_pickle=False) as gains:
        for config_index, name in enumerate(QUALITY_ORDER):
            if f"{name}.npz" in state["archives"]:
                continue
            model = task3.audit_frontend.load_model(config_index)
            operators = frontend.operators(model)
            thresholds = thresholds_for(config_index, health_run)
            quality_record = task3.read_quality_model(config_index)
            graph = task3.configure(config_index)[1]
            arrays = {}
            summaries = {"name": name, "request_q": target_request(quality_record),
                         "fixtures": {}, "thresholds": thresholds}
            for fixture_index, fixture in enumerate(FIXTURES):
                for purpose, path_count, label in ((1, N, "attack"), (2, HEALTH_N, "health")):
                    blocks = []
                    for batch in range(path_count//BATCH):
                        data = make_batch(config_index, fixture_index, purpose, batch,
                                          BATCH, model, operators, thresholds)
                        per_watermark = simulate_cells(model, graph, data,
                                                       target_request(quality_record))
                        block = {
                            "path_id": np.arange(batch*BATCH, (batch+1)*BATCH),
                            "configuration_index": np.full(BATCH, config_index, dtype=np.int16),
                            "x": data["paths"].x, "y": data["paths"].y,
                            "physical_probe": data["physical_probe"],
                            "uniform_key_probe": data["uniform_probe"],
                            "h1_scores": data["h1_scores"],
                        }
                        for watermark in WATERMARKS:
                            block[watermark+"_keys"] = data["keys"][watermark]
                            block[watermark+"_delta"] = data["deltas"][watermark]
                            block[watermark+"_scores"] = data["scores"][watermark]
                            block[watermark+"_gates"] = data["gates"][watermark].astype(np.uint8)
                            for policy in POLICIES:
                                result = per_watermark[watermark][policy]
                                tag = watermark+"_"+policy
                                block[tag+"_actions"] = result["actions"]
                                block[tag+"_losses"] = result["losses"]
                                block[tag+"_M7_violations"] = np.full(BATCH, result["M7_violations"],
                                                                      dtype=np.int64)
                                block[tag+"_M7_fallbacks"] = np.full(BATCH, result["M7_fallbacks"],
                                                                     dtype=np.int64)
                        blocks.append(block)
                    filename = f"{name}_{fixture}_{label}.npz"
                    packed = {key: np.concatenate([b[key] for b in blocks], axis=0)
                              for key in blocks[0]}
                    atomic_npz(out/filename, packed)
                    state["archives"][filename] = sha(out/filename)
                    if label == "attack":
                        summaries["fixtures"][fixture] = {"paths": path_count, "archive": filename}
                    else:
                        summaries["health_"+fixture] = {"paths": path_count, "archive": filename}
                    atomic_json(progress, state)
                    print("confirm", name, fixture, label, flush=True)
            state["configs"][name] = summaries
            atomic_json(progress, state)
    state["status"] = "complete_task4_confirmatory_data"
    state["wall_seconds"] = time.time()-state["start_time"]
    atomic_json(final, state)
    progress.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
    else:
        run()
