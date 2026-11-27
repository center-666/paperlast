"""Paired 4D development of hard rejection, fixed table and health/risk weights."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import audit_frontend
import frontend
import models_quality as quality

HERE = Path(__file__).resolve().parent
SEED = (20261126, 94017, 83)
TRAIN = 256
EVALUATE = 256
HEALTH = 256
BATCH = 32
POLICIES = ("A_q0_raw", "A_M7_min", "T_fixed_table",
            "H_fixed_epsilon", "V_updated_epsilon",
            "D0_duration", "D1_match_score", "q1_full")
PHASES = ("full", "active", "inactive")
Q = quality.frozen.Q
BITS = quality.frozen.BITS
ROOT = HERE / "task3_dev"
QUALITY_ORDER = [
    f"{topology}_{dynamics}_R35x{factor:g}"
    for dynamics, topology, factor in (
        (d, t, f) for d in ("unstable", "stable")
        for t in ("U", "R") for f in (.5, 1., 2.))
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    quality.frozen.atomic_json(path, value)


def identity():
    health_run = quality.json.loads(
        (HERE / "frontend_health/RUN.json").read_text(encoding="utf-8"))
    return {
        "registration_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        cwd=HERE, text=True).strip(),
        "namespace": SEED, "quality_configuration_order": QUALITY_ORDER,
        "paths_per_fixture_fit": TRAIN, "paths_per_fixture_evaluation": EVALUATE,
        "healthy_evaluation_paths": HEALTH, "policies": POLICIES,
        "source_sha256": {
            file.name: digest(file) for file in (
                HERE / "TASK3_DEVELOPMENT_PROTOCOL.md", HERE / "task3.py",
                HERE / "analyze_task3.py", HERE / "audit_task3.py",
                HERE / "models_quality.py", HERE / "frontend.py",
                HERE / "projection.c",
            )
        },
        "quality_model_index_sha256": digest(HERE / "quality_models/MODELS.json"),
        "frontend_health_run_sha256": digest(HERE / "frontend_health/RUN.json"),
        "frontend_calibration_archives": health_run["archives"],
        "legacy_h1_and_risk_inputs": quality.frozen.current_input_hashes(),
    }


def key_schedule(config_index, fixture_index, purpose, batch, n):
    root = [*SEED, config_index, fixture_index, purpose, batch]
    uniforms = np.random.default_rng(np.random.SeedSequence([*root, 2])).random((n, 150, 50))
    return frontend.keys(uniforms, True)[:, :, :]


def generate(config_index, fixture_index, purpose, batch, n, model, ops, thresholds):
    paths = quality.physical(
        model, n, [*SEED, config_index, fixture_index, purpose, batch, 0])
    keys_all = key_schedule(config_index, fixture_index, purpose, batch, n)
    keys = keys_all[:, :, model.links[:, 1]]
    raw = paths.y[:, :, model.links[:, 1]]
    active = np.ones(60) if fixture_index == 0 else (np.arange(60) % 10 < 6).astype(float)
    delta = np.zeros_like(raw)
    delta[:, 50:110, 6] = (keys[:, 50:110, 6] @
                             (active[:, None]*np.array([6., 0.]))[None, ..., None])[..., 0]
    if purpose == 6:
        delta.fill(0.)
    scores = quality.frozen.common.h1_score(model, paths, delta)
    matched = frontend.scores(raw+delta, keys, ops)
    h1_gate = scores > thresholds["H1"]
    matched_gate = matched > thresholds["no_repeat_matched"]
    # Target masks are the matched receiver stream. H1 remains a frozen diagnostic.
    gates = matched_gate
    return {
        "paths": paths, "delta": delta, "scores": scores, "matched": matched,
        "h1_gate": h1_gate, "gates": gates, "keys": keys, "active": active,
    }


def frontend_thresholds(config_index):
    health_dir = HERE / "frontend_health"
    with np.load(health_dir / f"config{config_index:02}_purpose2.npz") as data:
        return {
            family: float(np.sort(data["maxima"][:, i])[int(np.ceil(2049*.95))-1])
            for i, family in enumerate(("H1", "iid_matched", "no_repeat_matched"))
        }


def duration_code(run):
    return np.where(run == 1, 0, np.where(run <= 3, 1, 2))


def fit_risks(training, threshold, mode):
    # Inputs contain the same source-only configuration and independent physical paths.
    pooled_scores = np.concatenate([
        row["matched"][:, 50:91, 6][row["runs"][:, 50:91] > 0]/threshold
        for row in training
    ])
    boundaries = np.quantile(pooled_scores, (1/3, 2/3))
    path_cells = []
    duration_cells = []
    pooled_path_means = []
    for row in training:
        score = row["matched"][:, 50:91, 6]/threshold
        run = row["runs"][:, 50:91]
        alarm = run > 0
        duration = duration_code(run)
        score_bin = np.searchsorted(boundaries, score, side="right")
        labels = row["labels"][:, 50:91]
        values = np.full((len(score), 9 if mode == "D1_match_score" else 3, 7), np.nan)
        duration_values = np.full((len(score), 3, 7), np.nan)
        path_pooled = np.full((len(score), 7), np.nan)
        for path in range(len(score)):
            sums = np.zeros_like(values[path])
            counts = np.zeros(values.shape[1], dtype=int)
            duration_sums = np.zeros((3, 7))
            duration_counts = np.zeros(3, dtype=int)
            for tick in range(score.shape[1]):
                if not alarm[path, tick]:
                    continue
                d = int(duration[path, tick])
                duration_sums[d] += labels[path, tick]
                duration_counts[d] += 1
                cell = (3*score_bin[path, tick]+duration[path, tick]
                        if mode == "D1_match_score" else duration[path, tick])
                sums[cell] += labels[path, tick]
                counts[cell] += 1
            valid = counts > 0
            values[path, valid] = sums[valid]/counts[valid, None]
            valid_duration = duration_counts > 0
            duration_values[path, valid_duration] = (
                duration_sums[valid_duration]/duration_counts[valid_duration, None])
            if alarm[path].any():
                path_pooled[path] = labels[path, alarm[path]].mean(axis=0)
        path_cells.append(values)
        duration_cells.append(duration_values)
        pooled_path_means.append(path_pooled)
    joined = np.concatenate(path_cells, axis=0)
    total = np.zeros((9 if mode == "D1_match_score" else 3, 7))
    support = np.zeros(len(total), dtype=int)
    means = np.full_like(total, np.nan)
    for cell in range(len(total)):
        selected = joined[:, cell, 0]
        valid = np.isfinite(selected)
        support[cell] = int(valid.sum())
        if valid.any():
            means[cell] = np.nanmean(joined[valid, cell], axis=0)
    pooled = np.concatenate(pooled_path_means, axis=0)
    pooled_valid = np.isfinite(pooled[:, 0])
    overall = pooled[pooled_valid].mean(axis=0) if pooled_valid.any() else np.zeros(7)
    duration_joined = np.concatenate(duration_cells, axis=0)
    duration_means = np.zeros((3, 7))
    duration_support = np.zeros(3, dtype=int)
    for duration_bin in range(3):
        valid = np.isfinite(duration_joined[:, duration_bin, 0])
        duration_support[duration_bin] = int(valid.sum())
        if valid.any():
            duration_means[duration_bin] = np.nanmean(duration_joined[valid, duration_bin], axis=0)
    if mode == "H_pooled":
        return {"mode": mode, "means": overall[None, :],
                "support_paths": np.array([int(pooled_valid.sum())]),
                "fallback": ["all_alarm_paths"],
                "score_tertiles": boundaries}
    fallback = []
    if mode == "D0_duration":
        risk = np.zeros((3, 7))
        for d in range(3):
            if duration_support[d] >= 30:
                risk[d] = duration_means[d]
                fallback.append("duration")
            else:
                risk[d] = overall
                fallback.append("pooled")
        return {"mode": mode, "means": risk, "support_paths": support,
                "fallback": fallback, "score_tertiles": boundaries}
    risk = np.zeros((9, 7))
    for cell in range(9):
        duration = cell % 3
        if support[cell] >= 30:
            risk[cell] = means[cell]
            fallback.append("score_by_duration")
        elif duration_support[duration] >= 30:
            risk[cell] = duration_means[duration]
            fallback.append("duration_parent")
        else:
            risk[cell] = overall
            fallback.append("pooled")
    return {"mode": mode, "means": risk, "support_paths": support,
            "fallback": fallback, "score_tertiles": boundaries}


def risk_cell(risk, score, run, threshold):
    if risk["mode"] == "H_pooled":
        return 0
    d = int(duration_code(np.array([run]))[0])
    if risk["mode"] == "D0_duration":
        return d
    s = int(np.searchsorted(risk["score_tertiles"], score/threshold, side="right"))
    return 3*s+d


def simulate(model, graph, data, row, risks):
    paths, delta, scores, gates = (data[k] for k in ("paths", "delta", "scores", "gates"))
    n = len(paths.x)
    states = np.zeros((n, 8, 50, 4))
    histories = np.full((n, 8), 7**6-1, dtype=np.int32)
    runs = np.zeros((n, 150), dtype=np.int16)
    for tick in range(1, 150):
        runs[:, tick] = np.where(gates[:, tick, 6], runs[:, tick-1]+1, 0)
    actions = np.full((n, 8, 150), 6, dtype=np.int8)
    values = np.zeros((n, 8, 10))
    alarm_decisions = np.zeros((n, 150), dtype=np.uint8)
    selected_histories = np.zeros((n, 8, 150), dtype=np.int32)
    m7_violations = np.zeros(8, dtype=np.int64)
    budget_violations = np.zeros(8, dtype=np.int64)
    m7_fallbacks = np.zeros(8, dtype=np.int64)
    max_missing_error_sq = np.zeros((n, 8))
    masks = np.array(row["policy_masks"], dtype=bool)
    risks_map = {item["mode"]: item for item in risks}
    requested_q = float(row["fixed_T_q"])
    requested_index = int(np.argmin(abs(quality.Q-requested_q)))
    target_sender = int(model.links[6, 1])
    active = data["active"]
    raw = (paths.y.swapaxes(-1, -2) @ model.adjacency.T).swapaxes(-1, -2)
    thresholds = data["thresholds"]
    for tick in range(1, 150):
        bits = gates[:, tick]
        alarm = bits[:, 6]
        alarm_decisions[:, tick] = alarm
        y = paths.y[:, tick]
        received = raw[:, tick].copy()
        for edge in range(6):
            received[:, model.links[edge, 0]] -= bits[:, edge, None]*y[:, model.links[edge, 1]]
        legal = graph[histories]
        selected = np.full((n, 8), 6, dtype=np.int8)
        # A: raw q=0, irrespective of the M7 numerical screen.
        selected[alarm, 0] = 0
        # A-M7: smallest legal candidate.
        selected[alarm, 1] = legal[alarm, 1].argmax(axis=1)
        req = requested_index
        candidates = legal & (np.arange(7)[None, None, :] >= req)
        assert candidates.any(axis=2).all()
        selected[alarm, 2] = candidates[alarm, 2].argmax(axis=1)
        if alarm.any():
            m7_fallbacks[2] += int(np.count_nonzero(selected[alarm, 2] != req))
        # H and the two score policies use updated epsilon health-feasible masks.
        for policy, constraint in ((3, 1), (4, 2), (5, 2), (6, 2)):
            feasible = legal[:, policy] & masks[constraint, histories[:, policy]]
            if not feasible.any(axis=1).all():
                feasible[:, 6] = legal[:, policy, 6]
            mode = "H_pooled" if policy == 3 else ("H_pooled" if policy == 4 else
                   ("D0_duration" if policy == 5 else "D1_match_score"))
            table = risks_map[mode]
            cell_idx = np.array([risk_cell(table, data["matched"][p, tick, 6],
                                           runs[p, tick], thresholds["no_repeat_matched"])
                                 for p in range(n)])
            risk_by_path = table["means"][cell_idx]
            objective = np.where(feasible, risk_by_path, np.inf)
            tied = feasible & (objective <= objective.min(axis=1, keepdims=True)+1e-10)
            selected[alarm, policy] = 6-np.argmax(tied[alarm, ::-1], axis=1)
            if alarm.any():
                budget_violations[policy] += int(np.count_nonzero(~feasible[alarm, selected[alarm, policy]]))
                m7_fallbacks[policy] += int(np.count_nonzero(~legal[alarm, policy, selected[alarm, policy]]))
        selected[alarm, 7] = 6
        if not legal[np.arange(n)[:, None], np.arange(8)[None], selected].all():
            # Raw q=0 is a diagnostic and deliberately not M7-projected.
            invalid = ~legal[np.arange(n)[:, None], np.arange(8)[None], selected]
            m7_violations += invalid.sum(axis=0)
            assert invalid[:, 0].any() or not invalid.any()
            assert not invalid[:, 1:].any()
        actions[:, :, tick] = selected
        selected_histories[:, :, tick] = histories
        histories = (histories*7+selected) % (7**6)
        q_actual = np.where(alarm[:, None], quality.Q[selected], 1.)
        next_state = np.empty_like(states)
        for policy in range(8):
            mode_bit = alarm.astype(int)
            index = selected[:, policy]
            f = model.target["F"][mode_bit, index]
            k = model.target["K"][mode_bit, index]
            g = model.target["G"][mode_bit, index]
            packet = received[:, 26]+delta[:, tick, 6]+(q_actual[:, policy]-1)[:, None]*(
                y[:, target_sender]+delta[:, tick, 6])
            next_state[:, policy, 26] = (
                np.einsum("bij,bj->bi", f, states[:, policy, 26])
                + np.einsum("bij,bj->bi", k, y[:, 26])
                + np.einsum("bij,bj->bi", g, packet)
            )
        global_mode = bits.astype(np.int64) @ BITS
        other = (
            np.einsum("bnij,bnj->bni", model.base["F"][global_mode], states[:, 0])
            + np.einsum("bnij,bnj->bni", model.base["K"][global_mode], y)
            + np.einsum("bnij,bnj->bni", model.base["G"][global_mode], received)
        )
        next_state[:, :, :26] = other[:, None, :26]
        next_state[:, :, 27:] = other[:, None, 27:]
        states = next_state
        squared = (states-paths.x[:, tick, None, None, :])**2
        missing = squared[:, :, 26, 2:].sum(axis=-1)
        max_missing_error_sq = np.maximum(
            max_missing_error_sq, squared[:, :, 26, 2:].max(axis=-1))
        network = squared.sum(axis=(2, 3))/50
        attack_active = tick >= 50 and tick < 110
        if attack_active:
            values[:, :, 0] += missing/60
            values[:, :, 3] += network/60
            if active[tick-50]:
                values[:, :, 1] += missing/max(active.sum(), 1)
            else:
                inactive = 60-active.sum()
                values[:, :, 2] += missing/max(inactive, 1)
        elif tick >= 110:
            values[:, :, 4] += missing/40
        if tick >= 20:
            values[:, :, 5] += missing/130
            values[:, :, 6] += network/130
    values[:, :, 7] = (actions[:, :, 50:110] != 6).mean(axis=2)
    values[:, :, 8] = (actions[:, :, 50:110] == 0).mean(axis=2)
    values[:, :, 9] = max_missing_error_sq
    return {"losses": values, "actions": actions, "gates": gates.astype(np.uint8),
            "target_alarm": alarm_decisions, "histories": selected_histories,
            "M7_violations": m7_violations, "budget_violations": budget_violations,
            "M7_fallbacks": m7_fallbacks}

def batch_data(model, ops, config_index, fixture_index, purpose, n, thresholds, first_batch=0):
    if n % BATCH:
        raise ValueError("Path count must be divisible by the registered batch size.")
    parts = [generate(config_index, fixture_index, purpose, first_batch+b, BATCH,
                      model, ops, thresholds) for b in range(n//BATCH)]
    return {
        "paths": type(parts[0]["paths"])(
            x=np.concatenate([p["paths"].x for p in parts]),
            y=np.concatenate([p["paths"].y for p in parts])),
        **{key: np.concatenate([p[key] for p in parts])
           for key in ("delta", "scores", "matched", "h1_gate", "gates", "keys")},
        "active": parts[0]["active"], "thresholds": thresholds,
    }


def training_data(model, ops, config_index, fixture_index, thresholds):
    data = batch_data(model, ops, config_index, fixture_index, 4, TRAIN, thresholds)
    data["labels"] = quality.frozen.common.single_step_damage(
        model, data["paths"], data["delta"], data["gates"])
    runs = np.zeros((TRAIN, 150), dtype=np.int16)
    for tick in range(1, 150):
        runs[:, tick] = np.where(data["gates"][:, tick, 6], runs[:, tick-1]+1, 0)
    data["runs"] = runs
    return data


def read_quality_model(index):
    registry = quality.json.loads((HERE / "quality_models/MODELS.json").read_text(encoding="utf-8"))
    return registry["configurations"][index]


def configure(index):
    record = read_quality_model(index)
    path = HERE / "quality_models" / record["name"] / \
        f"{record['topology']}_{record['dynamics']}_target35_qwords.npz"
    with np.load(path) as archive:
        graph = archive["legal"].copy()
    model_archive = HERE / "quality_models" / f"{record['name']}.npz"
    with np.load(model_archive) as archive:
        masks = archive["policy_masks"].copy()
    return record, graph, masks


def summarize(data, simulated, health=False):
    losses = simulated["losses"]
    alarm = data["gates"][:, :, 6].astype(bool)
    action_counts = np.zeros((len(POLICIES), 7), dtype=np.int64)
    for policy in range(len(POLICIES)):
        action_counts[policy] = np.bincount(
            simulated["actions"][:, policy][alarm], minlength=7)
    pairs = ((0, 1), (1, 2), (3, 4), (5, 6))
    disagreements = {}
    for left, right in pairs:
        different = simulated["actions"][:, left] != simulated["actions"][:, right]
        disagreements[f"{POLICIES[left]}-{POLICIES[right]}"] = {
            "alarm_decisions": int((different & alarm).sum()),
            "all_decisions": int(different.sum()),
            "rate_given_alarm": float((different & alarm).sum()/max(1, alarm.sum())),
        }
    first = {}
    for left, right in pairs:
        found = None
        for tick in range(1, 150):
            path_index = np.flatnonzero(
                simulated["actions"][:, left, tick] != simulated["actions"][:, right, tick])
            if path_index.size:
                p = int(path_index[0])
                found = {"path": p, "tick": tick,
                         "matched_score": float(data["matched"][p, tick, 6]),
                         "H1_score": float(data["scores"][p, tick, 6]),
                         "matched_alarm": bool(data["gates"][p, tick, 6]),
                         "H1_alarm": bool(data["h1_gate"][p, tick, 6]),
                         "q_left": float(quality.Q[simulated["actions"][p, left, tick]]),
                         "q_right": float(quality.Q[simulated["actions"][p, right, tick]])}
                break
        first[f"{POLICIES[left]}-{POLICIES[right]}"] = found
    result = {
        "loss_columns": ["attack_full", "attack_active", "attack_inactive",
                         "network_attack", "recovery", "health_post20", "network_post20",
                         "nondefault_action_fraction", "q0_action_fraction",
                         "maximum_missing_block_error_sq"],
        "mean_path_metrics": losses.mean(axis=0).tolist(),
        "attack_median": np.median(losses[:, :, 0], axis=0).tolist(),
        "attack_p99": np.quantile(losses[:, :, 0], .99, axis=0).tolist(),
        "attack_max": losses[:, :, 0].max(axis=0).tolist(),
        "attack_gt1_counts": (losses[:, :, 0] > 1).sum(axis=0).tolist(),
        "action_counts_on_target_alarm_by_q": action_counts.tolist(),
        "target_alarm_decisions": int(alarm.sum()),
        "target_alarm_paths": int(alarm.any(axis=1).sum()),
        "action_disagreements": disagreements, "first_disagreement": first,
        "M7_violations_by_policy": simulated["M7_violations"].tolist(),
        "health_budget_violations_by_policy": simulated["budget_violations"].tolist(),
        "M7_fallback_counts_by_policy": simulated["M7_fallbacks"].tolist(),
        "health": bool(health),
    }
    return result


def run_configuration(index):
    model = audit_frontend.load_model(index)
    ops = frontend.operators(model)
    model_row, graph, masks = configure(index)
    decision_row = {**model_row, "policy_masks": masks}
    thresholds = frontend_thresholds(index)
    train = [training_data(model, ops, index, f, thresholds) for f in range(2)]
    risks = [
        fit_risks(train, thresholds["no_repeat_matched"], mode)
        for mode in ("H_pooled", "D0_duration", "D1_match_score")
    ]
    fixtures = []
    for fixture in range(2):
        data = batch_data(model, ops, index, fixture, 5, EVALUATE, thresholds)
        sim = simulate(model, graph, data, decision_row, risks)
        fixtures.append({"name": ("continuous", "burst")[fixture],
                         "data": data, "simulated": sim,
                         "summary": summarize(data, sim),
                         "risk_tables": [{
                             key: (value.tolist() if isinstance(value, np.ndarray) else value)
                             for key, value in risk.items()} for risk in risks]})
    health_data = batch_data(model, ops, index, 0, 6, HEALTH, thresholds, first_batch=100)
    health_data["delta"].fill(0.)
    health_sim = simulate(model, graph, health_data, decision_row, risks)
    return {"model_row": model_row, "thresholds": thresholds, "policy_masks": masks,
            "fixtures": fixtures, "health_data": health_data, "health_sim": health_sim,
            "health_summary": summarize(health_data, health_sim, health=True),
            "risk_summary": [{"mode": r["mode"], "support_paths": r["support_paths"].tolist(),
                              "fallback": r["fallback"],
                              "score_tertiles": r.get("score_tertiles", []).tolist()
                              if isinstance(r.get("score_tertiles"), np.ndarray)
                              else r.get("score_tertiles", [])} for r in risks]}


def save_configuration(index, result, archive_path):
    arrays = {"policy_masks": result["policy_masks"]}
    summaries = {"risk_tables": result["risk_summary"], "health": result["health_summary"]}
    for fixture in result["fixtures"]:
        name, data, simulated = fixture["name"], fixture["data"], fixture["simulated"]
        arrays[f"{name}_x"], arrays[f"{name}_y"] = data["paths"].x, data["paths"].y
        for key in ("delta", "scores", "matched", "h1_gate", "gates", "keys"):
            arrays[f"{name}_{key}"] = data[key]
        for key in ("losses", "actions", "target_alarm", "histories",
                    "M7_violations", "budget_violations", "M7_fallbacks"):
            arrays[f"{name}_{key}"] = simulated[key]
        summaries[name] = {"summary": fixture["summary"], "risk_tables": fixture["risk_tables"]}
    data, simulated = result["health_data"], result["health_sim"]
    arrays["health_x"], arrays["health_y"] = data["paths"].x, data["paths"].y
    for key in ("delta", "scores", "matched", "h1_gate", "gates", "keys"):
        arrays["health_"+key] = data[key]
    for key in ("losses", "actions", "target_alarm", "histories",
                "M7_violations", "budget_violations", "M7_fallbacks"):
        arrays["health_"+key] = simulated[key]
    quality.frozen.atomic_npz(archive_path, arrays)
    return summaries


def main():
    ROOT.mkdir(exist_ok=True)
    progress = ROOT / "PROGRESS.json"
    final = ROOT / "RUN.json"
    current_identity = identity()
    state = {
        "identity": current_identity, "configs": {}, "archives": {},
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    cwd=HERE, text=True).strip(),
        "start_time": time.time(),
    }
    if final.exists():
        completed = quality.json.loads(final.read_text(encoding="utf-8"))
        assert completed["identity"] == current_identity
        for filename, expected in completed["archives"].items():
            assert digest(ROOT / filename) == expected
        return
    if progress.exists():
        state = quality.json.loads(progress.read_text(encoding="utf-8"))
        assert state["identity"] == current_identity
        for filename, expected in state["archives"].items():
            assert digest(ROOT / filename) == expected
    else:
        atomic_json(progress, state)
    for index, name in enumerate(QUALITY_ORDER):
        archive = ROOT / f"config{index:02}.npz"
        if name in state["configs"]:
            continue
        result = run_configuration(index)
        summaries = save_configuration(index, result, archive)
        state["archives"][archive.name] = digest(archive)
        state["configs"][name] = {
            "quality_model": {k: v for k, v in result["model_row"].items()
                              if k not in ("policy_masks", "policy_choices",
                                           "policy_reachable", "covariance_traces")},
            "thresholds": result["thresholds"],
            "risk_and_loss": summaries,
        }
        atomic_json(progress, state)
        print("task3", name, flush=True)
    state["status"] = "complete_task3_development"
    state["wall_seconds"] = time.time()-state["start_time"]
    atomic_json(final, state)
    progress.unlink(missing_ok=True)


if __name__ == "__main__":
    main()


