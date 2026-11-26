"""Independent action reconstruction, literal replay and paired D2 transfer readout."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import d2b2_run as frozen

common = frozen.common
Q = np.array([0., .2, .4, .6, .8, .9, 1.])
POLICIES = ("R1", "HC_static", "Class_model", "Class_topology")
NAMESPACE = (20261126, 88403, 1)
REPS = 10000
SEED = [20261126, 88403, 9001]
METRICS = ("attack_missing_mse", "recovery_missing_mse", "post20_missing_mse",
           "attack_network_mse", "post20_network_mse")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cp(count, n):
    return (0. if count == 0 else float(beta.ppf(.025, count, n-count+1)),
            1. if count == n else float(beta.ppf(.975, count+1, n-count)))


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def reconstruct(graph, scores, masks, old_result):
    n = len(scores)
    history = np.full((n, 4), 7**6-1, dtype=np.int64)
    actions = np.full((n, 4, 150), 6, dtype=np.int8)
    consecutive = np.zeros(n, dtype=int)
    spec = old_result["spec"]
    requested = (4 if spec[0] == "unstable" else 3, 4) if spec[1] == "U" else (0, 0)
    absolute = np.array(old_result["static_nominal_health_cost_absolute"])
    allowed_health = absolute-absolute[-1] <= old_result["epsilon"]*absolute[-1]+1e-10
    table = old_result["risk_table"]
    bound = np.array(table["simultaneous_upper"])
    for tick in range(1, 150):
        alarm = masks[:, tick, 6]
        consecutive = np.where(alarm, consecutive+1, 0)
        cell = np.array(table["fallback_cell"])[common.cell_index(
            scores[:, tick, 6]/old_result["threshold"],
            consecutive, np.array(table["boundaries"]))]
        legal = graph[history]
        for path in np.flatnonzero(alarm):
            for policy in (0, 1):
                feasible = legal[path, policy].copy()
                if policy == 1:
                    feasible &= allowed_health
                best = np.min(bound[cell[path], feasible])
                actions[path, policy, tick] = np.flatnonzero(
                    feasible & (bound[cell[path]] <= best+1e-10))[-1]
            for policy in (2, 3):
                candidates = np.flatnonzero(legal[path, policy] &
                                           (np.arange(7) >= requested[policy-2]))
                actions[path, policy, tick] = candidates[0]
        assert np.take_along_axis(legal, actions[:, :, tick, None], axis=-1).all()
        history = (history*7+actions[:, :, tick]) % 7**6
    return actions


def main():
    folder = HERE / "results"
    run = json.loads((folder / "RUN.json").read_text(encoding="utf-8"))
    old = json.loads((HERE.parent / "b2_results/B2_RUN.json").read_text(encoding="utf-8"))
    assert sha(HERE / "transfer.py") == run["identity"]["source_sha256"]
    assert sha(HERE / "PROTOCOL.md") == run["identity"]["protocol_sha256"]
    assert sha(HERE.parent / "b2_results/B2_RUN.json") == run["identity"]["old_run_sha256"]
    assert frozen.current_input_hashes() == run["identity"]["old_input_hashes"]
    for filename, expected in run["archives"].items():
        assert sha(folder / filename) == expected
    attack_cells = []
    health_cells = []
    scene_rows = []
    fwer_rows = []
    mismatches = 0
    alarm_count = 0
    reproduced = 0
    max_loss_error = 0.
    with np.load(frozen.modes.OUTPUT / "mode_tables.npz", allow_pickle=False) as gains:
        for scene in range(12):
            result = old["results"][str(scene)]
            model, graph, _ = common.configure(result["spec"], gains)
            with np.load(folder / f"scene_{scene:02}.npz") as data:
                for fixture in ("continuous", "burst", "health"):
                    loss = data[fixture+"_loss"]
                    masks = data[fixture+"_masks"].astype(bool)
                    actions = data[fixture+"_actions"]
                    np.testing.assert_array_equal(actions[:, 1], actions[:, 2])
                    np.testing.assert_array_equal(loss[:, 1], loss[:, 2])
                    mismatches += int(np.count_nonzero(actions[:, 1] != actions[:, 2]))
                    alarm_count += int(masks[:, :, 6].sum())
                    if fixture == "health":
                        health_cells.append(loss)
                        for name, flags in (
                            ("target", masks[:, :, 6].any(axis=1)),
                            ("all_seven", masks.any(axis=(1, 2))),
                        ):
                            count = int(flags.sum())
                            lo, hi = cp(count, len(flags))
                            fwer_rows.append({
                                "scene": scene, "spec": json.dumps(result["spec"]),
                                "family": name, "alarms": count, "paths": len(flags),
                                "FWER": count/len(flags), "lower": lo, "upper": hi,
                            })
                        physical, score, small_masks = common.generate_health(
                            model, scene, result["threshold"], 1, 64, NAMESPACE)
                        delta = np.zeros((64, 150, 7, 2))
                    else:
                        attack_cells.append(loss)
                        physical, delta, score, small_masks = common.generate_paths(
                            model, result["spec"], scene, fixture, 64, 2,
                            threshold=result["threshold"], namespace=NAMESPACE)
                    np.testing.assert_array_equal(physical.y[:, 1, 0],
                                                  data[fixture+"_physical_probe"][:64])
                    np.testing.assert_array_equal(small_masks, masks[:64])
                    rebuilt_actions = reconstruct(graph, score, small_masks, result)
                    np.testing.assert_array_equal(rebuilt_actions, actions[:64])
                    literal = common.literal_policy_replay(
                        model, physical, delta, small_masks, rebuilt_actions)
                    np.testing.assert_allclose(literal, loss[:64], rtol=2e-9, atol=2e-8)
                    max_loss_error = max(max_loss_error, float(np.max(np.abs(literal-loss[:64]))))
                    reproduced += 64
                    for p, policy in enumerate(POLICIES):
                        for m, metric in enumerate(METRICS):
                            scene_rows.append({
                                "scene": scene, "spec": json.dumps(result["spec"]),
                                "fixture": fixture, "policy": policy, "metric": metric,
                                "mean": float(loss[:, p, m].mean()),
                            })
    attack_cells = np.stack(attack_cells)
    health_cells = np.stack(health_cells)
    groups = {
        "all_attacks": (attack_cells, tuple(range(24))),
        "unstable_U": (attack_cells, (0, 1, 2, 3)),
        "stable_U": (attack_cells, (12, 13, 14, 15)),
        "R35": (attack_cells, (4, 5, 6, 7, 16, 17, 18, 19)),
        "R1": (attack_cells, (8, 9, 10, 11, 20, 21, 22, 23)),
        "continuous": (attack_cells, tuple(range(0, 24, 2))),
        "burst": (attack_cells, tuple(range(1, 24, 2))),
        "health_all": (health_cells, tuple(range(12))),
        "health_unstable_U": (health_cells, (0, 1)),
        "health_stable_U": (health_cells, (6, 7)),
    }
    rng = np.random.default_rng(SEED)
    boot = {name: np.zeros((REPS, 4, 5)) for name in groups}
    for population, cells in (("attack", attack_cells), ("health", health_cells)):
        n = cells.shape[1]
        for cell, values in enumerate(cells):
            weights = rng.multinomial(n, np.full(n, 1/n), size=REPS)/n
            sample = (weights @ values.reshape(n, -1)).reshape(REPS, 4, 5)
            for name, (group_cells, indices) in groups.items():
                if group_cells is cells and cell in indices:
                    boot[name] += sample/len(indices)
    absolute = []
    pairs = []
    for name, (cells, indices) in groups.items():
        point = cells[list(indices)].mean(axis=(0, 1))
        lo, hi = np.quantile(boot[name], (.025, .975), axis=0)
        for p, policy in enumerate(POLICIES):
            for m, metric in enumerate(METRICS):
                absolute.append({"group": name, "policy": policy, "metric": metric,
                                 "mean": point[p, m], "lower": lo[p, m], "upper": hi[p, m]})
        for a, b in ((1, 2), (1, 0), (1, 3), (3, 0)):
            lo, hi = np.quantile(boot[name][:, a]-boot[name][:, b], (.025, .975), axis=0)
            for m, metric in enumerate(METRICS):
                pairs.append({"group": name, "comparison": POLICIES[a]+"-"+POLICIES[b],
                              "metric": metric, "difference": point[a, m]-point[b, m],
                              "lower": lo[m], "upper": hi[m]})
    for filename, rows in (
        ("absolute.csv", absolute), ("paired.csv", pairs),
        ("scenes.csv", scene_rows), ("health_fwer.csv", fwer_rows),
    ):
        save_csv(folder / filename, rows)
    output = {
        "status": "PASS", "run_sha256": sha(folder / "RUN.json"),
        "source_sha256": sha(Path(__file__)), "bootstrap_replicates": REPS,
        "bootstrap_seed": SEED, "HC_static_Class_model_action_mismatches": mismatches,
        "total_target_alarm_decisions": alarm_count,
        "independent_regenerated_paths": reproduced,
        "max_loss_error_vs_literal_per_neighbor_replay": max_loss_error,
        "all_edge_fwer_family": "All seven monitored edges, all scored times. Parent B2 summary used its masks[:, :, 4:] slice for its named all-edge column. This audit counts all seven explicitly.",
        "intervals": "Pointwise path-paired 95% bootstrap intervals stratified within fixed scene and fixture, conditional on frozen B2 thresholds and tables.",
        "outputs": {name: sha(folder / name) for name in (
            "absolute.csv", "paired.csv", "scenes.csv", "health_fwer.csv")},
    }
    common.atomic_json(folder / "VERIFY_ANALYSIS.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
