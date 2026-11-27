"""Replay frozen development policies on existing confirmation packets and alarms."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json

import numpy as np

import task3

HERE = Path(__file__).resolve().parent
RESULT = HERE / "fusion_attribution"
SEED = [20261127, 94017, 1, 2203]
REPS = 20000
COMPARISONS = {"H-T": (3, 2), "D0-T": (5, 2), "D1-D0": (6, 5), "D1-T": (6, 2)}
POLICIES = {"T": 2, "H": 3, "D0": 5, "D1": 6}


def csvout(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replay_config(args):
    index, folder = args
    config = task3.QUALITY_ORDER[index]
    run = json.loads((HERE / "task4_confirm/RUN.json").read_text())
    development = HERE / "task3_dev/ANALYSIS.json"
    assert sha(development) == run["identity"]["source_hashes"]["task3_dev/ANALYSIS.json"]
    saved = json.loads(development.read_text())["configuration_summaries"][config]
    tables = saved["continuous"]["risk_tables"]
    assert tables == saved["burst"]["risk_tables"]
    risks = [dict(row, means=np.asarray(row["means"])) for row in tables]
    model = task3.audit_frontend.load_model(index)
    record, graph, masks = task3.configure(index)
    decisions = dict(record, policy_masks=masks)
    threshold = run["configs"][config]["thresholds"]["no_repeat"]
    metrics = {}
    counts, path_rows, action_weights = [], [], []
    max_error = 0.
    for fixture, purpose in (("continuous", "attack"), ("burst", "attack"), ("continuous", "health")):
        path = folder / f"{config}_{fixture}_{purpose}.npz"
        assert sha(path) == run["archives"][path.name]
        with np.load(path) as arch:
            scores = arch["no_repeat_scores"]
            gates = arch["no_repeat_gates"].astype(bool)
            np.testing.assert_array_equal(scores > threshold, gates)
            data = dict(paths=SimpleNamespace(x=arch["x"], y=arch["y"]),
                        delta=arch["no_repeat_delta"], scores=arch["h1_scores"], matched=scores,
                        gates=gates, thresholds={"no_repeat_matched": threshold},
                        active=np.ones(60, dtype=bool) if fixture == "continuous" else np.arange(60)%10 < 6)
            sim = task3.simulate(model, graph, data, decisions, risks)
            actions, losses = sim["actions"], sim["losses"]
            assert not sim["M7_violations"][1:].any() and not sim["budget_violations"].any()
            for pol, column in (("A_M7_min", 1), ("S_fixed_table_T", 2)):
                np.testing.assert_array_equal(arch[f"no_repeat_{pol}_actions"], actions[:, column])
                idx = [5, 6] if purpose == "health" else list(range(7))
                err = losses[:, column, idx] - arch[f"no_repeat_{pol}_losses"][:, idx]
                max_error = max(max_error, float(np.abs(err).max()))
                np.testing.assert_allclose(err, 0., atol=2e-6, rtol=0)
            alarm = gates[:, 50:110, 6] if purpose == "attack" else gates[:, 20:, 6]
            ticks = slice(50, 110) if purpose == "attack" else slice(20, 150)
            for name, column in POLICIES.items():
                histogram = np.bincount(actions[:, column, ticks][alarm], minlength=len(task3.Q))
                for q, count in zip(task3.Q, histogram):
                    action_weights.append(dict(config=config, fixture=fixture, purpose=purpose,
                                        policy=name, q=q, alarm_decisions=int(count)))
            for label, (left, right) in COMPARISONS.items():
                diff = actions[:, left, ticks] != actions[:, right, ticks]
                assert not (diff & ~alarm).any()
                counts.append(dict(config=config, fixture=fixture, purpose=purpose, comparison=label,
                                   changed=int(diff.sum()), alarm_decisions=int(alarm.sum()),
                                   eligible_ticks=diff.size, paths_with_change=int(diff.any(axis=1).sum())))
                diffs = dict(health=losses[:, left, 5]-losses[:, right, 5]) if purpose == "health" else {
                    "full": losses[:, left, 0]-losses[:, right, 0],
                    "recovery": losses[:, left, 4]-losses[:, right, 4],
                    "action_disagreement": diff.mean(axis=1)}
                if purpose == "attack" and fixture == "burst":
                    for col, name in [(1, "burst_on"), (2, "burst_off")]:
                        diffs[name] = losses[:, left, col]-losses[:, right, col]
                for metric, values in diffs.items():
                    metrics.setdefault((label, metric), []).append(values)
            for i, path_id in enumerate(arch["path_id"]):
                row = dict(config=config, fixture=fixture, purpose=purpose, path_id=int(path_id))
                for name, pol in POLICIES.items():
                    row[name+"_full"] = losses[i, pol, 0] if purpose == "attack" else ""
                    row[name+"_recovery"] = losses[i, pol, 4] if purpose == "attack" else ""
                    row[name+"_health"] = losses[i, pol, 5] if purpose == "health" else ""
                    row[name+"_burst_on"] = losses[i, pol, 1] if purpose == "attack" and fixture == "burst" else ""
                    row[name+"_burst_off"] = losses[i, pol, 2] if purpose == "attack" and fixture == "burst" else ""
                path_rows.append(row)
    # Within each fixture a resampled path carries every contrast and endpoint.
    keys = list(metrics)
    boots = np.zeros((REPS, len(keys)))
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(index,)))
    for group in ("continuous", "burst", "health"):
        columns, samples = [], []
        for j, (_, metric) in enumerate(keys):
            arrays = metrics[keys[j]]
            if metric == "health":
                if group != "health":
                    continue
                array = arrays[0]
            elif metric.startswith("burst_"):
                if group != "burst":
                    continue
                array = arrays[0]
            else:
                if group == "health":
                    continue
                array = arrays[int(group == "burst")]/2
            columns.append(j)
            samples.append(array)
        matrix = np.column_stack(samples)
        n = len(matrix)
        for first in range(0, REPS, 500):
            weights = rng.multinomial(n, np.full(n, 1/n), size=500)/n
            boots[first:first+500, columns] += weights @ matrix
    paired = []
    for j,(label,metric) in enumerate(keys):
        arrays = metrics[(label,metric)]
        low, high = np.quantile(boots[:, j], [.025, .975])
        paired.append(dict(config=config, comparison=label, metric=metric,
                           diff=np.mean([a.mean() for a in arrays]), lower95=low, upper95=high,
                           independent_paths=sum(map(len, arrays))))
    print(config, "replayed", flush=True)
    return paired, counts, path_rows, action_weights, max_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    RESULT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(replay_config, [(i, args.archives.resolve()) for i in range(12)]))
    outputs = ["paired.csv", "action_counts.csv", "paired_paths.csv", "alarm_weights.csv"]
    for i, name in enumerate(outputs):
        csvout(RESULT/name, [row for result in results for row in result[i]])
    report = dict(max_abs_archive_endpoint_error=max(r[4] for r in results),
                  archive_policies_checked=["A_M7_min", "S_fixed_table_T"],
                  attack_paths=12288, health_paths=6144, refitted_models=0,
                  output_sha256={f: sha(RESULT/f) for f in outputs}, bootstrap_seed=SEED,
                  source_sha256=sha(Path(__file__)), protocol_sha256=sha(HERE/"TASK4_SUPPLEMENTAL_ABLATION_PROTOCOL.md"))
    (RESULT/"REPLAY.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
