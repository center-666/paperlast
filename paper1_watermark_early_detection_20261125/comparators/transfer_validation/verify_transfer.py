"""Independent endpoint readback and directly reconstructed small-path checks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import experiment as exp


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def independent_inputs(split, mi, scene, population, batch, n):
    root = [20261126, 61783, split, mi, scene, population, batch]
    source, model, generate, estimator, operators = exp.old.backend(("unstable", "stable")[mi])
    physical = generate(model, n, np.random.SeedSequence([*root, 0]))
    u = np.random.default_rng(np.random.SeedSequence([*root, 1])).random((n, 150, 50))
    keys = {}
    for label in ("iid", "no_repeat", "no_opposite"):
        state = np.floor(4*u).astype(int)
        if label != "iid":
            offsets = (1, 2, 3) if label == "no_repeat" else (0, 1, 3)
            for t in range(1, 150):
                selection = np.minimum(2, (u[:, t]*3).astype(int))
                step = np.choose(selection, offsets)
                state[:, t] = np.remainder(state[:, t-1]+step, 4)
        keys[label] = exp.ROTATION[state][:, :, model.links[:, 1]]
    draws = exp.old.make_key_draws(n, *[
        np.random.SeedSequence([*root, 2, i]) for i in range(3)])
    keys["negative"] = exp.old.matrices(draws, "negative_fixed")[:, :, model.links[:, 1]]
    return source, model, physical, estimator, operators, u, keys


def replay(folder, run, mi, scene):
    identity = run["identity"]
    source, model, physical, estimator, ops, u, keys = independent_inputs(
        identity["split"], mi, scene, 2, 0, 32)
    model_name = ("unstable", "stable")[mi]
    archive = np.load(folder / f"{model_name}_p2_s{scene}.npz")
    np.testing.assert_array_equal(physical.y[:, 1, 0], archive["physical_probe"][:32])
    np.testing.assert_array_equal(u[:, 1, 0], archive["key_probe"][:32])
    raw = physical.y[:, :, model.links[:, 1]]
    onset = (50, 65)[scene % 2]
    stop = onset+60
    fixture = source.rotated_raw_attack(model, (0., -45., 45.)[scene//2])[0][50:110]
    checks = 0
    for label, key in keys.items():
        for scorer in ("plugin", "exact"):
            arm = label+"_"+scorer
            a = identity["arms"].index(arm)
            threshold = run["thresholds"][model_name][arm]["threshold"]
            # Compute mixed-support scores directly, independently of runner's edge reuse.
            for s in (0, 6, 7, 8):
                edges = identity["supports"][s][1]
                delta = np.zeros_like(raw)
                delta[:, onset:stop, edges] = (
                    key[:, onset:stop, edges] @ fixture[None, :, edges, :, None])[..., 0]
                score = (exp.old.base.plugin_scores(model, physical, delta)
                         if scorer == "plugin" else exp.receiver.matched(raw+delta, key, ops))
                gate = score > threshold
                np.testing.assert_array_equal(gate[:, onset:onset+3].any(axis=1),
                                              archive["early"][:32, s, a])
                np.testing.assert_array_equal(score[:, onset:onset+3].max(axis=1),
                                              archive["early_max"][:32, s, a])
                world = estimator(model, physical, delta,
                                  sender_packets=physical.y, gate_override=gate)
                error = np.sum((world.post-physical.x[:, :, None, :])**2, axis=-1)
                receivers = np.unique(model.links[edges, 0])
                independent_mse = np.column_stack((
                    error[:, onset:stop, receivers].mean(axis=(1, 2)),
                    error[:, onset:stop, model.receiver_ids].mean(axis=(1, 2)),
                    error[:, onset:stop].mean(axis=(1, 2)),
                    error[:, stop:, receivers].mean(axis=(1, 2)),
                    error[:, stop:, model.receiver_ids].mean(axis=(1, 2)),
                    error[:, stop:].mean(axis=(1, 2)),
                ))
                np.testing.assert_array_equal(independent_mse, archive["values"][:32, s, a, 7:13])
                checks += 32
    archive.close()
    return checks


def verify(split):
    folder = HERE / split
    run = read(folder / "RUN.json")
    identity = run["identity"]
    analysis = read(folder / "ANALYSIS.json")
    for filename, expected in run["archives"].items():
        assert digest(folder / filename) == expected
    for filename, expected in identity["sources"].items():
        assert digest(HERE / filename) == expected, filename
    for filename, expected in analysis["table_sha256"].items():
        assert digest(folder / filename) == expected
    arms, supports, metrics = identity["arms"], identity["supports"], identity["metrics"]
    all_values = []
    scene_checks = 0
    event_totals = {}
    for mi, model in enumerate(("unstable", "stable")):
        with np.load(folder / f"{model}_p0_s0.npz") as cal, \
                np.load(folder / f"{model}_p1_s0.npz") as healthy, \
                np.load(folder / f"{model}_p2_s0.npz") as first:
            for probe in ("physical_probe", "key_probe"):
                for x, y in ((cal, healthy), (cal, first), (healthy, first)):
                    n = min(len(x[probe]), len(y[probe]))
                    assert not np.array_equal(x[probe][:n], y[probe][:n])
            for a, arm in enumerate(arms):
                rank = int(np.ceil((len(cal["path_id"])+1)*.95))
                expected = np.partition(cal["maxima"][:, a], rank-1)[rank-1]
                assert run["thresholds"][model][arm]["threshold"] == expected
            np.testing.assert_array_equal(cal["maxima"][:, 0::2],
                                          np.repeat(cal["maxima"][:, :1], 4, axis=1))
        for scene in range(6):
            with np.load(folder / f"{model}_p2_s{scene}.npz") as data:
                n = len(data["path_id"])
                np.testing.assert_array_equal(data["path_id"], np.arange(n))
                values = data["values"]
                all_values.append(values)
                transitions = data["transitions"].sum(axis=0)
                assert np.diag(transitions[1]).sum() == 0
                assert sum(transitions[2, state, (state+2) % 4] for state in range(4)) == 0
                for s, (support, edges) in enumerate(supports):
                    clean = list(set(range(6))-set(edges))
                    for a, arm in enumerate(arms):
                        early = data["early"][:, s, a].astype(bool)
                        full = data["full"][:, s, a].astype(bool)
                        expected = {
                            "attacked_early": early[:, edges].mean(axis=1),
                            "clean_edge_early": early[:, clean].mean(axis=1) if clean else np.zeros(n),
                            "clean_any_early": early[:, clean].any(axis=1),
                            "exact_early": early[:, edges].all(axis=1) & ~early[:, clean].any(axis=1),
                            "attacked_full": full[:, edges].mean(axis=1),
                            "clean_edge_full": full[:, clean].mean(axis=1) if clean else np.zeros(n),
                            "exact_full": full[:, edges].all(axis=1) & ~full[:, clean].any(axis=1),
                            "compatible_path": data["compatible_windows"][:, a//2, edges].any(axis=1),
                            "large_late_accept_path": data["large_late_count"][:, a, edges].any(axis=1),
                            "mse_gt1": values[:, s, a, metrics.index("attack_five_mse")] > 1,
                        }
                        negative = values[:, s, arms.index("negative_exact"),
                                          metrics.index("attack_five_mse")]
                        mse = values[:, s, a, metrics.index("attack_five_mse")]
                        expected["tail_event"] = (mse > 1) & (mse >= 10*negative)
                        for metric, x in expected.items():
                            np.testing.assert_array_equal(x, values[:, s, a, metrics.index(metric)])
                        threshold = run["thresholds"][model][arm]["threshold"]
                        np.testing.assert_array_equal(data["early_max"][:, s, a] > threshold, early)
                        scene_checks += 1
                        if support == "all_six":
                            event_totals.setdefault(arm, {"tail": 0, "compatible": 0, "large_late": 0})
                            for label, metric in (("tail", "tail_event"), ("compatible", "compatible_path"),
                                                  ("large_late", "large_late_accept_path")):
                                event_totals[arm][label] += int(expected[metric].sum())
    cells = np.stack(all_values)
    with (folder / "absolute.csv").open(encoding="utf-8", newline="") as stream:
        absolute = list(csv.DictReader(stream))
    groups = {"all": list(range(12)), "unstable": list(range(6)),
              "stable": list(range(6, 12)), "onset50": list(range(0, 12, 2)),
              "onset65": list(range(1, 12, 2))}
    support_names = [x[0] for x in supports]
    for row in absolute:
        expected = cells[groups[row["group"]], :, support_names.index(row["support"]),
                         arms.index(row["arm"]), metrics.index(row["metric"])].mean()
        np.testing.assert_allclose(float(row["mean"]), expected, atol=2e-12, rtol=1e-13)
    with (folder / "paired.csv").open(encoding="utf-8", newline="") as stream:
        contrasts = list(csv.DictReader(stream))
    for row in contrasts:
        left, right = row["comparison"].split("-")
        selected = cells[groups[row["group"]], :, support_names.index(row["support"])]
        difference = selected[:, :, arms.index(left), metrics.index(row["metric"])] - \
            selected[:, :, arms.index(right), metrics.index(row["metric"])]
        np.testing.assert_allclose(float(row["difference"]), difference.mean(), atol=3e-12, rtol=1e-13)
        assert float(row["bonferroni8_lower"]) <= float(row["lower"])
        assert float(row["bonferroni8_upper"]) >= float(row["upper"])
    replayed_worlds = replay(folder, run, 0, 0) + replay(folder, run, 1, 1)
    result = {
        "status": "PASS", "run_sha256": digest(folder / "RUN.json"),
        "source_sha256": digest(Path(__file__)),
        "archives_verified": len(run["archives"]), "endpoint_cells_recomputed": scene_checks,
        "absolute_rows_recomputed": len(absolute), "paired_rows_recomputed": len(contrasts),
        "independently_replayed_path_arm_support_worlds": replayed_worlds,
        "calibration_holdout_attack_probe_checks": True,
        "all_six_event_totals": event_totals,
    }
    target = folder / "VERIFY.json"
    target.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("dev", "confirm"))
    verify(parser.parse_args().split)
