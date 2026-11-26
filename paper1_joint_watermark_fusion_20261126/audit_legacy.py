"""Replay the frozen B2 decisions and enumerate the requested finite mask grid."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
D2 = HERE.parent / "paper1_D2_history_value_fusion_20261124"
sys.path.insert(0, str(D2))
import d2b2_run as frozen
import audit_b2 as independent
import audit_b2_budget_margin as margin


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = HERE / "legacy_audit"
    output.mkdir(exist_ok=True)
    independent.OUTPUT = output / "B2_INDEPENDENT_REPLAY.json"
    independent.main()
    margin.OUTPUT = output / "B2_BUDGET_MARGIN_REPLAY.json"
    margin.main()
    manifest = json.loads(independent.RUN_MANIFEST.read_text(encoding="utf-8"))
    comparisons = []
    actual_mismatch = 0
    actual_decisions = 0
    first_actual = None
    smoke = []
    risk_rows = []
    with np.load(frozen.modes.OUTPUT / "mode_tables.npz") as gains:
        for scene in range(12):
            record = manifest["results"][str(scene)]
            model, graph, _ = frozen.common.configure(record["spec"], gains)
            absolute = np.array(record["static_nominal_health_cost_absolute"])
            health_ok = absolute-absolute[-1] <= record["epsilon"]*absolute[-1]+1e-10
            table = record["risk_table"]
            risk = np.array(table["simultaneous_upper"])
            for cell in range(9):
                mapped = table["fallback_cell"][cell]
                for mask in range(64):
                    candidates = ((mask | 64) & (1 << np.arange(7))) != 0
                    feasible = candidates & health_ok
                    minimal = int(np.flatnonzero(feasible)[0])
                    best = risk[mapped, feasible].min()
                    original = int(np.flatnonzero(
                        feasible & (risk[mapped] <= best+1e-10))[-1])
                    comparisons.append({
                        "scene": scene, "fine_cell": cell, "fallback_cell": mapped,
                        "candidate_mask": mask | 64,
                        "minimum_legal_index": minimal, "original_index": original,
                    })
            risk_rows.append({
                "scene": scene, "spec": record["spec"], "threshold": record["threshold"],
                "normalized_score_tertiles": table["boundaries"],
                "support_paths": table["support_paths"],
                "support_paths_by_fixture": table["support_paths_by_fixture"],
                "fallback_cell": table["fallback_cell"], "fallback_level": table["fallback_level"],
                "fit_paths_per_fixture": 256, "evaluation_paths_per_fixture": 256,
                "old_namespace": manifest["seed_namespace"],
            })
            archive_path = independent.RUN_DIR / f"scene_{scene:02}.npz"
            with np.load(archive_path) as data:
                for fixture in ("continuous", "burst"):
                    actions = data[fixture+"_actions"][:, 1]
                    alarm = data[fixture+"_masks"][:, :, 6].astype(bool)
                    history = np.full(len(actions), 7**6-1, dtype=np.int32)
                    for tick in range(1, 150):
                        feasible = graph[history] & health_ok
                        expected = np.where(alarm[:, tick], feasible.argmax(axis=1), 6)
                        mismatch = np.flatnonzero(expected != actions[:, tick])
                        actual_mismatch += len(mismatch)
                        actual_decisions += int(alarm[:, tick].sum())
                        if len(mismatch) and first_actual is None:
                            first_actual = {"scene": scene, "fixture": fixture,
                                            "tick": tick, "path": int(mismatch[0])}
                        history = (history*7+actions[:, tick]) % 7**6

            paths, delta, scores, masks = frozen.common.generate_paths(
                model, record["spec"], scene, "burst", 64, 2,
                threshold=record["threshold"], namespace=(20261126, 94017, 99))
            paths = type(paths)(x=paths.x[:16], y=paths.y[:16])
            delta, scores, masks = delta[:16], scores[:16], masks[:16]
            inherited = frozen.simulate_constrained(
                model, graph, paths, scores, masks, delta, record["risk_table"],
                record["threshold"], record["epsilon"])
            literal = frozen.common.literal_policy_replay(
                model, paths, delta, masks, inherited[1])
            difference = float(np.max(np.abs(inherited[0]-literal)))
            np.testing.assert_allclose(inherited[0], literal, rtol=2e-9, atol=2e-8)
            q1 = frozen.common.simulate_full_accept_reference(model, paths, scores, masks, delta)
            q1_actions = np.full_like(inherited[1], 6)
            q1_literal = frozen.common.literal_policy_replay(model, paths, delta, masks, q1_actions)
            np.testing.assert_allclose(q1_literal[:, 0], q1, rtol=2e-9, atol=2e-8)
            # One-step full-innovation weighting, using all original binary modes.
            rng = np.random.default_rng([20261126, 94017, scene, 191])
            prior = rng.normal(size=(16, 4))
            y = paths.y[:, 31]
            one_step_errors = []
            for mode in (0, 1, 63, 64, 65, 127):
                target_bit = (mode >> 6) & 1
                for q_index in (0, 6):
                    weights = {j: 1. for j in model.neighbors[26]}
                    weights[model.links[6, 1]] = frozen.common.Q[q_index] if target_bit else 1.
                    predicted = prior @ model.A.T
                    local = y[:, 26]-predicted @ model.H[26].T
                    innovation = sum((weight*(y[:, j]-predicted @ model.H[j].T)
                                      for j, weight in weights.items()), start=np.zeros((16, 2)))
                    k, g, f = (model.target[name][target_bit, q_index] for name in ("K", "G", "F"))
                    direct = predicted+local @ k.T+innovation @ g.T
                    packet = sum((weight*y[:, j] for j, weight in weights.items()),
                                 start=np.zeros((16, 2)))
                    aggregate = prior @ f.T+y[:, 26] @ k.T+packet @ g.T
                    np.testing.assert_allclose(direct, aggregate, rtol=2e-12, atol=2e-12)
                    one_step_errors.append(float(np.max(np.abs(direct-aggregate))))
            smoke.append({
                "scene": scene, "paths": 16, "normal_entry_literal_max_abs_loss_error": difference,
                "q1_reference_max_abs_loss_error": float(np.max(np.abs(q1_literal[:, 0]-q1))),
                "q0_q1_full_innovation_one_step_max_abs_error": max(one_step_errors),
            })
    mismatch = [r for r in comparisons if r["original_index"] != r["minimum_legal_index"]]
    replay = json.loads(independent.OUTPUT.read_text(encoding="utf-8"))
    report = {
        "status": "PASS" if not mismatch and actual_mismatch == 0 else "COUNTEREXAMPLES",
        "source_sha256": sha(Path(__file__)),
        "old_run_sha256": sha(independent.RUN_MANIFEST),
        "grid_count": len(comparisons), "grid_mismatch_count": len(mismatch),
        "first_grid_counterexample": mismatch[0] if mismatch else None,
        "actual_alarm_decisions": actual_decisions,
        "actual_minimum_q_action_mismatches": actual_mismatch,
        "first_actual_counterexample": first_actual,
        "original_audit_alarm_decisions": replay["target_alarm_decisions_replayed"],
        "old_HC_static_attack_recovery_mse": replay["path_loss_weighted_summary"][1][:2],
        "smoke": smoke, "risk_cells_and_training": risk_rows,
        "candidate_mask_grid": comparisons,
        "source_entry": {
            "controller": "paper1_D2_history_value_fusion_20261124/d2b2_run.py:simulate_constrained",
            "literal": "paper1_D2_history_value_fusion_20261124/b_run.py:literal_policy_replay",
            "gain_order": "target[quantity][target_alarm_bit, q_index], K/G are 4x2",
            "edge_order_receiver_sender": [[0, 1], [2, 3], [15, 16], [17, 31], [14, 49], [17, 34], [26, "35 or 1"]],
            "state_initialization": "estimates zero, physical x0 normal, six-action history all q=1",
            "time": "filter k=1..149, H1 score k=5..149, attack [50,110), recovery [110,150)",
        },
    }
    frozen.atomic_json(output / "TASK1.json", report)
    print(json.dumps({k: report[k] for k in (
        "status", "grid_count", "grid_mismatch_count", "actual_alarm_decisions",
        "actual_minimum_q_action_mismatches", "old_HC_static_attack_recovery_mse")}, indent=2))


if __name__ == "__main__":
    main()
