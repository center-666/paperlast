"""D2-B2 development screen for risk minimization under nominal-health budgets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import beta as beta_distribution

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
D1 = Path(os.environ["D2_D1_DIR"]).resolve()
sys.path.insert(0, str(HERE))
import b_run as common

sys.path.insert(0, str(D1))
import b1_health as health
import b1_mode_models as modes

common.NAMESPACE = (20261125, 7319, 1)
common.SMOKE_NAMESPACE = (20261125, 7319, 99)
common.STRATEGIES = ("R1", "HC_static", "HC_scalar", "HC_dynamic")
common.FIT = 256

NAMESPACE = common.NAMESPACE
SMOKE_NAMESPACE = common.SMOKE_NAMESPACE
STRATEGIES = common.STRATEGIES
Q = common.Q
BITS = common.BITS
N = 512
FIT = 256
EVALUATION = 256
BATCH = 64
T = 150
START, STOP = 50, 110
HEALTH_START = 20
BOOTSTRAP_REPLICATES = 2048
OUT = HERE / "b2_results"
PROGRESS = OUT / "B2_RUN_PROGRESS.json"
PROTOCOL = HERE / "D2_B2_CONSTRAINED_PROTOCOL.md"
PREVIOUS_PROTOCOL = HERE / "D2_B_PROTOCOL.md"
MODEL_CONTRACT = HERE / "D2_MODEL_CONTRACT.md"
W_MISSING = np.diag([0., 0., 1., 1.])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_npz(path, arrays):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_value_budgets():
    audit_path = D1 / "results" / "B0_definition_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_deterministic_B0_definition_audit":
        raise ValueError("Frozen D1-B0 definition audit is not PASS.")
    budgets = {}
    for row in audit["results"]:
        key = (row["model"].split("_")[0], row["model"].split("_")[1], int(row["target_edge"]))
        budget = float(row["value_budget_epsilon"])
        expected = 0.05 - 0.04 * max(float(row["kappa_L60"]), 0.) / (
            1. + max(float(row["kappa_L60"]), 0.)
        )
        if abs(expected - budget) > 1e-14:
            raise ValueError(f"Frozen D1-B0 value budget mismatch for {key}.")
        budgets[key] = {"kappa_L60": float(row["kappa_L60"]), "epsilon": budget}
    if len(budgets) != 6:
        raise ValueError("Expected six frozen D1-B0 structural value budgets.")
    return budgets, audit_path


def current_input_hashes():
    graph_hashes = {}
    for scene, spec in enumerate(health.scenarios()):
        graph_path = common.GRAPH_DIR / f"{spec[1]}_{spec[0]}_target{spec[2]}_qwords.npz"
        graph_hashes[str(scene)] = digest(graph_path)
    return {
        "D2_B_PROTOCOL.md": digest(PREVIOUS_PROTOCOL),
        "D2_MODEL_CONTRACT.md": digest(MODEL_CONTRACT),
        "b_run.py": digest(HERE / "b_run.py"),
        "physical_configuration.json": digest(modes.CONFIG),
        "mode_tables.npz": digest(modes.OUTPUT / "mode_tables.npz"),
        "D1A_link_value.json": digest(D1 / "results" / "D1A_link_value.json"),
        "D1_mode_verification.json": digest(D1 / "b1_mode_results" / "verification.json"),
        "B1_HEALTH_RUN.json": digest(D1 / "B1_HEALTH_RUN.json"),
        "B1_POSTHOC_STATUS.md": digest(D1 / "B1_POSTHOC_STATUS.md"),
        "B1_policy_lock.json": digest(D1 / "b1_health_results" / "policy_lock.json"),
        "b1_health.py": digest(D1 / "b1_health.py"),
        "b1_mode_models.py": digest(D1 / "b1_mode_models.py"),
        "attack_waveform_generator": digest(
            D1.parent / "paper1_sscd_commonrandom_20261122" / "watson_crn.py"
        ),
        "H1_residual_and_score_source": digest(
            D1.parent / "paper1_history_ablation_20261123" / "ablation.py"
        ),
        "physical_path_generator": digest(
            D1.parent / "paper1_partial_observability_20261123" / "run_partial50.py"
        ),
        "M7_graphs": graph_hashes,
        "d2b2_run.py": digest(Path(__file__)),
        "analyze_b2.py": digest(HERE / "analyze_b2.py"),
        "D2_B2_CONSTRAINED_PROTOCOL.md": digest(PROTOCOL),
        "D2_B_PROTOCOL.md": digest(PREVIOUS_PROTOCOL),
        "D2_MODEL_CONTRACT.md": digest(MODEL_CONTRACT),
        "D1_B0_definition_audit.json": digest(D1 / "results" / "B0_definition_audit.json"),
        "D1_B0_registration.json": digest(D1 / "B0_REGISTRATION.json"),
        "D1_B0_source_commit": subprocess.check_output(
            ["git", "-C", str(D1), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def structural_key(spec):
    dynamics, topology, sender = spec[:3]
    return topology, dynamics, int(sender)


def configure_scene(spec, gains, budget_map):
    model, graph, graph_path = common.configure(spec, gains)
    key = structural_key(spec)
    try:
        budget = budget_map[key]
    except KeyError as error:
        raise ValueError(f"No frozen D1-B0 epsilon for structure {key}.") from error
    return model, graph, graph_path, budget


def prepare_health_curves(model):
    affine = common.prepare_health_cost_affine(model)
    initial = model.target["Pplus"][0]
    static_absolute = common.health_cost_vector(initial, affine)
    static_relative = static_absolute - static_absolute[-1]
    if static_absolute[-1] <= 0.:
        raise ValueError("The full-acceptance nominal health reference must be positive.")
    return affine, static_absolute, static_relative


def feasibility_sets(legal, epsilon, static_absolute, static_relative,
                     dynamic_absolute, dynamic_relative):
    """Construct four same-mode candidate sets before minimizing shared risk."""
    path_count = len(legal)
    sets = np.broadcast_to(legal[:, None, :], (path_count, 4, len(Q))).copy()
    sets[:, 1] &= (
        static_relative[None, :]
        <= epsilon * static_absolute[-1] + 1e-10
    )
    scalar = np.zeros(path_count)
    denominator = np.sum(static_relative[None, :] ** 2 * legal, axis=1)
    numerator = np.sum(dynamic_relative[:, 2] * static_relative[None, :] * legal, axis=1)
    nonzero = denominator > 1e-24
    scalar[nonzero] = np.maximum(numerator[nonzero] / denominator[nonzero], 0.)
    sets[:, 2] &= (
        scalar[:, None] * static_relative[None, :]
        <= epsilon * dynamic_absolute[:, 2, -1, None] + 1e-10
    )
    sets[:, 3] &= (
        dynamic_relative[:, 3]
        <= epsilon * dynamic_absolute[:, 3, -1, None] + 1e-10
    )
    if not np.all(sets[:, 1:, -1]):
        raise AssertionError("q=1 must remain feasible under each nominal-health budget.")
    return sets, scalar, int((~nonzero).sum())


def simulate_constrained(model, graph, paths, scores, masks, delta, table,
                         threshold, epsilon, curves=None):
    n = len(paths.x)
    if curves is None:
        affine, static_absolute, static_relative = prepare_health_curves(model)
    else:
        affine, static_absolute, static_relative = curves
    state = np.zeros((n, 4, 50, 4))
    state_cov = np.broadcast_to(model.target["Pplus"][0], (n, 4, 4, 4)).copy()
    history = np.full((n, 4), 7 ** 6 - 1, dtype=np.int32)
    alarm_run = np.zeros(n, dtype=np.int32)
    actions = np.full((n, 4, T), 6, dtype=np.int8)
    losses = np.zeros((n, 4, 5), dtype=float)
    scalar_zero = np.zeros((n, 4), dtype=np.int32)
    risk_fallback = np.zeros((n, 4), dtype=np.int32)
    health_forced_out = np.zeros((n, 3), dtype=np.int32)
    health_sets_saved = np.zeros((n, 4, T, len(Q)), dtype=np.uint8)
    health_delta_saved = np.zeros((n, 4, T, len(Q)), dtype=float)
    dynamic_absolute_saved = np.zeros((n, 4, T, len(Q)), dtype=float)
    scalar_saved = np.zeros((n, 4, T), dtype=float)
    budget_utilization = np.zeros((n, 4, T), dtype=float)
    budget_active = np.zeros((n, 3), dtype=np.int32)

    raw = (paths.y.swapaxes(-1, -2) @ model.adjacency.T).swapaxes(-1, -2)
    base_f, base_k, base_g = (model.base[name] for name in ("F", "K", "G"))
    target_f, target_k, target_g = (
        model.target[name] for name in ("F", "K", "G")
    )
    target_o = model.target["Omega"]
    target_sender = int(model.links[6, 1])
    bounds = np.asarray(table["boundaries"])
    lookup = np.asarray(table["fallback_cell"])
    risk_upper = np.asarray(table["simultaneous_upper"])
    if masks.shape != (n, T, 7):
        raise ValueError("All four policies must share the same seven-edge H1 mask.")

    for tick in range(1, T):
        bits = masks[:, tick]
        y = paths.y[:, tick]
        received = raw[:, tick].copy()
        for edge in range(6):
            received[:, model.links[edge, 0]] -= (
                bits[:, edge, None] * y[:, model.links[edge, 1]]
            )
        alarm = bits[:, 6]
        alarm_run = np.where(alarm, alarm_run + 1, 0)
        mode = alarm.astype(int)
        risk_cell = common.cell_index(scores[:, tick, 6] / threshold,
                                      alarm_run, bounds)
        mapped_cell = lookup[risk_cell]
        risk = risk_upper[mapped_cell]
        legal = graph[history]
        dynamic_absolute = common.health_cost_vector(state_cov, affine)
        dynamic_relative = dynamic_absolute - dynamic_absolute[..., -1:]
        # Each policy has its own M7 history. Recompute all candidate sets
        # against the matching policy history before selecting its action.
        legal_sets = np.empty((n, 4, len(Q)), dtype=bool)
        scalar_values = np.zeros((n, 4))
        for policy in range(4):
            policy_legal = legal[:, policy]
            policy_sets, policy_scalar, _ = feasibility_sets(
                policy_legal, epsilon, static_absolute, static_relative,
                dynamic_absolute, dynamic_relative
            )
            legal_sets[:, policy] = policy_sets[:, policy]
            scalar_values[:, policy] = policy_scalar
        scalar_zero[:, 2] += (np.sum(static_relative[None, :] ** 2 * legal[:, 2], axis=1)
                              <= 1e-24) & alarm
        health_sets_saved[:, :, tick] = legal_sets
        health_delta_saved[:, :, tick] = dynamic_relative
        dynamic_absolute_saved[:, :, tick] = dynamic_absolute
        scalar_saved[:, :, tick] = scalar_values
        chosen = np.full((n, 4), 6, dtype=np.int8)
        for path in np.flatnonzero(alarm):
            for policy in range(4):
                if policy == 0:
                    feasible = legal[path, policy]
                else:
                    feasible = legal_sets[path, policy]
                    health_forced_out[path, policy - 1] += int(
                        np.count_nonzero(legal[path, policy] & ~feasible)
                    )
                if not feasible.any():
                    raise AssertionError("A constrained action set unexpectedly became empty.")
                selected = common.select_q(risk[path], feasible)
                chosen[path, policy] = selected
                if policy == 1:
                    cost = static_relative[selected]
                    baseline = static_absolute[-1]
                elif policy == 2:
                    cost = scalar_values[path, policy] * static_relative[selected]
                    baseline = dynamic_absolute[path, policy, -1]
                elif policy == 3:
                    cost = dynamic_relative[path, policy, selected]
                    baseline = dynamic_absolute[path, policy, -1]
                else:
                    cost, baseline = 0.0, 1.0
                if policy > 0 and cost > epsilon * baseline + 1e-10:
                    raise AssertionError("Selected action exceeds its nominal-health budget.")
                utilization = cost / baseline
                budget_utilization[path, policy, tick] = utilization
                if policy > 0 and utilization >= epsilon - 1e-10:
                    budget_active[path, policy - 1] += 1
                risk_fallback[path, policy] += int(mapped_cell[path] >= 9)
        if not np.all(legal_sets[np.arange(n)[:, None], np.arange(4)[None], chosen]):
            raise AssertionError("A selected action violates its M7 or health constraint.")
        actions[:, :, tick] = chosen
        history = (history * 7 + chosen) % (7 ** 6)
        q_effective = np.where(alarm[:, None], Q[chosen], 1.0)

        next_state = np.empty_like(state)
        for policy in range(4):
            f = target_f[mode, chosen[:, policy]]
            k_gain = target_k[mode, chosen[:, policy]]
            g_gain = target_g[mode, chosen[:, policy]]
            packet = received[:, 26] + delta[:, tick, 6] + (
                q_effective[:, policy] - 1.0
            )[:, None] * (y[:, target_sender] + delta[:, tick, 6])
            next_state[:, policy, 26] = (
                np.einsum("bij,bj->bi", f, state[:, policy, 26])
                + np.einsum("bij,bj->bi", k_gain, y[:, 26])
                + np.einsum("bij,bj->bi", g_gain, packet)
            )
            state_cov[:, policy] = (
                f @ state_cov[:, policy] @ f.swapaxes(-1, -2)
                + target_o[mode, chosen[:, policy]]
            )

        global_mode = bits.astype(np.int64) @ BITS
        other_next = (
            np.einsum("bnij,bnj->bni", base_f[global_mode], state[:, 0])
            + np.einsum("bnij,bnj->bni", base_k[global_mode], y)
            + np.einsum("bnij,bnj->bni", base_g[global_mode], received)
        )
        for policy in range(4):
            next_state[:, policy, :26] = other_next[:, :26]
            next_state[:, policy, 27:] = other_next[:, 27:]
        state = next_state
        error = (state - paths.x[:, tick, None, None, :]) ** 2
        missing = error[:, :, 26, 2:].sum(axis=-1)
        network = error.sum(axis=(2, 3)) / 50.0
        if START <= tick < STOP:
            losses[:, :, 0] += missing / 60.
            losses[:, :, 3] += network / 60.
        if tick >= STOP:
            losses[:, :, 1] += missing / 40.
        if tick >= 20:
            losses[:, :, 2] += missing / 130.
            losses[:, :, 4] += network / 130.

    status = {
        "scalar_zero_denominator_counts": scalar_zero.sum(axis=0).tolist(),
        "health_forced_out_candidate_counts": health_forced_out.sum(axis=0).tolist(),
        "health_budget_active_decision_counts": budget_active.sum(axis=0).tolist(),
        "risk_fallback_counts": risk_fallback.sum(axis=0).tolist(),
        "target_alarm_events": int(masks[:, :, 6].sum()),
        "shared_h1_mask": True,
    }
    return (losses, actions, status, health_sets_saved, health_delta_saved,
            dynamic_absolute_saved, scalar_saved, budget_utilization)


def q1_health_reference(model, paths, masks):
    packed = np.sum(masks * BITS[None, None, :], axis=2).astype(np.uint8)
    frozen = health.evaluate(model, paths, packed)
    zero = np.zeros((len(paths.x), T, 7, 2))
    reference = common.full_accept_reference(model, paths, masks, zero)
    np.testing.assert_allclose(reference[:, 2], frozen["missing"][:, 6],
                               rtol=2e-9, atol=2e-8)
    np.testing.assert_allclose(reference[:, 4], frozen["network"][:, 6],
                               rtol=2e-9, atol=2e-8)
    return reference


def run_scene(scene, spec, gains, budget_map, path_count=N, fit_count=FIT,
              calibration_count=FIT, evaluation_count=EVALUATION,
              namespace=NAMESPACE, bootstrap_replicates=BOOTSTRAP_REPLICATES):
    model, graph, graph_path, budget = configure_scene(spec, gains, budget_map)
    threshold, calibration_maxima = common.threshold_calibration(
        model, scene, calibration_count, namespace
    )
    fit_records = []
    attack_records = {}
    for fixture in ("continuous", "burst"):
        paths, delta, scores, masks = common.generate_paths(
            model, spec, scene, fixture, path_count, 2,
            threshold=threshold, namespace=namespace
        )
        runs = common.alarm_runs(masks[:, :, 6])
        labels = common.single_step_damage(model, paths, delta, masks)
        fit_records.append({
            "fixture": fixture,
            "scores": scores[:fit_count],
            "runs": runs[:fit_count],
            "labels": labels[:fit_count],
        })
        attack_records[fixture] = (paths, delta, scores, masks)
    risk_table = common.fit_risk_table(
        fit_records, threshold, scene, bootstrap_replicates
    )

    health_paths, health_scores, health_masks = common.generate_health(
        model, scene, threshold, 1, evaluation_count, namespace
    )
    (health_loss, health_actions, health_status, health_sets, health_delta,
     health_abs, health_scalar, health_utilization) = simulate_constrained(
        model, graph, health_paths, health_scores, health_masks,
        np.zeros((evaluation_count, T, 7, 2)), risk_table, threshold, budget["epsilon"]
    )
    health_reference = q1_health_reference(model, health_paths, health_masks)
    target_fwer_paths = int(health_masks[:, :, 6].any(axis=1).sum())
    all_edge_fwer_paths = int(health_masks[:, :, 4:].any(axis=(1, 2)).sum())
    result = {
        "scene": scene,
        "spec": list(spec),
        "epsilon": budget["epsilon"],
        "kappa_L60": budget["kappa_L60"],
        "static_nominal_health_cost_absolute": static_absolute.tolist(),
        "static_nominal_health_cost_relative_to_q1": static_relative.tolist(),
        "threshold": threshold,
        "calibration_paths": calibration_count,
        "calibration_order_rank": int(np.ceil((calibration_count + 1) * .95)),
        "calibration_maxima_sha256": hashlib.sha256(calibration_maxima.tobytes()).hexdigest(),
        "health_evaluation_paths": evaluation_count,
        "target_fwer_paths": target_fwer_paths,
        "target_fwer": target_fwer_paths / evaluation_count,
        "target_fwer_CP95": common.clopper_pearson(target_fwer_paths, evaluation_count),
        "all_edge_fwer_paths": all_edge_fwer_paths,
        "all_edge_fwer": all_edge_fwer_paths / evaluation_count,
        "all_edge_fwer_CP95": common.clopper_pearson(all_edge_fwer_paths, evaluation_count),
        "health_loss_mean": health_loss.mean(axis=0).tolist(),
        "health_q1_reference_mean": health_reference.mean(axis=0).tolist(),
        "health_status": health_status,
        "risk_table": risk_table,
        "attacks": {},
    }
    arrays = {
        "health_masks": health_masks.astype(np.uint8),
        "health_scores": health_scores,
        "health_actions": health_actions,
        "health_loss": health_loss,
        "health_q1_reference": health_reference,
        "health_full_accept_loss": health_reference,
        "health_feasible_sets": health_sets,
        "health_relative_costs": health_delta,
        "health_absolute_costs": health_abs,
        "health_scalar_fit": health_scalar,
        "health_budget_utilization": health_utilization,
        "health_evaluation_effective_weights": Q[health_actions],
        "health_calibration_maxima": calibration_maxima,
    }

    for fixture, (paths, delta, scores, masks) in attack_records.items():
        evaluation = slice(fit_count, path_count)
        eval_paths = type(paths)(x=paths.x[evaluation], y=paths.y[evaluation])
        eval_delta = delta[evaluation]
        eval_scores = scores[evaluation]
        eval_masks = masks[evaluation]
        (losses, actions, status, feasible, relative_cost, absolute_cost,
         scalar, utilization) = simulate_constrained(
            model, graph, eval_paths, eval_scores, eval_masks,
            eval_delta, risk_table, threshold, budget["epsilon"]
        )
        q1_reference = common.simulate_full_accept_reference(
            model, eval_paths, eval_scores, eval_masks, eval_delta
        )
        active = common.active_attack(fixture)
        active_alarm = eval_masks[:, START:STOP, 6][:, active]
        target_alarm = eval_masks[:, :, 6]
        action_summary = common.action_summary(actions, target_alarm, slice(START, STOP))
        result["attacks"][fixture] = {
            "evaluation_paths": path_count - fit_count,
            "path_miss_count": int((~active_alarm.any(axis=1)).sum()),
            "path_miss_fraction": float((~active_alarm.any(axis=1)).mean()),
            "attack_tick_unalarmed_fraction": float((~active_alarm).mean()),
            "longest_target_alarm_run_mean": float(
                common.alarm_runs(target_alarm)[:, START:STOP].max(axis=1).mean()
            ),
            "loss_mean": losses.mean(axis=0).tolist(),
            "q1_reference_loss_mean": q1_reference.mean(axis=0).tolist(),
            "health_status": status,
            "action_summary": action_summary,
        }
        arrays[f"{fixture}_masks"] = eval_masks.astype(np.uint8)
        arrays[f"{fixture}_scores"] = eval_scores
        arrays[f"{fixture}_actions"] = actions
        arrays[f"{fixture}_loss"] = losses
        arrays[f"{fixture}_q1_reference"] = q1_reference
        arrays[f"{fixture}_full_accept_loss"] = q1_reference
        arrays[f"{fixture}_feasible_sets"] = feasible
        arrays[f"{fixture}_relative_health_costs"] = relative_cost
        arrays[f"{fixture}_absolute_health_costs"] = absolute_cost
        arrays[f"{fixture}_scalar_fit"] = scalar
        arrays[f"{fixture}_budget_utilization"] = utilization
        arrays[f"{fixture}_effective_weights"] = Q[actions]
    return result, arrays, graph_path


def deterministic_smoke():
    scene_indices = tuple(range(12))
    summaries = []
    budget_map, _ = load_value_budgets()
    with np.load(modes.OUTPUT / "mode_tables.npz", allow_pickle=False) as gains:
        for scene in scene_indices:
            spec = health.scenarios()[scene]
            model, graph, _, budget = configure_scene(spec, gains, budget_map)
            threshold, _ = common.threshold_calibration(
                model, scene, 256, SMOKE_NAMESPACE
            )
            paths, delta, scores, masks = common.generate_paths(
                model, spec, scene, "burst", 64, 2, threshold=threshold,
                namespace=SMOKE_NAMESPACE
            )
            runs = common.alarm_runs(masks[:, :, 6])
            labels = common.single_step_damage(model, paths, delta, masks)
            record = {"fixture": "burst", "scores": scores[:32],
                      "runs": runs[:32], "labels": labels[:32]}
            second = {"fixture": "continuous", "scores": scores[:32],
                      "runs": runs[:32], "labels": labels[:32]}
            risk_table = common.fit_risk_table(
                [record, second], threshold, scene, bootstrap_replicates=128
            )
            loss, actions, status, feasible, relative, absolute, scalar, utilization = simulate_constrained(
                model, graph, paths, scores, masks, delta, risk_table,
                threshold, budget["epsilon"]
            )
            np.testing.assert_array_equal(feasible[:, 1:, 1:, -1], True)
            selected_feasible = np.take_along_axis(feasible, actions[..., None], axis=-1)[..., 0]
            selected_feasible = selected_feasible[:, :, 1:]
            np.testing.assert_array_equal(selected_feasible, True)
            q1 = common.simulate_full_accept_reference(model, paths, scores, masks, delta)
            literal_actions = np.full_like(actions, 6)
            literal = common.literal_policy_replay(model, paths, delta, masks, literal_actions)
            np.testing.assert_allclose(literal[:, 0], q1, rtol=2e-9, atol=2e-8)
            literal_policy = common.literal_policy_replay(model, paths, delta, masks, actions)
            np.testing.assert_allclose(literal_policy, loss, rtol=2e-9, atol=2e-8)
            clean_delta = np.zeros_like(delta)
            frozen_b1_reference = q1_health_reference(model, paths, masks)
            direct_q1_health = common.simulate_full_accept_reference(
                model, paths, scores, masks, clean_delta
            )
            np.testing.assert_allclose(
                frozen_b1_reference, direct_q1_health, rtol=2e-9, atol=2e-8
            )
            for policy in range(4):
                assert np.all(np.isfinite(loss[:, policy]))
            summaries.append({
                "scene": scene,
                "epsilon": budget["epsilon"],
                "health_feasible_nonempty": True,
                "q1_regression_max_abs": float(np.max(np.abs(literal[:, 0] - q1))),
                "policy_actions": actions.shape[1],
                "formal_attack_paths_generated": False,
            })
    return {"status": "PASS_DETERMINISTIC_SMOKE", "scenes": summaries}


def preflight():
    common.NAMESPACE = NAMESPACE
    base = common.preflight()
    if base["D2_branch"] != "d2-history-value-fusion-20261124":
        raise ValueError("D2-B2 must run on the frozen D2 branch.")
    budgets, audit_path = load_value_budgets()
    d2b2_namespace = tuple(NAMESPACE)
    old_namespaces = {
        "B1": tuple(base["B1_seed_namespace"]),
        "B2": tuple(base["B2_seed_namespace"]),
        "D2-A-prime": tuple(base["D2_A_prime_seed_namespace"]),
        "D2-B": (20261124, 4401, 1),
    }
    for name, old in old_namespaces.items():
        if d2b2_namespace[:len(old)] == old or old[:len(d2b2_namespace)] == d2b2_namespace:
            raise ValueError(f"D2-B2 seed namespace overlaps {name}.")
    input_hashes = current_input_hashes()
    for name, expected in base["input_sha256"].items():
        if input_hashes.get(name) != expected:
            raise ValueError(f"D2-B2 preflight hash differs from the frozen B audit for {name}.")
    return {
        **base,
        "D2_B2_seed_namespace": list(NAMESPACE),
        "D2_B2_seed_namespaces_disjoint": True,
        "D1_B0_value_budgets": {
            f"{topology}_{dynamics}_target{sender}": value
            for (topology, dynamics, sender), value in budgets.items()
        },
        "input_sha256": input_hashes,
        "runtime_B2": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }


def validate_resume(progress):
    branch = subprocess.check_output(
        ["git", "-C", str(ROOT), "branch", "--show-current"], text=True
    ).strip()
    current_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if branch != "d2-history-value-fusion-20261124":
        raise ValueError("Partial D2-B2 run is on an unexpected branch.")
    if current_commit != progress["preflight"]["D2_commit_at_preflight"]:
        raise ValueError("Code commit changed since the partial D2-B2 run started.")
    if progress["input_sha256"] != current_input_hashes():
        raise ValueError("A D2-B2 input changed since the partial run started.")
    if any(scene not in range(12) for scene in progress.get("completed_scenes", [])):
        raise ValueError("Partial D2-B2 progress contains an invalid scene index.")
    for scene in progress.get("completed_scenes", []):
        filename = f"scene_{scene:02}.npz"
        path = OUT / filename
        if not path.exists() or digest(path) != progress["output_sha256"].get(filename):
            raise ValueError(f"Completed D2-B2 output is missing or corrupt: {filename}")
    return progress["preflight"]


def run_all():
    if OUT.exists():
        if not PROGRESS.exists() or (OUT / "B2_RUN.json").exists():
            raise FileExistsError(f"Refusing to overwrite completed or unregistered output {OUT}.")
        progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
        audit = validate_resume(progress)
    else:
        audit = preflight()
        OUT.mkdir()
        progress = {
            "status": "in_progress_D2_B2_development_screening",
            "preflight": audit,
            "seed_namespace": list(NAMESPACE),
            "completed_scenes": [],
            "input_sha256": audit["input_sha256"],
            "output_sha256": {},
            "results": {},
        }
        atomic_json(PROGRESS, progress)
    budget_map, _ = load_value_budgets()
    with np.load(modes.OUTPUT / "mode_tables.npz", allow_pickle=False) as gains:
        for scene, spec in enumerate(health.scenarios()):
            if scene in progress["completed_scenes"]:
                continue
            result, arrays, _ = run_scene(scene, spec, gains, budget_map)
            filename = f"scene_{scene:02}.npz"
            atomic_npz(OUT / filename, arrays)
            progress["completed_scenes"].append(scene)
            progress["completed_scenes"].sort()
            progress["output_sha256"][filename] = digest(OUT / filename)
            progress["results"][str(scene)] = result
            atomic_json(PROGRESS, progress)
            print("completed D2-B2 scene", scene, flush=True)
    manifest = {
        "status": "complete_D2_B2_development_screening",
        "protocol_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "execution_commit": audit["D2_commit_at_preflight"],
        "preflight": audit,
        "seed_namespace": list(NAMESPACE),
        "attack_paths_per_fixture": N,
        "attack_fit_paths_per_fixture": FIT,
        "attack_evaluation_paths_per_fixture": EVALUATION,
        "health_calibration_paths": FIT,
        "health_evaluation_paths": EVALUATION,
        "scene_count": len(health.scenarios()),
        "strategies": list(STRATEGIES),
        "claim_scope": "independent_development_screening_proxy_constraints_not_health_guarantees",
        "risk_table_is_closed_loop_mse_bound": False,
        "nominal_health_cost_is_formal_adaptive_alarm_bound": False,
        "results": progress["results"],
        "input_sha256": audit["input_sha256"],
        "output_sha256": progress["output_sha256"],
    }
    atomic_json(OUT / "B2_RUN.json", manifest)
    PROGRESS.unlink()
    print(json.dumps({"status": manifest["status"], "scenes": list(range(12)),
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(deterministic_smoke(), indent=2))
    elif args.preflight:
        print(json.dumps(preflight(), indent=2))
    else:
        run_all()
