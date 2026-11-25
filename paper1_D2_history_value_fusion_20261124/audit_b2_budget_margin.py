"""Posthoc margin audit for static, scalar, and dynamic health constraints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "b2_results"
RUN_FILE = RUN_DIR / "B2_RUN.json"
INDEPENDENT_AUDIT = RUN_DIR / "B2_INDEPENDENT_AUDIT.json"
OUTPUT = RUN_DIR / "B2_BUDGET_MARGIN_AUDIT.json"
Q = np.asarray([0., .2, .4, .6, .8, .9, 1.])
EPS_TOL = 1e-10
POLICIES = ("R1", "HC_static", "HC_scalar", "HC_dynamic")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_margin(values, eps, eligible):
    normalized = values[eligible] / eps
    slack = 1.0 - normalized
    flat_slack = slack.ravel()
    feasible = flat_slack >= -(EPS_TOL / eps)
    distance = np.abs(flat_slack)
    return {
        "candidate_evaluations": int(flat_slack.size),
        "feasible_fraction": float(feasible.mean()) if flat_slack.size else None,
        "normalized_margin_quantiles": np.quantile(
            flat_slack, [0., .01, .1, .5, .9, .99, 1.]
        ).tolist() if flat_slack.size else None,
        "absolute_normalized_margin_below_0_5_percent_count": int((distance < .005).sum()),
        "absolute_normalized_margin_below_1_percent_count": int((distance < .01).sum()),
        "absolute_normalized_margin_below_5_percent_count": int((distance < .05).sum()),
    }


def reconstruct_m7_actions(graph, actions):
    path_count, policy_count, time_count = actions.shape
    history = np.full((path_count, policy_count), 7 ** 6 - 1, dtype=np.int32)
    legal = np.zeros((path_count, policy_count, time_count, len(Q)), dtype=bool)
    for tick in range(1, time_count):
        legal[:, :, tick] = graph[history]
        history = (history * 7 + actions[:, :, tick]) % (7 ** 6)
    return legal


def main():
    manifest = json.loads(RUN_FILE.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_D2_B2_development_screening":
        raise ValueError("Budget-margin audit requires all corrected D2-B2 scenes.")
    records = []
    totals = {
        "alarm_decisions": 0,
        "static_dynamic_feasible_set_mismatches": 0,
        "static_scalar_feasible_set_mismatches": 0,
        "dynamic_only_feasible_actions": 0,
        "static_only_feasible_actions": 0,
        "scalar_only_vs_static_feasible_actions": 0,
        "static_only_vs_scalar_feasible_actions": 0,
        "static_vs_dynamic_candidate_margin_sign_flips": 0,
        "static_vs_scalar_candidate_margin_sign_flips": 0,
        "static_dynamic_common_M7_candidate_evaluations": 0,
        "static_scalar_common_M7_candidate_evaluations": 0,
    }
    scene_summary = {}
    for scene in range(12):
        result = manifest["results"][str(scene)]
        epsilon = float(result["epsilon"])
        static_relative = np.asarray(result["static_nominal_health_cost_relative_to_q1"])
        static_absolute = np.asarray(result["static_nominal_health_cost_absolute"])
        static_budget_base = float(static_absolute[-1])
        scene_sets = {"static_dynamic": 0, "static_scalar": 0}
        scene_sign_flips = {"static_dynamic": 0, "static_scalar": 0}
        scene_dynamic_only = 0
        scene_static_only = 0
        scene_scalar_only = 0
        scene_static_only_vs_scalar = 0
        margin_values = {"static": [], "scalar": [], "dynamic": []}
        alarm_count = 0
        archive_path = RUN_DIR / f"scene_{scene:02}.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            spec = result["spec"]
            dynamics, topology, sender = spec[:3]
            graph_path = HERE / "experiment_A" / "graphs" / f"{topology}_{dynamics}_target{sender}_qwords.npz"
            with np.load(graph_path, allow_pickle=False) as graph_archive:
                graph = graph_archive["legal"].copy()
            for fixture in ("continuous", "burst"):
                mask = archive[f"{fixture}_masks"].astype(bool)
                target_alarm = mask[:, :, 6]
                alarm_window = target_alarm[:, 50:110]
                alarm_count += int(alarm_window.sum())
                saved_sets = archive[f"{fixture}_feasible_sets"][:, :, 50:110].astype(bool)
                actions = archive[f"{fixture}_actions"].astype(np.int64)
                m7_legal = reconstruct_m7_actions(graph, actions)[:, :, 50:110]
                if np.any(saved_sets & ~m7_legal):
                    raise AssertionError("An archived health-feasible action violates its M7 history.")
                dynamic_relative = archive[f"{fixture}_relative_health_costs"][:, 3, 50:110]
                dynamic_absolute = archive[f"{fixture}_absolute_health_costs"][:, 3, 50:110]
                scalar = archive[f"{fixture}_scalar_fit"][:, 2, 50:110]
                alarm = alarm_window
                static_feasible = (
                    static_relative[None, None, :]
                    <= epsilon * static_budget_base + EPS_TOL
                )
                dynamic_base = dynamic_absolute[..., -1]
                dynamic_feasible = dynamic_relative <= (
                    epsilon * dynamic_base[..., None] + EPS_TOL
                )
                scalar_health = scalar[..., None] * static_relative[None, None, :]
                scalar_feasible = scalar_health <= (
                    epsilon * dynamic_base[..., None] + EPS_TOL
                )
                # Each policy receives M7 legality reconstructed from its own
                # executed action history before the health filter is applied.
                static_m7 = m7_legal[:, 1]
                scalar_m7 = m7_legal[:, 2]
                dynamic_m7 = m7_legal[:, 3]
                static_set = static_m7 & static_feasible
                dynamic_set = dynamic_m7 & dynamic_feasible
                scalar_set = scalar_m7 & scalar_feasible
                compare = alarm[..., None]
                scene_sets["static_dynamic"] += int(
                    np.any(static_set != dynamic_set, axis=-1)[alarm].sum()
                )
                scene_sets["static_scalar"] += int(
                    np.any(static_set != scalar_set, axis=-1)[alarm].sum()
                )
                scene_dynamic_only += int((dynamic_set & ~static_set & compare).sum())
                scene_static_only += int((static_set & ~dynamic_set & compare).sum())
                scene_scalar_only += int((scalar_set & ~static_set & compare).sum())
                scene_static_only_vs_scalar += int((static_set & ~scalar_set & compare).sum())

                static_slack = epsilon - static_relative[None, None, :] / static_budget_base
                dynamic_slack = epsilon - np.divide(
                    dynamic_relative,
                    dynamic_base[..., None],
                    out=np.full_like(dynamic_relative, np.inf),
                    where=dynamic_base[..., None] > 0,
                )
                scalar_slack = epsilon - np.divide(
                    scalar_health,
                    dynamic_base[..., None],
                    out=np.full_like(scalar_health, np.inf),
                    where=dynamic_base[..., None] > 0,
                )
                eligible_dynamic = alarm[..., None] & static_m7 & dynamic_m7
                eligible_scalar = alarm[..., None] & static_m7 & scalar_m7
                if eligible_dynamic.any():
                    static_broadcast = np.broadcast_to(static_slack, dynamic_slack.shape)
                    dynamic_sign_flip = (
                        (static_broadcast >= -EPS_TOL / max(epsilon, 1e-12))
                        != (dynamic_slack >= -EPS_TOL / max(epsilon, 1e-12))
                    ) & eligible_dynamic
                    scene_sign_flips["static_dynamic"] += int(dynamic_sign_flip.sum())
                    margin_values["static"].append(static_broadcast[eligible_dynamic])
                    margin_values["dynamic"].append(dynamic_slack[eligible_dynamic])
                if eligible_scalar.any():
                    scalar_sign_flip = (
                        (static_broadcast >= -EPS_TOL / max(epsilon, 1e-12))
                        != (scalar_slack >= -EPS_TOL / max(epsilon, 1e-12))
                    ) & eligible_scalar
                    scene_sign_flips["static_scalar"] += int(scalar_sign_flip.sum())
                    margin_values["scalar"].append(scalar_slack[eligible_scalar])

        def joined(name):
            return np.concatenate(margin_values[name]) if margin_values[name] else np.empty(0)

        scene_summary[str(scene)] = {
            "spec": result["spec"],
            "epsilon": epsilon,
            "alarm_decisions": alarm_count,
            "static_dynamic_feasible_set_mismatch_count": scene_sets["static_dynamic"],
            "static_scalar_feasible_set_mismatch_count": scene_sets["static_scalar"],
            "dynamic_only_feasible_actions": scene_dynamic_only,
            "static_only_feasible_actions": scene_static_only,
            "scalar_only_vs_static_feasible_actions": scene_scalar_only,
            "static_only_vs_scalar_feasible_actions": scene_static_only_vs_scalar,
            "static_dynamic_candidate_margin_sign_flips": scene_sign_flips["static_dynamic"],
            "static_scalar_candidate_margin_sign_flips": scene_sign_flips["static_scalar"],
            "M7_legal_candidate_margin_summaries": {
                name: summarize_margin(joined(name), epsilon, np.ones(len(joined(name)), dtype=bool))
                if len(joined(name)) else None
                for name in ("static", "scalar", "dynamic")
            },
        }
        totals["alarm_decisions"] += alarm_count
        totals["static_dynamic_feasible_set_mismatches"] += scene_sets["static_dynamic"]
        totals["static_scalar_feasible_set_mismatches"] += scene_sets["static_scalar"]
        totals["dynamic_only_feasible_actions"] += scene_dynamic_only
        totals["static_only_feasible_actions"] += scene_static_only
        totals["scalar_only_vs_static_feasible_actions"] += scene_scalar_only
        totals["static_only_vs_scalar_feasible_actions"] += scene_static_only_vs_scalar
        totals["static_vs_dynamic_candidate_margin_sign_flips"] += scene_sign_flips["static_dynamic"]
        totals["static_vs_scalar_candidate_margin_sign_flips"] += scene_sign_flips["static_scalar"]
        totals["static_dynamic_common_M7_candidate_evaluations"] += len(joined("static"))
        totals["static_scalar_common_M7_candidate_evaluations"] += len(joined("scalar"))

    report = {
        "status": "posthoc_D2_B2_budget_margin_mechanism_audit",
        "B2_RUN_sha256": sha256(RUN_FILE),
        "B2_INDEPENDENT_AUDIT_sha256": sha256(INDEPENDENT_AUDIT),
        "audit_source_sha256": sha256(Path(__file__)),
        "claim_scope": "descriptive_development_screening_only",
        "constraint_tolerance": EPS_TOL,
        "decision": totals,
        "scenes": scene_summary,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "decision": totals}, indent=2))


if __name__ == "__main__":
    main()
