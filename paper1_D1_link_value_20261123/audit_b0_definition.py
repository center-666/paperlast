"""Deterministic audit of the D1-B B0 health-switch definition.

This script reads only the archived D1-A matrices and JSON. It generates no
random paths, no H1 scores, and no attack trajectories.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
B0_REGISTRATION = HERE / "B0_REGISTRATION.json"
D1A_REPORT = RESULTS / "D1A_link_value.json"
GAIN_TABLES = RESULTS / "gain_tables.npz"
COVARIANCE_TRACES = RESULTS / "covariance_traces.npz"
Q_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
HORIZON = 60
HEALTH_START = 20
HEALTH_STOP = 150
W_MISSING = np.diag([0.0, 0.0, 1.0, 1.0])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def risk_from_trace(trace: np.ndarray) -> float:
    return float(sum(np.trace(W_MISSING @ trace[t])
                     for t in range(1, HORIZON + 1)))


def relative_delta(value: float, baseline: float) -> float:
    return float((value - baseline) / (baseline + 1e-12))


def value_budget(kappa: float) -> float:
    positive = max(float(kappa), 0.0)
    return float(0.05 - 0.04 * positive / (1.0 + positive))


def switched_risk(p0: np.ndarray, f_base: np.ndarray, omega_base: np.ndarray,
                  f_alarm: np.ndarray, omega_alarm: np.ndarray,
                  alarm_positions: set[int]) -> float:
    covariance = p0.copy()
    total = 0.0
    for step in range(HORIZON):
        if step in alarm_positions:
            f_matrix, omega = f_alarm, omega_alarm
        else:
            f_matrix, omega = f_base, omega_base
        covariance = f_matrix @ covariance @ f_matrix.T + omega
        covariance = (covariance + covariance.T) / 2.0
        total += float(np.trace(W_MISSING @ covariance))
    return total


def scenario_audit(report: dict, gains: np.lib.npyio.NpzFile,
                   covariances: np.lib.npyio.NpzFile,
                   model_name: str, edge_name: str) -> dict:
    scan = report["models"][model_name]["link_scans"][edge_name]
    prefix = f"{model_name}_edge{edge_name}_to26"
    p0 = gains[f"{prefix}_q1_Pplus"]
    f_base = gains[f"{prefix}_q1_F"]
    omega_base = gains[f"{prefix}_q1_Omega"]
    f_alarm = gains[f"{prefix}_q0.9_F"]
    omega_alarm = gains[f"{prefix}_q0.9_Omega"]

    traces = {
        q: covariances[f"{prefix}_q{q:g}_P_keep"]
        for q in Q_GRID
    }
    risks = {str(q): risk_from_trace(trace) for q, trace in traces.items()}
    baseline = risks["1.0"]
    deltas = {str(q): relative_delta(value, baseline)
              for q, value in ((float(key), value) for key, value in risks.items())}
    kappa = float(scan["kappa"]["keep"]["60"])
    value_epsilon = value_budget(kappa)
    switch_safety = {
        str(q): bool(float(scan["weights"][index]["norm_F_2"]) < 1.0)
        for index, q in enumerate(Q_GRID)
    }
    fixed_value_safe = [q for q in Q_GRID
                        if deltas[str(q)] <= value_epsilon and switch_safety[str(q)]]
    fixed_uniform_safe = [q for q in Q_GRID
                          if deltas[str(q)] <= 0.05 and switch_safety[str(q)]]

    all_positions = [set(position)
                     for position in itertools.combinations(range(HORIZON), 3)]
    switched_values = np.asarray([
        switched_risk(p0, f_base, omega_base, f_alarm, omega_alarm, positions)
        for positions in all_positions
    ])
    switched_deltas = 100.0 * (switched_values / baseline - 1.0)
    fixed_q09_pct = 100.0 * deltas["0.9"]

    if not np.all(np.diff([risks[str(q)] for q in Q_GRID]) <= 1e-9):
        raise AssertionError(f"D1-A risk grid is not non-increasing for {model_name} edge {edge_name}.")
    if 1.0 not in fixed_value_safe or 1.0 not in fixed_uniform_safe:
        raise AssertionError("q=1 must remain feasible under both health budgets.")
    if abs(float(switched_values.min()) - switched_risk(
            p0, f_base, omega_base, f_alarm, omega_alarm, all_positions[int(np.argmin(switched_values))])) > 1e-8:
        raise AssertionError("Switching extrema are not deterministic.")

    return {
        "model": model_name,
        "target_edge": int(edge_name),
        "kappa_L60": kappa,
        "value_budget_epsilon": value_epsilon,
        "switch_safety_P_equals_I": switch_safety,
        "fixed_weight_risk": risks,
        "fixed_weight_relative_delta": deltas,
        "fixed_value_budget_safe_weights": fixed_value_safe,
        "fixed_uniform_5pct_safe_weights": fixed_uniform_safe,
        "q09_fixed_relative_delta_percent": fixed_q09_pct,
        "three_single_alarm_positions": math.comb(HORIZON, 3),
        "three_single_alarm_switched_relative_delta_percent": {
            "minimum": float(switched_deltas.min()),
            "mean": float(switched_deltas.mean()),
            "maximum": float(switched_deltas.max()),
        },
    }


def run() -> None:
    registration = json.loads(B0_REGISTRATION.read_text(encoding="utf-8"))
    report = json.loads(D1A_REPORT.read_text(encoding="utf-8"))
    if registration["candidate_weights"] != list(Q_GRID):
        raise AssertionError("B0 candidate grid differs from D1-A.")
    if registration["health_interval"] != {
            "start_inclusive": HEALTH_START, "stop_exclusive": HEALTH_STOP}:
        raise AssertionError("B0 health interval mismatch.")
    if registration["simultaneous_inference"]["family_size"] != 84:
        raise AssertionError("B0 simultaneous family size must be 12*7=84.")
    if registration["alarm_policy"]["post_alarm_hold"] != 0:
        raise AssertionError("B0 primary policy must recover immediately.")

    expected_models = {
        "U_unstable": ("35",), "U_stable": ("35",),
        "R_unstable": ("35", "1"), "R_stable": ("35", "1"),
    }
    scenario_results = []
    with np.load(GAIN_TABLES) as gains, np.load(COVARIANCE_TRACES) as covariances:
        for model_name, edges in expected_models.items():
            for edge_name in edges:
                scenario_results.append(
                    scenario_audit(report, gains, covariances, model_name, edge_name))

    source_hashes = {
        "D1A_link_value.json": digest(D1A_REPORT),
        "gain_tables.npz": digest(GAIN_TABLES),
        "covariance_traces.npz": digest(COVARIANCE_TRACES),
        "B0_REGISTRATION.json": digest(B0_REGISTRATION),
    }
    output = {
        "status": "PASS_deterministic_B0_definition_audit",
        "scope": "No healthy paths, no H1 score generation, no attack Monte Carlo",
        "registration": registration,
        "source_sha256": source_hashes,
        "deterministic_scenarios_checked": len(scenario_results),
        "results": scenario_results,
        "interpretation": {
            "fixed_weight_value_safe_set": "all six D1-A deterministic cases reduce to q=1",
            "switching_sensitive_quantity": "three single alarm positions are deterministic occupancy sensitivity only",
            "health_fwer": "separate from the paired health-risk confidence family",
            "B_vs_D": "compare health-attack Pareto outcomes, not A_L dominance",
        },
    }
    output_path = RESULTS / "B0_definition_audit.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": output["status"],
        "deterministic_scenarios_checked": len(scenario_results),
        "output": str(output_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    run()
