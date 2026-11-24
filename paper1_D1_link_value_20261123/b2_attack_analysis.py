"""Descriptive paired analysis for the frozen B2 attack archives."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.stats import t

import b1_health as health


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = HERE / "b2_attack_results"
REGISTRATION = HERE / "B2_REGISTRATION.json"
POLICY = HERE / "b1_health_results" / "policy_lock.json"
N = 2048
METRICS = (
    "missing_window_mse", "missing_post20_mse", "missing_tail_mse",
    "full_window_mse", "network_window_mse", "network_post20_mse",
    "target_alarm_delay", "target_alarm_run_max",
    "target_attack_accept_fraction", "target_window_accept_fraction",
    "accepted_target_forcing",
)
MSE_METRICS = (
    "missing_window_mse", "missing_post20_mse", "missing_tail_mse",
    "full_window_mse", "network_window_mse", "network_post20_mse",
    "accepted_target_forcing",
)


def summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    critical = float(t.ppf(0.975, len(values) - 1))
    return {
        "mean": float(values.mean()),
        "path_se": se,
        "descriptive_ci95": [
            float(values.mean() - critical * se),
            float(values.mean() + critical * se),
        ],
        "n": int(len(values)),
    }


def load_condition(scene_index: int, fixture: str, watermark: str):
    path = OUTPUT / f"scene_{scene_index:02}_{fixture}_{watermark}.npz"
    with np.load(path, allow_pickle=False) as archive:
        data = {strategy: {metric: archive[f"{strategy}__{metric}"].copy()
                           for metric in METRICS}
                for strategy in ("A", "BD", "C")}
        masks = archive["masks"].copy()
        threshold = float(archive["threshold"])
    return data, masks, threshold


def alarm_summary(masks: np.ndarray, fixture: str) -> dict:
    bits = (masks[:, :, None] & (1 << np.arange(7))) != 0
    target = bits[:, 50:110, 6]
    active = np.ones(60, dtype=bool)
    if fixture == "burst":
        active = (np.arange(60) % 10) < 6
    active_alarm = target[:, active]
    detected = active_alarm.any(axis=1)
    first_active_index = np.argmax(active_alarm, axis=1)
    active_offsets = np.flatnonzero(active)
    first_delay = active_offsets[first_active_index]
    gap_alarm = target[:, ~active]
    runs = []
    window_runs = []
    for row in bits[:, 1:, 6]:
        cur = peak = 0
        for value in row:
            cur = cur + 1 if value else 0
            peak = max(peak, cur)
        runs.append(peak)
    for row in target:
        cur = peak = 0
        for value in row:
            cur = cur + 1 if value else 0
            peak = max(peak, cur)
        window_runs.append(peak)
    return {
        "target_alarm_fraction_on_attack_ticks": float(active_alarm.mean()),
        "target_attack_accept_fraction": float((~active_alarm).mean()),
        "target_attack_detected_path_fraction": float(detected.mean()),
        "target_attack_miss_fraction": float((~detected).mean()),
        "first_alarm_delay_conditional_mean": (
            float(first_delay[detected].mean()) if detected.any() else None),
        "first_alarm_delay_conditional_median": (
            float(np.median(first_delay[detected])) if detected.any() else None),
        "target_max_run_mean": float(np.mean(runs)),
        "target_max_run_max": int(np.max(runs)),
        "target_attack_window_max_run_mean": float(np.mean(window_runs)),
        "target_attack_window_max_run_max": int(np.max(window_runs)),
        "target_alarm_fraction_on_burst_gaps": (
            float(gap_alarm.mean()) if gap_alarm.shape[1] else None),
        "target_burst_gap_accept_fraction": (
            float((~gap_alarm).mean()) if gap_alarm.shape[1] else None),
        "target_window_accept_fraction": float((~target).mean()),
        "all_edge_alarm_fraction": bits[:, 50:110].mean(axis=(0, 1)).tolist(),
        "simultaneous_alarm_fraction": float((bits[:, 50:110].sum(axis=2) >= 2).mean()),
    }


def condition_report(scene_index: int, spec, fixture: str, policy: dict):
    dynamics, topology, target, watermark = spec
    data, masks, threshold = load_condition(scene_index, fixture, watermark)
    row = policy["rows"][scene_index]
    weights = {"A": float(row["A"]), "BD": float(row["B"]),
               "C": float(row["C"])}
    target_alarm = ((masks[:, :, None] & (1 << np.arange(7))) != 0)[:, :, 6]
    attack_active = np.ones(60, dtype=bool)
    if fixture == "burst":
        attack_active = (np.arange(60) % 10) < 6
    gap_active = ~attack_active
    report = {
        "scene": scene_index,
        "dynamics": dynamics,
        "topology": topology,
        "target_sender": target,
        "watermark": watermark,
        "fixture": fixture,
        "threshold": threshold,
        "alarm": alarm_summary(masks, fixture),
        "strategies": {},
        "effective_attack_weights": {},
        "effective_burst_gap_weights": {},
        "paired": {},
    }
    for strategy in ("A", "BD", "C"):
        effective_q = np.where(target_alarm, weights[strategy], 1.0)
        report["strategies"][strategy] = {
            metric: summary(data[strategy][metric]) for metric in METRICS
        }
        report["effective_attack_weights"][strategy] = summary(
            effective_q[:, 50:110][:, attack_active].mean(axis=1))
        report["effective_burst_gap_weights"][strategy] = (
            summary(effective_q[:, 50:110][:, gap_active].mean(axis=1))
            if gap_active.any() else None)
    for left, right, label in (("BD", "A", "BD_minus_A"),
                               ("C", "A", "C_minus_A"),
                               ("BD", "C", "BD_minus_C")):
        report["paired"][label] = {
            metric: summary(data[left][metric] - data[right][metric])
            for metric in METRICS
        }
    return report


def make_markdown(reports: list[dict]) -> str:
    lines = [
        "# B2 攻击性能描述性分析",
        "",
        "结果条件于 B1 冻结阈值和策略表，使用 B2 新攻击路径。B 与 D 在 B1 中策略等价，因此这里只保留 BD 一次执行。所有区间均为路径级配对差的描述性 95% t 区间，不构成确认性结论。",
        "",
        "## 逐条件结果",
        "",
        "| 场景 | 故障 | 水印 | 目标 | A 缺失块 MSE | BD 缺失块 MSE | C 缺失块 MSE | BD−A 配对差 [95%区间] | C−A 配对差 [95%区间] | 攻击漏检率 | 首次报警延迟 | 窗口最长报警串均值 | 污染拍未报警率 | A/BD/C 攻击拍有效权重 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in reports:
        s = item["strategies"]
        p = item["paired"]
        lines.append(
            f"| {item['dynamics']} {item['topology']} | {item['fixture']} | {item['watermark']} | {item['target_sender']} | "
            f"{s['A']['missing_window_mse']['mean']:.6g} | "
            f"{s['BD']['missing_window_mse']['mean']:.6g} | "
            f"{s['C']['missing_window_mse']['mean']:.6g} | "
            f"{p['BD_minus_A']['missing_window_mse']['mean']:.6g} "
            f"[{p['BD_minus_A']['missing_window_mse']['descriptive_ci95'][0]:.4g}, "
            f"{p['BD_minus_A']['missing_window_mse']['descriptive_ci95'][1]:.4g}] | "
            f"{p['C_minus_A']['missing_window_mse']['mean']:.6g} "
            f"[{p['C_minus_A']['missing_window_mse']['descriptive_ci95'][0]:.4g}, "
            f"{p['C_minus_A']['missing_window_mse']['descriptive_ci95'][1]:.4g}] | "
            f"{item['alarm']['target_attack_miss_fraction']:.4f} | "
            f"{item['alarm']['first_alarm_delay_conditional_mean']:.3f} | "
            f"{item['alarm']['target_attack_window_max_run_mean']:.3f} | "
            f"{item['alarm']['target_attack_accept_fraction']:.4f} | "
            f"{item['effective_attack_weights']['A']['mean']:.3f}/"
            f"{item['effective_attack_weights']['BD']['mean']:.3f}/"
            f"{item['effective_attack_weights']['C']['mean']:.3f} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "污染拍未报警率只在实际注入时刻计算。攻击有效权重对报警拍使用策略权重 q，对未报警拍使用 1，作用于完整水印后污染报文。analysis.json 另存逐路径攻击拍和突发间隙的有效权重。配对差按同一条新攻击路径计算，区间是未校正的双侧 95% t 区间，属于描述性统计。BD 是 B/D 的去重执行，不能由本轮结果估计 D−B 的攻击增量。C 与 BD 的差异只表示在冻结 B1 策略下的描述性差异。B1 的健康路径未进入本轮攻击路径。不同水印条件使用独立物理随机路径，因此本表不提供负耦合与 iid 的配对因果比较。R 的 1→26 条件与原监测边 1→0 共用发送节点和传感器量测，但注入仅发生在注册的 1→26 通信弧上，H1 对两条有向边保留独立的检测位。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    reports = [condition_report(i, spec, fixture, policy)
               for i, spec in enumerate(health.scenarios())
               for fixture in ("continuous", "burst")]
    output = {
        "status": "descriptive_b2_analysis",
        "path_count_per_condition": N,
        "strategy_set": ["A", "BD", "C"],
        "interpretation": "B and D are policy-equivalent in B1 and are executed once as BD. Intervals are descriptive paired intervals conditional on frozen B1 thresholds and policy lock.",
        "conditions": reports,
    }
    (OUTPUT / "analysis.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    (OUTPUT / "B2_ATTACK_REPORT.md").write_text(
        make_markdown(reports), encoding="utf-8", newline="\n")
    input_paths = [
        REGISTRATION,
        HERE / "B2_REGISTRATION_SHA256.txt",
        HERE / "B2_REPRODUCTION.md",
        HERE / "B1_HEALTH_RUN.json",
        HERE / "B1_POSTHOC_STATUS.md",
        HERE / "B2_METRIC_CORRECTION.md",
        HERE / "B2_EXECUTION_AMENDMENT.md",
        HERE / "b1_health.py",
        HERE / "b1_mode_models.py",
        HERE / "b1_health_results" / "policy_lock.json",
        HERE / "b1_mode_results" / "mode_tables.npz",
        ROOT / "paper1_common_g50_gain_acceptance_20261121" / "physical_configuration.json",
        ROOT / "paper1_partial_observability_20261123" / "run_partial50.py",
        ROOT / "paper1_history_ablation_20261123" / "ablation.py",
        ROOT / "paper1_sscd_commonrandom_20261122" / "watson_crn.py",
        ROOT / "paper1_original50_source_20261121" / "source" / "simulation_revised_detection_metrics_complete_v2.py",
        HERE / "b2_attack.py",
        HERE / "verify_b2_attack.py",
        HERE / "verify_b2_metric_correction.py",
        HERE / "verify_b2_input_correction.py",
        HERE / "b2_attack_analysis.py",
    ]
    input_hashes = {}
    for path in input_paths:
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        input_hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    scene_thresholds = {}
    for scene_index in range(12):
        scene_path = HERE / "b1_health_results" / f"scene_{scene_index:02}.npz"
        with np.load(scene_path, allow_pickle=False) as scene:
            scene_thresholds[scene_path.name] = float(scene["threshold"])
        input_hashes[str(scene_path.relative_to(ROOT))] = hashlib.sha256(
            scene_path.read_bytes()).hexdigest()
    output_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(OUTPUT.iterdir()) if path.is_file()
        and path.name != "run_manifest.json"
    }
    manifest = {
        "status": "complete_descriptive_B2_attack_replay",
        "authorization_and_claim_scope": "B2 executed after B1 under frozen definitions. B1 was posthoc under B0 and pending manual confirmation. B2 statistics are descriptive and conditional on B1 thresholds and policy lock.",
        "B1_commit": "023b6dbe7fff2b29fae7b297171cc69187393a42",
        "B2_registration_sha256": hashlib.sha256(REGISTRATION.read_bytes()).hexdigest(),
        "seed_namespace": [20261124, 2207],
        "paths": {"conditions": 24, "per_condition": 2048,
                  "total": 49152, "time_length": 150,
                  "attack_window_half_open": [50, 110]},
        "policies": ["A", "BD", "C"],
        "B1_thresholds": scene_thresholds,
        "pre_input_correction_manifest_sha256": hashlib.sha256(
            (HERE / "b2_attack_results_pre_input_correction" / "run_manifest.json").read_bytes()
        ).hexdigest(),
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "inputs_sha256": input_hashes,
        "outputs_sha256": output_hashes,
    }
    (OUTPUT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({"status": output["status"],
                      "conditions": len(reports),
                      "report": str(OUTPUT / "B2_ATTACK_REPORT.md")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
