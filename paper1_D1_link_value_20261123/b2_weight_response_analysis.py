"""Summarize exploratory seven-weight B2 path replays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, t

import b1_health as health


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "b2_weight_response_results"
SHADOW = HERE / "b2_shadow_decomposition_results"
Q = np.array([0., .2, .4, .6, .8, .9, 1.])
METRICS = (
    "missing_window_mse", "missing_active_mse", "missing_gap_mse",
    "missing_post20_mse", "missing_tail_mse", "missing_tail_110_120_mse",
    "missing_tail_120_130_mse", "missing_tail_130_150_mse",
    "full_window_mse", "network_window_mse", "network_post20_mse",
    "attack_forcing_window", "attack_forcing_active", "effective_attack_weight",
    "target_attack_unalarmed_fraction", "target_path_missed_fraction",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(data):
    n = len(data)
    mean = np.mean(data, axis=0)
    se = np.std(data, axis=0, ddof=1) / np.sqrt(n)
    critical = t.ppf(.975, n - 1)
    return {"mean": mean.tolist(), "path_se": se.tolist(),
            "descriptive_ci95_lower": (mean - critical * se).tolist(),
            "descriptive_ci95_upper": (mean + critical * se).tolist(), "n": n}


def rank_correlation(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def condition(scene, spec, fixture, d1a):
    dynamics, topology, sender, watermark = spec
    filename = f"scene_{scene:02}_{fixture}_{watermark}.npz"
    scan = d1a["models"][f"{topology}_{dynamics}"]["link_scans"][str(sender)]
    gain = [entry["attack_transfer_gain_by_horizon"]["60"]
            for entry in scan["weights"]]
    fixed_healthy = [entry["J_by_horizon"]["60"] / 60
                     for entry in scan["risk"]["keep"]]
    with np.load(OUT / filename, allow_pickle=False) as data, \
            np.load(HERE / "b1_health_results" / f"scene_{scene:02}.npz",
                    allow_pickle=False) as healthy:
        health_values = healthy["missing"]
        masks = healthy["masks"]
        target_event = ((masks & 64) != 0).any(axis=1)
        event_count = int(target_event.sum())
        paired_health = health_values - health_values[:, [-1]]
        health_conditional = (describe(paired_health[target_event]) if event_count else None)
        observed = {}
        contrasts = {}
        for metric in METRICS:
            values = data[metric]
            if metric == "missing_gap_mse" and fixture == "continuous":
                observed[metric] = None
                contrasts[metric] = None
                continue
            observed[metric] = describe(values)
            contrasts[metric] = describe(values - values[:, [-1]])
        window = np.asarray(observed["missing_window_mse"]["mean"])
        active = np.asarray(observed["missing_active_mse"]["mean"])
        tail = np.asarray(observed["missing_tail_mse"]["mean"])
        forcing = np.asarray(observed["attack_forcing_active"]["mean"])
        minimum = float(window.min())
        minimizing_q = Q[np.isclose(window, minimum, atol=1e-12, rtol=1e-12)].tolist()
        health_safety = healthy["safety"].tolist()
    with np.load(SHADOW / filename, allow_pickle=False) as shadow:
        shadow_terms = {}
        for period in ("window", "active", "gap", "post"):
            if period == "gap" and fixture == "continuous":
                shadow_terms[period] = None
                continue
            terms = shadow[period]
            shadow_terms[period] = {
                "clean_shadow": describe(terms[:, :, 0]),
                "twice_cross": describe(terms[:, :, 1]),
                "attack_response_squared": describe(terms[:, :, 2]),
                "paired_terms_vs_q1": {
                    "clean_shadow": describe(terms[:, :, 0] - terms[:, [-1], 0]),
                    "twice_cross": describe(terms[:, :, 1] - terms[:, [-1], 1]),
                    "attack_response_squared": describe(terms[:, :, 2] - terms[:, [-1], 2]),
                },
            }
    return {
        "scene": scene, "dynamics": dynamics, "topology": topology,
        "target_sender": sender, "watermark": watermark, "fixture": fixture,
        "q_values": Q.tolist(),
        "d1a": {"kappa60": scan["kappa"]["keep"]["60"],
                "observability_rank_q0": scan["weights"][0]["rank"],
                "fixed_q_healthy_60_step_mean": fixed_healthy,
                "fixed_q_attack_transfer_gain_60": gain,
                "fixed_q_transfer_spearman_with_window_mse": rank_correlation(gain, window),
                "fixed_q_transfer_spearman_with_active_mse": rank_correlation(gain, active),
                "fixed_q_transfer_spearman_with_tail_mse": rank_correlation(gain, tail),
                "fixed_q_transfer_spearman_with_empirical_forcing": rank_correlation(gain, forcing)},
        "healthy_B1_reused": {"missing_mse": describe(health_values),
                               "paired_difference_vs_q1": describe(paired_health),
                               "target_alarm_paths": event_count,
                               "target_alarm_fraction": event_count / len(health_values),
                               "difference_conditional_on_target_alarm": health_conditional,
                               "maximum_mode_norm": health_safety,
                               "passes_registered_P_I_screen": [x < 1 for x in health_safety]},
        "observed": observed, "paired_difference_vs_q1": contrasts,
        "fixed_attack_schedule_shadow_decomposition": shadow_terms,
        "exploratory_minimizing_q_window": minimizing_q,
    }


def values_line(values, digits=3):
    return " / ".join(f"{value:.{digits}f}" for value in values)


def markdown(rows):
    lines = [
        "# B2 七权重响应曲线与三阶段风险",
        "",
        "本报告按最终 B2 种子重构同一批物理路径及水印攻击，重放七个候选报警权重。每个条件的报警掩码逐位等于已归档 B2。健康曲线读取 B1 的留出路径，这些路径已参与策略筛选。结果为事后探索和同路径重放。",
        "",
        "报警时完整目标报文按 q 加权，未报警时权重为 1。下面每组数字依次对应 q = 0、0.2、0.4、0.6、0.8、0.9、1。攻击期完整窗口、攻击激活拍、正常间隙与攻击停止后恢复段分开报告。",
        "",
        "## 三阶段响应",
        "",
        "| 场景 | 故障 | B1 健康 MSE | 攻击窗口 MSE | 攻击激活拍 MSE | 突发正常间隙 MSE | 攻击后 MSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        obs = row["observed"]
        label = f"{row['dynamics']} {row['topology']} {row['target_sender']} {row['watermark']}"
        gap = "—" if obs["missing_gap_mse"] is None else values_line(obs["missing_gap_mse"]["mean"])
        lines.append(f"| {label} | {row['fixture']} | "
                     f"{values_line(row['healthy_B1_reused']['missing_mse']['mean'])} | "
                     f"{values_line(obs['missing_window_mse']['mean'])} | "
                     f"{values_line(obs['missing_active_mse']['mean'])} | {gap} | "
                     f"{values_line(obs['missing_tail_mse']['mean'])} |")
    lines.extend([
        "",
        "## 链路价值、攻击输入和响应排序",
        "",
        "| 场景 | 故障 | q=0 后观测秩 | D1-A κ⁽⁶⁰⁾ | 七点窗口 MSE 最小值对应 q | 固定权重攻击传递增益与窗口 MSE 的 Spearman ρ |",
        "|---|---|---:|---:|---|---:|",
    ])
    for row in rows:
        label = f"{row['dynamics']} {row['topology']} {row['target_sender']} {row['watermark']}"
        lines.append(f"| {label} | {row['fixture']} | {row['d1a']['observability_rank_q0']} | "
                     f"{row['d1a']['kappa60']:.4g} | {row['exploratory_minimizing_q_window']} | "
                     f"{row['d1a']['fixed_q_transfer_spearman_with_window_mse']:.3f} |")
    lines.extend([
        "",
        "排序相关仅在同一条件下的七个候选点之间计算，属于描述性拟合。κ 是每个物理场景的一项固定链路指标，单独无法给出同一链路七个权重的 MSE 排序。D1-A 的增益按固定 q 连续施加计算，本实验 q 只在报警拍生效。特别是 q=0 时，未报警的攻击拍仍以权重 1 进入估计器，因此固定 q=0 的零攻击传递增益不等于切换策略的攻击输入为零。",
        "",
        "攻击后恢复 MSE 由继承误差、继续变化的报警门控和后续噪声共同决定。实际攻击 forcing 是输入项平方的路径均值，其与 MSE 之间还包含时间传播及交叉项。攻击激活拍和正常间隙是时间分段，包含此前误差记忆。",
        "",
        "## 固定报警序列下的误差分解",
        "",
        "对同一条物理路径、同一权重和同一个受攻击报警序列，构造攻击估计误差 e 及将目标报文扰动置零后的反事实误差 e⁰。采用估计减真实状态的误差约定，令 b=e−e⁰，则逐拍满足 bₖ=F(qₖ)bₖ₋₁+qₖG(qₖ)δₖ。缺失块二次损失满足 eₖᵀWeₖ=(e⁰ₖ)ᵀWe⁰ₖ+2(e⁰ₖ)ᵀWbₖ+bₖᵀWbₖ。该恒等式说明输入 forcing 的大小无法单独确定最终 MSE，响应还取决于切换状态传播与交叉项。受攻击报警序列依赖原始攻击路径，反事实误差和 b 可以相关。",
        "",
        "本轮已对全部 24 个条件和七个权重重放清洁报文反事实，三个加项的路径级数组及均值、配对差与描述性区间存于 analysis.json。逐路径加和复现攻击 MSE。清洁参考使用受攻击数据产生的报警序列，因而包含门控选择效应。",
        "",
        "| 示例 | 时间段 | q | 清洁参考误差 | 两倍交叉项 | 攻击响应平方 | 三项之和 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for scene, fixture, period in ((0, "continuous", "active"),
                                   (0, "burst", "active"),
                                   (0, "burst", "gap"),
                                   (2, "continuous", "active"),
                                   (6, "continuous", "active")):
        row = next(item for item in rows if item["scene"] == scene and item["fixture"] == fixture)
        terms = row["fixed_attack_schedule_shadow_decomposition"][period]
        for index in (0, 2, 4, 6):
            clean = terms["clean_shadow"]["mean"][index]
            cross = terms["twice_cross"]["mean"][index]
            response = terms["attack_response_squared"]["mean"][index]
            lines.append(f"| {row['dynamics']} {row['topology']} {row['target_sender']} {fixture} | "
                         f"{period} | {Q[index]:g} | {clean:.3f} | {cross:.3f} | "
                         f"{response:.3f} | {clean + cross + response:.3f} |")
    lines.extend([
        "",
        "清洁参考的误差与攻击响应之间的交叉项允许正负变化。攻击响应平方包含历次已接纳扰动经过后续模式传播的结果，因而同一次的输入 forcing 平方不能直接替代该项。",
        "",
        "健康 MSE 是 B1 既有留出数据中的无条件路径平均。稀有目标误报警条件下的配对增量、各策略安全筛查结果、D1-A 固定权重协方差风险、每项配对差区间及三个攻击后时间分段均保存在 analysis.json。q=0 在不稳定 U 的单拍范数筛查未通过，曲线上保留该点用于观察风险，不作为已证实切换安全的候选。",
        "",
        "负耦合和 iid 使用不同物理路径，均值差未作跨水印配对。本阶段网格最小值用于描述这些既有路径，不作为新锁定权重、独立验证或普遍最优权重。",
        "",
    ])
    return "\n".join(lines)


def main():
    d1a = json.loads((HERE / "results" / "D1A_link_value.json").read_text(encoding="utf-8"))
    rows = [condition(i, spec, fixture, d1a)
            for i, spec in enumerate(health.scenarios())
            for fixture in ("continuous", "burst")]
    result = {"status": "exploratory_posthoc_replay", "conditions": rows,
              "B1_health_reuse": "B1 holdout paths were used to select strategies; summaries are descriptive",
              "scope": "No new independent random paths. Per-scene q minima are descriptive and have no selection adjustment."}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n",
                                         encoding="utf-8", newline="\n")
    (OUT / "B2_WEIGHT_RESPONSE_REPORT.md").write_text(markdown(rows), encoding="utf-8", newline="\n")
    files = [HERE / "B2_WEIGHT_RESPONSE_PLAN.md", HERE / "B2_WEIGHT_RESPONSE_REPRODUCTION.md",
             HERE / "b2_weight_response.py",
             HERE / "verify_b2_weight_response.py", HERE / "b2_weight_response_analysis.py",
             HERE / "b2_shadow_decomposition.py", HERE / "verify_b2_shadow_decomposition.py",
             OUT / "response_manifest.json", OUT / "independent_verification.json",
             SHADOW / "shadow_manifest.json", SHADOW / "independent_verification.json",
             OUT / "analysis.json", OUT / "B2_WEIGHT_RESPONSE_REPORT.md"]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    (OUT / "analysis_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "conditions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
