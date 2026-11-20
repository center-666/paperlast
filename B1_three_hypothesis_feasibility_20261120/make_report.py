"""Generate a standalone evidence figure and a small CSV from stored outputs."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    causal = json.loads((HERE / "results/causal_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((HERE / "results/validation.json").read_text(encoding="utf-8"))
    engineering = json.loads((HERE / "results/summary.json").read_text(encoding="utf-8"))["regional_gaussian_watermark_replay"]
    names = ["uniform", "development_fixed", "adaptive_health_0", "adaptive_health_0.02"]
    labels = ["Uniform", "Fixed", "Adaptive 0", "Adaptive 2%"]
    colors = ["#555555", "#2479b2", "#d48419", "#318469"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), layout="constrained")
    rows = []
    for i, h in enumerate(("0", "1", "2")):
        means, se = [], []
        for policy in names:
            row = causal["results"]["common"][h][policy]
            means.append(row["full_mse_change_vs_uniform"])
            se.append(row["full_mse_change_paired_se"])
            rows.append(dict(true_h=h, policy=policy, full_mse=row["full_mse"],
                             change=row["full_mse_change_vs_uniform"], paired_se=row["full_mse_change_paired_se"],
                             support_mse=row["support_mse"], mean_coding_energy=row["mean_coding_energy"]))
        positions = np.arange(4) + (i - 1) * .22
        axes[0].errorbar(positions, means, yerr=np.array(se) * 1.96, fmt="o", capsize=3,
                         label=["Healthy", "Attack onset 7", "Attack onset 8"][i])
    axes[0].axhline(0, color="#999999", linewidth=1)
    axes[0].set_xticks(np.arange(4), labels, rotation=25, ha="right")
    axes[0].set_ylabel("Full-horizon common-G MSE change")
    axes[0].set_title("Causal comparison\nn = 16 paired paths", fontsize=11)
    axes[0].legend(fontsize=8)

    methods = ["discard", "raw", "V1a", "common", "bayes"]
    values = [engineering["1"]["discard"]["total_mse"], engineering["1"]["raw"]["total_mse"],
              validation["regional_V1a_summary"]["1"]["total_mse"], engineering["1"]["common"]["total_mse"],
              engineering["1"]["bayes"]["total_mse"]]
    axes[1].bar(np.arange(len(methods)), values, color=["#b34d4d", "#888888", "#7c6ab0", "#2479b2", "#318469"])
    axes[1].set_xticks(np.arange(len(methods)), ["Always drop", "Raw CG", "Frozen V1a", "Known bank CG", "Known bank MMSE"], rotation=30, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Full-horizon state MSE (log scale)")
    axes[1].set_title("Regional scenario: onset 7\nGaussian watermarks", fontsize=11)
    for i, value in enumerate(values):
        axes[1].annotate(f"{value:.3g}", (i, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)

    refined = json.loads((HERE / "results/high_accuracy_refinement.json").read_text(encoding="utf-8"))
    selected = [r for r in refined if r["model"] == "regional_missing_unstable_mode"]
    x = np.arange(3)
    axes[2].bar(x - .17, [r["stages"][-1]["conditional_actual_bayes_change"] for r in selected], width=.34, label="Ideal Bayes")
    axes[2].bar(x + .17, [r["stages"][-1]["conditional_actual_common_change"] for r in selected], width=.34, label="Actual common-G")
    axes[2].axhline(0, color="#777777", linewidth=1)
    axes[2].set_xticks(x, ["Healthy", "Onset 7", "Onset 8"])
    axes[2].set_ylabel("Conditional one-step MSE change")
    axes[2].set_title("Same information-directed policy\nPath 16, tick 8", fontsize=11)
    axes[2].legend(fontsize=8)
    fig.suptitle("Exploratory mechanism screen: known-waveform bank; no adaptive superiority established", fontsize=11)
    fig.savefig(HERE / "results/evidence.png", dpi=180)
    fig.savefig(HERE / "results/evidence.pdf")
    plt.close(fig)
    with (HERE / "results/causal_common_g.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
