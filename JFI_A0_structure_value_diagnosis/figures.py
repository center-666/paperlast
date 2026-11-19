"""Static scientific figures with explicit data provenance."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PALETTE = {
    "V1a": ("#174a7e", "o"),
    "V1.2a": ("#915f9b", "D"),
    "V1.4": ("#cc730e", "^"),
    "saturation": ("#087e75", "s"),
    "activity-label diagnostic": ("#b55627", "^"),
    "true-onset diagnostic": ("#915f9b", "D"),
}


def save(fig, dest, title):
    fig.savefig(dest, dpi=150, metadata={"Title": title}, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def archive_frontier(models, dest):
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.4), layout="constrained", sharex=True)
    for i, (name, model) in enumerate(models.items()):
        data = model["saved_outcomes"]
        for field, label, ax, scale in (
            ("active_mse", "Attack-period MSE", axes[i, 0], 1),
            ("strict_pre_alarm_rate", "Strict pre-alarm danger (%)", axes[i, 1], 100),
        ):
            for method, values in data["methods"].items():
                if method == "V0":
                    continue
                kind = "saturation" if method.startswith("anchor@") else method
                color, marker = PALETTE[kind]
                x = values["nominal_health"]["relative_pct"]
                ax.scatter(x, scale * values[field], color=color, marker=marker, s=62,
                           linewidths=0.8, edgecolors="white", zorder=3)
            ax.set_title(f"{name.capitalize()} model (n = {data['path_count']})")
            ax.set_ylabel(label)
            ax.set_xlim(-0.23, 2.12)
            ax.set_xticks([0, 0.5, 1, 1.5, 2])
            ax.grid(alpha=0.25)
            if i == 1:
                ax.set_xlabel("Nominal healthy MSE change vs V0 (%)")
    handles = [Line2D([], [], color=color, marker=marker, ls="none", label=label, markersize=7)
               for label, (color, marker) in PALETTE.items()
               if label in ("V1a", "V1.2a", "V1.4", "saturation")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Saved A0 outcomes | onset 7, ten ticks", fontsize=14, fontweight="bold")
    save(fig, dest, "Saved A0 operating points")


def decomposition_figure(models, dest):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, layout="constrained")
    phases = ("first_packet", "first_three", "remaining_seven")
    titles = ("First packet", "First 3", "Next 7")
    names = ("Inactive mass", "Onset mixture", "True-onset fit", "Mass × mixture",
             "Mass × fit", "Mixture × fit")
    colors = ("#87b8da", "#5a82ad", "#345273", "#eed7a1", "#db9f51", "#b55c38")
    for ax, (name, model) in zip(axes, models.items(), strict=True):
        source = model["complete_record_replay"]["decomposition"]
        values = [source[key]["decoded_distortion"] for key in phases]
        pieces = np.array([list(x["squares"].values()) + list(x["signed_cross"].values()) for x in values])
        positive, negative = np.zeros(len(phases)), np.zeros(len(phases))
        for j, component in enumerate(pieces.T):
            ax.bar(titles, component, bottom=np.where(component >= 0, positive, negative),
                   width=0.59, color=colors[j], label=names[j])
            positive += np.maximum(component, 0)
            negative += np.minimum(component, 0)
        ax.scatter(titles, [x["total"] for x in values], marker="o", s=27, color="#202020",
                   label="Total squared error", zorder=4)
        ax.axhline(0, color="#555555", lw=0.7)
        ax.set_title(name.capitalize() + " model")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean squared decoded-distortion error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Complete A0 replays | decomposition with signed cross terms",
                 fontsize=14, fontweight="bold")
    save(fig, dest, "A0 identification error decomposition")


def information_figure(models, dest):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True, layout="constrained")
    methods = ("V1a", "activity-label diagnostic", "true-onset diagnostic", "anchor@2%")
    labels = ("V1a", "Known activity (diagnostic)", "Known onset (diagnostic)", "2% saturation")
    for ax, (name, model) in zip(axes, models.items(), strict=True):
        info = model["complete_record_replay"]["closed_loop"]["methods"]
        for method, label in zip(methods, labels, strict=True):
            data = info[method]
            color, marker = PALETTE["saturation" if method == "anchor@2%" else method]
            ax.scatter(data["support_mse"], data["event_counts"]["before_alarm"], color=color, marker=marker,
                       label=label, s=72, edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_title(name.capitalize() + " model (n = 12)")
        ax.set_xlabel("Attack-period MSE")
        ax.set_ylim(-0.55, 12.55)
        ax.set_yticks(range(0, 13, 2))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Paths with danger strictly before BF")
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Complete A0 replays | separate closed-loop state histories",
                 fontsize=14, fontweight="bold")
    save(fig, dest, "A0 label diagnostics in full closed loop")


def plot_all(report, folder):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    models = report["models"]
    archive_frontier(models, folder / "saved_outcome_frontier.png")
    decomposition_figure(models, folder / "replay_error_decomposition.png")
    information_figure(models, folder / "replay_information_value.png")
