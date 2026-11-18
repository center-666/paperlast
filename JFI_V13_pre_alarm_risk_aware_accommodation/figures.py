"""Publication-readable views of the recorded Pareto and calibration evidence."""
from matplotlib import pyplot as plt

from inputs import HERE, load_json


def pareto():
    src = load_json(HERE/"results/legacy_pareto.json")["models"]
    names = ("Unknown onset A0","A1-H replay","A1-C replay")
    codes = ("A0_onset_replay","A1_H_mismatch","A1_C_mismatch")
    fig, axs = plt.subplots(2,3,figsize=(11,6.5),layout="constrained")
    for i, (model, groups) in enumerate(src.items()):
        for j, case in enumerate(codes):
            ax = axs[i,j]
            rows = groups["96"][case]["methods"]
            ref = rows["V1a"]["risk_before_alarm"]
            def metric(key):
                row = rows[key]
                return 100*row["mse_relative"], 100*(row["risk_before_alarm"]-ref)/96
            fixed = [metric(key) for key in ("fixed-0","fixed-0.25","fixed-0.5","fixed-0.75","V1a")]
            ax.plot(*zip(*fixed), color="#607184", marker="o", ms=4, label="Fixed weights")
            ax.scatter(*metric("V1a"),color="#14517b",s=55,label="V1a")
            ax.scatter(*metric("two-mode"),color="#cf7131",marker="s",s=30,label="Two-mode")
            ax.scatter(*metric("conditional-MSE"),color="#268073",marker="D",s=29,label="Conditional MSE")
            ax.scatter(*metric("V1.2a"),color="#7851a0",marker="^",s=36,label="V1.2a")
            ax.set_xscale("symlog",linthresh=1)
            ax.axhline(0,color="#a6a6a6",lw=.7)
            ax.axvline(0,color="#a6a6a6",lw=.7)
            ax.set_title(model.capitalize()+" / "+names[j],fontsize=10)
            ax.set_xlabel("Support MSE change (%)")
            if j==0:
                ax.set_ylabel("Pre-alarm danger change (pp)")
            ax.grid(axis="y",alpha=.2)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=5,fontsize=8)
    fig.savefig(HERE/"results/legacy_pareto.png",dpi=170)
    plt.close(fig)


def calibration():
    data = load_json(HERE/"results/validation/summary.json")["models"]
    fig, axs = plt.subplots(2,4,figsize=(12,6.6),layout="constrained",sharex=True,sharey=True)
    names = ("Unknown onset A0","Shorter A0 replay","A1-H replay","A1-C replay")
    codes = ("unknown_start","timing_mismatch","A1_H_replay","A1_C_replay")
    for i,(model, rows) in enumerate(data.items()):
        for j,code in enumerate(codes):
            ax=axs[i,j]
            ax.plot([0,1],[0,1],color="#999",lw=.8,label="Calibrated")
            for which,clr in (("V1a","#8a7c99"),("V1.3","#06726c")):
                bins=rows["profiles"][code]["calibration"][which]["bins"]
                points=[(q["predicted"],q["observed"],q["episode_cluster_interval95"]) for q in bins]
                for k,(x,y,error) in enumerate(points):
                    ax.errorbar(x,y,yerr=[[y-error[0]],[error[1]-y]],fmt="o",ms=4,
                                lw=.8,alpha=.85,color=clr,label=which if not k else None)
            ax.set_xlim(-.01,1.01)
            ax.set_ylim(-.03,1.03)
            ax.set_title(model.capitalize()+" / "+names[j],fontsize=9)
            ax.set_xlabel("Predicted probability")
            if j==0:
                ax.set_ylabel("Observed frequency")
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=3,fontsize=8)
    fig.savefig(HERE/"results/independent_calibration.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    pareto()
    calibration()
