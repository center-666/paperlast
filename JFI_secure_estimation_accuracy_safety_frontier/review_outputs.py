"""Audit saved paths, export checked comparison points, and plot the two costs."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from frozen_inputs import BUDGETS, HERE, MODELS, V14, digest, load_json, write_json
from receiver import label
from replay import paired_risk


ROOT = HERE/"results"
PROFILES = ("unknown_start","timing_mismatch","A1_H_replay","A1_C_replay")


def check_runs(frontier, comparison):
    reg = load_json(HERE/"registration.json")
    assert comparison["calibration_sha256"]==digest(ROOT/"calibration_frozen.json")
    assert comparison["hashes"]["frontier"]==digest(ROOT/"frontier.json")
    for name,sha in comparison["hashes"]["candidate"].items():
        assert digest(HERE/name)==sha,name
    upstream = load_json(V14/"results/delivery_checks.json")["files_sha256"]
    for name,sha in upstream.items():
        assert digest(V14/name)==sha,name
    checks = {"V14_results_files_verified":len(upstream),"models":{}}
    ticks = reg["horizon"]
    Tpaths = reg["holdout"]["paths_per_model_and_profile"]
    d = reg["D_squared"]
    for model in MODELS:
        found = comparison["models"][model]
        names = [m["name"] for m in found["method_table"]]
        n0 = names.index("V0")
        with np.load(ROOT/(model+"_paired.npz"),allow_pickle=False) as item:
            for case,row in found["cases"].items():
                def data(field):
                    return item[case+"__"+field]
                assert len(data("pre_alarm"))==Tpaths==row["paths"]
                t_danger,t_alarm = data("first_danger"),data("first_alarm")
                np.testing.assert_array_equal(data("pre_alarm"),t_danger<t_alarm[:,None])
                np.testing.assert_array_equal(t_danger<ticks,data("peak_paired_squared")>=d)
                np.testing.assert_array_equal(data("first_packet"),
                                              (t_danger==data("onset")[:,None])&(t_danger<ticks))
                np.testing.assert_array_equal(data("first_packet_pre_alarm"),data("first_packet")&
                                              (data("onset")<t_alarm)[:,None])
                h = data("health_mean_mse").mean(axis=0)
                cond = data("conditional_health_squared").mean(axis=0)
                for j,name in enumerate(names):
                    r = row["methods"][name]
                    assert data("pre_alarm")[:,j].sum()==r["pre_alarm_count"]
                    assert data("first_packet")[:,j].sum()==r["first_packet_count"]
                    assert data("first_packet_pre_alarm")[:,j].sum()==r["first_strictly_pre_alarm_count"]
                    np.testing.assert_allclose([h[j],100*(h[j]/h[n0]-1)],
                                               [r["health_actual_mse"][0],r["health_actual_pct"]],atol=2e-12)
                    calc = frontier["models"][model]["full_MMSE"]+cond[j]
                    np.testing.assert_allclose(calc,r["health_expectation_MSE_conditional_MC"],atol=2e-13)
                    if case!="healthy":
                        np.testing.assert_allclose(data("active_mean_mse")[:,j].mean(),
                                                   r["active_actual_mse"][0],atol=2e-13)
                    if "diameter_squared_bound" in r:
                        assert data("peak_paired_squared")[:,j].max()<=r["diameter_squared_bound"]+2e-12
                for name in ("trusted-MMSE","anchor-safe"):
                    assert row["methods"][name]["pre_alarm_count"]==0
        checks["models"][model] = {"paths_per_profile":Tpaths,"profiles":len(found["cases"]),
             "methods":len(names),"all_times_from_each_method's_own_output":True,
             "max_paired_recurrence_roundoff":max(x["post_recurrence_roundoff"] for x in found["cases"].values()),
             "frozen_V14_endpoints_max_abs":found["regression_v14_max_abs"]}
    return checks


def expected(row):
    return row.get("health_expected_pct_exact_formula",row["health_expectation_pct_conditional_MC95"][0])


def save_table(comparison):
    columns=("model","profile","method","nominal_MSE_expectation_excess_pct",
             "analytic_nominal_upper_pct_if_available","sample_healthy_excess_pct",
             "sample_active_MSE","sample_tail_MSE","pre_alarm_risk","first_packet_count",
             "first_pre_alarm_count","pathwise_squared_diameter_if_available")
    with (ROOT/"operating_points.csv").open("w",encoding="utf-8",newline="") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        for model,entries in comparison["models"].items():
            for name,r in entries["cases"].items():
                for method,s in r["methods"].items():
                    writer.writerow((model,name,method,expected(s),s.get("analytic_nominal_health_penalty_upper_pct"),
                                     s["health_actual_pct"],s["active_actual_mse"][0] if s["active_actual_mse"] else None,
                                     s["tail_actual_mse"][0] if s["tail_actual_mse"] else None,
                                     s["pre_alarm_risk"],s["first_packet_count"],s["first_strictly_pre_alarm_count"],
                                     s.get("diameter_squared_bound")))


def simultaneous_risk(comparison):
    size = len(MODELS)*len(PROFILES)*len(BUDGETS)
    rows = []
    for model in MODELS:
        item = comparison["models"][model]
        names = [m["name"] for m in item["method_table"]]
        with np.load(ROOT/(model+"_paired.npz"),allow_pickle=False) as saved:
            for case in PROFILES:
                risk = saved[case+"__pre_alarm"]
                methods = item["cases"][case]["methods"]
                for budget in BUDGETS:
                    a,c = label("anchor",budget),label("innovation-clip",budget)
                    pair = paired_risk(risk[:,names.index(a)],risk[:,names.index(c)],tail=.05/(2*size))
                    rows.append({"model":model,"profile":case,"target_pct":100*budget,
                                 "risk_difference_pp":100*pair["difference"],
                                 "familywise95_pp":[100*x for x in pair["exact_interval"]],
                                 "paired_only_anchor":pair["only_new"],
                                 "paired_only_clip":pair["only_reference"],
                                 "active_mse_difference_approx95":methods[a]["active_mse_difference_clip_target"]})
    return {"comparisons":size,"posthoc_multiplicity_over_the_entire_registered_grid":True,
            "max_upper_interval_pp":max(r["familywise95_pp"][1] for r in rows),
            "results":rows}


def plot_boundary(frontier):
    fig,ax=plt.subplots(figsize=(8,4.6),layout="constrained")
    colors=("#117270","#6c6293")
    r=frontier["diameter_radius_at_equality"]
    ax.axvspan(0.,r,color="#ecf5f3",label="Radius with diameter guarantee")
    ax.axvline(r,color="#467f77",ls=":",lw=1.25)
    ax.axhline(2.,color="#ac6544",ls="--",lw=1.2,label="2% healthy MSE allowance")
    for model,color in zip(MODELS,colors,strict=True):
        v=frontier["models"][model]
        ax.plot([x["radius"] for x in v["radius_profile"]],
                [x["pct_vs_V0"] for x in v["radius_profile"]],lw=2.1,label=model.capitalize(),c=color)
        ax.scatter([r,v["minimum_radii"][-1]["radius"]],[v["ball_min_at_boundary_pct_vs_V0"],2.],
                   color=color,s=22,zorder=4)
    ax.set_xlim(0,.40)
    ax.set_ylim(-1,30)
    ax.set_xlabel("Radius about the trusted conditional state estimate")
    ax.set_ylabel("Minimum expected healthy MSE increase (%)")
    ax.grid(alpha=.16)
    ax.legend(loc="upper right",frameon=False)
    fig.savefig(ROOT/"nominal_frontier.png",dpi=175,facecolor="white")
    plt.close(fig)


def plot_comparison(comparison, metric, filename):
    fig,axes=plt.subplots(2,4,figsize=(14.5,7.1),sharex=True)
    colors={"anchor":"#116f66","innovation-clip":"#5f68a2","trusted-ledger":"#b56b35"}
    fixed={"V1a":("s","#54565b"),"V1.2a":("D","#6e7782"),"V1.4":("^","#0d3037"),
           "clip95":("v","#9a5b92")}
    titles=("A0 onset unknown","A0 timing mismatch","Archived A1-H","Archived A1-C")
    def y(row):
        return 100*row["pre_alarm_risk"] if metric=="risk" else row["active_actual_mse"][0]
    for model,panels in zip(MODELS,axes,strict=True):
        for case,ax,title in zip(PROFILES,panels,titles,strict=True):
            c = comparison["models"][model]["cases"][case]["methods"]
            for group,color in colors.items():
                curve=[c[label(group,e)] for e in BUDGETS]
                ax.plot([expected(x) for x in curve],[y(x) for x in curve],
                        "o-",color=color,ms=4,lw=1.6)
            for name,(mark,color) in fixed.items():
                ax.scatter(expected(c[name]),y(c[name]),marker=mark,color=color,s=24)
            ax.axvline(2.,color="#c3c8cb",lw=1,ls="--")
            ax.grid(alpha=.18)
            ax.set_title(model.capitalize()+" / "+title,fontsize=9)
            if metric=="risk":
                ax.set_ylim(0,85)
    handles=[Line2D([0],[0],marker="o",color=cl,lw=1.6,label=group) for group,cl in colors.items()]
    handles.extend(Line2D([0],[0],marker=marker,color=cl,ls="",label=name) for name,(marker,cl) in fixed.items())
    fig.supxlabel("Nominal expected healthy MSE increase (%)   |   analytic for anchor, estimated for other curves",y=.014)
    fig.supylabel("Observed pre-alarm danger (%)" if metric=="risk" else "Observed active-episode MSE",x=.013)
    fig.legend(handles=handles,loc="upper center",ncol=7,frameon=False)
    fig.subplots_adjust(left=.066,right=.987,top=.89,bottom=.13,wspace=.25,hspace=.29)
    fig.savefig(ROOT/filename,dpi=175,facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":9})
    frontier,comparison = (load_json(ROOT/file) for file in ("frontier.json","comparison.json"))
    checks=check_runs(frontier,comparison)
    save_table(comparison)
    write_json(ROOT/"paired_risk_intervals.json",simultaneous_risk(comparison))
    plot_boundary(frontier)
    plot_comparison(comparison,"risk","replay_risk_by_cost.png")
    plot_comparison(comparison,"mse","replay_mse_by_cost.png")
    checks["results_sha256"]={path.name:digest(path) for path in sorted(ROOT.iterdir())
                             if path.is_file() and path.name!="delivery_checks.json"}
    write_json(ROOT/"delivery_checks.json",checks)
    print("All frozen outputs, comparison counts, healthy losses, and archive hashes verified.")


if __name__=="__main__":
    main()
