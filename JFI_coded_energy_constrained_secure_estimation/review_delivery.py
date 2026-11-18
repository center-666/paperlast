"""Build the influence and health comparison figure and verify immutable inputs."""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from energy_inputs import BASE_COMMIT, HERE, MODELS, PREVIOUS, digest, load_json, write_json


def make_figure(affine, influence, previous, out):
    fig,(a,b) = plt.subplots(1,2,figsize=(11.6,4.6),layout="constrained")
    rows = influence["models"]["unstable"]
    t = [r["time"] for r in rows]
    for key,color,title in (("common_G","#136c62","Common G"),("full_nominal_conditional","#6170a3","Full nominal posterior")):
        vals = [r[key] for r in rows]
        a.fill_between(t,[r["expected_operator_sq_analytic_lower"] for r in vals],
                       [r["expected_operator_sq_analytic_upper"] for r in vals],alpha=.1,color=color)
        a.plot(t,[r["expected_operator_sq_watermark_qmc"][-1] for r in vals],c=color,lw=2,label=title)
        a.plot(t,[r["best_deterministic_open_loop_expected_sq_gain"] for r in vals],c=color,ls="--",lw=1.1)
    a.plot([],[],color="#58636a",ls="--",label="Fixed open-loop expected gain")
    a.set(title="A. Expected gain vs. fixed-waveform gain",
          xlabel="Time (unstable model)",ylabel="Squared gain from coded perturbations")
    a.set_ylim(0.,.27)
    a.legend(frameon=False,loc="lower right")
    a.grid(alpha=.15)
    x = np.arange(len(MODELS))
    ball = [previous["models"][name]["ball_min_at_boundary_pct_vs_V0"] for name in MODELS]
    affine_cost = [affine["models"][name]["no_danger_percent_lower_down"] for name in MODELS]
    b.bar(x-.17,ball,width=.29,color="#765e89",label="Minimum in the trusted ball class")
    b.bar(x+.17,affine_cost,width=.29,color="#378373",label="Lower bound in current-affine class")
    b.axhline(2,color="#aa643f",lw=1.2,ls="--",label="2% healthy MSE allowance")
    b.set(title="B. Different restrictions on zero danger",ylabel="Healthy MSE increase vs. V0 (%)",
          yticks=[0,2,4,6,8],ylim=(0,8.5),xticks=x,xticklabels=[n.capitalize() for n in MODELS])
    b.legend(frameon=False,loc="upper right",fontsize=8)
    b.grid(axis="y",alpha=.15)
    fig.savefig(out,dpi=175,facecolor="white")
    plt.close(fig)


def main():
    root = HERE/"results"
    previous = load_json(PREVIOUS/"results/frontier.json")
    frozen = load_json(PREVIOUS/"results/delivery_checks.json")
    for name,sha in frozen["results_sha256"].items():
        assert digest(PREVIOUS/"results"/name)==sha,name
    affine,influence,archive = (load_json(root/name) for name in
                                ("affine_frontier.json","linear_influence.json","archived_comparison.json"))
    assert {x["baseline"] for x in (affine,influence,archive)}=={BASE_COMMIT}
    assert affine["model_sha256"]==influence["model_sha256"]==previous["model_sha256"]
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":9})
    make_figure(affine,influence,previous,root/"gain_and_feasibility.png")
    source = {path.relative_to(HERE).as_posix():digest(path) for path in sorted(HERE.glob("*.py"))}
    records = {p.name:digest(p) for p in root.iterdir() if p.name!="delivery_checks.json"}
    write_json(root/"delivery_checks.json",{"base":BASE_COMMIT,"prior_result_hashes_verified":len(
                frozen["results_sha256"]),"new_physical_seeds":0,"source_sha256":source,"results_sha256":records})
    for file in HERE.glob("*.md"):
        for href in re.findall(r"\]\(([^)]+)\)",file.read_text(encoding="utf-8")):
            if not href.startswith(("https:","http:")):
                assert (HERE/href).exists(),(file.name,href)
    print("Independent results, document links, and previous result files verified.")


if __name__=="__main__":
    main()
