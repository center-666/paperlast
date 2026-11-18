"""Recompute saved event metrics and paired intervals from episode arrays."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v14_inputs import (DEFAULT_BASELINE, HERE, MODELS, digest, load_json, load_prepared,
                        setup, write_json)
from dual_receiver import DESIGN, METHODS
from replay import first, paired_risk
from run_screen import development_gate, inputs_manifest


BASES = ("V1a","V1.2a","trusted-budget")
CANDIDATE = "V1.4"


def recompute(path, groups, prepared, D, tail):
    T = prepared.settings.horizon
    idx = {name:METHODS.index(name) for name in METHODS}
    j0, j = idx["V0"],idx[CANDIDATE]
    a = prepared.nominal_mse()*.02
    maximum = 0.
    profile = {}
    with np.load(path,allow_pickle=False) as data:
        for name,row in groups.items():
            def read(field):
                return data[f"{name}__{field}"]

            w, b2 = read("weights"),read("b2")
            assert np.all((w>=0)&(w<=1)&np.isfinite(w))
            assert np.all(w.sum(axis=-1)<=1+8e-15)
            k_start = read("k_start")
            before = np.arange(T)[None]<k_start[:,None]
            assert np.max(b2*before[:,None])<1e-22
            alarm = first(read("log_bf")>=-np.log(prepared.settings.event_false_alarm))
            crossed = b2>=D
            danger = first(crossed)
            hit = danger<alarm[:,None]
            packet = np.arange(T)[None]==k_start[:,None]
            first_packet = np.any(crossed&packet[:,None],axis=2)
            np.testing.assert_array_equal(alarm,read("first_alarm"))
            np.testing.assert_array_equal(danger,read("first_danger"))
            np.testing.assert_array_equal(hit,read("risk"))
            np.testing.assert_array_equal(first_packet,read("first_packet"))
            np.testing.assert_array_equal(first_packet&(k_start<alarm)[:,None],read("first_packet_pre"))
            np.testing.assert_array_equal(w[:,j,:,1][read("score_post_S")<=chi2.ppf(
                DESIGN["trusted_score_level"],prepared.m)],0.)
            spent = np.cumsum(read("charge"),axis=2)
            error = read("remainder")-(np.cumsum(a)-spent)
            maximum = max(maximum,float(np.max(np.abs(error))))
            np.testing.assert_allclose(error,0.,atol=4e-11)
            assert read("remainder").min()>=0
            assert np.max(read("charge")-read("available"))<3e-12
            healthy,loss = read("health_mse"),read("active_mse")
            for other,m in idx.items():
                metrics = row["methods"][other]
                assert metrics["pre_alarm_danger"]==int(hit[:,m].sum())
                assert metrics["first_packet_danger"]==int(first_packet[:,m].sum())
                ratio = healthy[:,m].mean()/healthy[:,j0].mean()-1
                np.testing.assert_allclose(metrics["health_relative_vs_V0"],ratio,atol=5e-15)
                if name!="healthy":
                    np.testing.assert_allclose(metrics["active_mse"][0],loss[:,m].mean(),atol=1e-14)
            if name=="healthy":
                continue
            entries = {}
            for other in BASES:
                k = idx[other]
                entries[other] = {
                    "counts_base_new": [int(hit[:,k].sum()),int(hit[:,j].sum())],
                    "pointwise95":paired_risk(hit[:,j],hit[:,k]),
                    "familywise95":paired_risk(hit[:,j],hit[:,k],tail),
                    "first_packet95":paired_risk(first_packet[:,j],first_packet[:,k]),
                    "relative_active_mse_difference": float(loss[:,j].mean()/loss[:,k].mean()-1)
                }
            profile[name] = entries
    return profile,maximum


def geometry_recheck(geom):
    groups = geom["models"]
    keys = ("hits","structural_avoided","trusted_replacement_avoided",
            "trusted_segment_avoided","triangle_avoided","triangle_only_vs_both_segments")
    totals = {key:sum(row[key] for model in groups.values() for row in model.values()) for key in keys}
    totals["first_packet_hits"] = 0
    totals["first_packet_triangle_avoided"] = 0
    totals["triangle_only_first_packet"] = 0
    for model in MODELS:
        with np.load(HERE/f"results/development/{model}_96_paths.npz",allow_pickle=False) as arrays:
            for case,row in groups[model].items():
                expected = np.flatnonzero(arrays[f"{case}__risk"][:,METHODS.index("V1a")])
                np.testing.assert_array_equal(expected,[r["path"] for r in row["rows"]])
                starts = arrays[f"{case}__k_start"]
                dangers = arrays[f"{case}__first_danger"][:,METHODS.index("V1a")]
                for r in row["rows"]:
                    path,k = r["path"],r["k"]
                    assert k==dangers[path]
                    assert r["best_triangle_b2"]<=min(r["best_trusted_segment_b2"],
                                                      r["best_structural_b2"])+5e-12
                    if k==starts[path]:
                        totals["first_packet_hits"] += 1
                        if r["best_triangle_b2"]<DESIGN["danger_squared"]:
                            totals["first_packet_triangle_avoided"] += 1
                            if min(r["best_trusted_segment_b2"],r["best_structural_b2"])>=DESIGN["danger_squared"]:
                                totals["triangle_only_first_packet"] += 1
    return totals


def figure(geometry, tests, out):
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "font.family":"DejaVu Sans"})
    palette = {"V1a":"#9da9b4","V1.2a":"#d6ac63","trusted-budget":"#486e9a",CANDIDATE:"#137470"}
    fig,(left,right) = plt.subplots(1,2,figsize=(12,5.1),layout="constrained",
                                    gridspec_kw={"width_ratios":[.84,1.4]})
    names = ["Structural segment","Full trusted replacement","Trusted segment","Both directions","Free correction (oracle)"]
    values = [geometry[key] for key in ("structural_avoided","trusted_replacement_avoided",
                                         "trusted_segment_avoided","triangle_avoided","hits")]
    y = np.arange(5)
    left.barh(y,values,color=["#9da9b4","#9fb8cd","#486e9a","#137470","#d7dde3"])
    for i,x in enumerate(values):
        left.text(x+5,i,str(x),va="center")
    left.set(yticks=y,yticklabels=names,xlim=(0,407),xlabel="Crossings avoidable at the frozen update",
             title="A. Fixed V1a history (363 crossings)")
    left.invert_yaxis()
    left.grid(axis="x",alpha=.2)
    flat = [(model,case,row) for model in MODELS for case,row in tests[model].items()]
    y = np.arange(len(flat))
    labels = {"unknown_start":"A0, onset unknown","timing_mismatch":"A0, timing mismatch",
              "A1_H_replay":"A1-H replay","A1_C_replay":"A1-C replay"}
    method_names = (*BASES,CANDIDATE)
    for offset,method in enumerate(method_names):
        vals = [100*row[method]["counts_base_new"][0]/2048 if method!=CANDIDATE
                else 100*row["V1a"]["counts_base_new"][1]/2048 for _,_,row in flat]
        right.barh(y+(offset-1.5)*.19,vals,height=.17,label=method,color=palette[method])
    right.set(yticks=y,yticklabels=[f"{m.capitalize()} / {labels[c]}" for m,c,_ in flat],
              xlim=(0,100),xlabel="Observed strict pre-alarm risk (%)",
              title="B. Paired holdout (2,048 paths per row)")
    right.invert_yaxis()
    right.grid(axis="x",alpha=.2)
    right.legend(loc="upper center",bbox_to_anchor=(.54,-.16),ncol=4,frameon=False)
    fig.savefig(out,dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main(base):
    out = HERE/"results"
    geometry = geometry_recheck(load_json(out/"geometry.json"))
    code = inputs_manifest([*HERE.glob("*.py"),HERE/"registration.json"])
    dev = load_json(out/"development/summary.json")
    assert dev["gate"]==development_gate(dev["models"])
    assert code==dev["source_hashes"]
    hold = load_json(out/"holdout/summary.json")
    freeze = load_json(out/"holdout/freeze.json")
    assert dev["gate"]["passed"]
    assert digest(out/"development/summary.json")==freeze["development_sha256"]
    assert code==freeze["source_hashes"]==hold["source_hashes"]
    common,settings,spec,_ = setup(base)
    comparisons, dev_cmp = {}, {}
    worst = 0.
    for model in MODELS:
        prepared = load_prepared(base,model,common,settings)
        for n in (32,48,96):
            filename = out/f"development/{model}_{n}_paths.npz"
            got,residual = recompute(filename,dev["models"][model][str(n)],prepared,spec["danger_squared"],.025)
            worst = max(worst,residual)
            if n==96:
                dev_cmp[model] = got
        count = DESIGN["conditional_holdout"]["paths_per_model_and_profile"]
        assert count==2048
        filename = out/f"holdout/{model}_{count}_paths.npz"
        # Eight profiles (two models, four non-nominal groups) by three comparisons.
        got,residual = recompute(filename,hold["models"][model][str(count)],prepared,
                                 spec["danger_squared"],.05/(2*8*len(BASES)))
        comparisons[model] = got
        worst = max(worst,residual)
    write_json(out/"paired_comparisons.json",{"holdout":comparisons,"old96":dev_cmp,
               "geometry_totals":geometry,"maximum_ledger_replay_difference":worst,
               "simultaneous_family":"Eight non-nominal profiles times three comparators, two signs each."})
    figure(geometry,comparisons,out/"geometry_and_holdout.png")
    print("Verified held-out and archived results, weights, alarm timing, allocations and preservation hashes.")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    main(parser.parse_args().baseline)
