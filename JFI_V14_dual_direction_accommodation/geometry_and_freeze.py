"""Retrospective action geometry and reproduction of the frozen V1.3 findings."""
import argparse
from pathlib import Path

import numpy as np

from v14_inputs import (DEFAULT_BASELINE, HERE, MODELS, PRIMARY, V12, V13, check_hashes, digest,
                        legacy_cases, load_json, load_prepared, setup, write_json)
from replay import first, paired_risk, run_batches
from trusted_prediction import TrustedConditioning


def segment(target, a, b):
    d = b-a
    square = np.sum(d*d,axis=-1)
    t = np.divide(np.sum((target-a)*d,axis=-1),square,out=np.zeros_like(square),
                  where=square>1e-26).clip(0.,1.)
    residual = target-a-t[:,None]*d
    return np.sum(residual*residual,axis=-1), t


def project_triangle(target, structural, trusted):
    zero = np.zeros_like(target)
    s, u = segment(target,zero,structural)
    t, v = segment(target,zero,trusted)
    edge, e = segment(target,structural,trusted)
    columns = np.stack((structural,trusted),axis=-1)
    singular = np.linalg.svd(columns,compute_uv=False)
    rank = np.sum(singular>np.maximum(1e-12,1e-10*singular[:,:1]),axis=-1)
    full = rank==2
    bary = np.zeros((len(target),2))
    bary[full] = np.linalg.solve(columns[full],target[full,:,None])[...,0]
    inside = full & np.all(bary>=-1e-13,axis=-1) & (bary.sum(axis=-1)<=1+1e-13)
    w = np.stack((np.stack((u,np.zeros_like(u)),axis=-1),
                  np.stack((np.zeros_like(v),v),axis=-1),np.stack((1-e,e),axis=-1),bary),axis=1)
    costs = np.column_stack((s,t,edge,np.where(inside,0.,np.inf)))
    pick = np.argmin(costs,axis=1)
    ix = np.arange(len(target))
    win = w[ix,pick]
    np.testing.assert_allclose(np.sum((target-(columns @ win[...,None])[...,0])**2,axis=-1),
                               costs[ix,pick],atol=5e-11,rtol=5e-11)
    np.testing.assert_array_less(win.sum(axis=1),1+2e-12)
    return costs[ix,pick],win,s,t,rank,singular


def risk_freeze():
    previous = load_json(V13/"results/delivery_checks.json")
    check_hashes(V13,previous["verified_result_hashes"])
    summary = load_json(V13/"results/validation/summary.json")
    p3 = {}
    for model in MODELS:
        group = {}
        with np.load(V13/f"results/validation/{model}_paths.npz",allow_pickle=False) as kept:
            for case in ("unknown_start","timing_mismatch","A1_H_replay","A1_C_replay"):
                key = lambda field: kept[f"{case}__{field}"]
                base, new = key("risk")[:,0],key("risk")[:,-1]
                result = paired_risk(new,base)
                expected = summary["models"][model]["profiles"][case]["methods"]["V1.3"]["paired_risk"]
                np.testing.assert_allclose(result["exact_interval"],expected["exact_interval"],atol=1e-14)
                assert result["exact_interval"][0]<=0<=result["exact_interval"][1]
                group[case] = {"counts":[int(base.sum()),int(new.sum())],"paired95":result["exact_interval"]}
            onset = kept["A1_C_replay__start"]
            ix = np.arange(len(onset))
            score = kept["A1_C_replay__risk_score"][ix,-1,onset]
            eligible = onset<kept["A1_C_replay__first_alarm"]
            expected = summary["models"][model]["profiles"]["A1_C_replay"]["calibration"]["V1.3"]
            actual = kept["A1_C_replay__first_packet"][:,-1][eligible].mean()
            np.testing.assert_allclose([score[eligible].mean(),actual],
                                       [expected["first_packet_predicted"],expected["first_packet_observed"]],
                                       atol=2e-8)
            group["A1_C_first_packet"] = {"score":float(score[eligible].mean()),
                                         "observed":float(actual),"eligible":int(eligible.sum())}
        p3[model]=group
    return {"full_result_hashes_match":len(previous["verified_result_hashes"]),
            "V1a_frozen_artifact_hashes":{p:digest(V13/p) for p in previous["verified_result_hashes"]
                                         if p.endswith("paths.npz")},
            "paired_recalculation":p3}


def geometry(p, trust, spec, stream, clean, attacked):
    s, m = p.settings,p.m
    g = p.model.G[s.receiver]
    D = spec["danger_squared"]
    b = attacked["post"][:,0]-clean["post"][:,0]
    s_direction = attacked["delta"] @ g.T
    earliest = first(np.sum(b*b,axis=-1)>=D)
    alarm = first(attacked["log_bf"]>=-np.log(s.event_false_alarm))
    paths = np.flatnonzero(earliest<alarm)
    ts = earliest[paths]
    v = np.empty((len(paths),m))
    for t in np.unique(ts):
        select = ts==t
        obs = paths[select]
        forecast = trust.at(stream.trusted[obs],t)
        v[select] = (stream.target[obs,t]-forecast.mean_y) @ g.T
    dS, dT = s_direction[paths,ts],v
    base = b[paths,ts]+dS
    opt, weight, pure_s, pure_t, rank, singular = project_triangle(base,dS,dT)
    trust_full = np.sum((base-dT)**2,axis=-1)
    norms = np.linalg.norm(dS,axis=-1)*np.linalg.norm(dT,axis=-1)
    dot = np.sum(dS*dT,axis=-1)
    angle = np.degrees(np.arccos(np.divide(dot,norms,out=np.zeros_like(norms),
                                      where=norms>1e-26).clip(-1,1)))
    rows = []
    for i,j in enumerate(paths):
        rows.append({"path":int(j),"k":int(ts[i]),"first_alarm":int(alarm[j]),"rank":int(rank[i]),
                     "angle_degrees":float(angle[i]) if norms[i]>1e-26 else None,
                     "smallest_singular":float(singular[i,-1]),
                     "V1a_b2":float(b[j,ts[i]]@b[j,ts[i]]),
                     "best_structural_b2":float(pure_s[i]),"trusted_replacement_b2":float(trust_full[i]),
                     "best_trusted_segment_b2":float(pure_t[i]),"best_triangle_b2":float(opt[i]),
                     "unconstrained_state_b2":0.0,"offline_wS_wT":weight[i].tolist()})
    return {"hits":int(len(paths)),"rank_two":int(np.sum(rank==2)),
            "structural_avoided":int(np.sum(pure_s<D)),
            "trusted_replacement_avoided":int(np.sum(trust_full<D)),
            "trusted_segment_avoided":int(np.sum(pure_t<D)),
            "triangle_avoided":int(np.sum(opt<D)),
            "triangle_only_vs_both_segments":int(np.sum((opt<D)&(pure_s>=D)&(pure_t>=D))),
            "median_angle":float(np.median(angle[norms>1e-26])),"rows":rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    args = parser.parse_args()
    common,settings,spec,_ = setup(args.baseline)
    out = HERE/"results"
    out.mkdir(exist_ok=True)
    freeze = risk_freeze()
    hashes = load_json(V12/"results/delivery_checks.json")["validation_traces"]
    check_hashes(V12,{name:row["sha256"] for name,row in hashes.items()})
    models, mse, numerics = {}, {}, []
    for model in MODELS:
        p = load_prepared(args.baseline,model,common,settings)
        trust = TrustedConditioning(p)
        original = legacy_cases(args.baseline,p,spec,common)[1]
        clean = run_batches(p,original["healthy"],("V1a",))
        model_out = {}
        with np.load(V12/f"results/validation/{model}_paths.npz",allow_pickle=False) as before:
            np.testing.assert_allclose(clean["post"][:,0],before["healthy__post"][:,1],atol=2e-10)
            for scenario in PRIMARY:
                attack = original[scenario]
                saved = run_batches(p,attack,("V1a",))
                ref = before[scenario+"__post"][:,1]
                np.testing.assert_allclose(saved["post"][:,0],ref,atol=2e-10)
                numerics.append(float(np.max(np.abs(saved["post"][:,0]-ref))))
                model_out[scenario] = geometry(p,trust,spec,attack,clean,saved)
        models[model] = model_out
        mse[model] = {"V0_mean":float(p.nominal_mse().mean()),
                      "trusted_MMSE_floor":float(np.trace(trust.cov_x,axis1=1,axis2=2).mean()),
                      "target_current_cov":trust.cov_y.tolist()}
    total = sum(a["hits"] for group in models.values() for a in group.values())
    unable = sum(a["hits"]-a["structural_avoided"] for group in models.values() for a in group.values())
    assert total==unable==363
    freeze["V1a_replay_max_abs"] = max(numerics)
    freeze["V12_trace_files_read_only"] = len(hashes)
    freeze["old_pre_alarm_hits_unreachable_with_scalar"] = unable
    write_json(out/"V13_freeze_recalculation.json",freeze)
    write_json(out/"geometry.json",{"scope":"V1a histories and same current healthy V1a estimate held fixed",
                                   "models":models,"analytic_nominal":mse})
    print("Frozen V1.3 risk and 363 scalar limits reproduced", flush=True)
    for k,v in models.items():
        for case,row in v.items():
            print(k,case,{key:value for key,value in row.items() if key!="rows"},flush=True)


if __name__=="__main__":
    main()
