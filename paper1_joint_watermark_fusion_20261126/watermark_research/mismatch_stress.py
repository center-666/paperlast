"""Empirical frozen-receiver model-mismatch stress replay on existing W5 paths."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent.parent
RESEARCH=HERE/"watermark_research"
WATERMARK=Path(r"C:\Users\zhongxin\shixidoc\paperlast-watermark-w1\paper1_watermark_early_detection_20261125")
COMPARATOR=WATERMARK/"comparators"
sys.path[:0]=[str(COMPARATOR),str(WATERMARK),
              str(WATERMARK.parent/"paper1_sscd_commonrandom_20261122"),
              str(WATERMARK.parent/"paper1_50node_tv_gate_theory_20261121"),
              str(WATERMARK.parent/"paper1_common_g50_gain_acceptance_20261121")]
import experiment as exp
import rotation_markov as markov
from matched_glrt import build_sender_geometry, sender_template_maps, template_subspace_score_batch

MODELS=("unstable","stable")
METHODS=("quarter_no_repeat","quarter_iid","negative")
ANGLES=(0.,-45.,45.)
ONSETS=(50,65)
FRACTIONS=(0.,.5,.9,1.,1.1,2.)
SEED=[20261128,9521,78]
CHUNK=32
EARLY_REPORTS=3


def sha(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def atomic(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",
                    encoding="utf-8",newline="\n")


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")


def create_lock():
    margin=json.loads((RESEARCH/"MISMATCH_SENDER_SUMMARY.json").read_text(encoding="utf-8"))
    rng=np.random.default_rng(np.random.SeedSequence(SEED))
    directions={}
    for model in MODELS:
        directions[model]={}
        for row in margin["sender_results"]:
            if row["model"]!=model:continue
            phi=float(rng.uniform(0,2*np.pi))
            psi=float(rng.uniform(0,2*np.pi))
            directions[model][str(row["sender"])]={
                "u":[float(np.cos(phi)),float(np.sin(phi))],
                "w":[float(np.cos(psi)),float(np.sin(psi))],
                "rho_relative_D_bound":float(row["rho_relative_D_operator_bound"]),
                "error_over_v_bound":float(row["uniform_error_over_v_limit"]),
            }
    w5=json.loads((markov.HERE/"markov_confirm/RUN.json").read_text(encoding="utf-8"))
    record={"status":"LOCKED_mismatch_stress_directions_before_path_replay",
            "protocol_sha256":sha(RESEARCH/"MISMATCH_REPLAY_PROTOCOL.md"),
            "margin_summary_sha256":sha(RESEARCH/"MISMATCH_SENDER_SUMMARY.json"),
            "w5_RUN_sha256":sha(markov.HERE/"markov_confirm/RUN.json"),
            "split":16,"namespace":w5["identity"]["namespace"],
            "scenarios":w5["identity"]["scenarios"],"paths_per_scenario":512,
            "watermarks":list(METHODS),"fractions_of_positive_margin":list(FRACTIONS),
            "error_definition":"E_j=f*rho_j*||D_j||2*u_j*w_j.T, where rho_j=m_j/(b_j||D_j||2), and ||u_j||=||w_j||=1. Actual recurrence is v[k+1]=(D_j+E_j)v[k].",
            "rank_one_directions":directions,
            "receiver_model":"Correct frozen H,A,Q,R, parity space, residual covariance, whitening, templates and W5 thresholds. Only attack-generator D is replaced by D+E.",
            "seed_namespace":SEED,
            "source_sha256":{"script":sha(Path(__file__)),
                             "W5_REPLAY_RUN":sha(markov.HERE/"markov_confirm/RUN.json")},
            "interpretation":"Post-hoc stress testing of a fixed rank-one mismatch direction per sender on already exposed W5 paths; not a worst-case result and not independent confirmation."}
    write_json(RESEARCH/"MISMATCH_REPLAY_LOCK.json",record)
    print(json.dumps({"status":record["status"],"rho_by_model":{
        m:[v["rho_relative_D_bound"] for v in directions[m].values()] for m in MODELS}},indent=2))


def make_mismatched_delta(raw_attack,onset,model,locked_model,fraction,keys):
    count=len(keys)
    delta=np.zeros((count,150,6,2),dtype=float)
    stop=onset+60
    for edge,(_,sender) in enumerate(model.links):
        key=json.loads(json.dumps(locked_model[str(int(sender))]))
        h=model.H[int(sender)]
        D=h@model.A@np.linalg.inv(h)
        u=np.asarray(key["u"],dtype=float);w=np.asarray(key["w"],dtype=float)
        E=(fraction*key["rho_relative_D_bound"]*np.linalg.norm(D,2))*np.outer(u,w)
        # The frozen attack is a registered 60-report waveform raw_attack[50:110];
        # onset only moves that same waveform in time.
        vector=np.asarray(raw_attack[50,edge],dtype=float)
        signal=np.empty((60,2),dtype=float)
        signal[0]=vector
        for tick in range(59):
            signal[tick+1]=(D+E)@signal[tick]
        delta[:,onset:stop,edge]=np.einsum("btij,tj->bti",keys[:,onset:stop,edge],signal)
    return delta


def white_signal_projection(model,geometries,keys,delta,onset):
    n=len(delta)
    full=np.zeros((n,EARLY_REPORTS,6))
    projected=np.zeros_like(full)
    fraction=np.zeros_like(full)
    for step in range(EARLY_REPORTS):
        tick=onset+step
        window=delta[:,tick-7:tick+1]
        mats=keys[:,tick-7:tick+1]
        for edge,(_,sender) in enumerate(model.links):
            geom=geometries[int(sender)]
            record=window[:,:,edge].reshape(n,16)
            white=record@geom.whitening_map.T
            design=sender_template_maps(model,int(sender),mats[:,:,edge],geom.attack_state_maps)
            white_design=np.einsum("ri,bsic->bsrc",geom.whitening_map,design)
            scores=template_subspace_score_batch(white,white_design)
            energy=np.einsum("bi,bi->b",white,white)
            proj=scores.max(axis=1)
            full[:,step,edge]=energy
            projected[:,step,edge]=proj
            fraction[:,step,edge]=np.divide(proj,energy,out=np.zeros_like(proj),where=energy>1e-30)
    return full,projected,fraction


def evaluate_scenario(mi,scene,fraction,smoke=False):
    label=MODELS[mi]
    model=exp.old.backend(label)[1]
    common_g=exp.old.backend(label)[3]
    ops=exp.old.backend(label)[-1]
    geometries={int(sender):build_sender_geometry(model,int(sender))
                for sender in np.unique(model.links[:,1])}
    run=json.loads((markov.HERE/"markov_confirm/RUN.json").read_text(encoding="utf-8"))
    thresholds={method:float(run["thresholds"][label][family]["threshold"])
                for method,family in (("quarter_no_repeat","quarter_no_repeat_exact"),
                                      ("quarter_iid","quarter_iid_exact"),
                                      ("negative","negative_exact"))}
    lock=json.loads((RESEARCH/"MISMATCH_REPLAY_LOCK.json").read_text(encoding="utf-8"))
    locked_model=lock["directions"][label]
    angle=float(ANGLES[scene//2]);onset=int(ONSETS[scene%2])
    source=exp.old.backend(label)[0]
    base_attack=source.rotated_raw_attack(model,angle)[0]
    count=32 if smoke else 512
    path_ids=[];path_rows=[]
    index=0
    for batch,start in enumerate(range(0,count,CHUNK)):
        n=min(CHUNK,count-start)
        _,_,physical,_,uniforms,negative=markov.streams(n,16,mi,scene,batch,2)
        quarter_keys={
            "quarter_no_repeat":markov.quarter_keys(uniforms,True)[0][:,:,model.links[:,1]],
            "quarter_iid":markov.quarter_keys(uniforms,False)[0][:,:,model.links[:,1]],
            "negative":negative}
        raw=physical.y[:,:,model.links[:,1]]
        per_path={}
        for method in METHODS:
            keys=quarter_keys[method]
            delta=make_mismatched_delta(base_attack,onset,model,locked_model,fraction,keys)
            score=exp.receiver.matched(raw+delta,keys,ops)
            gates=score>thresholds[method]
            early=gates[:,onset:onset+3].any(axis=1).astype(np.uint8)
            world=common_g(model,physical,delta,sender_packets=physical.y,gate_override=gates)
            error=np.sum((world.post-physical.x[:,:,None,:])**2,axis=-1)
            attack=error[:,onset:onset+60,model.receiver_ids].mean(axis=(1,2))
            recovery=error[:,onset+60:,model.receiver_ids].mean(axis=(1,2))
            late=np.zeros(n,dtype=np.int16)
            for p in range(n):
                for edge in range(6):
                    alarms=gates[p,onset:onset+60,edge]
                    hits=np.flatnonzero(alarms)
                    if hits.size:
                        first=onset+int(hits[0])+1
                        if first<onset+60:
                            false=np.flatnonzero(~gates[p,first:onset+60,edge])+first
                            late[p]+=int(np.count_nonzero(
                                np.linalg.norm(delta[p,false,edge],axis=1)>=100))
            full,proj,projfrac=white_signal_projection(model,geometries,keys,delta,onset)
            if fraction==0.:
                baseline_name=f"{label}_attack_{scene}.npz"
                with np.load(markov.HERE/"markov_confirm"/baseline_name) as saved:
                    tag={"quarter_no_repeat":"quarter_no_repeat_exact",
                         "quarter_iid":"quarter_iid_exact","negative":"negative_exact"}[method]
                    np.testing.assert_array_equal(early.mean(axis=1),
                        saved[tag+"_early_link"][start:start+n].mean(axis=1))
                    np.testing.assert_allclose(attack,saved[tag+"_attack_mse"][start:start+n],
                                               atol=2e-10,rtol=0)
                    np.testing.assert_allclose(recovery,saved[tag+"_recovery_mse"][start:start+n],
                                               atol=2e-10,rtol=0)
            per_path[method]={
                "early":early.mean(axis=1),"attack":attack,"recovery":recovery,
                "late_large_accept":late,"full_white_first3":full.mean(axis=(1,2)),
                "projected_first3":proj.mean(axis=(1,2)),
                "projection_fraction_first3":projfrac.mean(axis=(1,2)),
                "path_id":np.arange(start,start+n),
                "score_max_first3":score[:,onset:onset+3].max(axis=(1,2)),
            }
            for p in range(n):
                path_rows.append({"model":label,"scene":scene,"angle":angle,"onset":onset,
                                  "fraction":fraction,"watermark":method,"path_id":start+p,
                                  "early_link_rate":float(early[p].mean()),
                                  "attack_mse":float(attack[p]),"recovery_mse":float(recovery[p]),
                                  "late_large_accept":int(late[p]),
                                  "white_energy_first3":float(full[p].mean()),
                                  "projection_energy_first3":float(proj[p].mean()),
                                  "projection_fraction_first3":float(projfrac[p].mean()),
                                  "score_max_first3":float(score[p,onset:onset+3].max())})
        index+=n
        print("MISMATCH",label,scene,fraction,"batch",batch+1,"of",count//CHUNK,flush=True)
    # Return per-path rows for later whole-path bootstrap; analysis keeps scenario strata.
    return path_rows


def run(smoke=False):
    lock_path=RESEARCH/"MISMATCH_REPLAY_LOCK.json"
    lock=json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["source_sha256"]["script"]==sha(Path(__file__))
    if smoke:
        cells=[(mi,0,f) for mi in range(2) for f in (0.,.5,1.1)]
    else:
        cells=[(mi,scene,f) for mi in range(2) for scene in range(6) for f in FRACTIONS]
    records=[]
    for mi,scene,fraction in cells:
        records.extend(evaluate_scenario(mi,scene,fraction,smoke))
    target=RESEARCH/("MISMATCH_SMOKE_PATHS.csv" if smoke else "MISMATCH_PATHS.csv")
    with target.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"status":"PASS_smoke" if smoke else "complete_mismatch_path_generation",
                      "records":len(records),"file_sha256":sha(target),
                      "lock_sha256":sha(lock_path)},indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("lock","smoke","run"),default="lock")
    args=parser.parse_args()
    if args.mode=="lock":
        if (RESEARCH/"MISMATCH_PATHS.csv").exists():
            raise FileExistsError("Cannot change the mismatch lock after path replay.")
        if (RESEARCH/"MISMATCH_REPLAY_LOCK.json").exists():
            raise FileExistsError("Mismatch lock already exists; do not overwrite it.")
        # Lock directions are generated once from the registered seed. This is
        # run as a separate source-level action before any mismatch output exists.
        margin=json.loads((RESEARCH/"MISMATCH_SENDER_SUMMARY.json").read_text())
        rng=np.random.default_rng(np.random.SeedSequence(SEED))
        directions={}
        for model in MODELS:
            directions[model]={}
            for row in margin["sender_results"]:
                if row["model"]!=model:continue
                phi=float(rng.uniform(0,2*np.pi));psi=float(rng.uniform(0,2*np.pi))
                directions[model][str(row["sender"])]={
                    "u":[float(np.cos(phi)),float(np.sin(phi))],
                    "w":[float(np.cos(psi)),float(np.sin(psi))],
                    "rho_relative_D_bound":float(row["rho_relative_D_operator_bound"]),
                    "sender_margin":float(row["m_min_sigma_WN"]),
                    "b_bound":float(row["b_max_norm_WRb"]),
                    "D_norm":float(row["D_spectral_norm"]),
                }
        lock={"status":"LOCKED_before_mismatch_paths",
              "protocol_sha256":sha(RESEARCH/"MISMATCH_REPLAY_PROTOCOL.md"),
              "margin_table_sha256":sha(RESEARCH/"MISMATCH_SENDER_MARGIN.csv"),
              "w5_RUN_sha256":sha(markov.HERE/"markov_confirm/RUN.json"),
              "namespace":[20261127,52387,16],
              "fractions":list(FRACTIONS),"seed":SEED,
              "directions":directions,
              "definition":"E_j=f*rho_j*||D_j||_2*u_j*w_j.T; then v[t+1]=(D_j+E_j)v[t].",
              "watermarks":list(METHODS),
              "receiver":"Frozen W5 exact matched GLRT thresholds, physical covariance, parity map and whitening.",
              "stage":"Post-hoc development replay on previously evaluated paths."}
        lock["source_sha256"]={"script":sha(Path(__file__)),
                               "margins":sha(RESEARCH/"MISMATCH_SENDER_MARGIN.csv")}
        atomic(RESEARCH/"MISMATCH_REPLAY_LOCK.json",lock)
        print("LOCKED",sha(RESEARCH/"MISMATCH_REPLAY_LOCK.json"))
    elif args.mode=="smoke":run(smoke=True)
    else:run(False)


if __name__=="__main__":main()
