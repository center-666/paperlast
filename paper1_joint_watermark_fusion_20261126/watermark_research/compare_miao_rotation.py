"""Post-hoc local Miao-Givens adaptation on the frozen W5 split-16 paths."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent.parent
RESEARCH=HERE/"watermark_research"
COMPARATOR=Path(r"C:\Users\zhongxin\shixidoc\paperlast-watermark-w1\paper1_watermark_early_detection_20261125\comparators")
sys.path.insert(0,str(COMPARATOR))
import experiment as exp
import rotation_markov as markov

MODELS=("unstable","stable")
METHODS=("miao_local","quarter_iid","quarter_no_repeat","negative")
ANGLES=(0.,-45.,45.)
ONSETS=(50,65)
SPLIT=16
CHUNK=32
BOOT=20000
SEED=[20261128,9521,77]
LOCK=RESEARCH/"MIAO_STATIC_CODE_LOCK.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def atomic_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")


def coding_record():
    rng=np.random.default_rng(np.random.SeedSequence([*SEED,0]))
    records={}
    for model_index,label in enumerate(MODELS):
        model=exp.old.backend(label)[1]
        eigenvalues,eigenvectors=np.linalg.eig(model.A)
        unstable=[eigenvectors[:,i].real/np.linalg.norm(eigenvectors[:,i].real)
                  for i,value in enumerate(eigenvalues) if abs(value)>1+1e-10]
        angles=rng.uniform(np.nextafter(0.,1.),np.pi/2,size=50)
        sigmas=np.empty((50,2,2))
        supports={}
        for sender in range(50):
            projected=[model.H[sender]@vector for vector in unstable]
            coordinates=(np.flatnonzero(np.max(np.abs(projected),axis=0)>1e-12)
                         if projected else np.asarray([],dtype=int))
            supports[str(sender)]=coordinates.tolist()
            theta=float(angles[sender])
            sigmas[sender]=[[np.cos(theta),-np.sin(theta)],
                            [np.sin(theta),np.cos(theta)]]
        monitored=[int(sender) for sender in model.links[:,1]]
        records[label]={
            "model_A":model.A.tolist(),
            "eigenvalues":[[float(v.real),float(v.imag)] for v in eigenvalues],
            "unstable_eigenvector_count":len(unstable),
            "monitored_senders":monitored,
            "H_sender_matrices":{str(j):model.H[j].tolist() for j in monitored},
            "unstable_projected_support":{str(j):supports[str(j)] for j in monitored},
            "angle_radians":{str(j):float(angles[j]) for j in monitored},
            "Sigma":{str(j):sigmas[j].tolist() for j in monitored},
            "decode_M":{str(j):sigmas[j].T.tolist() for j in monitored},
            "coding_source":"Algorithm-1 local p=2 Givens adaptation. Static sender-shared Sigma; the coded message is Sigma@y and the receiver decodes with Sigma.T.",
            "original_theorem_applicable":bool(unstable),
            "distributed_adaptation":True,
        }
    return {
        "status":"LOCKED_pre_metrics_miao_local_rotation",
        "protocol_sha256":sha(RESEARCH/"MIAO_ADAPTATION_PROTOCOL.md"),
        "source_pdf_sha256":hashlib.sha256(
            Path(r"C:\Users\zhongxin\AppData\Local\Temp\miao_tcns17.pdf").read_bytes()).hexdigest(),
        "paper":"Miao et al., Coding Schemes for Securing Cyber-Physical Systems Against Stealthy Data Injection Attacks, IEEE TCNS 2017, 4(1):106–117, DOI 10.1109/TCNS.2016.2573039.",
        "angle_seed":[*SEED,0],
        "update_frequency_per_sender":0,
        "per_packet_entropy_bits_after_installation":0,
        "models":records,
    }


def write_lock():
    if LOCK.exists():
        raise FileExistsError(f"Refusing to change pre-existing Miao lock: {LOCK}")
    record=coding_record()
    record["lock_source_sha256"]=sha(Path(__file__))
    atomic_json(LOCK,record)
    smoke={}
    for label in MODELS:
        source,model,physical_fn,common_g,ops=exp.old.backend(label)
        meta=record["models"][label]
        sigma=np.tile(np.eye(2),(50,1,1))
        for sender,value in meta["Sigma"].items():
            sigma[int(sender)]=np.asarray(value)
        rng=np.random.default_rng(np.random.SeedSequence([*SEED,0,99,MODELS.index(label)]))
        y=rng.normal(size=(4,150,50,2))
        coded=np.einsum("sij,btsj->btsi",sigma,y)
        decoded=np.einsum("sij,btsj->btsi",sigma.swapaxes(1,2),coded)
        smoke[label]={
            "roundtrip_max_abs":float(np.max(abs(decoded-y))),
            "energy_ratio":float(np.sum(coded*coded)/np.sum(y*y)),
            "orthogonality_max_error":float(np.max(abs(sigma.swapaxes(1,2)@sigma-np.eye(2)))),
        }
        np.testing.assert_allclose(decoded,y,rtol=0,atol=2e-15)
        np.testing.assert_allclose(smoke[label]["energy_ratio"],1.,rtol=0,atol=2e-15)
        np.testing.assert_allclose(smoke[label]["orthogonality_max_error"],0.,rtol=0,atol=2e-15)
    atomic_json(RESEARCH/"MIAO_INTERFACE_SMOKE.json",smoke)
    print(json.dumps({"status":record["status"],"smoke":smoke},indent=2))


def load_miao_keys(label):
    lock=json.loads(LOCK.read_text(encoding="utf-8"))
    meta=lock["models"][label]
    model=exp.old.backend(label)[1]
    matrices=np.broadcast_to(np.eye(2),(50,2,2)).copy()
    for sender,matrix in meta["decode_M"].items():
        matrices[int(sender)]=np.asarray(matrix,dtype=float)
    senders=model.links[:,1].astype(int)
    return np.ascontiguousarray(matrices[senders]),meta


def get_keys(model,label,method,uniforms,negative,miao_static):
    n=len(uniforms);senders=model.links[:,1].astype(int)
    if method=="miao_local":
        return np.broadcast_to(miao_static[None,None],(n,150,len(senders),2,2))
    if method=="quarter_iid":
        return markov.quarter_keys(uniforms,False)[0][:,:,senders]
    if method=="quarter_no_repeat":
        return markov.quarter_keys(uniforms,True)[0][:,:,senders]
    return negative


def calibrated_threshold(label,mi,miao_static):
    maxima=[]
    # Use the exact frozen W5 geometry object returned by its backend.
    ops=exp.old.backend(label)[-1]
    for batch,start in enumerate(range(0,4096,CHUNK)):
        n=min(CHUNK,4096-start)
        _,model,physical,_,uniforms,negative=markov.streams(n,SPLIT,mi,0,batch,0)
        raw=physical.y[:,:,model.links[:,1]]
        key=get_keys(model,label,"miao_local",uniforms,negative,miao_static)
        score=exp.receiver.matched(raw,key,ops)
        maxima.extend(score[:,5:].max(axis=(1,2)).tolist())
    cutoff=exp.old.path_fwer_cutoff(np.asarray(maxima),.05)
    return float(cutoff["threshold"]),np.asarray(maxima)


def score_arm(model,physical,angle,onset,method,key,threshold,ops,common_g,source):
    n=len(physical.x);senders=model.links[:,1].astype(int)
    raw=physical.y[:,:,senders]
    raw_attack=source.rotated_raw_attack(model,float(angle))[0]
    fixture=raw_attack[50:110]
    delta=np.zeros((n,150,6,2))
    delta[:,onset:onset+60]=np.einsum("bteij,atej->btei",
        key[:,onset:onset+60],fixture[None,...])
    score=exp.receiver.matched(raw+delta,key,ops)
    gates=score>threshold
    early=gates[:,onset:onset+3].any(axis=1).astype(np.uint8)
    world=common_g(model,physical,delta,sender_packets=physical.y,gate_override=gates)
    error=np.sum((world.post-physical.x[:,:,None,:])**2,axis=-1)
    affected=error[:,:,model.receiver_ids].mean(axis=(1,2))
    attack=error[:,onset:onset+60,model.receiver_ids].mean(axis=(1,2))
    recovery=error[:,onset+60:,model.receiver_ids].mean(axis=(1,2))
    network_attack=error[:,onset:onset+60].mean(axis=(1,2))
    network_recovery=error[:,onset+60:].mean(axis=(1,2))
    return {
        "early_link":early,
        "early_fraction":early.mean(axis=1),
        "any_link_early":early.any(axis=1).astype(np.uint8),
        "attack_mse":attack,
        "recovery_mse":recovery,
        "attack_network_mse":network_attack,
        "recovery_network_mse":network_recovery,
        "prior_alarm":gates[:,:onset].any(axis=(1,2)).astype(np.uint8),
        "large_late_accept":large_late_accept(gates,delta,onset),
        "score_max":score[:,5:].max(axis=(1,2)),
        "gates":gates,
        "delta":delta,
        "key":key,
    }


def large_late_accept(gates,delta,onset):
    n=len(gates);counts=np.zeros(n,dtype=np.int16);stop=onset+60
    for edge in range(6):
        alarm=gates[:,onset:stop,edge]
        for path in range(n):
            found=np.flatnonzero(alarm[path])
            if found.size:
                first=onset+int(found[0])+1
                if first<stop:
                    late=np.flatnonzero(~gates[path,first:stop,edge])+first
                    counts[path]+=int(np.count_nonzero(np.linalg.norm(delta[path,late,edge],axis=1)>=100.))
    return counts


def evaluate_model(mi):
    label=MODELS[mi]
    source,model,physical_fn,common_g,base_ops=exp.old.backend(label)
    ops=exp.operators(mi,1.)
    static,meta=load_miao_keys(label)
    threshold,cal_max=calibrated_threshold(label,mi,static)
    run=json.loads((markov.HERE/"markov_confirm/RUN.json").read_text())
    baseline_thresholds={method:float(run["thresholds"][label][key]["threshold"])
        for method,key in (("quarter_iid","quarter_iid_exact"),
                           ("quarter_no_repeat","quarter_no_repeat_exact"),
                           ("negative","negative_exact"))}
    thresholds={"miao_local":threshold,**baseline_thresholds}
    scene_data={}
    for scene in range(6):
        angle=float(ANGLES[scene//2]);onset=int(ONSETS[scene%2])
        values={m:[] for m in METHODS}
        for batch,start in enumerate(range(0,512,CHUNK)):
            n=min(CHUNK,512-start)
            _,_,physical,_,uniforms,negative=markov.streams(n,SPLIT,mi,scene,batch,2)
            for method in METHODS:
                key=get_keys(model,label,method,uniforms,negative,static)
                result=score_arm(model,physical,angle,onset,method,key,thresholds[method],ops,common_g,source)
                values[method].append(result)
            baseline_path=markov.HERE/"markov_confirm"/f"{label}_attack_{scene}.npz"
            with np.load(baseline_path) as a:
                for method,w5tag in (("quarter_iid","quarter_iid_exact"),
                                     ("quarter_no_repeat","quarter_no_repeat_exact"),
                                     ("negative","negative_exact")):
                    rebuilt=values[method][-1]
                    np.testing.assert_array_equal(rebuilt["early_link"],a[w5tag+"_early_link"][start:start+n])
                    np.testing.assert_allclose(rebuilt["attack_mse"],a[w5tag+"_attack_mse"][start:start+n],rtol=0,atol=2e-10)
                    np.testing.assert_allclose(rebuilt["recovery_mse"],a[w5tag+"_recovery_mse"][start:start+n],rtol=0,atol=2e-10)
        scene_data[str(scene)]={m:{
            key:np.concatenate([x[key] for x in blocks],axis=0)
            for key in ("early_link","early_fraction","any_link_early","attack_mse","recovery_mse",
                        "attack_network_mse","recovery_network_mse","prior_alarm","large_late_accept","score_max")
        } for m,blocks in values.items()}
        print(label,"attack",scene,"baseline replay verified",flush=True)
    # Independent registered healthy holdout, reused read-only from split 16.
    health={}
    for batch,start in enumerate(range(0,4096,CHUNK)):
        n=min(CHUNK,4096-start)
        _,_,physical,_,uniforms,negative=markov.streams(n,SPLIT,mi,0,batch,1)
        senders=model.links[:,1].astype(int);raw=physical.y[:,:,senders]
        for method in METHODS:
            key=get_keys(model,label,method,uniforms,negative,static)
            score=exp.receiver.matched(raw,key,ops)
            gate=score>thresholds[method]
            delta=np.zeros((n,150,6,2))
            world=common_g(model,physical,delta,sender_packets=physical.y,gate_override=gate)
            error=np.sum((world.post-physical.x[:,:,None,:])**2,axis=-1)
            item=health.setdefault(method,{"alarms":0,"affected_mse":[],"network_mse":[],"maxima":[]})
            item["alarms"]+=int(gate[:,5:].any(axis=(1,2)).sum())
            item["affected_mse"].extend(error[:,20:,model.receiver_ids].mean(axis=(1,2)).tolist())
            item["network_mse"].extend(error[:,20:].mean(axis=(1,2)).tolist())
            item["maxima"].extend(score[:,5:].max(axis=(1,2)).tolist())
    return {"model":label,"thresholds":thresholds,"calibration_paths":4096,
            "calibration_path_maxima_miao":cal_max,"healthy":health,
            "scenes":scene_data,"static_code":meta}


def summarize(outputs):
    metric_names=("early_fraction","attack_mse","recovery_mse","large_late_accept")
    rng=np.random.default_rng(np.random.SeedSequence([*SEED,77]))
    contrasts=[]
    for result in outputs:
        scenario_arrays=result["scenes"]
        for method in METHODS:
            if method=="quarter_iid":continue
            for reference in ("quarter_no_repeat","negative"):
                for metric in metric_names:
                    diffs=[scenario_arrays[str(scene)][method][metric]
                           -scenario_arrays[str(scene)][reference][metric] for scene in range(6)]
                    point=float(np.mean([d.mean() for d in diffs]))
                    draws=np.zeros(BOOT)
                    for d in diffs:
                        n=len(d)
                        for start in range(0,BOOT,500):
                            size=min(500,BOOT-start)
                            weights=rng.multinomial(n,np.full(n,1/n),size=size)/n
                            draws[start:start+size]+=weights@d/6.
                    contrasts.append({"model":result["model"],"method":method,"reference":reference,
                                      "metric":metric,"difference":point,
                                      "lower95":float(np.quantile(draws,.025)),
                                      "upper95":float(np.quantile(draws,.975)),
                                      "scenarios":6,"paths_per_scenario":512})
    return contrasts


def run():
    RESEARCH.mkdir(exist_ok=True)
    lock=json.loads(LOCK.read_text())
    if lock["lock_source_sha256"]!=sha(Path(__file__)):
        raise ValueError("Miao static-code lock was generated by a different runner. Re-lock before scores.")
    outputs=[evaluate_model(mi) for mi in range(2)]
    contrasts=summarize(outputs)
    output={}
    for result in outputs:
        output[result["model"]]={"thresholds":result["thresholds"],"healthy":{
            m:{"paths":len(v["affected_mse"]),"alarm_count":v["alarms"],
               "FWER":v["alarms"]/len(v["affected_mse"]),
               "affected_mse":float(np.mean(v["affected_mse"])),
               "network_mse":float(np.mean(v["network_mse"]))}
            for m,v in result["healthy"].items()},
            "scenes":{s:{m:{k:{"mean":float(np.asarray(v[k]).mean()),
                              "median":float(np.median(v[k])),
                              "p99":float(np.quantile(v[k],.99)),
                              "max":float(np.max(v[k]))}
                          for k in v} for m,v in scene.items()}
                      for s,scene in result["scenes"].items()},
            "baseline_w5_replayed":True,
            "miao_calibration_maxima_sha_inputs":"registered split-16 namespace population 0"}
    report={"status":"complete_posthoc_w5_path_replay","protocol_sha256":sha(RESEARCH/"MIAO_ADAPTATION_PROTOCOL.md"),
            "static_lock_sha256":sha(LOCK),"w5_RUN_sha256":sha(markov.HERE/"markov_confirm/RUN.json"),
            "w5_archives_verified":{f"{model}_attack_{scene}.npz":sha(markov.HERE/"markov_confirm"/f"{model}_attack_{scene}.npz") for model in MODELS for scene in range(6)},
            "information_boundary":"Existing W5 namespace-16 physical paths and health calibration/holdout are reused. Miao-Local was added after those paths were viewed. All results are post-hoc development, not independent confirmation.",
            "models":output,"contrasts":contrasts,
            "performance_claim":"Common matched GLRT and common-G receiver. The local two-dimensional Givens construction is an adaptation, and stable-model performance does not satisfy the source method's unstable-eigenvector premise."}
    atomic_json(RESEARCH/"MIAO_RESULTS.json",report)
    rows=[]
    for result in outputs:
        for scene,values in result["scenes"].items():
            scenario_index=MODELS.index(result["model"])*6+int(scene)
            angle=float(ANGLES[scenario_index%6//2])
            onset=int(ONSETS[scenario_index%6%2])
            for method,metrics in values.items():
                for path in range(512):
                    rows.append({"model":result["model"],"scene":scene,"angle":angle,
                                 "onset":onset,"method":method,"path_id":path,
                                 **{key:float(metrics[key][path].mean()) if np.asarray(metrics[key][path]).ndim else float(metrics[key][path])
                                    for key in metric_names_for_csv()}})
    with (RESEARCH/"MIAO_PATH_RESULTS.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    with (RESEARCH/"MIAO_CONTRASTS.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(contrasts[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(contrasts)
    print(json.dumps({"status":report["status"],"contrasts":contrasts},indent=2))


def metric_names_for_csv():
    return ("early_fraction","attack_mse","recovery_mse","large_late_accept","prior_alarm","score_max")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("lock","run"),default="lock")
    args=parser.parse_args()
    RESEARCH.mkdir(exist_ok=True)
    if args.mode=="lock":
        if (RESEARCH/"MIAO_RESULTS.json").exists():
            raise FileExistsError("Result file already exists; lock cannot be changed after score evaluation.")
        value=coding_record()
        value["lock_source_sha256"]=sha(Path(__file__))
        atomic_json(LOCK,value)
        interface={}
        for mi,label in enumerate(MODELS):
            model=exp.old.backend(label)[1]
            static=np.tile(np.eye(2),(50,1,1))
            for sender,matrix in value["models"][label]["Sigma"].items():
                static[int(sender)]=np.asarray(matrix)
            rng=np.random.default_rng(np.random.SeedSequence([*SEED,99,mi]))
            x=rng.normal(size=(4,150,50,2))
            coded=np.einsum("sij,btsj->btsi",static,x)
            decoded=np.einsum("sij,btsj->btsi",static.swapaxes(1,2),coded)
            interface[label]={"roundtrip_max_abs":float(np.max(abs(decoded-x))),
                              "norm_ratio":float(np.linalg.norm(coded)/np.linalg.norm(x)),
                              "sigma_orthogonality_max_abs":float(np.max(abs(static.swapaxes(1,2)@static-np.eye(2))))}
            np.testing.assert_allclose(decoded,x,rtol=0,atol=2e-15)
            np.testing.assert_allclose(interface[label]["norm_ratio"],1.,rtol=0,atol=2e-15)
        atomic_json(RESEARCH/"MIAO_INTERFACE_SMOKE.json",interface)
        print(json.dumps({"locked":LOCK.exists(),"interface":interface},indent=2))
    else:run()


if __name__=="__main__":main()
