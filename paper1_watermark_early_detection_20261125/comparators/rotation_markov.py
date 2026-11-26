"""No-repeat four-state rotation screening on the frozen 50-node estimator."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.stats import beta
import experiment as exp

HERE=exp.HERE
ANGLES=(0.,-45.,45.)
ONSETS=(50,65)
SCENES=tuple(range(12))
SCHEMES=("quarter_iid","quarter_no_repeat","negative")
SCORERS=("plugin","exact")
ARMS=tuple(s+"_"+r for s in SCHEMES for r in SCORERS)
ROT=exp.ROTATION
CHUNK=32
BASE="f97748fcba7c45981878352aa03bc1b15f78dc65"
ALPHA=.05


def streams(n,split,mi,scene,batch):
    root=(20261127,52387,split,mi,scene,batch)
    source,model,physical_fn,_,ops=exp.old.backend(exp.MODEL[mi])
    physical=physical_fn(model,n,np.random.SeedSequence([*root,0]))
    uniforms=np.random.default_rng(np.random.SeedSequence([*root,1])).random((n,150,50))
    seeds=[np.random.SeedSequence([*root,2,i]) for i in range(3)]
    draws=exp.old.make_key_draws(n,*seeds)
    negative_all=exp.old.matrices(draws,"negative_fixed")
    senders=model.links[:,1]
    negative=negative_all[:,:,senders]
    return source,model,physical,ops,uniforms,negative


def quarter_keys(uniforms,no_repeat):
    n,ticks,senders=uniforms.shape
    codes=np.empty((n,ticks,senders),dtype=np.uint8)
    codes[:,0]=np.floor(4*uniforms[:,0]).astype(np.uint8)
    if no_repeat:
        for t in range(1,ticks):
            move=np.floor(3*uniforms[:,t]).astype(np.uint8)
            codes[:,t]=(codes[:,t-1]+1+move)%4
    else:
        codes[:,1:]=np.floor(4*uniforms[:,1:]).astype(np.uint8)
    return ROT[codes],codes


def scores(scheme,scorer,model,physical,raw,delta,key,ops):
    if scorer=="plugin":
        return exp.old.base.plugin_scores(model,physical,delta)
    return exp.receiver.matched(raw+delta,key,ops)


def path_outputs(split,mi,scene,batch,n,thresholds=None,kind="attack"):
    source,model,physical,ops,uniforms,negative=streams(n,split,mi,scene,batch)
    raw=physical.y[:,:,model.links[:,1]]
    onset=ONSETS[scene%2] if kind=="attack" else 0
    stop=onset+60 if kind=="attack" else 150
    senders=model.links[:,1]
    keys={"quarter_iid":quarter_keys(uniforms,False)[0][:,:,senders],
          "quarter_no_repeat":quarter_keys(uniforms,True)[0][:,:,senders],
          "negative":negative}
    transitions=np.zeros((4,4),dtype=np.int64)
    codes_nr=quarter_keys(uniforms,True)[1]
    flat=codes_nr[:,:-1].ravel()*4+codes_nr[:,1:].ravel()
    transitions=np.bincount(flat,minlength=16).reshape(4,4)
    fixture=source.rotated_raw_attack(model,ANGLES[scene//2])[0][50:110] if kind=="attack" else None
    codes_iid=quarter_keys(uniforms,False)[1]
    output={"path_id":np.arange(batch*CHUNK,batch*CHUNK+n),
            "transition_count":transitions,
            "no_repeat_state_count":np.bincount(codes_nr.ravel(),minlength=4),
            "iid_state_count":np.bincount(codes_iid.ravel(),minlength=4)}
    for scheme,key in keys.items():
        delta=np.zeros_like(raw)
        if kind=="attack":
            delta[:,onset:stop]=(key[:,onset:stop]@fixture[None,...,None])[...,0]
        if kind=="attack":
            psi=[];blocks=ops[8,False][0]
            for t in range(onset+7,stop):
                window=delta[:,t-7:t+1].transpose(0,2,1,3)
                white=np.einsum("bewm,ewrm->ber",window,blocks)
                psi.append((white*white).sum(axis=-1))
            psi=np.stack(psi,axis=1)
            rawnorm=(delta[:,onset+7:stop]**2).sum(axis=-1)
            relative=psi/np.maximum(1.,rawnorm)
            output[scheme+"_min_relative_psi"]=relative.min(axis=1)
            output[scheme+"_state_compatible_windows"]=(relative<1e-20).sum(axis=1)
        for scorer in SCORERS:
            tag=scheme+"_"+scorer
            score=scores(scheme,scorer,model,physical,raw,delta,key,ops)
            if thresholds is None:
                output[tag+"_max"]=score[:,5:].max(axis=(1,2))
                continue
            threshold=thresholds[tag]["threshold"]
            gate=score>threshold
            output[tag+"_max"]=score[:,5:].max(axis=(1,2))
            output[tag+"_fwer"]=gate[:,5:].any(axis=(1,2)).astype(np.uint8)
            if kind=="health":
                world=exp.old.backend(exp.MODEL[mi])[3](model,physical,delta,sender_packets=physical.y,gate_override=gate)
                error=np.sum((world.post-physical.x[:,:,None,:])**2,axis=-1)
                output[tag+"_post20_affected_mse"]=error[:,20:,model.receiver_ids].mean(axis=(1,2))
                output[tag+"_post20_network_mse"]=error[:,20:].mean(axis=(1,2))
            elif kind=="attack":
                attack_gate=gate[:,onset:stop]
                delay=np.full((n,6),60,dtype=np.uint8)
                hit=attack_gate.any(axis=1)
                if hit.any():delay[hit]=np.argmax(attack_gate.transpose(0,2,1)[hit],axis=1)
                output[tag+"_early_link"]=attack_gate[:,:3].any(axis=1).astype(np.uint8)
                output[tag+"_delay60"]=delay
                output[tag+"_prior"]=gate[:,:onset].any(axis=(1,2)).astype(np.uint8)
                world=exp.old.backend(exp.MODEL[mi])[3](model,physical,delta,sender_packets=physical.y,gate_override=gate)
                error=np.sum((world.post-physical.x[:,:,None,:])**2,axis=-1)
                rec=slice(stop,None)
                output[tag+"_attack_mse"]=error[:,onset:stop,model.receiver_ids].mean(axis=(1,2))
                output[tag+"_attack_network_mse"]=error[:,onset:stop].mean(axis=(1,2))
                output[tag+"_recovery_mse"]=error[:,rec,model.receiver_ids].mean(axis=(1,2))
                output[tag+"_recovery_network_mse"]=error[:,rec].mean(axis=(1,2))
                output[tag+"_large_late_accept_count"]=np.zeros(n,dtype=np.uint8)
                for p in range(n):
                    cnt=0
                    for e in range(6):
                        alarm=np.flatnonzero(attack_gate[p,:,e])
                        if alarm.size:
                            first=onset+int(alarm[0])
                            later=np.arange(first+1,stop)
                            false=later[~gate[p,later,e]]
                            cnt+=int(np.count_nonzero(np.linalg.norm(delta[p,false,e],axis=1)>=100))
                    output[tag+"_large_late_accept_count"][p]=cnt
    return output


def worker(task):
    split,mi,scene,batch,n,pop,thresholds=task
    return path_outputs(split,mi,scene,batch,n,thresholds,kind=pop)


def run(split,workers):
    label="markov_dev" if split==5 else "markov_confirm"
    folder=HERE/label;folder.mkdir(exist_ok=True)
    n=256 if split==5 else 512
    calibration=512 if split==5 else 4096
    holdout=512 if split==5 else 4096
    hashes={"W5_MARKOV_PROTOCOL.md":exp.old.base.digest(HERE/"W5_MARKOV_PROTOCOL.md"),
            "MARKOV_DESIGN_AUDIT.json":exp.old.base.digest(HERE/"MARKOV_DESIGN_AUDIT.json"),
            "rotation_markov.py":exp.old.base.digest(HERE/"rotation_markov.py"),
            "analyze_markov.py":exp.old.base.digest(HERE/"analyze_markov.py"),
            "experiment.py":exp.old.base.digest(HERE/"experiment.py"),
            "projection.c":exp.old.base.digest(HERE/"projection.c"),
            "online_scores.py":exp.old.base.digest(exp.BASE/"online_scores.py")}
    identity={"parent":BASE,"frozen_physics":exp.BASE_SHA,"split":split,
              "paths_per_cell":n,"calibration_per_model":calibration,"holdout_per_model":holdout,
              "scenarios":[[exp.MODEL[mi],ANGLES[s//2],ONSETS[s%2]] for mi in range(2) for s in SCENES],
              "namespace":[20261127,52387,split],"arms":ARMS,
              "W1_threshold_sha256":exp.old.base.digest(HERE/"confirm/RUN.json"),
              "source_sha256":hashes,"conditional_entropy_bits":{"quarter_iid":2.,"quarter_no_repeat":float(np.log2(3))},
              "transition_matrix_no_repeat":[[0.,1/3,1/3,1/3],[1/3,0.,1/3,1/3],[1/3,1/3,0.,1/3],[1/3,1/3,1/3,0.]]}
    if split==6:
        lock_path=HERE/"MARKOV_LOCK.json"
        if not lock_path.exists():raise ValueError("W5 development has not passed its locked gate.")
        lock=json.loads(lock_path.read_text())
        if lock["source_sha256"]!=hashes or lock["development_manifest_sha256"]!=exp.old.base.digest(HERE/"markov_dev/RUN.json"):
            raise ValueError("W5 confirmation differs from its development lock.")
    progress=folder/"progress.json";complete=folder/"RUN.json"
    if complete.exists():
        old=json.loads(complete.read_text())
        if old["identity"]!=identity:raise ValueError("Markov completed identity changed.")
        for f,h in old["archives"].items():
            if exp.old.base.digest(folder/f)!=h:raise ValueError(f"Markov archive changed: {f}")
        return
    state={"identity":identity,"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip(),
           "python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,
           "host":platform.platform(),"workers":workers,"archives":{},"thresholds":{}}
    if progress.exists():
        state=json.loads(progress.read_text())
        if state["identity"]!=identity:raise ValueError("Markov restart identity changed.")
        for f,h in state["archives"].items():
            if exp.old.base.digest(folder/f)!=h:raise ValueError(f"Markov partial archive changed: {f}")
    else:exp.old.base._atomic_json(progress,state)
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for mi,model in enumerate(exp.MODEL):
            if model not in state["thresholds"]:
                jobs=[(split,mi,0,b,min(CHUNK,calibration-i),"cal",None)
                      for b,i in enumerate(range(0,calibration,CHUNK))]
                rows=list(pool.map(worker,jobs))
                maxima={arm:np.concatenate([r[arm+"_max"] for r in rows]) for arm in ARMS}
                state["thresholds"][model]={arm:exp.old.path_fwer_cutoff(vals,.05) for arm,vals in maxima.items()}
                calibration_arrays={"path_id":np.arange(calibration)}
                calibration_arrays.update({"max_"+arm:vals for arm,vals in maxima.items()})
                cal_file=f"{model}_calibration.npz"
                exp.old.base._atomic_npz(folder/cal_file,calibration_arrays)
                state["archives"][cal_file]=exp.old.base.digest(folder/cal_file)
                exp.old.base._atomic_json(progress,state)
            thresholds=state["thresholds"][model]
            for pop,count,scenes in (("health",holdout,[0]),("attack",n,list(SCENES))):
                for scene in scenes:
                    filename=f"{model}_{pop}_{scene}.npz"
                    if filename in state["archives"]:continue
                    jobs=[(split,mi,scene,b,min(CHUNK,count-i),pop,thresholds)
                          for b,i in enumerate(range(0,count,CHUNK))]
                    rows=list(pool.map(worker,jobs))
                    arrays={key:np.concatenate([r[key] for r in rows]) for key in rows[0]}
                    np.testing.assert_array_equal(arrays["path_id"],np.arange(count))
                    exp.old.base._atomic_npz(folder/filename,arrays)
                    state["archives"][filename]=exp.old.base.digest(folder/filename)
                    exp.old.base._atomic_json(progress,state)
                    print(label,filename,flush=True)
    state["wall_seconds"]=time.perf_counter()-started
    state["archive_sha256"]=dict(state["archives"])
    exp.old.base._atomic_json(complete,state);progress.unlink()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=("dev","confirm"));p.add_argument("--workers",type=int,default=8)
    a=p.parse_args();run(5 if a.split=="dev" else 6,a.workers)
