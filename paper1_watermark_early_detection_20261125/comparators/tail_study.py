"""Independent frequency check of the fixed W1 catastrophic-loss mechanism."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
import experiment as exp

HERE=exp.HERE
SCENES=[(m,a,o) for m in (0,1) for a in range(3) for o in range(2)]
SCHEMES=("quarter","diagonal","sign","iid","negative")
CHUNK=32
W1=exp.read(HERE/"confirm/RUN.json")
THRESHOLD=W1["thresholds"]


def namespace(split,mi,scene,batch):
    return (20261127,52387,split,mi,scene,2,batch)


def trial(task):
    split,mi,scene,batch,n=task
    angle=(0.,-45.,45.)[scene//2]
    onset=(50,65)[scene%2]
    stop=onset+60
    ns=namespace(split,mi,scene,batch)
    source,model,generate,estimator,ops=exp.old.backend(exp.MODEL[mi])
    physical,draws,bits=exp.physical_and_keys(n,ns,mi)
    edge_senders=model.links[:,1]
    raw=physical.y[:,:,edge_senders]
    fixture=source.rotated_raw_attack(model,angle)[0][50:110]
    white=ops[8,False][0]
    record={"path_id":np.arange(batch*CHUNK,batch*CHUNK+n)}
    deltas={}
    score_cache={}
    for scheme in SCHEMES:
        key=exp.schedule(scheme,draws,bits,1.)
        delta=np.zeros_like(raw)
        delta[:,onset:stop]=(key[:,onset:stop]@fixture[None,...,None])[...,0]
        deltas[scheme]=delta
        psi=[]
        for tick in range(onset+7,stop):
            recent=delta[:,tick-7:tick+1].transpose(0,2,1,3)
            z=np.einsum("bewm,ewrm->ber",recent,white)
            psi.append((z*z).sum(axis=-1))
        psi=np.stack(psi,axis=1)
        rawnorm=(delta[:,onset+7:stop]**2).sum(axis=-1)
        relative=psi/np.maximum(1.,rawnorm)
        record[scheme+"_min_relative_residual_energy"]=relative.min(axis=1)
        record[scheme+"_min_relative_time"]=relative.argmin(axis=1)+onset+7
        record[scheme+"_state_compatible_window_count"]=(relative<1e-20).sum(axis=1)
        record[scheme+"_large_key_run_edge0"]=np.zeros(n,dtype=np.int16)
        for path in range(n):
            best=run=0
            for tick in range(onset,stop):
                if np.array_equal(key[path,tick,0],np.eye(2)) or np.array_equal(key[path,tick,0],-np.eye(2)):
                    run+=1;best=max(best,run)
                else:run=0
            record[scheme+"_large_key_run_edge0"][path]=best
        for scorer in ("plugin","exact"):
            tag=scheme+"_"+scorer
            if scorer=="plugin":
                score=exp.old.base.plugin_scores(model,physical,delta)
            else:
                score=exp.receiver.matched(raw+delta,key,ops)
            score_cache[tag]=score
            threshold=THRESHOLD[exp.MODEL[mi]][tag][1]
            gate=score>threshold
            world=estimator(model,physical,delta,sender_packets=physical.y,gate_override=gate)
            error=np.sum((world.post-physical.x[:,:,None,:])**2,axis=-1)
            affected=error[:,:,model.receiver_ids]
            record[tag+"_attack_mse"]=affected[:,onset:stop].mean(axis=(1,2))
            record[tag+"_network_attack_mse"]=error[:,onset:stop].mean(axis=(1,2))
            record[tag+"_recovery_mse"]=affected[:,stop:].mean(axis=(1,2))
            record[tag+"_node0_attack_peak"]=error[:,onset:stop,0].max(axis=1)
            attack_gate=gate[:,onset:stop]
            delay=np.full((n,6),60,dtype=np.uint8)
            hit=attack_gate.any(axis=1)
            if hit.any():
                delay[hit]=np.argmax(attack_gate.transpose(0,2,1)[hit],axis=1)
            record[tag+"_delay60"]=delay
            record[tag+"_prior_alarm"]=gate[:,:onset].any(axis=(1,2))
            after_alarm=[]
            large_after_alarm=[]
            accepted_last=[]
            for p in range(n):
                count=large=last=0
                for e in range(6):
                    alarm=np.flatnonzero(attack_gate[p,:,e])
                    if not alarm.size:continue
                    first=onset+int(alarm[0])
                    later=np.arange(first+1,stop)
                    false=later[~gate[p,later,e]]
                    count+=len(false)
                    large+=int(np.count_nonzero(np.linalg.norm(delta[p,false,e],axis=1)>=100.))
                    last+=int((stop-1 in false))
                after_alarm.append(count);large_after_alarm.append(large);accepted_last.append(last)
            record[tag+"_accepted_after_alarm_count"]=np.asarray(after_alarm)
            record[tag+"_large_after_alarm_count"]=np.asarray(large_after_alarm)
            record[tag+"_accepted_last_report_count"]=np.asarray(accepted_last)
            record[tag+"_gate_count_attack"]=attack_gate.sum(axis=(1,2))
            if scorer=="exact" and scheme=="quarter":
                latch=gate.copy()
                for p in range(n):
                    for e in range(6):
                        alarm=np.flatnonzero(attack_gate[p,:,e])
                        if alarm.size:latch[p,onset+int(alarm[0]):stop,e]=True
                latched=estimator(model,physical,delta,sender_packets=physical.y,gate_override=latch)
                latch_error=np.sum((latched.post-physical.x[:,:,None,:])**2,axis=-1)[:,:,model.receiver_ids]
                record["quarter_exact_latch_attack_mse"]=latch_error[:,onset:stop].mean(axis=(1,2))
                record["quarter_exact_latch_recovery_mse"]=latch_error[:,stop:].mean(axis=(1,2))
    return record


def run(split,workers):
    folder=HERE/("tail_dev" if split==1 else "tail_confirm")
    folder.mkdir(exist_ok=True)
    n=256 if split==1 else 512
    source_files=("tail_study.py","experiment.py","projection.c","NEXT_PROTOCOL.md",
                  "high_loss_trace.py","high_loss_mechanism.py",
                  "HIGH_LOSS_REPLAY.json","HIGH_LOSS_COUNTERFACTUAL.json")
    hashes={f:exp.old.base.digest(HERE/f) for f in source_files}
    identity={"parent":"f97748fcba7c45981878352aa03bc1b15f78dc65",
              "physical_baseline":exp.old.BASE_SHA,"split":split,"paths_per_cell":n,
              "namespace":[20261127,52387,split],"scenario_order":SCENES,
              "threshold_manifest_sha256":exp.old.base.digest(HERE/"confirm/RUN.json"),
              "source_sha256":hashes,"arms":[s+"_"+x for s in SCHEMES for x in ("plugin","exact")],
              "thresholds":THRESHOLD}
    if split==2:
        development=HERE/"tail_dev/RUN.json"
        if not development.exists():raise ValueError("Tail development must complete before confirmation.")
        dev=json.loads(development.read_text())
        if dev["identity"]["source_sha256"]!=hashes:
            raise ValueError("Tail confirmation source differs from development.")
    state={"identity":identity,"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip(),
           "python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,
           "host":platform.platform(),"workers":workers,"archives":{}}
    oldfile=folder/"RUN.json"
    if oldfile.exists():
        existing=json.loads(oldfile.read_text())
        if existing["identity"]!=identity:raise ValueError("Tail run identity changed.")
        for f,h in existing["archives"].items():
            if exp.old.base.digest(folder/f)!=h:raise ValueError("Tail archive changed.")
        return
    progress=folder/"progress.json"
    if progress.exists():
        state=json.loads(progress.read_text())
        if state["identity"]!=identity:raise ValueError("Tail progress identity changed.")
        for f,h in state["archives"].items():
            if exp.old.base.digest(folder/f)!=h:raise ValueError("Tail partial archive changed.")
    else:
        exp.old.base._atomic_json(progress,state)
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for mi,name in enumerate(exp.MODEL):
            for scene in range(6):
                filename=f"{name}_{scene}.npz"
                if filename in state["archives"]:continue
                jobs=[(split,mi,scene,b,min(CHUNK,n-i)) for b,i in enumerate(range(0,n,CHUNK))]
                rows=list(pool.map(trial,jobs))
                arrays={key:np.concatenate([r[key] for r in rows]) for key in rows[0]}
                np.testing.assert_array_equal(arrays["path_id"],np.arange(n))
                exp.old.base._atomic_npz(folder/filename,arrays)
                state["archives"][filename]=exp.old.base.digest(folder/filename)
                exp.old.base._atomic_json(progress,state)
                print(folder.name,filename,flush=True)
    state["wall_seconds"]=time.perf_counter()-started
    state["archive_sha256"]=dict(state["archives"])
    exp.old.base._atomic_json(oldfile,state)
    progress.unlink()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=("dev","confirm"));p.add_argument("--workers",type=int,default=8)
    a=p.parse_args();run(1 if a.split=="dev" else 2,a.workers)
