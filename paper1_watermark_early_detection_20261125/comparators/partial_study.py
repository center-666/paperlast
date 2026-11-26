"""Partial-support detection/localization on the frozen 50-node common-G model."""
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
import experiment as exp

HERE=exp.HERE
SCHEMES=("quarter","negative","iid")
SCORERS=("plugin","exact")
ARMS=tuple(s+"_"+d for s in SCHEMES for d in SCORERS)
CHUNK=32
SUPPORTS=[(f"single_{i}",(i,)) for i in range(6)]+[
    ("same_receiver_17",(3,5)),("different_receivers_0_2",(0,1))]
MASKS={name:np.asarray(edges,dtype=int) for name,edges in SUPPORTS}
W1=exp.read(HERE/"confirm/RUN.json")
THRESHOLDS=W1["thresholds"]
PARENT="f97748fcba7c45981878352aa03bc1b15f78dc65"


def namespace(split,mi,scene,batch):
    return (20261127,52387,split,mi,scene,4,batch)


def task(args):
    split,mi,scene,batch,n=args
    onset=(50,65)[scene%2]
    angle=(0.,-45.,45.)[scene//2]
    source,model,_,estimator,ops=exp.old.backend(exp.MODEL[mi])
    ns=namespace(split,mi,scene,batch)
    physical,draws,bits=exp.physical_and_keys(n,ns,mi)
    senders=model.links[:,1]
    raw=physical.y[:,:,senders]
    fixture=source.rotated_raw_attack(model,angle)[0][50:110]
    record={"path_id":np.arange(batch*CHUNK,batch*CHUNK+n)}
    keys={s:exp.schedule(s,draws,bits,1.) for s in SCHEMES}
    for mask_name,attacked in MASKS.items():
        clean=np.setdiff1d(np.arange(6),attacked)
        for scheme in SCHEMES:
            delta=np.zeros_like(raw)
            key=keys[scheme]
            local_key=key[:,onset:onset+60][:,:,attacked]
            local_fixture=fixture[:,attacked,:]
            delta[:,onset:onset+60][:,:,attacked,:]=(
                local_key@local_fixture[None,...,None]
            )[...,0]
            for scorer in SCORERS:
                tag=scheme+"_"+scorer
                if scorer=="plugin":
                    score=exp.old.base.plugin_scores(model,physical,delta)
                else:
                    score=exp.receiver.matched(raw+delta,key,ops)
                threshold=THRESHOLDS[exp.MODEL[mi]][tag][1]
                gates=score>threshold
                early=gates[:,onset:onset+3].any(axis=1)
                full=gates[:,onset:onset+60].any(axis=1)
                record[f"{mask_name}__{tag}_early_link"]=early.astype(np.uint8)
                record[f"{mask_name}__{tag}_full_link"]=full.astype(np.uint8)
                delay=np.full((n,6),60,dtype=np.uint8)
                at=gates[:,onset:onset+60]
                hit=at.any(axis=1)
                if hit.any():
                    delay[hit]=np.argmax(at.transpose(0,2,1)[hit],axis=1)
                record[f"{mask_name}__{tag}_delay60"]=delay
                record[f"{mask_name}__{tag}_pre_alarm"]=gates[:,:onset].any(axis=(1,2)).astype(np.uint8)
                world=estimator(model,physical,delta,sender_packets=physical.y,gate_override=gates)
                error=np.sum((world.post-physical.x[:,:,None,:])**2,axis=-1)
                direct=np.unique(model.links[attacked,0])
                target=slice(onset,onset+60);recovery=slice(onset+60,None)
                record[f"{mask_name}__{tag}_attack_target_mse"]=error[:,target][:,:,direct].mean(axis=(1,2))
                record[f"{mask_name}__{tag}_attack_five_mse"]=error[:,target][:,:,model.receiver_ids].mean(axis=(1,2))
                record[f"{mask_name}__{tag}_attack_network_mse"]=error[:,target].mean(axis=(1,2))
                record[f"{mask_name}__{tag}_recovery_target_mse"]=error[:,recovery][:,:,direct].mean(axis=(1,2))
                record[f"{mask_name}__{tag}_recovery_five_mse"]=error[:,recovery][:,:,model.receiver_ids].mean(axis=(1,2))
                record[f"{mask_name}__{tag}_recovery_network_mse"]=error[:,recovery].mean(axis=(1,2))
    return record


def run(split,workers):
    label="partial_dev" if split==3 else "partial_confirm"
    folder=HERE/label;folder.mkdir(exist_ok=True)
    n=256 if split==3 else 512
    sources=("NEXT_PROTOCOL.md","partial_study.py","experiment.py","projection.c","online_scores.py")
    source_hashes={f:exp.old.base.digest(HERE/f) for f in sources}
    identity={"parent":PARENT,"physical_baseline":exp.BASE_SHA,"split":split,
              "paths_per_scene":n,"support_masks":{k:list(map(int,v)) for k,v in MASKS.items()},
              "scenario_order":[[exp.MODEL[mi],(0,-45,45)[s//2],(50,65)[s%2]]
                                for mi in range(2) for s in range(6)],
              "namespace":[20261127,52387,split],"arms":ARMS,
              "W1_threshold_manifest_sha256":exp.old.base.digest(HERE/"confirm/RUN.json"),
              "W1_thresholds":THRESHOLDS,"source_sha256":source_hashes}
    if split==4:
        lock_path=HERE/"PARTIAL_LOCK.json"
        if not lock_path.exists():raise ValueError("Partial-link development is not locked.")
        frozen=json.loads(lock_path.read_text())
        if frozen["source_sha256"]!=source_hashes or frozen["development_manifest_sha256"]!=exp.old.base.digest(HERE/"partial_dev/RUN.json"):
            raise ValueError("Partial-link confirmation differs from its development lock.")
    state={"identity":identity,"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip(),
           "python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,
           "host":platform.platform(),"workers":workers,"archives":{}}
    progress=folder/"progress.json"
    completed=folder/"RUN.json"
    if completed.exists():
        old=json.loads(completed.read_text())
        if old["identity"]!=identity:raise ValueError("Partial-link run identity changed")
        for f,h in old["archives"].items():
            if exp.old.base.digest(folder/f)!=h:raise ValueError(f"Partial archive changed: {f}")
        return
    if progress.exists():
        state=json.loads(progress.read_text())
        if state["identity"]!=identity:raise ValueError("Partial-link restart identity changed")
        for f,h in state["archives"].items():
            if exp.old.base.digest(folder/f)!=h:raise ValueError(f"Partial archive changed: {f}")
    else:
        exp.old.base._atomic_json(progress,state)
    exp.receiver.kernel()
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for mi,name in enumerate(exp.MODEL):
            for scene in range(6):
                filename=f"{name}_{scene}.npz"
                if filename in state["archives"]:continue
                jobs=[(split,mi,scene,b,min(CHUNK,n-i)) for b,i in enumerate(range(0,n,CHUNK))]
                rows=list(pool.map(task,jobs))
                arrays={key:np.concatenate([r[key] for r in rows]) for key in rows[0]}
                np.testing.assert_array_equal(arrays["path_id"],np.arange(n))
                exp.old.base._atomic_npz(folder/filename,arrays)
                state["archives"][filename]=exp.old.base.digest(folder/filename)
                exp.old.base._atomic_json(progress,state)
                print(label,filename,flush=True)
    state["wall_seconds"]=time.perf_counter()-started
    exp.old.base._atomic_json(completed,state)
    progress.unlink()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=("dev","confirm"));p.add_argument("--workers",type=int,default=8)
    a=p.parse_args();run(3 if a.split=="dev" else 4,a.workers)
