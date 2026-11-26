"""W1/W2 paired benchmark; all physical and receiver models are pinned upstream."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0,str(BASE))
import study as old

# Isolate the rank-aware kernel; the original receiver remains available for W0.
spec = importlib.util.spec_from_file_location("w_receiver",BASE/"online_scores.py")
receiver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receiver)
receiver.HERE = HERE
SCHEMES = ("none","sign","diagonal","quarter","iid","negative","iid_eq","negative_eq")
MODES = ("plugin","exact","minus","plus")
ARMS = [(s,m) for s in SCHEMES for m in MODES
        if m in ("plugin","exact") or s in ("iid","negative","sign")]
TAGS = [s+"_"+m for s,m in ARMS]
MODEL = ("unstable","stable")
BASE_SHA="da7a6c7d92b4bbf3514e86dbe51cca15e84660a5"
CHUNK=32
ALPHA=(.025,.05,.075)
SCALES=(.98,1.,1.02)
ROTATION = np.array([[[1,0],[0,1]],[[0,-1],[1,0]],[[-1,0],[0,-1]],[[0,1],[-1,0]]],float)
METRICS=("health_or_post20_affected","health_or_post20_network","disturbance_affected",
         "disturbance_network","recovery_affected","recovery_network")
LOSSES=("early","any3","prior","delay60","eligible",*METRICS)


def read(file):
    return json.loads(Path(file).read_text(encoding="utf-8-sig"))


def namespace(split,mi,scene,pop,b):
    return [20261126,17291,split,mi,scene,pop,b]


def physical_and_keys(n,ns,mi):
    src,model,generate,estimator,ops=old.backend(MODEL[mi])
    # High-dimensional entropy passed directly to physical generator, never truncated to 32 bits.
    phy=generate(model,n,np.random.SeedSequence([*ns,1]))
    j=model.links[:,1]
    seeds=[np.random.SeedSequence([*ns,2,k]) for k in range(3)]
    draws=old.make_key_draws(n,*seeds)
    draws=old.CommonDraws(draws.radial_chi2_half[:,:,j],draws.initial_angle[:,j],draws.quantile_u[:,:,j],draws.branch[:,:,j])
    bits=np.random.default_rng(np.random.SeedSequence([*ns,3])).integers(0,4,(n,150,50),dtype=np.uint8)[:,:,j]
    return phy,draws,bits


def schedule(s,draws,bits,c):
    if s.startswith(("iid","negative")):
        return old.matrices(draws,"iid" if s.startswith("iid") else "negative_fixed")/(c if s.endswith("_eq") else 1.)
    if s=="quarter":
        return ROTATION[bits]
    eye=np.eye(2)
    if s=="none":
        return np.broadcast_to(eye,(*bits.shape,2,2))
    first=1.-2*(bits%2)
    if s=="sign":
        return first[...,None,None]*eye
    out=np.zeros((*bits.shape,2,2))
    out[...,0,0]=first
    out[...,1,1]=1.-2*(bits//2)
    return out


@lru_cache(None)
def operators(mi,factor):
    ops=old.backend(MODEL[mi])[-1]
    return {key:(blocks,np.ascontiguousarray(ha*(factor**np.arange(key[0]))[None,:,None,None]))
            for key,(blocks,ha) in ops.items()}


def energy(raw,key):
    y=raw[:,1:]
    tx=np.linalg.solve(key[:,1:],y[...,None])[...,0]
    return np.sum(y*y,axis=(1,2,3)),np.sum(tx*tx,axis=(1,2,3))


def fit():
    result={}
    for mi,name in enumerate(MODEL):
        sums=np.zeros(2)
        for b in range(32):
            phy,draws,bits=physical_and_keys(CHUNK,namespace(3,mi,0,0,b),mi)
            j=old.backend(name)[1].links[:,1]
            sums+=np.asarray(energy(phy.y[:,:,j],schedule("iid",draws,bits,1.))).sum(1)
        ratio=sums[1]/sums[0]
        assert abs(ratio-1)>.05
        result[name]={"paths":1024,"ratio":ratio,"encode_multiplier":1/np.sqrt(ratio)}
    old.base._atomic_json(HERE/"ENERGY_LOCK.json",result)


def scores(mode,mi,phy,model,delta,raw,key):
    if mode=="plugin":
        return old.base.plugin_scores(model,phy,delta)
    return receiver.matched(raw+delta,key,operators(mi,{"exact":1.,"minus":.98,"plus":1.02}[mode]))


def cell(task):
    split,mi,scene,pop,b,n,c,cutoff=task
    ns=namespace(split,mi,scene,pop,b)
    src,model,_,estimator,_=old.backend(MODEL[mi])
    phy,draws,bits=physical_and_keys(n,ns,mi)
    raw=phy.y[:,:,model.links[:,1]]
    onset=(50,65)[scene%2]
    stop=onset+60
    result={"path_id":np.arange(b*CHUNK,b*CHUNK+n)}
    fixture=src.rotated_raw_attack(model,(0.,-45.,45.)[scene//2])[0][50:110] if pop==2 else None
    healthy_scores={}
    healthy_errors={}
    for s in SCHEMES:
        begin=time.perf_counter()
        key=schedule(s,draws,bits,c)
        result[s+"_key_s"]=np.full(n,(time.perf_counter()-begin)/n)
        delta=np.zeros_like(raw)
        if pop==2:
            delta[:,onset:stop]=(key[:,onset:stop]@fixture[None,...,None])[...,0]
        if pop==1:
            before,after=energy(raw,key)
            result["raw_energy"]=before
            result[s+"_tx_energy"]=after
        for ss,mode in ARMS:
            if ss!=s:continue
            tag=s+"_"+mode
            cachekey="plugin" if mode=="plugin" else s.removesuffix("_eq")+"_"+mode
            begin=time.perf_counter()
            if pop!=2 and cachekey in healthy_scores:
                score=healthy_scores[cachekey]
            else:
                score=scores(mode,mi,phy,model,delta,raw,key)
                if pop!=2:healthy_scores[cachekey]=score
            result[tag+"_score_s"]=np.full(n,(time.perf_counter()-begin)/n)
            result[tag+"_max"]=score.max((1,2))
            if pop==0:continue
            gates=score>cutoff[tag][1]
            if pop==2:
                seg=gates[:,onset:stop]
                result[tag+"_delay"]=np.where(seg.any(1),seg.argmax(1),60).astype(np.uint8)
                result[tag+"_peak"]=score[:,onset:onset+3].max(1)
                result[tag+"_prior"]=gates[:,:onset].any((1,2))
                result[tag+"_eligible"]=seg[:,7:].mean((1,2))
            begin=time.perf_counter()
            if pop==1 and cachekey in healthy_errors:
                losses=healthy_errors[cachekey]
            else:
                world=estimator(model,phy,delta,sender_packets=phy.y,gate_override=gates)
                error=((world.post-phy.x[:,:,None])**2).sum(-1)
                windows=[error[:,20:]]
                if pop==2:windows.extend((error[:,onset:stop], error[:,stop:]))
                losses=np.column_stack([v for w in windows for v in (w[:,:,model.receiver_ids].mean((1,2)),w.mean((1,2)))])
                if pop==1:healthy_errors[cachekey]=losses
            result[tag+"_loss"]=losses
            result[tag+"_filter_s"]=np.full(n,(time.perf_counter()-begin)/n)
    return result


def run(split,workers):
    folder=HERE/("dev" if split==1 else "confirm")
    folder.mkdir(exist_ok=True)
    n=256 if split==1 else read(HERE/"CONFIRM_LOCK.json")["n"]
    health=512 if split==1 else 4096
    scale=read(HERE/"ENERGY_LOCK.json")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip()
    hashes={f:old.base.digest(HERE/f) for f in ("PROTOCOL.md","experiment.py","projection.c","analyze.py","ENERGY_LOCK.json")}
    if split==2:
        lock=read(HERE/"CONFIRM_LOCK.json")
        if hashes!=lock["frozen_sources"] or lock["dev_hash"]!=old.base.digest(HERE/"dev/RUN.json"):
            raise ValueError("Confirmation inputs differ from the development lock")
    state={"base":BASE_SHA,"head":head,"source_hashes":hashes,"paths_per_scene":n,"health_per_population":health,
           "split":split,"arms":TAGS,"namespace":[20261126,17291,split],"python":sys.version,"host":platform.platform(),
           "scipy":scipy.__version__,"numpy":np.__version__,"workers":workers,"thresholds":{},"archives":{}}
    identity=state.copy()
    for f in ("RUN.json","progress.json"):
        if (folder/f).exists():
            state=read(folder/f)
            for k in ("source_hashes","paths_per_scene","namespace"):
                if identity[k]!=state[k]:raise ValueError(f"Restart identity changed: {k}")
            for name,digest in state["archives"].items():
                if old.base.digest(folder/name)!=digest:raise ValueError(name)
            if f=="RUN.json":return
            break
    old.backend("stable")[0].validate_fixtures()
    receiver.kernel()
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for mi,name in enumerate(MODEL):
            for pop,scene in [(0,0),(1,0)]+[(2,i) for i in range(6)]:
                f=f"{name}_{pop}_{scene}.npz"
                if f in state["archives"]:continue
                count=n if pop==2 else health
                jobs=[(split,mi,scene,pop,b,min(CHUNK,count-i),scale[name]["encode_multiplier"],state["thresholds"].get(name,{}))
                      for b,i in enumerate(range(0,count,CHUNK))]
                rows=list(pool.map(cell,jobs))
                arrays={k:np.concatenate([v[k] for v in rows]) for k in rows[0]}
                np.testing.assert_array_equal(arrays["path_id"],np.arange(count))
                if pop==0:
                    state["thresholds"][name]={a:[old.path_fwer_cutoff(arrays[a+"_max"],v)["threshold"] for v in ALPHA] for a in TAGS}
                old.base._atomic_npz(folder/f,arrays)
                state["archives"][f]=old.base.digest(folder/f)
                old.base._atomic_json(folder/"progress.json",state)
                print(split,f,flush=True)
    state["wall_seconds_last_invocation"]=time.perf_counter()-start
    old.base._atomic_json(folder/"RUN.json",state)
    (folder/"progress.json").unlink()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("job",choices=("fit","dev","confirm"))
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    if args.job=="fit":fit()
    else:run(1 if args.job=="dev" else 2,args.workers)
