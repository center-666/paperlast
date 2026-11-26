"""Paired per-link temporal-watermark study on the frozen common-G receiver."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy

import run_stage1 as base
from calibration_rule import path_fwer_cutoff
from online_scores import matched, operators, kernel
from watermark_keys import beta_for_scheme, make_key_draws
from watson_crn import CommonDraws

HERE = base.HERE
CHUNK = 32
ARMS1 = (("A","iid","plugin"), ("B","negative_fixed","plugin"),
         ("C","iid","matched"), ("D","negative_fixed","matched"),
         ("E","iid","wrong"), ("F","negative_fixed","wrong"))
ARMS2 = tuple((short+"_"+kind, scheme, kind) for short,scheme in (
    ("fixed","negative_fixed"), ("radial","negative_radial_bin"), ("random","negative_random_bin"))
    for kind in ("plugin","matched"))
LABELS = {1:"attribution_dev",2:"attribution_confirm",3:"radial_dev",4:"radial_confirm",99:"test"}
ALPHAS = (.025, .05, .075)
LOSSES = ("attack_affected_mse", "attack_network_mse", "post20_affected_mse", "post20_network_mse",
          "postattack_affected_mse", "postattack_network_mse")
SOURCES = ("study.py","online_scores.py","projection.c","watermark_keys.py","run_stage1.py",
           "matched_glrt.py","REGISTRATION.json","PROTOCOL_CORRECTION.md","analyze_stage1.py")


@lru_cache(None)
def backend(name):
    old, load, physical, estimator, _ = base.load_backend()
    model = load(name)
    return old, model, physical, estimator, operators(model)


def short_draws(n, namespace, mi, scene, stage, batch, senders, wrong=False, index=0):
    draws, seeds = base.make_crn_draws(n, namespace, mi, scene, stage, batch, wrong, index)
    sub = CommonDraws(draws.radial_chi2_half[:,:,senders], draws.initial_angle[:,senders],
                      draws.quantile_u[:,:,senders], draws.branch[:,:,senders])
    labels = base.seed_word(namespace,mi,scene,stage,batch,90)
    sub.radial_bin_labels = (np.random.default_rng(labels).random(draws.quantile_u.shape)[:,:,senders] < .25)
    return sub


def matrices(draws, scheme):
    beta = beta_for_scheme(draws, scheme)
    return .5*np.eye(2)+beta[...,None]*beta[...,None,:]


def trial(task):
    split, kind, model_index, scene, batch, n, arms, cutoffs = task
    namespace = (20261125,8317,split)
    old, model, physical_paths, estimator, ops = backend(base.MODELS[model_index])
    senders = model.links[:,1]
    assert len(np.unique(senders)) == len(senders)
    healthy = kind != "eval"
    onset = 50 if healthy else base.ONSETS[scene%2]
    angle = 0 if healthy else base.ANGLES[scene//2]
    if healthy:
        kphase = 1 if kind=="cal" else 5
        pseed = base.seed_word(namespace,model_index,0,kphase-1,batch,1)
    else:
        kphase = 12
        pseed = base.seed_word(namespace,model_index,scene,11,batch,1)
    phy = physical_paths(model,n,pseed)
    draws = short_draws(n,namespace,model_index,scene,kphase,batch,senders)
    raw = phy.y[:,:,senders]
    zero = np.zeros_like(raw)
    actual = {}
    result = {"path_id":np.arange(batch*CHUNK,batch*CHUNK+n), "physical_seed":np.full(n,pseed)}
    score_cache = {}
    for label,scheme,scorer in arms:
        if scheme not in actual:
            key = matrices(draws,scheme)
            delta = np.zeros_like(raw)
            if not healthy:
                fixture,_ = old.rotated_raw_attack(model,angle)
                delta[:,onset:onset+60] = (key[:,onset:onset+60] @ fixture[None,50:110,:,:,None])[...,0]
                if split in (3,4,99):
                    signal = []
                    contamination = []
                    for h in (2,3,5):
                        recent = delta[:,onset+h-8:onset+h].transpose(0,2,1,3)
                        blocks,_ = ops[8,False]
                        white = np.einsum("bewm,ewrm->ber",recent,blocks)
                        signal.append((white**2).sum(axis=-1))
                        contamination.append((delta[:,onset:onset+h]**2).sum(-1).mean(1))
                    result[scheme+"_psi"] = np.stack(signal,axis=1)
                    result[scheme+"_ungated_packet_sq"] = np.stack(contamination,axis=1)
            actual[scheme] = (key,delta)
        key, delta = actual[scheme]
        tick = time.perf_counter()
        score_key = ("shared" if healthy and scorer=="plugin" else scheme,scorer)
        if score_key in score_cache:
            score = score_cache[score_key]
            score_seconds = 0.
        else:
            if scorer=="plugin":
                score = base.plugin_scores(model,phy,delta)
            else:
                if scorer=="wrong":
                    index = int(scheme=="negative_fixed")
                    wrong = short_draws(n,namespace,model_index,scene,kphase+1+index,batch,senders,True,index)
                    key = matrices(wrong,scheme)
                score = matched(raw+delta,key,ops)
            score_cache[score_key] = score
            score_seconds = time.perf_counter()-tick
        result[label+"_score_cpu_wall_s"] = np.full(n,score_seconds/n)
        result[label+"_max"] = score[:,5:].max(axis=(1,2))
        if kind=="cal":
            continue
        threshold = cutoffs[label][1]
        gates = score > threshold
        start = time.perf_counter()
        world = estimator(model,phy,delta,sender_packets=phy.y,gate_override=gates)
        result[label+"_replay_cpu_wall_s"] = np.full(n,(time.perf_counter()-start)/n)
        error = ((world.post-phy.x[:,:,None])**2).sum(-1)
        for metric in LOSSES:
            window = slice(20,None) if metric.startswith("post20") else (
                slice(onset+60,None) if metric.startswith("postattack") else slice(onset,onset+60))
            sample = error[:,window]
            if "affected" in metric:
                sample = sample[:,:,model.receiver_ids]
            result[label+"_"+metric] = sample.mean((1,2))
        if not healthy:
            full = gates[:,onset:onset+60]
            result[label+"_delay"] = np.where(full.any(1),full.argmax(1),60).astype("uint8")
            result[label+"_early_max"] = score[:,onset:onset+3].max(1)
            result[label+"_eligible"] = full[:,7:].mean((1,2))
            result[label+"_prior"] = gates[:,:onset].any((1,2))
    return result


def collect(pool, jobs, path):
    rows = list(pool.map(trial,jobs))
    arrays = {key:np.concatenate([row[key] for row in rows]) for key in rows[0]}
    np.testing.assert_array_equal(arrays["path_id"],np.arange(len(arrays["path_id"])))
    base._atomic_npz(path,arrays)
    return arrays


def run(split, directory=None, smoke=False, workers=6):
    wall = time.perf_counter()
    directory = directory or HERE / LABELS[split]
    directory.mkdir(exist_ok=True)
    confirmation = split in (2,4)
    lock = json.loads((HERE/"CONFIRM_LOCK.json").read_text()) if confirmation else {}
    if confirmation and base.digest(HERE/"REGISTRATION.json") != lock["registration_sha256"]:
        raise ValueError("Confirmation registration changed")
    arms = ARMS1 if split in (1,2) else ARMS2
    if split==4:
        if not lock["stage2_receiver"]:
            raise ValueError("Radial independent confirmation was not selected")
        arms=tuple(row for row in arms if row[2]==lock["stage2_receiver"])
    health = 512 if not confirmation else 4096
    size = 256 if not confirmation else lock[str(split)+"_n"]
    if smoke:
        health,size=48,8
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip()
    identity = {"split":split,"arms":arms,"calibration_and_holdout_each":health,"paths_per_scene":size,
                "seed_namespace":[20261125,8317,split],"chunk":CHUNK,
                "sources":{f:base.digest(HERE/f) for f in SOURCES},
                "upstream":base.BASELINE_COMMIT,"upstream_inputs":base.source_hashes(),
                "fixture_hashes":backend("stable")[0].validate_fixtures(),
                "lock":base.digest(HERE/"CONFIRM_LOCK.json") if confirmation else None}
    identity=json.loads(json.dumps(identity))
    if (directory/"RUN.json").exists():
        state=json.loads((directory/"RUN.json").read_text())
        if state["identity"]!=identity: raise ValueError("Completed archive has a different identity")
        for f,h in state["archives"].items():
            if base.digest(directory/f)!=h: raise ValueError(f)
        return
    progress=directory/"progress.json"
    if progress.exists():
        state=json.loads(progress.read_text())
        if state["identity"]!=identity: raise ValueError("Partial archive identity differs")
    else:
        state={"identity":identity,"execution_commit":head,"python":sys.version,"numpy":np.__version__,
               "scipy":scipy.__version__,"host":platform.platform(),"workers":workers,"archives":{},"thresholds":{}}
        base._atomic_json(progress,state)
    kernel()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for mi,name in enumerate(base.MODELS):
            for kind,scene in [("cal",0),("health",0)]+[("eval",s) for s in range(6)]:
                file=f"{name}_{kind}_{scene}.npz"
                path=directory/file
                if file in state["archives"]:
                    if base.digest(path)!=state["archives"][file]: raise ValueError(file)
                    continue
                n=size if kind=="eval" else health
                cutoffs=state["thresholds"].get(name,{})
                jobs=[(split,kind,mi,scene,b,min(CHUNK,n-i),arms,cutoffs)
                      for b,i in enumerate(range(0,n,CHUNK))]
                arrays=collect(pool,jobs,path)
                if kind=="cal":
                    # A 48-path smoke supports all three registered ranks.
                    state["thresholds"][name]={arm:[path_fwer_cutoff(arrays[arm+"_max"],a)["threshold"]
                        for a in ALPHAS] for arm,_,_ in arms}
                state["archives"][file]=base.digest(path)
                base._atomic_json(progress,state)
                print(LABELS[split],file,flush=True)
    state["wall_seconds_this_invocation"]=time.perf_counter()-wall
    base._atomic_json(directory/"RUN.json",state)
    progress.unlink()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("split",type=int,choices=(1,2,3,4))
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    run(args.split,workers=args.workers)
