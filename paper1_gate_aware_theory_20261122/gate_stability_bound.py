"""Paper 1: pathwise bounded-gain audit for data-dependent common-G gates.

This diagnostic does not change the original estimator, raw plugin-KL detector,
attack, gain table, or watermarks. The model and arrays must come from the
frozen first-paper repository.

Run without --execute for a dependency-free synthetic proof-of-implementation
check. Use --execute only in the full paperlast repository.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent
THETAS = {"iid": 0.0, "negative": -1.315236976827179, "positive": 1.315236976827179}


def summary(v):
    a = np.asarray(v, dtype=float)
    n = len(a)
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return {"mean": float(a.mean()), "path_se": se, "n_paths": n}


def audit_arrays(*, healthy_error, attacked_error, selection, attack_forcing,
                 paired_bias, F_lookup, masks, tolerance=1e-8):
    """Audit exact recursion and l2 bound; returns per-path and summary values.

    Shapes: error/forcing/bias [B,T,N,2]; F_lookup [n_masks,N,2,2];
    masks [B,T]. No statistical independence is assumed by the theorem.
    """
    e0 = np.asarray(healthy_error, float)
    ea = np.asarray(attacked_error, float)
    s = np.asarray(selection, float)
    a = np.asarray(attack_forcing, float)
    b = np.asarray(paired_bias, float)
    F = np.asarray(F_lookup, float)
    masks = np.asarray(masks, int)
    if e0.ndim != 4 or e0.shape != ea.shape or e0.shape != s.shape or e0.shape != a.shape or e0.shape != b.shape:
        raise ValueError("All trajectories must have identical [B,T,N,d] shapes")
    B, T, N, d = b.shape
    if F.ndim != 4 or F.shape[1:] != (N,d,d) or masks.shape != (B,T):
        raise ValueError("F_lookup or masks have incompatible dimensions")
    c = float(np.linalg.svd(F.reshape(-1,d,d), compute_uv=False)[:,0].max())
    if not c < 1:
        raise ValueError(f"Cannot apply contraction theorem: max norm is {c}")
    u = s+a
    if np.max(np.abs(b[:,0])) > tolerance or np.max(np.abs(u[:,0])) > tolerance:
        raise AssertionError("The zero-initial-bias / zero-initial-input theorem assumptions fail")
    Fb = np.einsum('btnij,btnj->btni',F[masks[:,1:]],b[:,:-1])
    rec_res = b[:,1:] - Fb - u[:,1:]
    rec_max = float(np.max(np.abs(rec_res)))
    if rec_max > tolerance:
        raise AssertionError(f"Paired bias recurrence residual {rec_max} exceeds tolerance")
    paired_res = float(np.max(np.abs((ea-e0)-b)))
    if paired_res > tolerance:
        raise AssertionError(f"Attack vs healthy pair does not match b, max residual {paired_res}")
    sq = lambda x: np.sum(x*x, axis=-1).mean(axis=(1,2))
    J0, Ja, Jb = sq(e0), sq(ea), sq(b)
    Us, Ua, Ucross, U = sq(s), sq(a),2*np.sum(s*a,axis=-1).mean(axis=(1,2)),sq(u)
    if not np.allclose(Us+Ua+Ucross,U,rtol=1e-10,atol=1e-10):
        raise AssertionError("Selection/attack forcing energy decomposition failed")
    bound_b = U/(1-c)**2
    bound_j = (np.sqrt(J0)+np.sqrt(bound_b))**2
    if np.max(Jb-bound_b) > tolerance*max(1.,float(bound_b.max())):
        raise AssertionError("Pathwise l2 contraction bound violated")
    if np.max(Ja-bound_j) > tolerance*max(1.,float(bound_j.max())):
        raise AssertionError("Pathwise final MSE upper bound violated")
    return {
        "c_spectral_max":c, "exact_recurrence_max_abs":rec_max,
        "paired_identity_max_abs":paired_res,
        "J_healthy":summary(J0), "J_attacked":summary(Ja),
        "J_pair_bias":summary(Jb), "J_pair_bias_bound":summary(bound_b),
        "J_attacked_bound":summary(bound_j),
        "U_selection":summary(Us), "U_attack_forcing":summary(Ua),
        "U_selection_attack_cross":summary(Ucross), "U_total":summary(U),
        "bound_to_attacked_mse_ratio":float(bound_j.mean()/Ja.mean()),
        "path_arrays": {"J_healthy":J0,"J_attacked":Ja,"J_pair_bias":Jb,
                        "U_selection":Us,"U_attack_forcing":Ua,"U_cross":Ucross,
                        "U_total":U,"bound_pair_bias":bound_b,"bound_attacked":bound_j}
    }


def self_test():
    rng=np.random.default_rng(2026112207)
    B,T,N,d=8,50,3,2
    # Each F has operator norm <= .08. Modes switch in a data-dependent way.
    lookup=rng.normal(size=(4,N,d,d))
    sig=np.linalg.svd(lookup.reshape(-1,d,d),compute_uv=False)[:,0].max()
    lookup *= .08/sig
    masks=rng.integers(0,4,size=(B,T))
    s=rng.normal(size=(B,T,N,d))*.10
    a=rng.normal(size=(B,T,N,d))*.05
    s[:,0]=0; a[:,0]=0
    b=np.zeros_like(s)
    for k in range(1,T):
        b[:,k]=np.einsum('bnij,bnj->bni',lookup[masks[:,k]],b[:,k-1])+s[:,k]+a[:,k]
    e0=rng.normal(size=s.shape)*.2
    ea=e0+b
    r=audit_arrays(healthy_error=e0,attacked_error=ea,selection=s,attack_forcing=a,
                   paired_bias=b,F_lookup=lookup,masks=masks)
    assert r["exact_recurrence_max_abs"]<1e-13
    assert r["J_pair_bias"]["mean"] <= r["J_pair_bias_bound"]["mean"]
    print(json.dumps({"status":"PASS", "synthetic_test_only":True,
                      "max_c":r["c_spectral_max"],
                      "recurrence_residual":r["exact_recurrence_max_abs"]},indent=2))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_physical(n_cal=512,n_paths=64,seed=2026112411):
    for folder in ["paper1_common_g50_gain_acceptance_20261121",
                   "paper1_50node_tv_gate_theory_20261121"]:
        d=ROOT/folder
        if not d.is_dir():
            raise FileNotFoundError(f"Missing frozen GitHub source directory: {d}")
        sys.path.insert(0,str(d))
    from run_common_g_50 import load_common_g, physical_paths, run_common_g,paired_identity
    from replay import attack_watermarks
    from calibration_rule import path_fwer_cutoff

    out={"kind":"paper1_commonG_final_risk_bound_audit",
         "note":"Independent pilot for source-frozen 50-node common-G; not a final efficacy study",
         "n_calibration":n_cal,"n_paths":n_paths,"seed":seed,
         "models":{}, "hashes":{}}
    d=ROOT/"paper1_common_g50_gain_acceptance_20261121"
    for filename in ["gain_table.json","physical_configuration.json","run_common_g_50.py"]:
        out["hashes"][filename]=sha256(d/filename)
    out["hashes"]["replay.py"]=sha256(ROOT/"paper1_50node_tv_gate_theory_20261121"/"replay.py")
    paths_to_save={}
    for j,label in enumerate(["unstable","stable"]):
        m=load_common_g(label)
        cal_physical=physical_paths(m,n_cal,seed+100_000*j)
        cal_world=run_common_g(m,cal_physical,gate_on=False)
        maxscore=cal_world.scores[:,4:].max(axis=(1,2))
        cutoff=path_fwer_cutoff(maxscore,.05)
        physical=physical_paths(m,n_paths,seed+100_000*j+1)
        healthy=run_common_g(m,physical,gate_on=True,threshold=cutoff["threshold"])
        e0=healthy.post-physical.x[:,:,None,:]
        result={"threshold":float(cutoff["threshold"]),"policies":{}}
        for pi,(name,theta) in enumerate(THETAS.items()):
            delta,_=attack_watermarks(m,n_paths,theta,seed+100_000*j+20_000,
                                       seed+100_000*j+30_000,
                                       seed+100_000*j+40_000+100*pi)
            attacked=run_common_g(m,physical,delta,gate_on=True,threshold=cutoff["threshold"])
            identity=paired_identity(m,physical,healthy,attacked,delta)
            res=audit_arrays(healthy_error=e0,
                             attacked_error=attacked.post-physical.x[:,:,None,:],
                             selection=identity.mismatch,attack_forcing=identity.forcing,
                             paired_bias=identity.b,F_lookup=m.F_lookup,masks=attacked.masks)
            for key,value in res.pop("path_arrays").items():
                paths_to_save[f"{label}_{name}_{key}"]=value
            result["policies"][name]=res
        out["models"][label]=result
    dst=THIS/"gate_risk_bound_results.json"
    dst.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    np.savez_compressed(THIS/"gate_risk_bound_paths.npz",**paths_to_save)
    print(json.dumps({"status":"PASS","models":list(out["models"]),
                      "results":str(dst),"archive":"gate_risk_bound_paths.npz"},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--calibration',type=int,default=512)
    ap.add_argument('--paths',type=int,default=64)
    ap.add_argument('--seed',type=int,default=2026112411)
    args=ap.parse_args()
    if args.execute:
        if args.calibration<100 or args.paths<10: raise ValueError("insufficient paths")
        run_physical(args.calibration,args.paths,args.seed)
    else:
        self_test()
