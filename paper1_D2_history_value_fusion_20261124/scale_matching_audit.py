"""A-prime offline replay, exclusively from the frozen A cache and fitted tables."""
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from analyze_audit import Data, argmin, classes
from audit_replay import D1, HERE, HISTORIES, OUT, Q, healthy, matrices, modes, mv
from prepare_audit import check_provenance

BASE = "fde98e086b2d722b2b0315ecf784ffa7350310aa"
DEST = HERE / "experiment_A_prime"
NAMES = ("R1", "R60", "RH1-SM", "RH60", "RH1-SM-V", "RH60-V")
LABEL = np.array([0, 1, 0, 1, 0, 1])
# Each pair is baseline, alternative. All differences are alternative minus baseline.
PAIRS = ((2, 3), (4, 5), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5))
FIELDS = ("disagree", "sum_delta_H", "H_lower", "H_same", "H_higher")


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf8", newline="\n")


def scale(tables):
    cells = []
    for row in tables:
        u = np.asarray(row["table"]["upper"])[:9]
        spans = np.ptp(u, axis=2)
        assert (spans > 0).all()
        ratios = spans[:, 0] / spans[:, 1]
        cells.append({"scene": row["scene"], "spec": row["spec"], "cell_ratios": ratios.tolist(),
                      "median": float(np.median(ratios))})
    medians = np.array([r["median"] for r in cells]).reshape(6, 2)
    config_gm = np.exp(np.log(medians).mean(1))
    raw = float(np.exp(np.log(config_gm).mean()))
    c = round(raw, 1)
    if c != 55.2:
        raise ValueError(f"Scale differs from the frozen protocol: {raw}")
    return dict(cells=cells, physical_gm=config_gm.tolist(), raw=raw, applied=c)


def frozen_a_hashes():
    # Check the actual Git objects, independent of CRLF conventions in this checkout.
    if subprocess.check_output(["git","diff","--name-only",BASE,"--","experiment_A", "A_PROTOCOL.md",
                                "A_RESULTS.md","analyze_audit.py","audit_graph.py","audit_replay.py",
                                "prepare_audit.py","verify_audit.py","summarize_audit.py"],cwd=HERE):
        raise ValueError("Historical A inputs differ from the frozen commit.")
    files = subprocess.check_output(["git","ls-tree","-r","-z",BASE,"--","."],cwd=HERE)
    result = {}
    for entry in files.split(b"\0"):
        if entry:
            header,name=entry.split(b"\t",1)
            blob=header.split()[-1].decode()
            data=subprocess.check_output(["git","cat-file","blob",blob],cwd=HERE)
            result[name.decode()]={"blob":blob,"sha256":hashlib.sha256(data).hexdigest()}
    return result


def check_mask(scene, spec, source, fixture, sl, g):
    dyn,_,_,wm=spec
    if source=="D0":
        p=D1.parent/"paper1_history_ablation_20261123/results/D0_locked/paths.npz"
        key=f"{dyn}__{fixture}__{wm}__gates"
    elif source=="B1":
        p=D1/"b1_health_results"/f"scene_{scene:02}.npz";key="masks"
    else:
        p=D1/"b2_attack_results"/f"scene_{scene:02}_{fixture}_{wm}.npz";key="masks"
    with np.load(p) as z:
        np.testing.assert_array_equal(g,(z[key][sl] & 64)!=0)


def replay(data, sl, table, m, legal, phi, c):
    x,local,other,sender,delta,score = (data[k][sl] for k in ("x","local","other","sender","delta","score"))
    g_all=score>1.
    n=len(x)
    state=np.zeros((n,6,4))
    cov=np.broadcast_to(m.P,(n,6,4,4)).copy()
    hist=np.full((n,6),HISTORIES-1)
    run=np.zeros(n,int)
    loss=np.zeros((n,6,3))
    count=np.zeros((n,2),np.int32)  # alarms, eligible
    records=np.zeros((n,len(PAIRS),len(FIELDS)))
    cuts=np.array(table["boundaries"]); lookup=np.array(table["cells"]); u=np.array(table["upper"])
    weights=np.array([0.,0.,c,1.,c*phi,phi])
    adj=m.F.transpose(0,2,1) @ m.gram @ m.F
    offset=np.einsum("ij,qji->q",m.gram,m.O)
    reference_disagreements=0
    for k in range(1,150):
        g=g_all[:,k]
        run=(run+1)*g
        a=np.full((n,6),6,int)
        if g.any():
            box=lookup[classes(score[g,k],run[g],cuts)[0]]
            risks=u[box][:,LABEL,:]
            h=np.einsum("qij,bpji->bpq",adj,cov[g])+offset
            h=h-h[:,:,-1:]
            a[g]=argmin(risks+weights[:,None]*h,legal[hist[g]])
            if 50<=k<91:
                allowed=legal[hist[g,0]]
                use=allowed.sum(1)>=2
                count[g,0]+=1
                count[g,1]+=use
                h_r=h[:,0,:]
                common=argmin(risks+weights[:,None]*h_r[:,None,:],allowed[:,None,:])
                np.testing.assert_array_equal(common[:,0],a[g,0])
                rr=np.arange(len(h_r))
                for p,(i,j) in enumerate(PAIRS):
                    flip=use & (common[:,j]!=common[:,i])
                    diff=h_r[rr,common[:,j]]-h_r[rr,common[:,i]]
                    records[g,p,0]+=flip
                    records[g,p,1]+=np.where(flip,diff,0)
                    records[g,p,2]+=flip & (diff < -1e-10)
                    records[g,p,3]+=flip & (np.abs(diff) <= 1e-10)
                    records[g,p,4]+=flip & (diff > 1e-10)
        assert legal[hist,a].all()
        reference_disagreements+=int((a[:,0]!=a[:,1]).sum())
        hist=(7*hist+a)%HISTORIES
        f=m.F[a]
        packet=other[:,k,None]+Q[a][...,None]*(sender[:,k,None]+delta[:,k,None])
        state=mv(f,state)+mv(m.K[a],local[:,k,None])+mv(m.G[a],packet)
        cov=f@cov@f.swapaxes(-1,-2)+m.O[a]
        error=((state[:,:,2:]-x[:,k,None,2:])**2).sum(2)
        if 50<=k<110:loss[:,:,0]+=error/60
        if k>=110:loss[:,:,1]+=error/40
        if k>=20:loss[:,:,2]+=error/130
    np.testing.assert_array_equal(records[:,:,0],records[:,:,2:].sum(2))
    return loss,count,records,reference_disagreements


def main():
    if DEST.exists():raise FileExistsError(DEST)
    frozen=frozen_a_hashes()
    check_provenance()
    rows=[json.loads((OUT/f"decisions_{i:02}.json").read_text()) for i in range(12)]
    scales=scale(rows)
    audit=dict(frozen_A_commit=BASE, frozen_A_inputs=frozen, cached_inputs_sha256={},
               labels=LABEL.tolist(), policy_order=NAMES, pairs=PAIRS, action_fields=FIELDS,
               periods=["window","tail","healthy_post20"], scale=scales,
               checks={}, scope="Exploratory supplement conditional on frozen A fit")
    with TemporaryDirectory(dir=HERE) as tmp, np.load(modes.OUTPUT/"mode_tables.npz") as gains:
        target=Path(tmp)/"result";target.mkdir()
        for scene,(spec,row) in enumerate(zip(healthy.scenarios(),rows,strict=True)):
            assert list(spec)==row["spec"] and scene==row["scene"]
            dyn,top,sender,wm=spec
            model=healthy.setup(spec,gains);m=matrices(gains,spec,model)
            with np.load(OUT/"graphs"/f"{top}_{dyn}_target{sender}_qwords.npz") as z:
                legal=z["legal"]
            with np.load(OUT/f"path_losses_{scene:02}.npz") as old:
                for key in row["conditions"]:
                    source,fixture=key.split("_")
                    sl=slice(None) if source=="D0" else slice(1024,None)
                    data=Data(scene,source,fixture)
                    name=f"{scene:02}_{key}"
                    for field in ("x","local","other","sender","delta","score"):
                        p=data.folder/(field+".npy")
                        audit["cached_inputs_sha256"][p.parent.name+"/"+p.name]=digest(p)
                    check_mask(scene,spec,source,fixture,sl,data["score"][sl]>1.)
                    loss,n,actions,r_diff=replay(data,sl,row["table"],m,legal,row["phi"],scales["applied"])
                    expected=old[key][:,[0,3,4,5],1:]
                    current=loss[:,[0,1,3,5],:]
                    np.testing.assert_allclose(current,expected,atol=2e-9,rtol=2e-11)
                    # Previous A measured RH60 -> RH60-V on the R60 reference.
                    # Compare with its counts and health contrasts if R1 and R60 matched.
                    if source!="B1" and r_diff==0:
                        a=row["conditions"][key]["common_R_history"]["1_1_2"]
                        assert n[:,0].sum()==a["alarm"] and n[:,1].sum()==a["eligible"]
                        assert actions[:,-1,0].sum()==a["flips"]
                        np.testing.assert_allclose(actions[:,-1,1].sum(),a["sums_Y1_Y60_observed_H"][-1],
                                                   rtol=1e-12,atol=1e-9)
                    entries=dict(loss=loss,counts=n,action=actions)
                    if source=="B1":entries["q1_reference"]=old["B1_reference"]
                    np.savez_compressed(target/(name+".npz"),**entries)
                    audit["checks"][name]=dict(path_count=len(n),max_A_loss_difference=float(np.abs(current-expected).max()),
                                               R1_R60_own_history_action_differences=r_diff,archived_target_bits_equal=True)
            print("finished scene",scene,flush=True)
        audit["path_outputs_sha256"]={p.name:digest(p) for p in sorted(target.glob("*.npz"))}
        write_json(target/"manifest.json",audit)
        target.rename(DEST)


if __name__=="__main__":
    main()
