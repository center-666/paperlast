"""Offline reconstruction and paired counterfactuals; D1 is a read-only dependency."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from audit_graph import Q, HISTORIES

HERE = Path(__file__).resolve().parent
D1 = Path(os.environ["D2_D1_DIR"]).resolve()
sys.path.insert(0, str(D1))
import b1_health as healthy
import b1_mode_models as modes
import b2_attack as b2
import locked_weight_check as d0

CURVE = D1 / "b2_weight_response_results"
OUT = HERE / "experiment_A"
CACHE = Path(os.environ["D2_A_CACHE"])
W = np.diag([0., 0., 1., 1.])
REF = 4
WINDOW = slice(50, 110)
COMMON = slice(50, 91)


def mv(m, x):
    return (m @ x[..., None])[..., 0]


def matrices(gains, spec, model):
    dyn, top, sender, _ = spec
    base = f"{top}_{dyn}_target{sender}_q1"
    prior = gains[base + "_Pplus"][0, 26]
    F, K, G, omega = (model.target[name][1] for name in ("F", "K", "G", "Omega"))
    powers = np.array([np.linalg.matrix_power(F[-1], i) for i in range(60)])
    gram = np.mean(powers.transpose(0, 2, 1) @ W @ powers, axis=0)
    return SimpleNamespace(F=F, K=K, G=G, O=omega, P=prior, powers=powers, gram=gram)


def read_paths(scene, spec, model, fixture, source):
    dyn, _, sender, wm = spec
    fi = int(fixture == "burst")
    wi = int(wm == "iid")
    if source == "B2":
        name = f"scene_{scene:02}_{fixture}_{wm}.npz"
        with np.load(b2.OUT / name) as z, np.load(CURVE / name) as curve:
            masks, threshold = z["masks"], float(z["threshold"])
            np.testing.assert_array_equal(masks, curve["masks"])
            check = curve["missing_window_mse"]
        size = 64
    elif source == "D0":
        reg = json.loads((d0.OUTPUT / "analysis.json").read_text())
        threshold = reg["models"][dyn]["threshold"]
        prefix = f"{dyn}__{fixture}__{wm}__"
        with np.load(d0.OUTPUT / "paths.npz") as z:
            masks = z[prefix+"gates"]
            check = np.stack([z[prefix+name+"__missing_fault"]
                              for name in ("hard", "locked", "accept")], axis=1)
        size = d0.BATCH
    else:
        with np.load(healthy.OUT / f"scene_{scene:02}.npz") as z:
            masks, threshold, check = z["masks"], float(z["threshold"]), z["missing"]
        size = 64
    neighbors = [j for j in model.neighbors[26] if j != sender]
    for batch in range(len(masks)//size):
        if source == "B2":
            paths = healthy.physical(model, size, b2.seed(scene, fi, wi, 2, batch))
            delta = b2.attack_delta(size, model, wm, fixture, scene, fi, wi, batch)
        elif source == "D0":
            mi = int(dyn == "stable")
            paths = healthy.physical(model, size, d0.seed(mi, fi+2, batch, 0))
            draws = d0.d0.make_common_draws(size, 60, 1, *[d0.seed(mi, fi+2, batch, j) for j in (1, 2, 3)])
            active = np.zeros(150, bool)
            active[WINDOW] = b2.active_attack_ticks(fixture)
            delta = d0.make_fault(draws, active, d0.d0.THETAS[wm])
        else:
            paths = healthy.physical(model, size, healthy.seed(scene, 1, batch))
            delta = np.zeros((size, 150, 7, 2))
        residual = healthy.residuals(model, paths, delta)
        scores = np.zeros((size, 150, 7))
        scores[:, 5:] = .5*np.einsum("btei,eij,btej->bte", residual[:, 1:], model.precision,
                                     residual[:, 1:], optimize=True)
        sl = slice(size*batch,size*(batch+1))
        np.testing.assert_array_equal(((scores > threshold) @ b2.BITS).astype(np.uint8), masks[sl])
        yield SimpleNamespace(x=paths.x, local=paths.y[:,:,26], sender=paths.y[:,:,sender],
                              other=paths.y[:,:,neighbors].sum(2), delta=delta[:,:,6],
                              s=scores[:,:,6]/threshold, alarm=(masks[sl] & 64)!=0,
                              check=check[sl])


def offsets(p, m):
    # Clean affine part for each action, with true state used only by the evaluator.
    return (np.einsum("qij,btj->btqi",m.K,p.local)
            + np.einsum("qij,btqj->btqi",m.G,p.other[:,:,None]+Q[None,None,:,None]*p.sender[:,:,None]))


def stage_labels(p, m, legal, source, crosscheck):
    n = len(p.x)
    clean_input = offsets(p,m)
    direct = np.einsum("qij,btj->btqi",m.G*Q[:,None,None],p.delta)
    all_input = clean_input + direct
    clean_base = np.zeros_like(p.x)
    for k in range(1,150):
        clean_base[:,k] = mv(m.F[-1],clean_base[:,k-1])+clean_input[:,k,-1]
    # Common clean continuation: error(q,h)=base_error(k+h)+F1^h*(candidate_clean-base(k)).
    clean_error = clean_base-p.x
    projected = np.zeros((n,41,4))
    for h,power in enumerate(m.powers):
        projected += clean_error[:,50+h:91+h] @ W @ power / 60.
    state = np.zeros((n,7,4))
    covariance = np.broadcast_to(m.P,(n,4,4)).copy()
    hist = np.full(n,HISTORIES-1)
    run = np.zeros((n,150),int)
    arrays = {name:np.empty((n,60,7)) for name in ("Y1","H","loss1","actual")}
    arrays["Y60"] = np.empty((n,41,7))
    arrays["mask"] = np.empty((n,60),np.uint8)
    check_sum = np.zeros((n,7))
    row = np.arange(n)
    for k in range(1,150):
        alarm = p.alarm[:,k]
        run[:,k] = (run[:,k-1]+1)*alarm
        u = np.where(alarm[:,None],np.arange(7),6)
        if 50 <= k <110:
            t = k-50
            baseline = np.einsum("qij,bj->bqi",m.F,state[:,REF])+clean_input[:,k]
            d = direct[:,k]
            e = baseline-p.x[:,k,None]
            y1 = ((2*e[:,:,2:]+d[:,:,2:])*d[:,:,2:]).sum(2)
            arrays["Y1"][:,t] = y1
            arrays["loss1"][:,t] = ((e+d)[:,:,2:]**2).sum(2)
            pred_cov = m.F @ covariance[:,None] @ m.F.transpose(0,2,1) + m.O
            cost = np.einsum("ij,bqji->bq",m.gram,pred_cov)
            arrays["H"][:,t] = cost-cost[:,[-1]]
            arrays["mask"][:,t] = legal[hist] @ (1 << np.arange(7))
            if t<41:
                diff = baseline-clean_base[:,k,None]
                arrays["Y60"][:,t] = (2*np.einsum("bi,bqi->bq",projected[:,t],d)
                    + np.sum(((2*diff+d)@m.gram)*d,axis=2))
                if crosscheck and t in (0,40):
                    # Direct propagation from two clean/impulsed posteriors using future clean packets.
                    a=baseline[:2].copy()
                    b=a+d[:2]
                    total=np.zeros((len(a),7))
                    for h in range(60):
                        if h:
                            a=a @ m.F[-1].T+clean_input[:2,k+h,None,-1]
                            b=b @ m.F[-1].T+clean_input[:2,k+h,None,-1]
                        xa=p.x[:2,k+h,None,2:]
                        total+=((b[:,:,2:]-xa)**2-(a[:,:,2:]-xa)**2).sum(2)/60
                    np.testing.assert_allclose(total,arrays["Y60"][:2,t],atol=2e-7,rtol=2e-8)
            if crosscheck and t == 1:
                v=pred_cov[0].copy(); total=np.zeros(7)
                for h in range(60):
                    total += np.trace(v[:,2:,2:],axis1=1,axis2=2)/60
                    v=m.F[-1] @ v @ m.F[-1].T + m.O[-1]
                np.testing.assert_allclose(total-total[-1],arrays["H"][0,t],atol=1e-11)
        state=mv(m.F[u],state)+all_input[row[:,None],k,u]
        loss=((state[:,:,2:]-p.x[:,k,None,2:])**2).sum(2)
        if 50<=k<110: arrays["actual"][:,k-50]=loss
        if source=="B1" and k>=20: check_sum+=loss/130
        elif source!="B1" and 50<=k<110: check_sum+=loss/60
        action=u[:,REF]
        covariance=m.F[action] @ covariance @ m.F[action].transpose(0,2,1)+m.O[action]
        assert legal[hist,action].all()
        hist=(hist*7+action)%HISTORIES
    checked=check_sum[:,[0,4,6]] if source=="D0" else check_sum
    np.testing.assert_allclose(checked,p.check,atol=3e-7,rtol=3e-9)
    arrays.update(s=p.s[:,WINDOW],run=run[:,WINDOW],fixed=check_sum)
    return arrays,float(np.abs(checked-p.check).max())


def materialize(scene, spec, model, m, legal, source, fixture):
    out=CACHE/f"{scene}_{source}_{fixture}"
    out.mkdir(parents=True,exist_ok=True)
    # Kept outside the repository. Store only the target-node path inputs and offline audit arrays.
    size=1024 if source=="D0" else 2048
    files={};offset=0;worst=0.
    for p in read_paths(scene,spec,model,fixture,source):
        n=len(p.x)
        data,error=stage_labels(p,m,legal,source,offset==0)
        worst=max(worst,error)
        data.update(x=p.x,local=p.local,other=p.other,sender=p.sender,delta=p.delta,score=p.s)
        if not files:
            for name,values in data.items():
                files[name]=np.lib.format.open_memmap(out/(name+".npy"),mode="w+",dtype=values.dtype,
                                                      shape=(size,*values.shape[1:]))
        for name,values in data.items():files[name][offset:offset+n]=values
        offset+=n
    assert offset==size
    return worst
