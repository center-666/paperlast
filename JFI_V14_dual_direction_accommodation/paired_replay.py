"""Closed-loop replay and metrics. Only the evaluator reads offline truth."""
import numpy as np
from scipy.stats import chi2

from dual_receiver import (DESIGN, METHODS, RAYS, DirectionLedger, correction,
                           trusted_center)
from batched_observer import BatchedObserver
from replay import Observations, first, interval, paired_risk
from trusted_prediction import TrustedConditioning


def scan_gls(frame, p, k):
    """Existing full-rank GLS reconstruction with the frozen active-window gate."""
    begin = max(0,k+1-p.settings.template_duration)
    posterior = frame.inverse_precision[:,begin:]
    beta = frame.posterior_coef[:,begin:]
    eta = np.linalg.solve(np.eye(p.n)-posterior,beta[...,None])[...,0]
    score = np.sum(np.linalg.solve(posterior,beta[...,None])[...,0]*eta,axis=-1)
    selected = score.argmax(axis=1)
    rows = np.arange(len(selected))
    triggered = chi2.sf(score[rows,selected],p.n)<=p.settings.event_false_alarm/p.active_test_count
    delta = (frame.transfer[rows,selected+begin] @ eta[rows,selected,:,None])[...,0]
    return delta*triggered[:,None]


def play(p, obs):
    s, model = p.settings, p.model
    F,G,K = model.F(s.receiver),model.G[s.receiver],model.K[s.receiver]
    T, n, m, count, methods = s.horizon,p.n,p.m,len(obs),len(METHODS)
    if np.linalg.matrix_rank(model.H[s.target])!=n:
        raise ValueError("The numerical GLS baseline uses a full-column-rank current packet.")
    observer, trusted = BatchedObserver(p),TrustedConditioning(p)
    cutoff = chi2.ppf(DESIGN["trusted_score_level"], m)
    caps = {
        "V1.2a": DirectionLedger(p,count,[[1.,0.]]),
        "trusted-budget": DirectionLedger(p,count,[[0.,1.]]),
        "V1.4": DirectionLedger(p,count,RAYS)
    }
    book = ("proposed_scale","cap","available","charge","remainder")
    ledger = {key:np.zeros((count,methods,T)) for key in book}
    post = np.zeros((count,methods,T,n))
    change = np.zeros_like(post)
    relative_parts = np.zeros((count,2,T,n))
    weights = np.zeros((count,methods,T,2))
    evidence = np.zeros((count,T,m))
    base_directions = np.zeros((count,T,2,n))
    log_bf = np.zeros((count,T))
    score = np.zeros((count,T))
    raw_score = np.zeros_like(score)
    current = np.zeros((count,methods,n))
    for k in range(T):
        ev = observer.at(obs,k,risk=True)
        pred = trusted.at(obs.trusted,k)
        deviation = obs.target[:,k]-pred.mean_y
        score[:,k] = trusted.score(deviation-ev.delta,k)
        raw_score[:,k] = trusted.score(deviation,k)
        loss_center, trust = trusted_center(score[:,k],ev.mean_working,pred.mean_x,cutoff)
        allowed = (RAYS[:,1]==0)[None] | (trust>0)[:,None]
        d = np.stack((ev.delta,deviation),axis=1) @ G.T
        evidence[:,k] = ev.delta
        base_directions[:,k] = d
        log_bf[:,k] = ev.log_bf
        local, neighbor = obs.trusted[:,k,:m], obs.trusted[:,k,m:]
        raw_update = local @ K.T + (neighbor+obs.target[:,k]) @ G.T
        u = current @ F.T+raw_update[:,None]
        for j,method in enumerate(METHODS):
            w = weights[:,j,k]
            if method in caps:
                center = {"V1.2a":ev.mean_working,"trusted-budget":pred.mean_x,"V1.4":loss_center}[method]
                action = caps[method].decide(k,u[:,j],center,ev.mean_null,d,allowed if method=="V1.4" else None)
                w[:] = action.weights
                for field in book:
                    ledger[field][:,j,k] = getattr(action,field)
            elif method=="V1a":
                w[:,0] = 1.
            elif method=="trusted-replacement":
                w[:,1] = 1.
            elif method=="innovation-clipping":
                w[:,1] = 1.-np.sqrt(cutoff/np.maximum(cutoff,raw_score[:,k]))
            corr = correction(d,w)
            if method=="GLS":
                corr = scan_gls(ev,p,k) @ G.T
            elif method=="trusted-MMSE":
                corr = u[:,j]-pred.mean_x
            elif method=="V1.4":
                relative_parts[:,:,k] = d*w[...,None]
            change[:,j,k] = corr
            current[:,j] = u[:,j]-corr
        post[:,:,k] = current
    return dict(post=post,change=change,parts=relative_parts,weights=weights,delta=evidence,
                directions=base_directions,log_bf=log_bf,score=score,raw_score=raw_score,**ledger)


def collect(p, streams, batch=64):
    outputs = [play(p,Observations(streams.trusted[i:i+batch],streams.target[i:i+batch],
                                  streams.decoder[i:i+batch])) for i in range(0,len(streams),batch)]
    return {key:np.concatenate([a[key] for a in outputs]) for key in outputs[0]}


def masked_average(values, mask):
    return np.sum(values*mask[:,None],axis=2)/np.maximum(mask.sum(axis=1)[:,None],1)


def evaluate(p, stream, reference, sample, D):
    """Compute events from the two actual recursions, with one shared raw score."""
    s, m, model = p.settings, p.m, p.model
    F,G = model.F(s.receiver),model.G[s.receiver]
    posts, clean = sample["post"],reference["post"]
    b = posts-clean
    prior = np.concatenate((np.zeros_like(b[:,:,:1]),b[:,:,:-1]),axis=2)
    raw = prior @ F.T + (stream.offset @ G.T)[:,None]
    change = sample["change"]-reference["change"]
    drift_error = float(np.max(np.abs(b-(raw-change))))
    np.testing.assert_allclose(b,raw-change,atol=7e-12,rtol=6e-11)
    e0 = clean-stream.state[:,None]
    ea = posts-stream.state[:,None]
    b2 = np.sum(b*b,axis=-1)
    health = np.sum(e0*e0,axis=-1)
    error = np.sum(ea*ea,axis=-1)
    np.testing.assert_allclose(error,health+b2+2*np.sum(e0*b,axis=-1),atol=3e-11,rtol=6e-11)
    T = s.horizon
    alarm = first(sample["log_bf"]>=-np.log(s.event_false_alarm))
    ticks = np.arange(T)[None]
    before, packet = ticks<stream.start[:,None],ticks==stream.start[:,None]
    active = (ticks>=stream.start[:,None]) & (ticks<stream.stop[:,None])
    tail = ticks>=stream.stop[:,None]
    assert float(np.max(b2*before[:,None]))<1e-22
    danger = first(b2>=D)
    risk = danger<alarm[:,None]
    first_hits = np.any((b2>=D)&packet[:,None],axis=2)
    first_eligible = packet & (ticks<alarm[:,None])
    first_pre = np.any((b2>=D)&first_eligible[:,None],axis=2)
    window = masked_average(error,active)
    after = masked_average(error,tail)
    path_health = health.mean(axis=2)
    j0 = METHODS.index("V0")
    j1 = METHODS.index("V1a")
    j2 = METHODS.index("V1.2a")
    rows = {}
    for j,name in enumerate(METHODS):
        weights = sample["weights"][:,j]
        healthy_wt = reference["weights"][:,j,:,1]>1e-10
        cut = sample["cap"][:,j]+1e-10<sample["proposed_scale"][:,j]
        phases = {}
        for label,mask in (("before",before),("first",packet),("active_later",active&~packet),("after",tail)):
            if mask.any():
                phases[label] = {
                    "n":int(mask.sum()),
                    "weights_mean":weights[mask].mean(axis=0).tolist(),
                    "weights_p10_p50_p90":np.quantile(weights[mask],[.1,.5,.9],axis=0).tolist(),
                    "trusted_execution_rate":float(np.mean(weights[mask,1]>1e-10)),
                    "selected_ray_cap_incidence":float(cut[mask].mean()) if name in
                        ("V1.2a","trusted-budget","V1.4") else None,
                    "mse":float(error[:,j][mask].mean()),
                    "relative_displacement_squared":float(b2[:,j][mask].mean())
                }
        rows[name] = {
            "health_mse":interval(path_health[:,j]),
            "health_relative_vs_V0":float(path_health[:,j].mean()/path_health[:,j0].mean()-1.),
            "health_MSE_difference_vs_V0":interval(path_health[:,j]-path_health[:,j0]),
            "healthy_trusted_execution_rate":float(healthy_wt.mean()),
            "active_mse":interval(window[:,j]) if active.any() else None,
            "active_MSE_difference_vs_V1a":interval(window[:,j]-window[:,j1]) if active.any() else None,
            "tail_mse":interval(after[:,j]) if tail.any() else None,
            "tail_any_danger":int(np.sum(np.any((b2[:,j]>=D)&tail,axis=1))),
            "average_peak_bias_squared":float(b2[:,j].max(axis=1).mean()),
            "first_packet_danger":int(first_hits[:,j].sum()),
            "first_packet_strictly_pre_alarm_danger":int(first_pre[:,j].sum()),
            "pre_alarm_danger":int(risk[:,j].sum()),
            "paired_pre_alarm_vs_V1a":paired_risk(risk[:,j],risk[:,j1]),
            "paired_pre_alarm_vs_V12a":paired_risk(risk[:,j],risk[:,j2]),
            "same_raw_alarm_by_horizon":int(np.sum(alarm<T)),
            "phases":phases
        }
    j = METHODS.index("V1.4")
    pair_parts = sample["parts"]-reference["parts"]
    np.testing.assert_allclose(b[:,j],raw[:,j]-pair_parts.sum(axis=1),atol=7e-12)
    v = raw[:,j]
    v_length = np.linalg.norm(v,axis=-1)
    dots = np.sum(pair_parts*v[:,None],axis=-1)
    projections = np.divide(dots,v_length[:,None],out=np.zeros_like(dots),where=v_length[:,None]>1e-12)
    V14_phases = rows["V1.4"]["phases"]
    for label,mask in (("before",before),("first",packet),("active_later",active&~packet),("after",tail)):
        if mask.any():
            V14_phases[label]["relative_S_T_projection_mean"] = projections.transpose(0,2,1)[mask].mean(axis=0).tolist()
            V14_phases[label]["relative_projection_T_negative_fraction"] = float(np.mean(projections[:,1][mask]<-1e-12))
    cutoff = chi2.ppf(DESIGN["trusted_score_level"],m)
    h_exec = reference["weights"][:,j,:,1]>1e-10
    clean_r = reference["score"]>cutoff
    c_s = reference["parts"][:,0]
    c_t = reference["parts"][:,1]
    # Offline only: remove T from this update without changing the healthy history or S.
    e_without_t = e0[:,j]+c_t
    inst = health[:,j]-np.sum(e_without_t*e_without_t,axis=-1)
    healthy_diagnostic = {
        "post_structural_score_gate_rate":float(clean_r.mean()),
        "trusted_execution_count":int(h_exec.sum()),
        "healthy_updates":int(h_exec.size),
        "mean_actual_same_history_increment_when_executed":float(inst[h_exec].mean()) if h_exec.any() else None,
        "mean_structural_correction_squared_when_T_executed":float(np.sum(c_s*c_s,axis=-1)[h_exec].mean())
            if h_exec.any() else None
    }
    details = {
        "weights":sample["weights"],"health_mse":path_health,"active_mse":window,"tail_mse":after,
        "first_alarm":alarm,"first_danger":danger,"risk":risk,"first_packet":first_hits,
        "first_packet_pre":first_pre,"b2":b2,"k_start":stream.start,"k_stop":stream.stop,
        "score_post_S":sample["score"],"score_raw":sample["raw_score"],"log_bf":sample["log_bf"],
        "S_T_relative_projections":projections,
    }
    for f in ("proposed_scale","cap","available","charge","remainder"):
        details[f] = sample[f][:,[METHODS.index(k) for k in ("V1.2a","trusted-budget","V1.4")]]
    return {"methods":rows,"healthy_trust":healthy_diagnostic,"paths":len(stream),
            "pair_max_abs":drift_error},details
