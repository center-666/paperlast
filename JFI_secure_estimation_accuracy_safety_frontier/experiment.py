"""Independent healthy calibration and paired receiver-only holdout."""
import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import brentq

from frozen_inputs import (BASELINE_COMMIT, BUDGETS, DEFAULT_BASELINE, HERE, MODELS, MODEL_PATH,
                           V14, TrustedConditioning, digest, fresh_streams, load_json, load_prepared,
                           setup, write_json)
from frontier import angular_excess
from nominal_audit import regression_cache
from receiver import clipping_removal, collect, label, method_table
from replay import first, interval, paired_risk


REG_PATH = HERE/"registration.json"
AUDIT_ROOT = HERE/"results"


def require_frontier(base, registration):
    proof = load_json(AUDIT_ROOT/"frontier.json")
    assert proof["baseline"]==BASELINE_COMMIT==registration["baseline"]
    assert proof["model_sha256"]==digest(base/MODEL_PATH)
    assert len(proof["models"])==2
    for row in proof["models"].values():
        cert = row["exact_certificate"]
        assert cert["stages_with_verified_min_eigenvalue_bound"]==registration["horizon"]==32
        assert cert["strict_margin_rounded_down"]>0
    assert registration["strict_ball_radius"] < proof["diameter_radius_at_equality"]
    assert tuple(registration["target_nominal_health_penalties"])==BUDGETS
    assert registration["D_squared"]==proof["danger_squared"]
    return proof


def input_hashes():
    names = ("frozen_inputs.py","frontier.py","receiver.py","experiment.py","registration.json")
    current = {name:digest(HERE/name) for name in names}
    frozen = load_json(V14/"results/development/summary.json")["source_hashes"]
    for name,sha in frozen.items():
        if digest(V14/name)!=sha:
            raise ValueError(f"Frozen V1.4 source failed verification: {name}")
    return {"candidate":current,"V14_verified":frozen,"frontier":digest(AUDIT_ROOT/"frontier.json")}


def calibration_features(p, streams):
    model,s = p.model,p.settings
    trust = TrustedConditioning(p)
    whites = regression_cache(p)
    null, dT, size = [],[],[]
    for k,prefix in enumerate(p.prefixes):
        muT = trust.at(streams.trusted,k)
        T = streams.trusted[:,:k+1].reshape(len(streams),-1)
        Y = streams.target[:,:k+1].reshape(len(streams),-1)
        v = (Y-T @ prefix.target_from_trusted.T) @ whites[k].T
        null.append(muT.mean_x+v @ prefix.state_from_white.T)
        dv = streams.target[:,k]-muT.mean_y
        dT.append(dv @ model.G[s.receiver].T)
        size.append(np.sqrt(trust.score(dv,k)))
    K,G = model.K[s.receiver],model.G[s.receiver]
    raw = streams.trusted[:,:,:p.m] @ K.T+(streams.trusted[:,:,p.m:]+streams.target) @ G.T
    return raw,np.stack(null,axis=1),np.stack(dT,axis=1),np.stack(size,axis=1)


def fit_clip(p, streams):
    s = p.settings
    F = p.model.F(s.receiver)
    raw,mean0,dT,norm = calibration_features(p,streams)
    mean_base = float(p.nominal_mse().mean())
    def losses(threshold):
        state = np.zeros_like(mean0[:,0])
        total = np.zeros(len(state))
        for k in range(s.horizon):
            state = state @ F.T+raw[:,k]-clipping_removal(norm[:,k],threshold)[:,None]*dT[:,k]
            total += np.sum((state-mean0[:,k])**2,axis=1)
        return total/s.horizon
    baseline = losses(float(norm.max()+1))
    # The full posterior covariance cancels in the paired conditional-MSE contrast.
    def estimate(t):
        return float((losses(t)-baseline).mean()/mean_base)
    result = []
    end = float(norm.max()+1)
    for budget in BUDGETS:
        t = brentq(lambda candidate:estimate(candidate)-budget,0.,end,xtol=2e-10)
        loss = (losses(t)-baseline)/mean_base
        result.append({"target_pct":100*budget,"norm_threshold":float(t),
                       "calibration_pct_raw_and_approx95":[100*x for x in interval(loss)]})
    return result


def register(base, p_settings):
    reg = load_json(REG_PATH)
    assert reg["horizon"]==p_settings.horizon
    assert reg["BF_event_budget"]==p_settings.event_false_alarm
    assert reg["coded_energy_limit"]==p_settings.coded_energy_reference
    seeds = [*reg["calibration"]["seed"].values(),*reg["holdout"]["seed"].values()]
    assert len(seeds)==len(set(seeds))==4
    return reg,require_frontier(base,reg)


def calibrate(base):
    common,settings,spec,_ = setup(base)
    reg,proof = register(base,settings)
    rows = {}
    for model in MODELS:
        p = load_prepared(base,model,common,settings)
        cfg = reg["calibration"]
        streams = fresh_streams(base,p,spec,cfg["seed"][model],cfg["paths_per_model"],healthy_only=True)["healthy"]
        fits = fit_clip(p,streams)
        rows[model] = fits
        print(model,[round(x["norm_threshold"],6) for x in fits],flush=True)
    write_json(AUDIT_ROOT/"calibration_frozen.json",{"input_hashes":input_hashes(),"models":rows,
                "nominal_only_selection":True,"holdout_used_for_selection":False})


def regress_v14(p, stream, methods, actual):
    from paired_replay import collect as frozen_collect, METHODS
    view = stream.take(slice(0,4))
    frozen = frozen_collect(p,view)
    mapping = {"V0":"V0","V1a":"V1a","V1.2a":"V1.2a","V1.4":"V1.4",
               "clip95":"innovation-clipping","trusted-MMSE":"trusted-MMSE",
               label("trusted-ledger",.02):"trusted-budget"}
    names = [a.name for a in methods]
    differences = []
    for new,old in mapping.items():
        a,b = actual.post[:4,names.index(new)], frozen["post"][:,METHODS.index(old)]
        np.testing.assert_allclose(a,b,atol=8e-11,rtol=5e-11)
        differences.append(float(np.max(np.abs(a-b))))
    np.testing.assert_allclose(frozen["log_bf"],actual.log_bf[:4],atol=8e-11,rtol=5e-11)
    return max(differences)


def mask_average(x,mask):
    return (x*mask[:,None]).sum(axis=-1)/np.maximum(1,mask.sum(axis=1))[:,None]


def measure(p, stream, run, ref, methods, threshold, proof):
    T = p.settings.horizon
    F = p.model.F(p.settings.receiver)
    G = p.model.G[p.settings.receiver]
    actual,healthy = run.post,ref.post
    difference = actual-healthy
    previous = np.concatenate((np.zeros_like(difference[:,:,:1]),difference[:,:,:-1]),axis=2)
    balance = previous @ F.T+(stream.offset @ G.T)[:,None]-run.change+ref.change
    np.testing.assert_allclose(difference,balance,atol=9e-12,rtol=7e-11)
    target = stream.state[:,None]
    error2 = np.sum((actual-target)**2,axis=-1)
    healthy2 = np.sum((healthy-target)**2,axis=-1)
    b2 = np.sum(difference*difference,axis=-1)
    np.testing.assert_allclose(error2,healthy2+b2+2*np.sum((healthy-target)*difference,axis=-1),
                               atol=2e-11,rtol=7e-11)
    alarm = first(run.log_bf>=-np.log(p.settings.event_false_alarm))
    cross = b2>=threshold
    dangerous = first(cross)
    prealarm = dangerous<alarm[:,None]
    ticks = np.arange(T)[None,:]
    before = ticks<stream.start[:,None]
    onset = ticks==stream.start[:,None]
    active = (ticks>=stream.start[:,None]) & (ticks<stream.stop[:,None])
    tail = ticks>=stream.stop[:,None]
    assert float((b2*before[:,None]).max())<2e-21
    first_packet = np.any(cross&onset[:,None],axis=2)
    first_pre = first_packet & (stream.start<alarm)[:,None]
    peak = b2.max(axis=2)
    active_loss = mask_average(error2,active)
    tail_loss = mask_average(error2,tail)
    conditional_sq = np.square(healthy-ref.null_mean[:,None]).sum(axis=-1).mean(axis=-1)
    healthy_loss = healthy2.mean(axis=-1)
    h = proof["v0_mse"]
    tr = proof["full_MMSE"]
    names = [a.name for a in methods]
    j0,struct = names.index("V0"),names.index("V1a")
    rows = {}
    for j,a in enumerate(methods):
        row = {"health_actual_mse":interval(healthy_loss[:,j]),
               "health_actual_pct":float(100*(healthy_loss[:,j].mean()/healthy_loss[:,j0].mean()-1)),
               "health_expectation_pct_conditional_MC95":[100*x for x in interval(
                   (conditional_sq[:,j]-conditional_sq[:,j0])/h)],
               "health_expectation_MSE_conditional_MC":float(tr+conditional_sq[:,j].mean()),
               "pre_alarm_risk":float(prealarm[:,j].mean()),"pre_alarm_count":int(prealarm[:,j].sum()),
               "first_packet_count":int(first_packet[:,j].sum()),
               "first_strictly_pre_alarm_count":int(first_pre[:,j].sum()),
               "pre_alarm_contrast_V1a":paired_risk(prealarm[:,j],prealarm[:,struct]),
               "max_peak_paired_squared":float(peak[:,j].max()),
               "active_actual_mse":interval(active_loss[:,j]) if active.any() else None,
               "tail_actual_mse":interval(tail_loss[:,j]) if tail.any() else None,
               "tail_cross_count":int(np.any(cross[:,j] & tail,axis=1).sum())}
        if a.operation=="anchor":
            cap = 4*a.parameter**2
            assert np.max(peak[:,j])<=cap+2e-12
            if cap<threshold:
                assert row["pre_alarm_count"]==row["first_packet_count"]==0
            exact = tr+float(np.mean([angular_excess(a.parameter,pair) for pair in
                                     proof["per_stage"]["C_eigenvalues"]]))
            row.update({"diameter_squared_bound":cap,"health_expected_MSE_exact_formula":exact,
                        "health_expected_pct_exact_formula":float(100*(exact/h-1)),
                        "absolute_all_time_output_bound_within_D":cap<threshold})
        if a.operation in ("trusted-ledger","structure-ledger","dual"):
            row["analytic_nominal_health_penalty_upper_pct"] = 100*a.parameter
        if a.name=="V0":
            row["health_expected_MSE_exact_formula"] = h
            row["health_expected_pct_exact_formula"] = 0.
        if a.name.startswith("anchor@"):
            cmp = names.index("innovation-clip@"+a.name.split("@")[1])
            row["pre_alarm_contrast_clip_target"] = paired_risk(prealarm[:,j],prealarm[:,cmp])
            row["active_mse_difference_clip_target"] = interval(active_loss[:,j]-active_loss[:,cmp]) if active.any() else None
        rows[a.name] = row
    data = {"health_mean_mse":healthy_loss,"conditional_health_squared":conditional_sq,
            "active_mean_mse":active_loss,"tail_mean_mse":tail_loss,"peak_paired_squared":peak,
            "first_packet":first_packet,"first_packet_pre_alarm":first_pre,
            "first_alarm":alarm,"first_danger":dangerous,"pre_alarm":prealarm,
            "onset":stream.start,"stop":stream.stop}
    return {"paths":len(stream),"common_BF_alarm_count":int((alarm<T).sum()),
            "post_recurrence_roundoff":float(np.max(np.abs(difference-balance))),
            "max_ball_violation_roundoff":run.ball_violation,"methods":rows},data


def holdout(base):
    common,settings,spec,_ = setup(base)
    reg,proof = register(base,settings)
    frozen = load_json(AUDIT_ROOT/"calibration_frozen.json")
    if frozen["input_hashes"]!=input_hashes():
        raise ValueError("Configuration, receiver code, or the deterministic first phase changed after calibration.")
    start = perf_counter()
    result = {"calibration_sha256":digest(AUDIT_ROOT/"calibration_frozen.json"),
              "hashes":frozen["input_hashes"],"models":{}}
    for name in MODELS:
        p = load_prepared(base,name,common,settings)
        row = proof["models"][name]
        radii = [entry["radius"] for entry in row["minimum_radii"]]
        thresholds = [entry["norm_threshold"] for entry in frozen["models"][name]]
        methods = method_table(p,reg["strict_ball_radius"],radii,thresholds)
        cfg = reg["holdout"]
        profiles = fresh_streams(base,p,spec,cfg["seed"][name],cfg["paths_per_model_and_profile"])
        assert list(profiles)==cfg["profiles"]
        health = collect(p,profiles["healthy"],methods)
        regerr = regress_v14(p,profiles["healthy"],methods,health)
        cases, arrays = {},{}
        for profile,stream in profiles.items():
            test = health if profile=="healthy" else collect(p,stream,methods)
            if profile=="unknown_start":
                regerr = max(regerr,regress_v14(p,stream,methods,test))
            cases[profile],data = measure(p,stream,test,health,methods,reg["D_squared"],row)
            arrays.update({profile+"__"+key:field for key,field in data.items()})
            print(name,profile,[cases[profile]["methods"][m]["pre_alarm_count"]
                for m in ("V1a","V1.4","trusted-ledger@2%","innovation-clip@2%","anchor@2%")],flush=True)
        np.savez_compressed(AUDIT_ROOT/f"{name}_paired.npz",**arrays)
        result["models"][name] = {"method_table":[asdict(m) for m in methods],"cases":cases,
                                  "regression_v14_max_abs":regerr}
    result["wall_seconds"] = perf_counter()-start
    write_json(AUDIT_ROOT/"comparison.json",result)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    parser.add_argument("--phase",choices=("calibrate","holdout"),required=True)
    a = parser.parse_args()
    {"calibrate":calibrate,"holdout":holdout}[a.phase](a.baseline)
