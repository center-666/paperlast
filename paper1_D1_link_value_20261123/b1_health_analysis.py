"""Paired B1 health contrasts and alarm audit, conditional on frozen per-scenario calibration."""
import json
import math

import numpy as np
from scipy.stats import beta, t

from b1_health import B0, BITS, HERE, N, OUT, PLAN, Q, modes, scenarios, seed


def summarize_alarms(mask):
    flags = (mask[...,None] & BITS) != 0
    length = np.zeros((N,7), int)
    longest = np.zeros_like(length)
    for step in flags.transpose(1,0,2):
        length = (length+1)*step
        longest = np.maximum(longest, length)
    starts = flags & ~np.concatenate([np.zeros_like(flags[:,:1]), flags[:,:-1]],axis=1)
    counts = flags.sum(axis=1)
    hits = int((counts.sum(axis=1)>0).sum())
    multi = flags.sum(axis=2)>1
    observed, num = np.unique(mask, return_counts=True)
    run_counts = starts.sum(axis=(0,1))
    return {
        "paths_with_any": hits, "fwer": hits/N,
        "fwer_exact_95": [float(beta.ppf(.025,hits,N+1-hits)) if hits else 0.,
                          float(beta.ppf(.975,hits+1,N-hits)) if hits<N else 1.],
        "paths_with_target": int((counts[:,-1]>0).sum()),
        "paths_with_simultaneous": int(multi.any(axis=1).sum()),
        "simultaneous_ticks": int(multi.sum()),
        "target_and_other_ticks": int((multi & flags[:,:,-1]).sum()),
        "counts_per_edge": counts.sum(axis=0).tolist(),
        "runs_per_edge": run_counts.tolist(),
        "max_run_per_edge": longest.max(axis=0).tolist(),
        "mean_run_per_edge": [float(a/b) if b else None for a,b in zip(counts.sum(axis=0),run_counts)],
        "observed_masks_counting_all_150_ticks": dict(zip(map(str,observed), map(int,num))),
    }


def schedule_diagnostic(z, gains, key):
    """Propagate unconditional mode noise along observed schedules as an exogenous replay only."""
    matrix = {}
    for name in ("F","Omega"):
        matrix[name] = np.stack([gains[f"{key}{q:g}_{name}"][[0,64],26] for q in Q],axis=1)
    p = np.broadcast_to(np.eye(4), (N,len(Q),4,4)).copy()
    total = np.zeros((N,len(Q)))
    for k in range(1,150):
        g = (z["masks"][:,k] >> 6).astype(int)
        f = matrix["F"][g]
        p = f @ p @ f.swapaxes(-1,-2) + matrix["Omega"][g]
        if k>=20:
            total += p[...,2,2]+p[...,3,3]
    return total/130


def paired_stats(loss, eps, critical):
    contrast = loss - (1+eps)*loss[:,-1,None]
    mean = contrast.mean(axis=0)
    se = contrast.std(axis=0,ddof=1)/np.sqrt(N)
    return contrast, mean, se, mean+critical*se


def bootstrap_statistics(values, rng, reps):
    """Path resampling preserves all raw candidate and budget outcomes within a scenario."""
    means = values.mean(axis=0)
    sq = values**2
    maxima = np.empty((reps,2))
    for i in range(0,reps,32):
        size = min(32,reps-i)
        indices = rng.integers(N,size=(size,N))
        counts = np.stack([np.bincount(row,minlength=N) for row in indices])
        avg = counts @ values/N
        se = np.sqrt(np.maximum(0., counts @ sq/N-avg**2)/(N-1))
        if np.any(se==0):
            raise ArithmeticError("Degenerate studentized bootstrap; report before inference.")
        maxima[i:i+size] = ((means-avg)/se).reshape(size,2,len(Q)).max(axis=2)
    return maxima


def main():
    report = json.loads((HERE/"results/D1A_link_value.json").read_text(encoding="utf-8"))
    critical = float(t.ppf(1-.05/84,N-1))
    reps = PLAN["bootstrap"]["replicates"]
    max_t = np.full((reps,2),-np.inf)
    rows, paired = [], {}
    with np.load(modes.OUTPUT/"mode_tables.npz") as gains:
        for idx,spec in enumerate(scenarios()):
            dynamics,topology,target,label = spec
            scan = report["models"][f"{topology}_{dynamics}"]["link_scans"][str(target)]
            kappa = max(scan["kappa"]["keep"]["60"],0.)
            eps = .05-.04*kappa/(1+kappa)
            with np.load(OUT/f"scene_{idx:02}.npz") as z:
                masks = z["masks"]
                alarm = summarize_alarms(masks)
                loss = z["missing"]
                baseline = loss[:,-1]
                rank = math.ceil(.95*(N+1))
                assert np.sort(z["calibration_max"])[rank-1] == z["threshold"]
                np.testing.assert_array_equal(z["holdout_max"]>z["threshold"], masks.any(axis=1))
                assert not masks[:,:5].any()
                never = (masks & 64).sum(axis=1)==0
                for metric in ("missing","full","network"):
                    vals = z[metric]
                    np.testing.assert_allclose(vals[never],np.broadcast_to(vals[never,-1,None],vals[never].shape),
                                               atol=1e-10,rtol=1e-12)
                key = f"{topology}_{dynamics}_target{target}_q"
                schedule = schedule_diagnostic(dict(masks=masks),gains,key)
                paired[f"s{idx}_exogenous_schedule_diagnostic"] = schedule
                eligible = z["safety"]<1
                row = dict(scene=idx,dynamics=dynamics,topology=topology,target=target,watermark=label,
                           threshold=float(z["threshold"]),rank=rank,alarms=alarm,
                           safety_max_mode_norm=z["safety"].tolist(),value_budget=eps,
                           mean_missing=loss.mean(axis=0).tolist(),mean_full=z["full"].mean(axis=0).tolist(),
                           mean_network=z["network"].mean(axis=0).tolist(),
                           relative_mean_missing=(loss.mean(axis=0)/baseline.mean()-1).tolist(),
                           exogenous_schedule_missing_mean=schedule.mean(axis=0).tolist(),
                           mean_effective_target_weight=(1-(1-Q)*(masks[:,20:]>>6).mean()).tolist())
            budget_vectors = []
            gain = np.array([w["attack_transfer_gain_by_horizon"]["60"] for w in scan["weights"]])
            for name, budget in (("uniform",.05),("value",eps)):
                contrast,mean,se,upper = paired_stats(loss,budget,critical)
                budget_vectors.append(contrast)
                paired[f"s{idx}_Z_{name}"] = contrast
                safe = eligible & (upper<=0)
                assert safe[-1]
                best = np.max(Q[safe & (gain <= gain[safe].min()+1e-10)])
                row[name] = dict(mean=mean.tolist(),se=se.tolist(),upper=upper.tolist(),
                                 feasible=Q[safe].tolist(),proxy_minimizer=float(best),
                                 minimum_feasible=float(Q[safe].min()))
            max_t = np.maximum(max_t, bootstrap_statistics(np.column_stack(budget_vectors),
                                                          np.random.default_rng(seed(idx,3)), reps))
            rows.append(row)
    boot = np.quantile(max_t,.95,axis=0,method="higher")
    for row in rows:
        for i,name in enumerate(("uniform","value")):
            z = row[name]
            bounds = np.array(z["mean"])+boot[i]*np.array(z["se"])
            z["bootstrap_upper"] = bounds.tolist()
            z["bootstrap_feasible"] = Q[(bounds<=0)&(np.array(row["safety_max_mode_norm"])<1)].tolist()
    conclusion = dict(n_cal=N,n_holdout=N, weights=Q.tolist(), critical=critical,
                      alpha_H=B0["simultaneous_inference"]["alpha"], family_size=84,
                      bootstrap_reps=reps, bootstrap_critical_by_uniform_value=boot.tolist(),
                      empirical_bootstrap_coverage_at_t_critical=(max_t<=critical).mean(axis=0).tolist(),
                      bootstrap_coverage_scope="empirical path distribution conditional on thresholds; sensitivity only",
                      scenarios=rows)
    paired["bootstrap_max_studentized"] = max_t
    np.savez_compressed(OUT/"paired_contrasts.npz",**paired)
    (OUT/"health_audit.json").write_text(json.dumps(conclusion,indent=2,allow_nan=False)+"\n",
                                          encoding="utf-8",newline="\n")
    policy = {
        "scope": "B1 healthy selection for track M; bit 6 is the target alarm",
        "off_alarm_q": 1.0, "other_six_edges": "frozen_binary_gates",
        "registration_commit": "7868bab", "B0_commit": PLAN["b0_commit"],
        "rows": [dict(scene=r["scene"], dynamics=r["dynamics"], topology=r["topology"],
                      sender=r["target"], receiver=26, watermark=r["watermark"], A=0.8,
                      B=r["uniform"]["proxy_minimizer"], C=r["value"]["minimum_feasible"],
                      D=r["value"]["proxy_minimizer"]) for r in rows],
    }
    (OUT/"policy_lock.json").write_text(json.dumps(policy,indent=2)+"\n",encoding="utf-8",newline="\n")
    for r in rows:
        print(r["scene"],r["alarms"]["paths_with_any"],r["alarms"]["paths_with_target"],
              r["value"]["feasible"],r["uniform"]["proxy_minimizer"],r["value"]["proxy_minimizer"])


if __name__=="__main__":
    main()
