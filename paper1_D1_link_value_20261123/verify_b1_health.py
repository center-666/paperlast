"""Regenerate healthy scores for both splits, audit archived losses and replay selected paths."""
import hashlib
import json

import numpy as np
from scipy.stats import beta, t

from b1_health import BATCH, BITS, HERE, N, OUT, PLAN, Q, literal_check, modes, physical, scenarios, seed, setup


def independent_scores(model, paths):
    packets = paths.y[:,:,model.links[:,1]].copy()
    packets[:,0] = 0
    scores = np.zeros((len(packets),150,7))
    bounds = np.zeros(len(packets))
    for edge in range(7):
        windows = np.lib.stride_tricks.sliding_window_view(packets[:,:,edge],5,axis=1)
        stack = windows.transpose(0,1,3,2).reshape(len(packets),146,10)
        residual = stack[:,1:] @ model.Z[edge].T
        # Theta is positive definite. Cholesky whitening gives the same H1 quadratic form.
        whitening = np.linalg.solve(np.linalg.cholesky(model.Theta[edge]), np.eye(8))
        white = residual @ whitening.T
        scores[:,5:,edge] = .5*(white**2).sum(axis=-1)
        # Both raw-window contractions sum ten signed terms. Account for cancellation
        # before comparing quadratic forms, especially late in the unstable model.
        machine = np.finfo(float).eps
        dr = 32*machine*np.linalg.norm(model.Z[edge], "fro")*np.linalg.norm(stack[:,1:],axis=-1)
        nr = np.linalg.norm(residual,axis=-1)
        precision = model.precision[edge]
        pn = np.linalg.norm(precision,2)
        pd = np.linalg.norm(precision - whitening.T @ whitening,2)
        error = pn*dr*(nr+dr) + (pd+128*machine*pn)*(nr+dr)**2
        bounds = np.maximum(bounds,error.max(axis=1))
    return scores, bounds


def check():
    for name, expected in PLAN["frozen_files_sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected, name
    summary = json.loads((OUT/"health_audit.json").read_text(encoding="utf-8"))
    lock = json.loads((OUT/"policy_lock.json").read_text(encoding="utf-8"))
    design = json.loads((HERE/"results/D1A_link_value.json").read_text(encoding="utf-8"))
    assert len(lock["rows"]) == len(summary["scenarios"]) == len(scenarios()) == 12
    critical = t.isf(.05/(12*7), N-1)
    assert abs(summary["critical"]-critical)<1e-12
    results = []
    with np.load(modes.OUTPUT/"mode_tables.npz") as gains, np.load(OUT/"paired_contrasts.npz") as contrasts:
        for idx,spec in enumerate(scenarios()):
            row = summary["scenarios"][idx]
            model = setup(spec,gains)
            with np.load(OUT/f"scene_{idx:02}.npz") as saved:
                data = dict(saved)
            masks = data["masks"]
            threshold = np.sort(data["calibration_max"])[int(np.ceil(.95*(N+1)))-1]
            assert threshold == row["threshold"] == data["threshold"]
            target_paths = np.flatnonzero((masks&64).any(axis=1))
            replay_ids = np.unique([0,*target_paths[:1],*target_paths[-1:]])
            max_score_diff = 0.
            max_loss_diff = 0.
            for stage in (0,1):
                for batch in range(N//BATCH):
                    group = slice(batch*BATCH,(batch+1)*BATCH)
                    paths = physical(model,BATCH,seed(idx,stage,batch))
                    scores, bounds = independent_scores(model,paths)
                    maxima = scores[:,4:].max(axis=(1,2))
                    expected = data["calibration_max" if stage==0 else "holdout_max"][group]
                    max_score_diff = max(max_score_diff,float(abs(expected-maxima).max()))
                    assert np.all(abs(maxima-expected) <= bounds), (idx,stage,batch)
                    if stage==1:
                        np.testing.assert_array_equal((scores>threshold)@BITS,masks[group])
                        ids = replay_ids[(replay_ids>=group.start)&(replay_ids<group.stop)]
                        if len(ids):
                            paths.y,paths.x = paths.y[ids-group.start],paths.x[ids-group.start]
                            max_loss_diff = max(max_loss_diff,literal_check(model,paths,masks[ids],
                                {n:data[n][ids] for n in ("missing","full","network")}))
            loss = data["missing"]
            baseline = loss[:,-1,None]
            np.testing.assert_allclose(data["network"]-data["network"][:,-1,None],
                (data["full"]-data["full"][:,-1,None])/50,rtol=1e-9,atol=2e-15)
            for name,eps in (("uniform",.05),("value",row["value_budget"])):
                z = loss - (1+eps)*baseline
                np.testing.assert_array_equal(z, contrasts[f"s{idx}_Z_{name}"])
                high = z.mean(axis=0)+critical*np.sqrt(np.var(z,axis=0,ddof=1)/N)
                np.testing.assert_allclose(high,row[name]["upper"],rtol=1e-12,atol=1e-15)
                assert Q[(high<=0)&(data["safety"]<1)].tolist()==row[name]["feasible"]
            dynamics,topology,target,label = spec
            policy = lock["rows"][idx]
            assert [policy["dynamics"],policy["topology"],policy["sender"],policy["watermark"]] == list(spec)
            costs = design["models"][f"{topology}_{dynamics}"]["link_scans"][str(target)]["weights"]
            cost = {w["q"]:w["attack_transfer_gain_by_horizon"]["60"] for w in costs}
            for name,strategy in (("uniform","B"),("value","D")):
                feasible = row[name]["feasible"]
                best_cost = min(cost[q] for q in feasible)
                assert policy[strategy] == max(q for q in feasible if cost[q] <= best_cost+1e-10)
            assert policy["A"] == 0.8 and policy["C"] == min(row["value"]["feasible"])
            counts = np.count_nonzero(masks.any(axis=1))
            interval = beta.ppf([.025,.975],[counts,counts+1],[N-counts+1,N-counts])
            np.testing.assert_allclose(interval,row["alarms"]["fwer_exact_95"])
            flags=(masks[...,None]&BITS)!=0
            multi=(flags.sum(axis=-1)>1)
            assert int(multi.sum())==row["alarms"]["simultaneous_ticks"]
            if spec[2]==1:
                np.testing.assert_array_equal(flags[...,0],flags[...,6])
            # Conditional event decomposition for target-only policies, including late post-alarm recovery.
            diff = loss - baseline
            never = ~flags[...,6].any(axis=1)
            np.testing.assert_allclose(diff[never],0,atol=1e-11,rtol=0)
            conditional = diff[~never].mean(axis=0)
            np.testing.assert_allclose(diff.mean(axis=0), (~never).mean()*conditional, atol=1e-13,rtol=0)
            results.append(dict(scene=idx,split_paths_checked=2*N,threshold=threshold,
                                max_path_score_maximum_error=max_score_diff,
                                literal_filter_path_ids=replay_ids.tolist(),max_path_mse_difference=max_loss_diff,
                                conditional_mean_difference_given_any_target_alarm=conditional.tolist(),
                                repeated_sender_identical_gate_bits=spec[2]==1))
    record = dict(status="PASS", actual_fwer_and_mask_checks="all paths",
                  family_size=84, literal_mse_path_count=sum(len(x["literal_filter_path_ids"]) for x in results),
                  results=results)
    (OUT/"verification.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",
                                         encoding="utf-8",newline="\n")
    print(record["status"], record["literal_mse_path_count"])


if __name__=="__main__":
    check()
