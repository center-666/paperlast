"""Paired, path-cross-fitted one-step prediction on frozen T histories."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from prediction_histories import task3, frontend, replay_T, target_replay, ridge, apply

OUT = Path(__file__).resolve().parent
CONFIG = "U_unstable_R35x2"
ARMS = ("zero", "history", "key_history", "key_prior", "key_template")
VARIANTS = {"base": (6, 0), "minus45": (18**.5, -18**.5), "plus45": (18**.5, 18**.5),
            "half": (3, 0), "one_half": (9, 0)}
GRID = (.001, .01, .1, 1., 10., 100.)
START = 8


def past(values):
    # Slice [k-8,k) for every k from 8 to the last report.
    window = np.lib.stride_tricks.sliding_window_view(values[:, :-1], 8, axis=1)
    return np.moveaxis(window, -1, 2)


def features(residual, keys, alarms):
    previous_run = np.zeros(alarms.shape, int)
    for k in range(1, alarms.shape[1]):
        previous_run[:, k] = (previous_run[:, k-1] + 1) * alarms[:, k-1]
    encoded = np.einsum("...ji,...j->...i", keys, residual)
    shape = (len(keys), keys.shape[1] - START, 16)
    return (past(residual).reshape(shape), past(encoded).reshape(shape),
            np.digitize(previous_run[:, START:], [1, 2, 4]))


def templates(report, keys, white, threshold):
    # No observation, key or alarm beyond the prediction time enters this fit.
    # Group identical design masks while retaining conflicting continuations.
    masks = {}
    for activity in frontend.template_activities(9):
        for start in range(8):
            mask = tuple(activity[:8] * (np.arange(8) >= start))
            if any(mask):
                masks.setdefault(mask, set()).add(activity[8])
    n = len(keys)
    obs = past(report).reshape(n, -1, 16)
    observed = obs @ white.T
    history = past(keys)
    # A non-repeating four-state sequence has at most 4*3**7 distinct histories.
    rotations = frontend.ROTATIONS
    state_codes = np.argmax(np.all(keys[...,None,:,:] == rotations, axis=(-1,-2)), axis=-1)
    word = past(state_codes) @ (4**np.arange(8))
    words, first, inverse = np.unique(word.ravel(), return_index=True, return_inverse=True)
    histories = history[np.unravel_index(first, word.shape)]
    masks_list = list(masks)
    continuing = np.array([1. in masks[m] for m in masks_list])
    stopping = np.array([0. in masks[m] for m in masks_list])
    predicted = np.zeros((word.size, 2))
    accepted = np.zeros(word.size, bool)
    # Fit each history design by SVD, including rank-deficient competitors.
    order = np.argsort(inverse)
    groups = np.split(order, np.flatnonzero(np.diff(inverse[order]))+1)
    for offset in range(0, len(words), 96):
        hs = histories[offset:offset+96]
        design = np.einsum("rti,ntij,mt->nmrj", white.reshape(-1,8,2), hs,
                           np.asarray(masks_list), optimize=True)
        u, s, vt = np.linalg.svd(design, full_matrices=False)
        keep = s > np.maximum(1e-10*s[..., :1], 1e-11)
        pseudo = np.divide(1, s, out=np.zeros_like(s), where=keep)
        for j in range(len(hs)):
            ids = groups[offset+j]
            y = observed.reshape(-1, white.shape[0])[ids]
            coeff = np.einsum("br,mri->bmi", y, u[j]) * keep[j]
            score = (coeff**2).sum(-1)
            best = score.argmax(1)
            peak = score[np.arange(len(ids)), best]
            near = score >= peak[:, None]-2
            beta = np.einsum("bmi,mij->bmj", coeff*pseudo[j], vt[j])
            candidate = beta[np.arange(len(ids)), best] * continuing[best,None]
            tolerance = .25*np.maximum(1., np.linalg.norm(candidate, axis=1))
            conflict = ((np.linalg.norm(beta-candidate[:,None], axis=-1)>tolerance[:,None]) & continuing)
            conflict |= ((np.linalg.norm(candidate, axis=1)>tolerance)[:,None] & stopping)
            ok = (peak>threshold) & ~np.any(near & (conflict | ~keep[j].all(-1)),axis=1)
            predicted[ids] = candidate*ok[:,None]
            accepted[ids] = ok
    return (np.einsum("...ij,...j->...i", keys[:, START:], predicted.reshape(*word.shape,2)),
            accepted.reshape(word.shape))


def extract(trace):
    residual, keys, alarm, delta = trace
    raw, encoded, duration = features(residual, keys, alarm)
    return dict(raw=raw, encoded=encoded, duration=duration, key=keys[:,START:],
                alarm=alarm[:,START:], delta=delta[:,START:],
                latent=np.einsum("...ji,...j->...i",keys[:,START:],delta[:,START:]))


def fit(traces, fold):
    def training(name, remainder=None, validation=False):
        rows = []
        for t in traces:
            paths = np.arange(len(t["alarm"]))
            selection = paths%2 != fold
            if remainder is not None:
                selection &= (paths%4==remainder) == validation
            mask = t["alarm"] & selection[:,None] & (np.arange(8,150)>=20)
            rows.append(t[name][mask])
        return np.concatenate(rows)
    fitted = {}
    for method, feature, label in (("history","raw","delta"), ("key_history","encoded","latent")):
        rem = 1-fold
        x, y = training(feature,rem), training(label,rem)
        xv, yv = training(feature,rem,True), training(label,rem,True)
        alpha = min(GRID, key=lambda a: np.mean((apply(ridge(x,y,a),xv)-yv)**2))
        fitted[method] = ridge(training(feature),training(label),alpha)
        fitted[method+"_alpha"] = alpha
    duration, latent = training("duration"), training("latent")
    fitted["key_prior"] = np.stack([latent[duration==i].mean(0) if np.any(duration==i)
                                     else np.zeros(2) for i in range(4)])
    return fitted


def forecasts(t, fits):
    shape = t["delta"].shape
    out = {m:np.zeros(shape) for m in ARMS}
    for fold, f in enumerate(fits):
        ix = np.arange(shape[0])%2==fold
        for m, feature in (("history","raw"),("key_history","encoded")):
            v = apply(f[m], t[feature][ix].reshape(-1,16)).reshape(-1,shape[1],2)
            out[m][ix] = np.einsum("...ij,...j->...i", t["key"][ix],v) if m=="key_history" else v
        out["key_prior"][ix] = np.einsum("...ij,...j->...i",t["key"][ix],f["key_prior"][t["duration"][ix]])
    return out


def record(rows, t, fixture, variant, preds, supported):
    ticks = np.arange(8,150)
    attack = (ticks>=50)&(ticks<110)
    windows = {"health":ticks>=20} if fixture=="health" else {"whole":attack}
    if fixture=="burst":
        active = (ticks-50)%10<6
        windows.update(active=attack&active, closed=attack&~active)
    for phase, window in windows.items():
        for arm in ARMS:
            alarm = t["alarm"] & window
            execute = alarm & (supported if arm=="key_template" else arm!="zero")
            effect = ((t["delta"]-preds[arm])**2-t["delta"]**2).sum(-1)
            total = (effect*execute).sum(1)
            for p,(a,c,s) in enumerate(zip(alarm.sum(1),execute.sum(1),total)):
                rows.append(dict(fixture=fixture,variant=variant,window=phase,path=p,method=arm,
                                 alarms=int(a),executed=int(c),change=s/max(1,a),
                                 executed_change=s/c if c else "",sum_change=s))


def write_csv(name, rows):
    with (OUT/name).open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    rng = np.random.default_rng(2026112808)
    for group in dict.fromkeys((r["fixture"],r["variant"],r["window"]) for r in rows):
        group_rows = [r for r in rows if (r["fixture"],r["variant"],r["window"])==group]
        columns = []
        for m in ARMS:
            arm = [r for r in group_rows if r["method"]==m]
            assert [r["path"] for r in arm]==list(range(512))
            columns.append(np.array([r["change"] for r in arm]))
        values = np.column_stack(columns+[columns[2]-columns[1],columns[2]-columns[3],
                                           columns[4]-columns[1],columns[4]-columns[3]])
        draws = np.concatenate([values[rng.integers(0,512,(200,512))].mean(1) for _ in range(100)])
        for i, m in enumerate((*ARMS,"key_history-history","key_history-key_prior",
                               "key_template-history","key_template-key_prior")):
            subset = [r for r in group_rows if r["method"]==m]
            a = sum(r["alarms"] for r in subset)
            c = sum(r["executed"] for r in subset)
            lo, hi = np.quantile(draws[:,i],[.025,.975])
            summary.append(dict(fixture=group[0],variant=group[1],window=group[2],method=m,paths=512,
                                change=values[:,i].mean(),lower95=lo,upper95=hi,alarm_ticks=a if subset else "",
                                executed=c if subset else "",fallback=(1-c/a if a else ""),
                                executed_tick_change=sum(r["sum_change"] for r in subset)/c if c else ""))
    write_csv("PREDICTION_SUMMARY.csv",summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive",type=Path)
    args = parser.parse_args()
    model = task3.audit_frontend.load_model(2)
    ops = frontend.operators(model)
    white = ops[8][1][6]["whitening"]
    threshold = task3.frontend_thresholds(2)["no_repeat_matched"]
    base = {}
    for fixture in ("continuous","burst","health"):
        name = "continuous_health" if fixture=="health" else fixture+"_attack"
        with np.load(args.archive/f"{CONFIG}_{name}.npz") as archive:
            trace = replay_T(archive,"no_repeat_",fixture,model)
            base[fixture] = (trace, archive["y"][:,:,35]+trace[3])
    fits = [fit([extract(base[f][0]) for f in ("continuous","burst")], fold) for fold in (0,1)]
    rows = []
    for fixture in base:
        for variant, vector in VARIANTS.items():
            if fixture=="health" and variant!="base":
                continue
            if variant=="base":
                trace, reports = base[fixture]
            else:
                with np.load(args.archive/f"{CONFIG}_{fixture}_attack.npz") as archive:
                    world = target_replay(archive,fixture,np.asarray(vector),model,ops,threshold)
                    trace = world["residual"],world["keys"],world["gates"],world["delta"]
                    reports = archive["y"][:,:,35]+world["delta"]
                    del world
            t = extract(trace)
            preds = forecasts(t,fits)
            preds["key_template"], supported = templates(reports,trace[1],white,threshold)
            if variant=="base":
                # Current and later report/innovation/alarm edits leave the forecast at k unchanged.
                k = 70
                altered = [v[:3].copy() for v in trace]
                altered[0][:,k:] += 1e3
                altered[2][:,k:] = ~altered[2][:,k:]
                altered[1][:,k+1:] *= -1
                changed = forecasts(extract(altered),fits)
                for m in ("history","key_history","key_prior"):
                    np.testing.assert_allclose(changed[m][:,k-8],preds[m][:3,k-8],atol=1e-12)
                altered_report = reports[:3].copy()
                altered_report[:,k:] += 1e3
                new, valid = templates(altered_report,altered[1],white,threshold)
                np.testing.assert_allclose(new[:,k-8],preds["key_template"][:3,k-8],atol=1e-12)
                np.testing.assert_array_equal(valid[:,k-8],supported[:3,k-8])
            record(rows,t,fixture,variant,preds,supported)
            print(fixture,variant,flush=True)
    write_csv("PREDICTION_PATHS.csv", rows)
    summarize(rows)
    files = ["PREDICTION_REPLAY_DESIGN.md","prediction_histories.py","causal_prediction.py",
             "PREDICTION_PATHS.csv","PREDICTION_SUMMARY.csv"]
    receipt = {"usage":"post-hoc namespace 85; fold by path ID; predictions on frozen T histories",
               "paths_per_fixture":512,"bootstrap":20000,"causality_checks":"passed on three paths per base fixture",
               "ridge_alpha_by_fold":[{m:f[m+"_alpha"] for m in ("history","key_history")} for f in fits],
               "sha256":{p:hashlib.sha256((OUT/p).read_bytes()).hexdigest() for p in files},
               "source_archives":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [args.archive/f"{CONFIG}_{n}.npz" for n in ("continuous_attack","burst_attack","continuous_health")]}}
    (OUT/"PREDICTION_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8",newline="\n")


if __name__=="__main__":
    main()
