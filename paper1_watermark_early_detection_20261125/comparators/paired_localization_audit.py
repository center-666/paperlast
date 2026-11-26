"""Post-run correction preserving shared physical paths across methods and supports."""
import argparse
import csv
import json

import numpy as np
import analyze_partial as original
import experiment as exp
import partial_study as study

HERE=study.HERE
REPS=10000
GROUPS={"all":range(12),"unstable":range(6),"stable":range(6,12),
        "onset50":range(0,12,2),"onset65":range(1,12,2)}


def main(split):
    folder=HERE/("partial_dev" if split=="dev" else "partial_confirm")
    manifest=exp.read(folder/"RUN.json")
    for f,h in manifest["archives"].items():
        assert exp.old.base.digest(folder/f)==h
    masks=list(study.MASKS)
    metrics=original.METRICS
    arrays=[]
    for mi in range(2):
        for scene in range(6):
            supports=[]
            for mask in masks:
                values,_,_=original.scene_values(folder,manifest,mi,scene,mask,study.ARMS)
                supports.append(np.stack([np.column_stack([values[a][m] for m in metrics])
                                          for a in study.ARMS],axis=1))
            arrays.append(np.stack(supports,axis=1))
    cells=np.stack(arrays) # [scene,path,support,arm,metric]
    n=cells.shape[1]
    rng=np.random.default_rng([20261127,52387,3 if split=="dev" else 4,9907])
    shape=(REPS,*cells.shape[2:])
    boot={g:np.zeros(shape) for g in GROUPS}
    for scene,x in enumerate(cells):
        weights=rng.multinomial(n,np.full(n,1/n),size=REPS).astype(float)/n
        y=(weights@x.reshape(n,-1)).reshape(shape)
        for g,indices in GROUPS.items():
            if scene in indices:boot[g]+=y/len(indices)
    absolute=[];contrasts=[]
    for group,indices in GROUPS.items():
        mean=cells[list(indices)].mean((0,1))
        lower,upper=np.quantile(boot[group],[.025,.975],axis=0)
        for i,mask in enumerate(masks):
            for j,arm in enumerate(study.ARMS):
                for k,m in enumerate(metrics):
                    absolute.append({"group":group,"support":mask,"arm":arm,"metric":m,
                                     "mean":mean[i,j,k],"lower":lower[i,j,k],"upper":upper[i,j,k]})
            for label,coef in original.CONTRASTS.items():
                d=sum(w*boot[group][:,i,study.ARMS.index(a)] for a,w in coef.items())
                point=sum(w*mean[i,study.ARMS.index(a)] for a,w in coef.items())
                lo,hi=np.quantile(d,[.025,.975],axis=0)
                for k,m in enumerate(metrics):
                    contrasts.append({"group":group,"support":mask,"comparison":label,"metric":m,
                                      "difference":point[k],"lower":lo[k],"upper":hi[k]})
    for filename, rows in (("paired_absolute.csv", absolute), ("paired_contrasts.csv", contrasts)):
        with (folder/filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    # Independent scalar contrast check for the first support.
    a=study.ARMS.index("quarter_exact");b=study.ARMS.index("negative_exact")
    check=np.mean(cells[:,:,0,a,3]-cells[:,:,0,b,3])
    expected=next(r["difference"] for r in contrasts if r["group"]=="all"
                  and r["support"]==masks[0] and r["metric"]=="exact_support_early"
                  and r["comparison"]=="quarter-negative")
    np.testing.assert_allclose(check,expected,atol=1e-14)
    exp.old.base._atomic_json(folder/"PAIRED_ANALYSIS_AUDIT.json",{
        "run_sha256":exp.old.base.digest(folder/"RUN.json"),
        "analysis_source_sha256":exp.old.base.digest(HERE/"paired_localization_audit.py"),
        "bootstrap_replicates":REPS,"seed":[20261127,52387,3 if split=="dev" else 4,9907],
        "correction":"The original analyzer resampled each arm independently before subtracting bootstrap means. This audit uses one common resampling weight per physical path and fixed scene across all methods, supports and outcomes.",
        "physical_units":12*n,"paired_support_worlds_per_unit":8,
        "paired_absolute_sha256":exp.old.base.digest(folder/"paired_absolute.csv"),
        "paired_contrasts_sha256":exp.old.base.digest(folder/"paired_contrasts.csv")})
    print(split,"shared-path resampling complete")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=("dev","confirm"))
    main(p.parse_args().split)
