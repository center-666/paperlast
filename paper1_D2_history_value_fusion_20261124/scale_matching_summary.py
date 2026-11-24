"""Paired path-level summaries of the A-prime offline replay."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
from scale_matching_audit import DEST, NAMES, PAIRS, digest, write_json


def interval(blocks, denominator=None, factor=1.):
    """Equal stratum weights; preserve covariance of numerator and denominator per path."""
    means=[b.mean() for b in blocks]
    numerator=np.mean(means)
    if denominator is None:
        value=numerator
        residual=blocks
    else:
        den=np.mean([b.mean() for b in denominator])
        if den==0.:return None
        value=numerator/den
        residual=[(a-value*b)/den for a,b in zip(blocks,denominator,strict=True)]
    components=np.array([v.var(ddof=1)/len(v)/len(blocks)**2 for v in residual])
    variance=components.sum()
    if variance==0:half=0.
    else:
        df=variance**2/sum(v**2/(len(b)-1) for v,b in zip(components,blocks,strict=True))
        half=t.ppf(.975,df)*np.sqrt(variance)
    return (factor*np.array([value,value-half,value+half])).tolist()


def summarize(items, pairs):
    # A block comprises all watermark sisters when D0 shares a physical path.
    blocks=[{k:np.mean([m[k] for m in item],axis=0) for k in item[0]} for item in items]
    all_files=[m for item in items for m in item]
    single=sum(int((m["counts"][:,0]-m["counts"][:,1]).sum()) for m in all_files)
    eligible=sum(int(m["counts"][:,1].sum()) for m in all_files)
    out=dict(condition_path_observations=sum(len(m["counts"]) for m in all_files),
             independent_paths=sum(len(m["counts"]) for m in blocks),
             singleton=single,eligible=eligible,
             mean_MSE=np.mean([m["loss"].mean(0) for m in blocks],0).tolist(),pairs={})
    for p,(i,j) in enumerate(pairs):
        name=NAMES[j]+" minus "+NAMES[i]
        event=[m["action"][:,p,0] for m in blocks]
        tot=np.sum([m["action"][:,p].sum(0) for m in all_files],axis=0)
        summary=dict(flip_count=int(tot[0]),rate=interval(event,[m["counts"][:,1] for m in blocks]),
                     nominal_H_change_on_flips=interval([m["action"][:,p,1] for m in blocks],event),
                     lower_same_higher=tot[2:].astype(int).tolist())
        for k,period in enumerate(("window","tail","post20")):
            base=[m["loss"][:,i,k] for m in blocks]
            difference=[m["loss"][:,j,k]-m["loss"][:,i,k] for m in blocks]
            summary[period]=dict(delta=interval(difference),relative=interval(difference,base))
        out["pairs"][name]=summary
    if "q1_reference" in blocks[0]:
        out["healthy_vs_q1"]=[interval([v["loss"][:,i,2]-v["q1_reference"] for v in blocks]) for i in range(6)]
    return out


def main():
    manifest=json.loads((DEST/"manifest.json").read_text())
    conditions={}; observations={}
    for name in manifest["checks"]:
        path=DEST/(name+".npz")
        if digest(path)!=manifest["path_outputs_sha256"][path.name]:raise ValueError(name)
        with np.load(path) as z:
            item={k:z[k] for k in z.files}
        observations[name]=item
        conditions[name]=summarize([[item]],PAIRS)
    summary=dict(policy_order=NAMES,physical={},sources={},conditions=conditions)
    for source in ("B2","D0","B1"):
        total=[]
        for pair in range(6):
            structures=[]
            for fixture in (["healthy"] if source=="B1" else ["continuous","burst"]):
                names=[f"{i:02}_{source}_{fixture}" for i in (2*pair,2*pair+1)]
                arr=[observations[name] for name in names if name in observations]
                if source=="D0" and arr:structures.append(arr)
                else:structures.extend([[v] for v in arr])
            if not structures:continue
            total.extend(structures)
            spec=manifest["scale"]["cells"][2*pair]["spec"]
            summary["physical"][f"{source}_{spec[0]}_{spec[1]}_{spec[2]}"]=summarize(structures,PAIRS)
        summary["sources"][source]=summarize(total,PAIRS)
    write_json(DEST/"summary.json",summary)


if __name__=="__main__":
    main()
