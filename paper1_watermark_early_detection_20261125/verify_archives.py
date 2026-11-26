"""Read back primary estimates, old thresholds, conditioning and paired geometry."""
import csv
import json
from pathlib import Path
import numpy as np

import study
from matched_glrt import build_sender_geometry


def legacy():
    result=json.loads((study.base.V3/"crn_final_results.json").read_text(encoding="utf-8"))
    out={}
    with np.load(study.base.V3/"crn_final_path_arrays.npz") as ar:
        for name in study.base.MODELS:
            _,model,physical,_,_=study.backend(name)
            record=result["models"][name]
            n=result["calibration_paths_per_model"]
            phy=physical(model,n,result["test_seed"] + 500_000_000 + study.base.MODELS.index(name)*10_000_000)
            score=study.base.plugin_scores(model,phy,np.zeros((n,150,6,2)))
            maxima=score[:,4:].max((1,2))
            archived=ar[name+"_calibration_maxscore"]
            np.testing.assert_allclose(maxima,archived,rtol=0,atol=1e-8)
            np.testing.assert_array_equal(np.argsort(maxima),np.argsort(archived))
            threshold=study.path_fwer_cutoff(maxima)["threshold"]
            assert abs(threshold-record["calibration_threshold"])<1e-8
            np.testing.assert_array_equal(maxima>threshold,archived>record["calibration_threshold"])
            out[name]={"calibration_paths":n,"max_abs_score_error":float(np.max(abs(maxima-archived))),
                       "ranks_and_decisions_exact":True,"threshold":threshold,"threshold_abs_error":abs(threshold-record["calibration_threshold"])}
    return out


def check_split(split):
    folder=study.HERE/study.LABELS[split]
    state=json.loads((folder/"RUN.json").read_text())
    summary=json.loads((folder/"analysis.json").read_text())
    for filename, expected in state["archives"].items():
        if study.base.digest(folder/filename) != expected:
            raise ValueError(f"Archive SHA-256 differs: {folder/filename}")
    arms=state["identity"]["arms"]
    row_means={a:[] for a,_,_ in arms}
    sensitivity={};conditioning={};geom={};all_seeds={}
    for mi,name in enumerate(study.base.MODELS):
        _,model,_,_,_=study.backend(name)
        ns=tuple(state["identity"]["seed_namespace"])
        senders=model.links[:,1]
        keys=study.short_draws(16,ns,mi,0,1,0,senders)
        conditions={}
        for scheme in dict.fromkeys(a[1] for a in arms):
            M=study.matrices(keys,scheme)
            rows={}
            for t in (5,6,7,8,50,65,149):
                w=min(8,t+1); first=t-w+1; cond=[]
                for e,j in enumerate(senders):
                    g=build_sender_geometry(model,int(j),first==0,w)
                    for start in range(w):
                        design=M[:,first:t+1,e]@g.attack_state_maps[start]
                        if first==0:design[:,0]=0.
                        x=g.whitening_map@design.reshape(16,-1,2)
                        sv=np.linalg.svd(x,compute_uv=False)
                        assert (sv[:,0]>0).all()
                        ratio=sv[:,1]/sv[:,0]
                        cond.extend((1/ratio[ratio>1e-10]).tolist())
                rows[t]={"rank2_count":len(cond),"N":16*w*6,"condition_q50_q95_max":np.quantile(cond,[.5,.95,1.]).tolist()}
            conditions[scheme]=rows
        conditioning[name]=conditions
        sensitivity[name]={}
        with np.load(folder/f"{name}_cal_0.npz") as ar,np.load(folder/f"{name}_health_0.npz") as hold:
            for a,_,_ in arms:
                maxima=np.sort(ar[a+"_max"]); n=len(maxima)
                thresholds=state["thresholds"][name][a]
                for alpha,t in zip(study.ALPHAS,thresholds):
                    rank=int(np.ceil((n+1)*(1-alpha)))
                    assert t==maxima[rank-1]
                k=int(np.count_nonzero(hold[a+"_max"]>thresholds[1]))
                assert k==summary["health"][name][a]["alarms"]
                sensitivity[name][a]={"FWER":summary["health"][name][a]["sensitivity_FWER"],"early3":[]}
        for i in range(6):
            scene=f"{name}_eval_{i}"
            with np.load(folder/(scene+".npz")) as ar:
                seeds=ar["physical_seed"]; all_seeds[scene]=np.unique(seeds).astype(int).tolist()
                for a,_,_ in arms:
                    delay=ar[a+"_delay"]; peaks=ar[a+"_early_max"]
                    c=state["thresholds"][name][a]
                    compare=(peaks>c[1]);np.testing.assert_array_equal(compare,delay<3)
                    row_means[a].append(compare.sum()/compare.size)
                    sensitivity[name][a]["early3"].append([(peaks>t).mean() for t in c])
                if split==3:
                    # Pool both onset strata only after computing each cell's independent mean and q10.
                    for alt in ("negative_radial_bin","negative_random_bin"):
                        one={}
                        for metric in ("psi","ungated_packet_sq"):
                            left=ar[alt+"_"+metric]; right=ar["negative_fixed_"+metric]
                            one[metric]={"mean_difference":(left-right).mean((0,2)).tolist(),
                                "q10_difference":(np.quantile(left,.1,axis=(0,2))-np.quantile(right,.1,axis=(0,2))).tolist()}
                        geom.setdefault(scene,{})[alt]=one
        for a,_,_ in arms:
            sensitivity[name][a]["early3"]=np.mean(sensitivity[name][a]["early3"],axis=0).tolist()
    for a,_,_ in arms:
        np.testing.assert_allclose(np.mean(row_means[a]),summary["means"]["all"][a]["early3"],atol=1e-13)
    return {"full_path_checks":"passed","physical_seeds":all_seeds,"geometry_differences":geom,
            "threshold_sensitivity":sensitivity,"template_conditioning_sample":conditioning}


if __name__=="__main__":
    data={"historical_calibration_replay":legacy(),"splits":{i:check_split(i) for i in (1,2,3)}}
    seeds=[v for sp in data["splits"].values() for rows in sp["physical_seeds"].values() for v in rows]
    assert len(set(seeds))==len(seeds)
    data["unique_evaluation_physical_batch_seeds"]=len(seeds)
    study.base._atomic_json(study.HERE/"ARCHIVE_VERIFICATION.json",data)
    radial=json.loads((study.HERE/"radial_dev/analysis.json").read_text())
    with (study.HERE/"RADIAL_GEOMETRY.csv").open("w",encoding="utf-8",newline="") as stream:
        table=csv.writer(stream)
        table.writerow(["model","angle_deg","onset","reports","scheme","measure","mean","q10",
                        "mean_minus_fixed","q10_minus_fixed"])
        for scene,schemes in radial["geometry"].items():
            model,_,s=scene.split("_")
            index=int(s)
            for scheme,metrics in schemes.items():
                for metric,v in metrics.items():
                    ref=schemes["negative_fixed"][metric]
                    for i,h in enumerate((2,3,5)):
                        table.writerow([model,study.base.ANGLES[index//2],study.base.ONSETS[index%2],
                                        h,scheme,metric,v["mean"][i],v["q10"][i],
                                        v["mean"][i]-ref["mean"][i],v["q10"][i]-ref["q10"][i]])
