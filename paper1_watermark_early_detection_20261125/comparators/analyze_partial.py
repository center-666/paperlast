"""Path-level partial-support localization analysis with fixed scene weights."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import partial_study as study
import experiment as exp

HERE=study.HERE
BOOT=10000
CONTRASTS={"quarter-negative":{"quarter_exact":1,"negative_exact":-1},
           "negative-iid":{"negative_exact":1,"iid_exact":-1},
           "quarter_plugin-negative_plugin":{"quarter_plugin":1,"negative_plugin":-1},
           "negative_plugin-iid_plugin":{"negative_plugin":1,"iid_plugin":-1}}
METRICS=("attacked_edge_early","healthy_edge_early_false_alarm","healthy_path_any_early_false_alarm",
         "exact_support_early","attacked_edge_full","healthy_edge_full_false_alarm","exact_support_full",
         "attack_target_mse","attack_five_mse","attack_network_mse",
         "recovery_target_mse","recovery_five_mse","recovery_network_mse")


def csvout(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def scene_values(folder,run,mi,scene,mask,arms):
    attacked=study.MASKS[mask];healthy=np.setdiff1d(np.arange(6),attacked)
    path=folder/f"{exp.MODEL[mi]}_{scene}.npz"
    result={arm:[] for arm in arms}; perlink=[]; scenario=[]
    with np.load(path) as archive:
        n=len(archive["path_id"])
        for arm in arms:
            early=archive[f"{mask}__{arm}_early_link"].astype(bool)
            full=archive[f"{mask}__{arm}_full_link"].astype(bool)
            result[arm]={
                "attacked_edge_early":early[:,attacked].mean(1),
                "healthy_edge_early_false_alarm":early[:,healthy].mean(1),
                "healthy_path_any_early_false_alarm":early[:,healthy].any(1).astype(float),
                "exact_support_early":(early[:,attacked].all(1)&~early[:,healthy].any(1)).astype(float),
                "attacked_edge_full":full[:,attacked].mean(1),
                "healthy_edge_full_false_alarm":full[:,healthy].mean(1),
                "exact_support_full":(full[:,attacked].all(1)&~full[:,healthy].any(1)).astype(float),
            }
            for metric in ("attack_target_mse","attack_five_mse","attack_network_mse",
                           "recovery_target_mse","recovery_five_mse","recovery_network_mse"):
                result[arm][metric]=archive[f"{mask}__{arm}_{metric}"]
            delay=archive[f"{mask}__{arm}_delay60"]
            for edge in range(6):
                perlink.append({"model":exp.MODEL[mi],"scene":scene,"support":mask,"arm":arm,
                                "edge":edge,"receiver":int(exp.old.backend(exp.MODEL[mi])[1].links[edge,0]),
                                "attacked":int(edge in attacked),
                                "early_detection":float(early[:,edge].mean()),
                                "full_detection":float(full[:,edge].mean()),
                                "mean_delay60":float(delay[:,edge].mean())})
            scenario.append({"model":exp.MODEL[mi],"scene":scene,"support":mask,"arm":arm,
                             "attacked_edges":list(map(int,attacked)),"healthy_edges":list(map(int,healthy)),
                             **{k:float(v.mean()) for k,v in result[arm].items()}})
    return result,perlink,scenario


def summarize(split):
    folder=HERE/("partial_dev" if split=="dev" else "partial_confirm")
    run=json.loads((folder/"RUN.json").read_text())
    for f,h in run["archives"].items():
        if exp.old.base.digest(folder/f)!=h:raise ValueError(f"SHA mismatch: {f}")
    split_code=3 if split=="dev" else 4
    arms=study.ARMS;rng=np.random.default_rng([20261127,52387,split_code,9001])
    absolute=[];contrasts=[];scenes=[];perlink=[];health=[]
    for mask in study.MASKS:
        cells={metric:{arm:[] for arm in arms} for metric in METRICS}
        cell_scene=[]
        for mi,name in enumerate(exp.MODEL):
            for scene in range(6):
                values,links,scene_rows=scene_values(folder,run,mi,scene,mask,arms)
                perlink.extend(links);scenes.extend(scene_rows)
                cell_scene.append((name,scene,values))
                for arm in arms:
                    for metric in METRICS:cells[metric][arm].append(values[arm][metric])
        for metric in METRICS:
            scene_boot={}
            for arm in arms:
                vals=cells[metric][arm];boot=[]
                for x in vals:
                    ix=rng.integers(0,len(x),size=(BOOT,len(x)))
                    boot.append(x[ix].mean(1))
                scene_boot[arm]=np.mean(boot,axis=0)
                point=float(np.mean([x.mean() for x in vals]))
                ci=np.quantile(scene_boot[arm],[.025,.975])
                absolute.append({"support":mask,"arm":arm,"metric":metric,"mean":point,
                                 "lower":ci[0],"upper":ci[1],"paths_per_scene":run["identity"]["paths_per_scene"]})
            for comp,weights in CONTRASTS.items():
                point=sum(w*np.mean([x.mean() for x in cells[metric][arm]]) for arm,w in weights.items())
                boot=sum(w*scene_boot[arm] for arm,w in weights.items())
                ci=np.quantile(boot,[.025,.975])
                contrasts.append({"support":mask,"comparison":comp,"metric":metric,
                                 "difference":point,"lower":ci[0],"upper":ci[1]})
        for model in exp.MODEL:
            # Report exact support as a path-level event, with exact binomial interval as a secondary check.
            file=folder/f"{model}_0_0.npz"
            # FWER belongs to the fully healthy W1 paths, and is read separately below.
    for model in exp.MODEL:
        for arm in arms:
            source=exp.HERE/"confirm/health.csv"
            with source.open(newline="",encoding="utf-8") as stream:
                rows=list(csv.DictReader(stream))
            record=next(x for x in rows if x["model"]==model and x["arm"]==arm)
            health.append({"model":model,"arm":arm,"W1_full_healthy_FWER":float(record["FWER"]),
                           "CP95_lower":float(record["lower"]),"CP95_upper":float(record["upper"]),
                           "calibration_source_sha256":run["identity"]["W1_threshold_manifest_sha256"]})
    csvout(folder/"absolute.csv",absolute);csvout(folder/"contrasts.csv",contrasts)
    csvout(folder/"per_link.csv",perlink);csvout(folder/"scenes.csv",scenes);csvout(folder/"healthy_fwer.csv",health)
    analysis={"split":split,"run_sha256":exp.old.base.digest(folder/"RUN.json"),
              "bootstrap_replicates":BOOT,"bootstrap_seed":[20261127,52387,split_code,9001],
              "weighting":"12 fixed model-angle-onset cells equally; whole paths resampled inside cells",
              "support_masks":{k:list(map(int,v)) for k,v in study.MASKS.items()},
              "threshold_source":"W1 full six-edge healthy calibration, frozen unchanged",
              "exact_support_event":"all attacked links alarm within first three attack reports and no healthy link alarms in the same window",
              "tables":{"absolute":"absolute.csv","contrasts":"contrasts.csv","per_link":"per_link.csv",
                        "scenes":"scenes.csv","healthy_FWER":"healthy_fwer.csv"}}
    exp.old.base._atomic_json(folder/"ANALYSIS.json",analysis)
    if split=="dev":
        lock={"development_manifest_sha256":exp.old.base.digest(folder/"RUN.json"),
              "development_analysis_sha256":exp.old.base.digest(folder/"ANALYSIS.json"),
              "source_sha256":run["identity"]["source_sha256"],
              "threshold_manifest_sha256":run["identity"]["W1_threshold_manifest_sha256"],
              "paths_per_scene":run["identity"]["paths_per_scene"],
              "confirmation_paths_per_scene":512,
              "prespecified_method_set":list(arms),
              "prespecified_support_masks":{k:list(map(int,v)) for k,v in study.MASKS.items()}}
        exp.old.base._atomic_json(HERE/"PARTIAL_LOCK.json",lock)
    return analysis


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=("dev","confirm"));a=p.parse_args()
    print(json.dumps(summarize(a.split),indent=2))

\n