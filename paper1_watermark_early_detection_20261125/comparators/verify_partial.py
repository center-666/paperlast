"""Independent endpoint and archive audit for partial-link localization."""
import csv
import json
from pathlib import Path
import numpy as np
import experiment as exp
import partial_study as study

HERE=study.HERE


def audit(split):
    folder=HERE/("partial_dev" if split=="dev" else "partial_confirm")
    run=json.loads((folder/"RUN.json").read_text())
    for filename,digest in run["archives"].items():
        if exp.old.base.digest(folder/filename)!=digest:
            raise ValueError(f"Path archive SHA mismatch: {filename}")
    rows={}
    with (folder/"scenes.csv").open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream):
            rows[(row["model"],int(row["scene"]),row["support"],row["arm"])]=row
    checked=0
    for mi,model in enumerate(exp.MODEL):
        for scene in range(6):
            with np.load(folder/f"{model}_{scene}.npz",allow_pickle=False) as archive:
                n=len(archive["path_id"])
                np.testing.assert_array_equal(archive["path_id"],np.arange(n))
                for support,edges in study.MASKS.items():
                    attacked=np.asarray(edges,dtype=int);healthy=np.setdiff1d(np.arange(6),attacked)
                    for arm in study.ARMS:
                        early=archive[f"{support}__{arm}_early_link"].astype(bool)
                        full=archive[f"{support}__{arm}_full_link"].astype(bool)
                        delay=archive[f"{support}__{arm}_delay60"]
                        np.testing.assert_array_equal(delay<3,early)
                        edge_early=early[:,attacked].mean(1)
                        healthy_early=early[:,healthy].mean(1)
                        exact=(early[:,attacked].all(1)&~early[:,healthy].any(1))
                        record=rows[(model,scene,support,arm)]
                        checks={
                            "attacked_edge_early":edge_early.mean(),
                            "healthy_edge_early_false_alarm":healthy_early.mean(),
                            "healthy_path_any_early_false_alarm":early[:,healthy].any(1).mean(),
                            "exact_support_early":exact.mean(),
                            "attacked_edge_full":full[:,attacked].mean(),
                            "healthy_edge_full_false_alarm":full[:,healthy].mean(),
                            "exact_support_full":(full[:,attacked].all(1)&~full[:,healthy].any(1)).mean(),
                            "attack_target_mse":archive[f"{support}__{arm}_attack_target_mse"].mean(),
                            "attack_five_mse":archive[f"{support}__{arm}_attack_five_mse"].mean(),
                            "attack_network_mse":archive[f"{support}__{arm}_attack_network_mse"].mean(),
                            "recovery_target_mse":archive[f"{support}__{arm}_recovery_target_mse"].mean(),
                            "recovery_five_mse":archive[f"{support}__{arm}_recovery_five_mse"].mean(),
                            "recovery_network_mse":archive[f"{support}__{arm}_recovery_network_mse"].mean(),
                        }
                        for metric,value in checks.items():
                            if abs(float(record[metric])-float(value))>1e-11:
                                raise AssertionError((model,scene,support,arm,metric,record[metric],value))
                        checked+=1
    return {"split":split,"checked_cells":checked,"paths_per_scene":run["identity"]["paths_per_scene"],
            "W1_threshold_manifest_sha256":run["identity"]["W1_threshold_manifest_sha256"],
            "run_manifest_sha256":exp.old.base.digest(folder/"RUN.json"),
            "archive_hashes_verified":len(run["archives"])}


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("split",choices=("dev","confirm"));a=p.parse_args()
    result=audit(a.split)
    exp.old.base._atomic_json(HERE/("partial_dev" if a.split=="dev" else "partial_confirm")/"VERIFY_INDEPENDENT.json",result)
    print(json.dumps(result,indent=2))
