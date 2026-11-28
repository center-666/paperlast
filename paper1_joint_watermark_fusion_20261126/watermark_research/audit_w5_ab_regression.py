"""Independent read-only reduction of the already-open W5 A/B endpoint archives."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = {"iid":"quarter_iid_exact", "no_repeat":"quarter_no_repeat_exact"}
BOOT = 20000
SEED = [20261128, 45201, 16]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run_path = args.archive/"RUN.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    verify = {}
    all_scene = []
    for model in ("unstable", "stable"):
        for scene in range(6):
            filename = f"{model}_attack_{scene}.npz"
            path = args.archive/filename
            digest = sha(path)
            if run["archives"][filename] != digest:
                raise ValueError(f"W5 archive hash mismatch: {filename}")
            with np.load(path) as archive:
                ids = archive["path_id"]
                np.testing.assert_array_equal(ids, np.arange(512))
                for arm, tag in ARMS.items():
                    early = archive[f"{tag}_early_link"].astype(bool)
                    if early.shape != (512, 6):
                        raise ValueError(f"Unexpected early-link mask shape for {filename} {arm}")
                    mse = archive[f"{tag}_attack_mse"].copy()
                    recovery = archive[f"{tag}_recovery_mse"].copy()
                    large = archive[f"{tag}_large_late_accept_count"].copy()
                    ref = archive["negative_exact_attack_mse"]
                    tail = (mse > 1.) & (mse >= 10*ref)
                    all_scene.append({
                        "model":model, "scene":scene, "arm":arm, "mask":early,
                        "early_path":early.mean(axis=1), "attack_mse":mse,
                        "recovery_mse":recovery, "late_large_count":large,
                        "late_large_path":large > 0, "tail":tail,
                    })
                verify[filename] = digest
    strata = [(m,s) for m in ("unstable","stable") for s in range(6)]
    metrics = ("early_path","attack_mse","recovery_mse","late_large_path","tail")
    rng = np.random.default_rng(np.random.SeedSequence(SEED))
    draws = {metric:np.zeros((BOOT,2)) for metric in metrics}
    points = {metric:np.zeros(2) for metric in metrics}
    scene_rows = []
    for stratum_index,(model,scene) in enumerate(strata):
        rows = {r["arm"]:r for r in all_scene if r["model"]==model and r["scene"]==scene}
        left,right = rows["no_repeat"],rows["iid"]
        for metric in metrics:
            a,b = left[metric],right[metric]
            points[metric] += np.array([a.mean(),b.mean()])/len(strata)
            scene_rows.append({"model":model,"scene":scene,"endpoint":metric,
                               "no_repeat":float(a.mean()),"iid":float(b.mean()),
                               "difference_no_repeat_minus_iid":float(a.mean()-b.mean()),
                               "n_paths":len(a)})
        combos = np.column_stack((left["late_large_path"],right["late_large_path"]))
        scene_rows.append({"model":model,"scene":scene,"endpoint":"late_large_event_pair_counts",
                           "no_repeat_only":int(np.sum((combos[:,0]==1)&(combos[:,1]==0))),
                           "iid_only":int(np.sum((combos[:,0]==0)&(combos[:,1]==1))),
                           "both":int(np.sum(combos.all(axis=1))),
                           "neither":int(np.sum(~combos.any(axis=1))),"n_paths":len(combos)})
        tail_pair=np.column_stack((left["tail"],right["tail"]))
        scene_rows.append({"model":model,"scene":scene,"endpoint":"registered_tail_pair_counts",
                           "no_repeat_only":int(np.sum((tail_pair[:,0]==1)&(tail_pair[:,1]==0))),
                           "iid_only":int(np.sum((tail_pair[:,0]==0)&(tail_pair[:,1]==1))),
                           "both":int(np.sum(tail_pair.all(axis=1))),
                           "neither":int(np.sum(~tail_pair.any(axis=1))),"n_paths":len(tail_pair)})
        for start in range(0,BOOT,250):
            count=min(250,BOOT-start)
            index=rng.integers(0,512,size=(count,512))
            for metric in metrics:
                a=left[metric][index].mean(axis=1)
                b=right[metric][index].mean(axis=1)
                draws[metric][start:start+count] += np.column_stack((a,b))/len(strata)
    overall=[]
    for metric in metrics:
        delta=draws[metric][:,0]-draws[metric][:,1]
        overall.append({"endpoint":metric,"no_repeat":float(points[metric][0]),
                        "iid":float(points[metric][1]),
                        "difference_no_repeat_minus_iid":float(points[metric][0]-points[metric][1]),
                        "lower95":float(np.quantile(delta,.025)),
                        "upper95":float(np.quantile(delta,.975)),
                        "unit":"12 scenes equally weighted; path-paired bootstrap within scene"})
    for arm,tag in ARMS.items():
        pass
    args.out.mkdir(parents=True,exist_ok=True)
    for filename, rows in (("W5_AB_REGRESSION_SCENES.csv",scene_rows),
                           ("W5_AB_REGRESSION_OVERALL.csv",overall)):
        with (args.out/filename).open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=sorted({key for row in rows for key in row}),lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    receipt={"status":"PASS_readonly_W5_A_B_endpoint_reduction",
             "stage":"post-hoc regression on already-viewed W5 split-16 archives; not a replayed physical simulation or new confirmation",
             "W5_RUN_sha256":sha(run_path),"archives_sha256":verify,
             "scenario_count":12,"paths_per_scenario":512,
             "stored_endpoints":["per-edge first-three-packet at-least-once mask","five-node attack MSE",
                                 "five-node recovery MSE","first-alarm-following large-message accept count",
                                 "registered tail event mask"],
             "not_stored":"Per-tick score/gate time series are not present in the W5 NPZ endpoint archives and are not reconstructed here.",
             "bootstrap_replicates":BOOT,"bootstrap_seed":SEED,
             "summary":overall,
             "outputs_sha256":{name:sha(args.out/name) for name in
                               ("W5_AB_REGRESSION_SCENES.csv","W5_AB_REGRESSION_OVERALL.csv")}}
    (args.out/"W5_AB_REGRESSION.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":receipt["status"],"summary":overall},indent=2))


if __name__ == "__main__":
    main()
