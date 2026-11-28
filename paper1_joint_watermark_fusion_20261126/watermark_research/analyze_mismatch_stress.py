"""Audit zero-mismatch regression and summarize W5 mismatch paths."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent.parent
OUT=HERE/"watermark_research"
BASE=Path(r"C:\Users\zhongxin\shixidoc\paperlast-watermark-w1\paper1_watermark_early_detection_20261125\comparators\markov_confirm")
MAPPING={"quarter_no_repeat":"quarter_no_repeat_exact","quarter_iid":"quarter_iid_exact","negative":"negative_exact"}
METHODS=tuple(MAPPING)
FRACTIONS=(0.,.5,.9,1.,1.1,2.)
BOOT=20000
SEED=[20261128,9521,78,9901]


def sha(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    with (OUT/"MISMATCH_PATHS.csv").open(encoding="utf-8",newline="") as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=2*6*6*3*512:raise ValueError(f"Expected complete mismatch matrix, got {len(rows)} rows.")
    regressions=[]
    for row in rows:
        if float(row["fraction"])!=0.:continue
        path=BASE/f'{row["model"]}_attack_{row["scene"]}.npz'
        with np.load(path) as archive:
            method=MAPPING[row["watermark"]]
            i=int(row["path_id"])
            for field,key in (("early_link_rate",method+"_early_link"),
                              ("attack_mse",method+"_attack_mse"),
                              ("recovery_mse",method+"_recovery_mse")):
                arr=archive[key][i]
                if field=="early_link_rate":
                    val=float(arr.mean())
                else:val=float(arr)
                regressions.append(abs(float(row[field])-val))
    if max(regressions)>3e-10:
        raise ValueError(f"Zero-mismatch W5 replay failed: {max(regressions)}")
    groups=[]
    for model in ("unstable","stable"):
        for fraction in FRACTIONS:
            for method in METHODS:
                strata=[]
                for scene in range(6):
                    path_rows=[r for r in rows if r["model"]==model and int(r["scene"])==scene
                               and float(r["fraction"])==fraction and r["watermark"]==method]
                    path_rows.sort(key=lambda r:int(r["path_id"]))
                    strata.append({k:np.asarray([float(x[k]) for x in path_rows])
                                   for k in ("early_link_rate","attack_mse","recovery_mse",
                                             "late_large_accept","white_energy_first3",
                                             "projection_energy_first3","projection_fraction_first3",
                                             "score_max_first3")})
                record={"model":model,"fraction":fraction,"watermark":method}
                for metric in strata[0]:
                    point=float(np.mean([s[metric].mean() for s in strata]))
                    rng=np.random.default_rng(np.random.SeedSequence([*SEED,0 if model=="unstable" else 1,
                                                                        FRACTIONS.index(fraction),METHODS.index(method)]))
                    draws=np.zeros(BOOT)
                    for values in strata:
                        n=len(values[metric])
                        for start in range(0,BOOT,500):
                            size=min(500,BOOT-start)
                            weights=rng.multinomial(n,np.full(n,1/n),size=size)/n
                            draws[start:start+size]+=weights@values[metric]/len(strata)
                    record[metric]=point
                    record[metric+"_lower95"]=float(np.quantile(draws,.025))
                    record[metric+"_upper95"]=float(np.quantile(draws,.975))
                groups.append(record)
    path=OUT/"MISMATCH_SCENARIO_SUMMARY.csv"
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(groups[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(groups)
    # Explicit mismatch minus zero paired by path, scene, method.
    paired=[]
    for model in ("unstable","stable"):
        for fraction in FRACTIONS[1:]:
            for method in METHODS:
                for metric in ("early_link_rate","attack_mse","recovery_mse",
                               "late_large_accept","white_energy_first3",
                               "projection_energy_first3","projection_fraction_first3"):
                    differences=[]
                    for scene in range(6):
                        lookup={(int(r["path_id"]),float(r["fraction"])):float(r[metric]) for r in rows
                                if r["model"]==model and int(r["scene"])==scene and r["watermark"]==method}
                        differences.append(np.array([lookup[p,fraction]-lookup[p,0.]
                                                     for p in range(512)]))
                    point=float(np.mean([x.mean() for x in differences]))
                    rng=np.random.default_rng(np.random.SeedSequence([*SEED,9,0 if model=="unstable" else 1,
                                                                        FRACTIONS.index(fraction),METHODS.index(method)]))
                    draws=np.zeros(BOOT)
                    for values in differences:
                        n=len(values)
                        for start in range(0,BOOT,500):
                            size=min(500,BOOT-start)
                            weights=rng.multinomial(n,np.full(n,1/n),size=size)/n
                            draws[start:start+size]+=weights@values/6.
                    paired.append({"model":model,"fraction":fraction,"watermark":method,
                                   "contrast":f"{metric}_minus_zero_mismatch","difference":point,
                                   "lower95":float(np.quantile(draws,.025)),
                                   "upper95":float(np.quantile(draws,.975)),
                                   "scenarios":6,"paths_per_scenario":512})
    p2=OUT/"MISMATCH_PAIRED_MINUS_ZERO.csv"
    with p2.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(paired)
    report={"status":"PASS_zero_mismatch_W5_regression_and_summary",
            "path_rows":len(rows),"scenarios":12,"paths_per_scenario":512,
            "methods":list(METHODS),"mismatch_fractions":list(FRACTIONS),
            "zero_mismatch_max_abs_endpoint_error":max(regressions),
            "bootstrap_replicates":BOOT,"seed":SEED,
            "direction_lock_sha256":sha(OUT/"MISMATCH_REPLAY_LOCK.json"),
            "path_file_sha256":sha(OUT/"MISMATCH_PATHS.csv"),
            "summary_sha256":sha(path),"paired_zero_contrast_sha256":sha(p2),
            "scope":"Post-hoc stress replay on existing W5 split-16 paths. Whole-path pairing within each fixed scene. Not independent confirmation."}
    (OUT/"MISMATCH_RESULTS.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
