"""Corrected window-only reanalysis of the frozen evaluation paths."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent.parent
OUT=HERE/"class_weight_development"
METHODS=("unified","topology_two","three_class","frozen_T","saturation")
PAIRS=(("topology_two","unified","two_minus_unified"),
("three_class","topology_two","three_minus_two"),
("three_class","unified","three_minus_unified"),
("frozen_T","unified","T_minus_unified"),
("frozen_T","three_class","T_minus_three"),
("saturation","frozen_T","saturation_minus_T"),
("saturation","topology_two","saturation_minus_two"))
METRICS=("full_mixed","continuous_conditional","burst_conditional",
"burst_active","burst_inactive","recovery_mixed")
BOOT=20000
SEED=[20261127,94017,271,991]
def sha(path):
 with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
 configs=("U_unstable_R35x0.5","U_unstable_R35x1","U_unstable_R35x2","R_unstable_R35x0.5","R_unstable_R35x1","R_unstable_R35x2","U_stable_R35x0.5","U_stable_R35x1","U_stable_R35x2","R_stable_R35x0.5","R_stable_R35x1","R_stable_R35x2")
 groups={"all":range(12),"U":(0,1,2,6,7,8),"R":(3,4,5,9,10,11),"U_unstable":(0,1,2),"U_stable":(6,7,8),"primary":(2,)}
 rng=np.random.default_rng(np.random.SeedSequence(SEED));rows=[];burst_path_error=0.
 archive=np.load(OUT/"pathwise_actions.npz")
 receipt=json.loads((OUT/"EVALUATION.json").read_text(encoding="utf-8"))
 assert sha(OUT/"pathwise_actions.npz")==receipt["pathwise_actions_sha256"]
 for group,indices in groups.items():
  strata=[(i,f) for i in indices for f in ("continuous","burst")]
  health_indices=list(indices)
  attack_draw=np.zeros((BOOT,len(METHODS),len(METRICS)))
  health_draw=np.zeros((BOOT,len(METHODS)))
  action_draw=np.zeros((BOOT,len(PAIRS),2))
  attack_point=np.zeros((len(METHODS),len(METRICS)))
  health_point=np.zeros(len(METHODS))
  for i,fixture in strata:
   loss=np.stack([archive[f"losses_{i:02}_{fixture}_{m}"][:,[0,1,2,4]] for m in METHODS],axis=1)
   actions=np.stack([archive[f"actions_{i:02}_{fixture}_{m}"][:,50:110] for m in METHODS],axis=1)
   alarm=archive[f"gates_{i:02}_{fixture}"][:,50:110,6].astype(bool)
   if fixture=="burst":
    err=float(np.max(abs(loss[:,:,0]-.6*loss[:,:,1]-.4*loss[:,:,2])))
    burst_path_error=max(burst_path_error,err)
    if err>2e-12:raise AssertionError(f"six-open/four-closed path identity failed at config {i}: {err}")
    block=np.stack((loss[:,:,0]/(2*len(indices)),np.zeros_like(loss[:,:,0]),
                    loss[:,:,0]/len(indices),loss[:,:,1]/len(indices),loss[:,:,2]/len(indices),
                    loss[:,:,3]/(2*len(indices))),axis=2)
   else:
    block=np.stack((loss[:,:,0]/(2*len(indices)),loss[:,:,0]/len(indices),
                    np.zeros_like(loss[:,:,0]),np.zeros_like(loss[:,:,0]),
                    np.zeros_like(loss[:,:,0]),loss[:,:,3]/(2*len(indices))),axis=2)
   attack_point+=block.mean(axis=0)
   acts=[]
   for left,right,_ in PAIRS:
    changed=(actions[:,METHODS.index(left)]!=actions[:,METHODS.index(right)]).sum(axis=1)
    n_alarm=alarm.sum(axis=1)
    acts.append(np.stack((changed/60,np.divide(changed,n_alarm,out=np.zeros(len(changed),float),where=n_alarm>0)),axis=1))
   acts=np.stack(acts,axis=1);n=len(loss)
   for start in range(0,BOOT,500):
    size=min(500,BOOT-start);w=rng.multinomial(n,np.full(n,1/n),size=size)/n
    attack_draw[start:start+size]+=(w@block.reshape(n,-1)).reshape(size,len(METHODS),len(METRICS))
    action_draw[start:start+size]+=(w@acts.reshape(n,-1)).reshape(size,len(PAIRS),2)/(2*len(indices))
  for i in health_indices:
   values=np.stack([archive[f"losses_{i:02}_health_{m}"][:,5] for m in METHODS],axis=1)
   health_point+=values.mean(axis=0)/len(health_indices);n=len(values)
   for start in range(0,BOOT,500):
    size=min(500,BOOT-start);w=rng.multinomial(n,np.full(n,1/n),size=size)/n
    health_draw[start:start+size]+=w@values/len(health_indices)
  for pi,(left,right,label) in enumerate(PAIRS):
   li,ri=METHODS.index(left),METHODS.index(right)
   for mi,metric in enumerate(METRICS):
    draws=attack_draw[:,li,mi]-attack_draw[:,ri,mi]
    rows.append({"group":group,"comparison":label,"metric":metric,
       "difference":float(attack_point[li,mi]-attack_point[ri,mi]),
       "lower95":float(np.quantile(draws,.025)),"upper95":float(np.quantile(draws,.975)),
       "paths":256*len(strata),"strata":len(strata),"bootstrap_replicates":BOOT})
   draws=health_draw[:,li]-health_draw[:,ri]
   rows.append({"group":group,"comparison":label,"metric":"health_independent",
       "difference":float(health_point[li]-health_point[ri]),
       "lower95":float(np.quantile(draws,.025)),"upper95":float(np.quantile(draws,.975)),
       "paths":256*len(health_indices),"strata":len(health_indices),"bootstrap_replicates":BOOT})
   for ai,metric in enumerate(("action_disagreement_per_tick","action_disagreement_given_alarm")):
    draws=action_draw[:,pi,ai]
    rows.append({"group":group,"comparison":label,"metric":metric,"difference":float(draws.mean()),
       "lower95":float(np.quantile(draws,.025)),"upper95":float(np.quantile(draws,.975)),
       "paths":256*len(strata),"strata":len(strata),"bootstrap_replicates":BOOT})
 out=OUT/"window_corrected_paired.csv"
 with out.open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
 lookup={(r["group"],r["comparison"],r["metric"]):r for r in rows}
 mixed_identity=max(abs(float(lookup[g,c,"full_mixed"]["difference"])
   -.5*float(lookup[g,c,"continuous_conditional"]["difference"])
   -.5*float(lookup[g,c,"burst_conditional"]["difference"]))
   for g in groups for _,_,c in PAIRS)
 result={"status":"PASS_corrected_fusion_window_reanalysis",
  "pathwise_archive_sha256":sha(OUT/"pathwise_actions.npz"),
  "fit_sha256":sha(OUT/"FIT_PARAMETERS.json"),"evaluation_receipt_sha256":sha(OUT/"EVALUATION.json"),
  "bootstrap_replicates":BOOT,"bootstrap_seed":SEED,
  "unit":"Whole path within configuration x fixture; one shared bootstrap draw across all methods, windows and action outcomes.",
  "burst_pathwise_identity_max_abs_error":burst_path_error,
  "mixed_equals_half_each_conditional_mean_max_error":mixed_identity,
  "activity_closed_only_from_burst_fixture":True,
  "health_only_from_independent_purpose6_paths":True,
  "output_sha256":sha(out),"source_sha256":sha(Path(__file__))}
 (OUT/"WINDOW_CORRECTION_FINAL.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
