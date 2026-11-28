import csv,json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent.parent;OUT=HERE/"class_weight_development";B=20000;SEED=[20261127,94017,271,991]
GROUPS={"all":range(12),"U":list(range(0,3))+list(range(6,9)),"R":list(range(3,6))+list(range(9,12)),"U_unstable":[0,1,2],"U_stable":[6,7,8],"primary":[2]}
PAIRS=(("topology_two","unified","two_minus_unified"),("three_class","topology_two","three_minus_two"),("three_class","unified","three_minus_unified"),("frozen_T","unified","T_minus_unified"),("frozen_T","three_class","T_minus_three"),("saturation","frozen_T","saturation_minus_T"),("saturation","topology_two","saturation_minus_two"))
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
 z=np.load(OUT/"pathwise_actions.npz");path=OUT/"window_corrected_paired.csv"
 with path.open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
 rng=np.random.default_rng(np.random.SeedSequence(SEED))
 for group,indices in GROUPS.items():
  for left,right,label in PAIRS:
   strata=[]
   for i in indices:
    values=z[f"losses_{i:02}_health_{left}"][:,5]-z[f"losses_{i:02}_health_{right}"][:,5]
    strata.append(values)
   point=float(np.mean([v.mean() for v in strata]))
   draws=np.zeros(B)
   for v in strata:
    n=len(v)
    for start in range(0,B,500):
     size=min(500,B-start);w=rng.multinomial(n,np.full(n,1/n),size=size)/n
     draws[start:start+size]+=w@v/len(strata)
   row=next(r for r in rows if r["group"]==group and r["comparison"]==label and r["metric"]=="health_independent")
   row["difference"]=str(point);row["lower95"]=str(float(np.quantile(draws,.025)));row["upper95"]=str(float(np.quantile(draws,.975)))
 with path.open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
 meta=OUT/"WINDOW_CORRECTION.json";report=json.loads(meta.read_text())
 report["output_sha256"]=sha(path);report["health_bootstrap"]="Direct bootstrap of paired method differences within independent purpose-6 paths and configuration strata."
 report["source_sha256"]=sha(Path(__file__));meta.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
 print("health paired CIs updated",report["output_sha256"])
if __name__=="__main__":main()
