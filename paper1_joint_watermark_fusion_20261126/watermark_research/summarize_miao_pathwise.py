"""Reduce pathwise regressions and intervals into endpoint-specific Miao contrasts."""
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent.parent
p=HERE/"watermark_research/MIAO_PATH_RESULTS.csv"
rows=list(csv.DictReader(p.open(encoding="utf-8")))
out=[]
for model in ("unstable","stable"):
 for reference in ("quarter_iid","quarter_no_repeat","negative"):
  for metric in ("early_fraction","attack_mse","recovery_mse","large_late_accept"):
   diffs=[]
   for scene in range(6):
    a={(r["path_id"],r["method"]):float(r[metric]) for r in rows if r["model"]==model and int(r["scene"])==scene}
    ids=sorted({k[0] for k in a})
    diffs.append([a[(i,"miao_local")]-a[(i,reference)] for i in ids])
   point=sum(sum(v)/len(v) for v in diffs)/6
   out.append({"model":model,"comparison":f"miao_local_minus_{reference}","endpoint":metric,
               "difference":point,"paths_per_scenario":512,"scenarios":6})
with (HERE/"watermark_research/MIAO_ENDPOINT_CONTRASTS.csv").open("w",encoding="utf-8",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]),lineterminator="\n");w.writeheader();w.writerows(out)
