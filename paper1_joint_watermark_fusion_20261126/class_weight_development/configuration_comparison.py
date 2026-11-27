"""Extract fixed-weight versus class-weight development results by configuration."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent.parent
OUT=HERE/"class_weight_development"


def main():
 fit=json.loads((OUT/"FIT_PARAMETERS.json").read_text(encoding="utf-8"))
 rows=list(csv.DictReader((OUT/"evaluation.csv").open(encoding="utf-8")))
 configs=[]
 for config in sorted({row["config"] for row in rows}):
  entry=next(row for row in rows if row["config"]==config)
  two_q=fit["topology_two_q"]["U" if entry["topology"]=="U" else "R"]
  three_key=("U_unstable" if entry["dynamics"]=="unstable" else "U_stable") if entry["topology"]=="U" else "R"
  three_q=fit["three_class_q"][three_key]
  result={"configuration":config,"topology":entry["topology"],"dynamics":entry["dynamics"],
          "R35_factor":entry["factor"],"unified_q":fit["unified_q"],"two_class_q":two_q,
          "three_class_q":three_q}
  for method,name in (("unified","unified"),("topology_two","two_class"),("three_class","three_class"),("frozen_T","frozen_T")):
   attacks=[row for row in rows if row["config"]==config and row["fixture"] in ("continuous","burst") and row["method"]==method]
   health=next(row for row in rows if row["config"]==config and row["fixture"]=="health" and row["method"]==method)
   for key,col in (("attack_mse","full"),("burst_active_mse","burst_on"),("burst_closed_mse","burst_off"),("recovery_mse","recovery")):
    result[name+"_"+key]=sum(float(row[col]) for row in attacks)/2
   result[name+"_health_mse"]=float(health["health"])
  configs.append(result)
 target=OUT/"CONFIGURATION_COMPARISON.csv"
 with target.open("w",encoding="utf-8",newline="") as stream:
  writer=csv.DictWriter(stream,fieldnames=list(configs[0]),lineterminator="\n")
  writer.writeheader();writer.writerows(configs)
 print(json.dumps({"status":"PASS","configurations":len(configs),"path_means":"Continuous and burst equally weighted within each configuration.","output":str(target)}))


if __name__=="__main__": main()
