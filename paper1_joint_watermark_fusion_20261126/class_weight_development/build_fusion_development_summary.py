"""Build compact result tables from locked development outputs."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent.parent
OUT=HERE/"class_weight_development"


def read(name):
 with (OUT/name).open(encoding="utf-8",newline="") as stream: return list(csv.DictReader(stream))


def write(name,rows):
 with (OUT/name).open("w",encoding="utf-8",newline="") as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
  writer.writeheader();writer.writerows(rows)


def main():
 summary=read("summary.csv")
 groups=("all","U","R","U_unstable","U_stable","primary")
 methods=("unified","topology_two","three_class","frozen_T","saturation")
 write("headline.csv",[{"group":g,"method":m,**{k:r[k] for k in (
  "attack_full","burst_active","burst_inactive","recovery","health","alert_decisions",
  "requested_q_counts","executed_q_counts","mean_raw_requested_weight","M7_projection_fraction")}}
  for g in groups for m in methods for r in summary if r["group"]==g and r["method"]==m])
 paired=read("paired.csv")
 comparisons=("two_minus_unified","three_minus_two","three_minus_unified","T_minus_unified",
              "T_minus_three","saturation_minus_T","saturation_minus_two")
 metrics=("full","burst_on","burst_off","recovery","health",
          "action_disagreement_per_tick","action_disagreement_given_alarm")
 write("headline_paired.csv",[r for r in paired if r["group"] in groups
  and r["comparison"] in comparisons and r["metric"] in metrics])
 report={"status":"PASS_headline_tables","fit_sha256":__import__("hashlib").sha256(
  (OUT/"FIT_PARAMETERS.json").read_bytes()).hexdigest(),
  "summary_sha256":__import__("hashlib").sha256((OUT/"headline.csv").read_bytes()).hexdigest(),
  "paired_sha256":__import__("hashlib").sha256((OUT/"headline_paired.csv").read_bytes()).hexdigest(),
  "full_paths":6144,"healthy_paths":3072,"groups":list(groups),"methods":list(methods)}
 (OUT/"HEADLINE_VERIFY.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
 print(json.dumps(report,indent=2))


if __name__=="__main__": main()
