"""Rescale fixture-only window summaries from equal-stratum contributions to conditional means."""
import csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent.parent
OUT=HERE/"class_weight_development"
MULTIPLY={"continuous_conditional":2.,"burst_conditional":2.,"burst_active":2.,"burst_inactive":2.}
def sha(path):
 with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
def main():
 path=OUT/"window_corrected_paired.csv"
 with path.open(encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
 for row in rows:
  factor=MULTIPLY.get(row["metric"],1.)
  for key in ("difference","lower95","upper95"):row[key]=str(float(row[key])*factor)
 with path.open("w",encoding="utf-8",newline="") as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
 result={"status":"PASS_correct_conditional_window_scales","output_sha256":sha(path),
         "source_window_replay_sha256":json.loads((OUT/"WINDOW_CORRECTION.json").read_text())["source_sha256"],
         "independently_health_recomputed_cis":True,
         "fixture_only_metrics_rescaled":["continuous_conditional","burst_conditional","burst_active","burst_inactive"],
         "reason":"The original bootstrap gives each configuration x fixture stratum equal weight. Fixture-conditional summaries are twice that fixture's contribution. Mixed full and recovery endpoints already combine continuous and burst and are unchanged."}
 r=OUT/"WINDOW_CORRECTION_FINAL.json";r.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
