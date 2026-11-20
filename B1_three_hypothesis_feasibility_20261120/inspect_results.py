"""Compact output review; no changes to experiment inputs or policies."""
import json
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
summary = json.loads((here / "results/summary.json").read_text(encoding="utf-8"))
histories = json.loads((here / "results/histories.json").read_text(encoding="utf-8"))
for model, record in summary["one_step"].items():
    print(model, "fixed", record["development_fixed_v"])
    for policy, row in record["comparisons"].items():
        print(policy, "active", row["active_histories"], "/", row["histories"],
              "info", row["information_gain"], "Bayes actual", row["true_conditional_bayes_change"],
              "CG actual", row["true_conditional_common_change"],
              "healthy CG", row["healthy_common_change"], "dispersion", row["max_abs_dispersion_change"])
    subset = [r for r in histories if r["model"] == model and r["partition"] == "evaluation"]
    identity = np.array([r["stats"]["variance_identity_sensitivity_error"] for r in subset])
    print("identity sensitivity max", np.max(np.linalg.norm(identity, axis=1)))
print("refinements")
for row in summary["quadrature_refinement"]:
    print(row["model"], row["true_h"], "gain", row["coarse_gain"], row["refined_gain"], "error", row["gI_error"])
print("regional")
print(json.dumps(summary["regional_gaussian_watermark_replay"], indent=2))
