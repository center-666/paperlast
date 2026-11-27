"""Correct contrast labels while preserving locked bootstrap magnitudes and samples."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE/"task4_confirm"
INVERT = {
    "iid_S_minus_A", "no_repeat_S_minus_A",
    "iid_S_minus_A_M7", "no_repeat_S_minus_A_M7",
    "no_repeat_minus_iid_A", "no_repeat_minus_iid_S",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    original = read_csv(OUT/"paired.csv")
    corrected = []
    for source in original:
        row = dict(source)
        if row["comparison"] in INVERT:
            old = row["difference"]
            row["difference"] = repr(-float(old))
            row["lower95"] = repr(-float(source["upper95"]))
            row["upper95"] = repr(-float(source["lower95"]))
            row["direction_correction"] = "The frozen analyzer computed left minus right while the label states right minus left. Negated the point and reversed interval endpoints. Bootstrap draws and magnitude are unchanged."
        else:
            row["direction_correction"] = "The registered difference direction already matches this interaction label."
        corrected.append(row)
    path = OUT/"paired_direction_corrected.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(corrected[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(corrected)

    absolute = read_csv(OUT/"absolute.csv")
    errors = []
    for row in corrected:
        group, label, metric = row["group"], row["comparison"], row["metric"]
        if metric not in ("attack_full", "attack_active", "attack_inactive",
                          "network_attack", "recovery", "health_post20",
                          "network_post20", "max_missing_sq"):
            continue
        means = {}
        for water, fusion in (("iid", "A_q0_raw"), ("iid", "S_fixed_table_T"),
                              ("iid", "A_M7_min"), ("no_repeat", "A_q0_raw"),
                              ("no_repeat", "S_fixed_table_T"), ("no_repeat", "A_M7_min")):
            source = next(a for a in absolute if a["configuration"] == group
                          and a["fixture"] == "equal_weighted"
                          and a["watermark"] == water and a["fusion"] == fusion
                          and a["metric"] == metric)
            means[water, fusion] = float(source["mean"])
        definitions = {
            "iid_S_minus_A": means["iid", "S_fixed_table_T"]-means["iid", "A_q0_raw"],
            "no_repeat_S_minus_A": means["no_repeat", "S_fixed_table_T"]-means["no_repeat", "A_q0_raw"],
            "iid_S_minus_A_M7": means["iid", "S_fixed_table_T"]-means["iid", "A_M7_min"],
            "no_repeat_S_minus_A_M7": means["no_repeat", "S_fixed_table_T"]-means["no_repeat", "A_M7_min"],
            "no_repeat_minus_iid_A": means["no_repeat", "A_q0_raw"]-means["iid", "A_q0_raw"],
            "no_repeat_minus_iid_S": means["no_repeat", "S_fixed_table_T"]-means["iid", "S_fixed_table_T"],
            "interaction_new_vs_old_S_minus_A":
                (means["no_repeat", "S_fixed_table_T"]-means["no_repeat", "A_q0_raw"])
                -(means["iid", "S_fixed_table_T"]-means["iid", "A_q0_raw"]),
        }
        error = abs(float(row["difference"])-definitions[label])
        errors.append(error)
        np.testing.assert_allclose(float(row["difference"]), definitions[label],
                                   rtol=1e-10, atol=1e-9)
    result = {
        "status": "PASS", "source_sha256": sha(Path(__file__)),
        "confirm_run_sha256": sha(OUT/"RUN.json"),
        "locked_original_analyzer_sha256": json.loads((OUT/"ANALYSIS.json").read_text())["source_sha256"],
        "locked_original_paired_sha256": sha(OUT/"paired.csv"),
        "corrected_paired_sha256": sha(path),
        "rows": len(corrected), "contrasts_recomputed_from_absolute_means": len(errors),
        "maximum_point_mean_consistency_error": max(errors, default=0.),
        "correction": "Only six sign labels were reversed to match the declared estimand. The original paired.csv and ANAYSIS.json remain unchanged. A_M7 comparisons and interaction are preserved.",
        "confirmatory_paths_modified": False,
        "source_lock_modified": False,
    }
    (OUT/"DIRECTION_CORRECTION.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
