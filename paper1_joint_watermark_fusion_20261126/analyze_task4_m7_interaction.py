"""Post-run paired interaction with the M7-feasible hard-rejection comparator."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import task3

HERE = Path(__file__).resolve().parent
OUT = HERE/"task4_confirm"
REPS = 20000
SEED = [20261126, 94017, 85, 9917]
GROUPS = {
    "U_unstable_R35x2_primary": (4, 5),
    "U_unstable_all_quality": tuple(range(0, 6)),
    "U_all_dynamics_quality": tuple(range(0, 6))+tuple(range(12, 18)),
    "all_12_configurations": tuple(range(24)),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    draws_out = {}
    for group, strata in GROUPS.items():
        paths_by_stratum = []
        for stratum in strata:
            config = stratum//2
            fixture = ("continuous", "burst")[stratum % 2]
            name = task3.QUALITY_ORDER[config]
            data = {}
            for watermark in ("iid", "no_repeat"):
                with np.load(OUT/f"{name}_{fixture}_attack.npz") as archive:
                    soft = archive[watermark+"_S_fixed_table_T_losses"][:, 0]
                    hard = archive[watermark+"_A_M7_min_losses"][:, 0]
                    q1 = archive[watermark+"_q1_full_losses"][:, 0]
                    data[watermark] = {"soft": soft, "hard": hard, "q1": q1}
            # Negative means the joint new-watermark plus soft-fusion gain is
            # larger relative to M7-feasible hard rejection.
            interaction = ((data["no_repeat"]["soft"]-data["no_repeat"]["hard"])
                           -(data["iid"]["soft"]-data["iid"]["hard"]))
            water_effect_soft = data["no_repeat"]["soft"]-data["iid"]["soft"]
            water_effect_hard = data["no_repeat"]["hard"]-data["iid"]["hard"]
            paired_m7_gain_no_repeat = data["no_repeat"]["soft"]-data["no_repeat"]["hard"]
            soft_minus_q1_iid = data["iid"]["soft"]-data["iid"]["q1"]
            soft_minus_q1_nr = data["no_repeat"]["soft"]-data["no_repeat"]["q1"]
            paths_by_stratum.append((interaction, water_effect_soft,
                                     water_effect_hard, paired_m7_gain_no_repeat,
                                     soft_minus_q1_iid, soft_minus_q1_nr))
        point = np.mean([item[0].mean() for item in paths_by_stratum])
        boot = np.zeros(REPS)
        boot_soft = np.zeros(REPS)
        boot_hard = np.zeros(REPS)
        boot_gain = np.zeros(REPS)
        boot_q1_iid = np.zeros(REPS)
        boot_q1_nr = np.zeros(REPS)
        for item in paths_by_stratum:
            n = len(item[0])
            first = 0
            while first < REPS:
                count = min(1000, REPS-first)
                weights = rng.multinomial(n, np.full(n, 1/n), size=count)/n
                boot[first:first+count] += weights @ item[0]/len(paths_by_stratum)
                boot_soft[first:first+count] += weights @ item[1]/len(paths_by_stratum)
                boot_hard[first:first+count] += weights @ item[2]/len(paths_by_stratum)
                boot_gain[first:first+count] += weights @ item[3]/len(paths_by_stratum)
                boot_q1_iid[first:first+count] += weights @ item[4]/len(paths_by_stratum)
                boot_q1_nr[first:first+count] += weights @ item[5]/len(paths_by_stratum)
                first += count
        for metric, value, sample in (
            ("interaction_soft_minus_M7_hard", point, boot),
            ("no_repeat_minus_iid_soft", np.mean([x[1].mean() for x in paths_by_stratum]), boot_soft),
            ("no_repeat_minus_iid_M7_hard", np.mean([x[2].mean() for x in paths_by_stratum]), boot_hard),
            ("no_repeat_soft_minus_M7_hard", np.mean([x[3].mean() for x in paths_by_stratum]), boot_gain),
            ("iid_soft_minus_q1", np.mean([x[4].mean() for x in paths_by_stratum]), boot_q1_iid),
            ("no_repeat_soft_minus_q1", np.mean([x[5].mean() for x in paths_by_stratum]), boot_q1_nr),
        ):
            rows.append({"group": group, "comparison": metric, "difference": value,
                         "lower95": float(np.quantile(sample, .025)),
                         "upper95": float(np.quantile(sample, .975)),
                         "independent_paths": sum(len(x[0]) for x in paths_by_stratum),
                         "bootstrap_replicates": REPS})
    path = OUT/"paired_m7_interaction.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "PASS_postrun_M7_interaction_audit",
        "confirm_run_sha256": sha(OUT/"RUN.json"),
        "source_sha256": sha(Path(__file__)),
        "bootstrap_seed": SEED, "bootstrap_replicates": REPS,
        "unit": "Whole physical path within fixed configuration and fixture, shared weights across all four cells.",
        "interpretation": "Supplementary contrast. Main lock interaction uses raw q=0 A as originally registered. This table adds the M7-feasible minimum-q diagnostic.",
        "paired_m7_interaction_sha256": sha(path),
    }
    (OUT/"M7_INTERACTION_AUDIT.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(rows[:4], indent=2))


if __name__ == "__main__":
    main()
