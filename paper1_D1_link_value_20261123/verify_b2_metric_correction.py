"""Check that the B2 metric correction changed only the declared estimands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CURRENT = HERE / "b2_attack_results_pre_input_correction"
PREVIOUS = HERE / "b2_attack_results_pre_metric_correction"
STRATEGIES = ("A", "BD", "C")
METRICS = (
    "missing_window_mse", "missing_post20_mse", "missing_tail_mse",
    "full_window_mse", "network_window_mse", "network_post20_mse",
    "target_alarm_delay", "target_alarm_run_max",
    "accepted_target_forcing",
)


def check(filename: str, fixture: str) -> dict:
    with np.load(PREVIOUS / filename, allow_pickle=False) as old, \
            np.load(CURRENT / filename, allow_pickle=False) as new:
        masks = new["masks"]
        np.testing.assert_array_equal(old["masks"], masks)
        np.testing.assert_array_equal(old["threshold"], new["threshold"])
        alarms = ((masks[:, :, None] & (1 << np.arange(7))) != 0)[:, :, 6]
        active = np.ones(60, dtype=bool)
        if fixture == "burst":
            active = (np.arange(60) % 10) < 6
        attack_accept = (~alarms[:, 50:110][:, active]).mean(axis=1)
        window_accept = (~alarms[:, 50:110]).mean(axis=1)
        for strategy in STRATEGIES:
            for metric in METRICS:
                np.testing.assert_array_equal(
                    old[f"{strategy}__{metric}"],
                    new[f"{strategy}__{metric}"],
                    err_msg=f"unexpected change in {filename}/{strategy}/{metric}",
                )
            np.testing.assert_array_equal(
                old[f"{strategy}__target_attack_accept_fraction"],
                new[f"{strategy}__target_window_accept_fraction"],
                err_msg=f"legacy acceptance must equal corrected window acceptance in {filename}",
            )
            np.testing.assert_array_equal(
                new[f"{strategy}__target_attack_accept_fraction"],
                attack_accept,
                err_msg=f"corrected attack acceptance must use active attack ticks in {filename}",
            )
            np.testing.assert_array_equal(
                new[f"{strategy}__target_window_accept_fraction"],
                window_accept,
                err_msg=f"window acceptance must use all attack-window ticks in {filename}",
            )
            if fixture == "continuous":
                np.testing.assert_array_equal(
                    old[f"{strategy}__target_attack_accept_fraction"],
                    new[f"{strategy}__target_attack_accept_fraction"],
                    err_msg=f"continuous attack denominator should be unchanged in {filename}",
                )
    return {"file": filename, "status": "all_nonacceptance_arrays_unchanged"}


def main() -> None:
    records = []
    for scene_index in range(12):
        watermark = "negative" if scene_index % 2 == 0 else "iid"
        for fixture in ("continuous", "burst"):
            filename = f"scene_{scene_index:02}_{fixture}_{watermark}.npz"
            records.append(check(filename, fixture))
    result = {
        "status": "PASS",
        "archives_compared": len(records),
        "correction_scope": "Only the burst target_attack_accept_fraction denominator changed. The new target_window_accept_fraction equals the original full-window rate. All masks, thresholds, and remaining strategy metrics are bitwise identical.",
        "checks": records,
    }
    (HERE / "b2_attack_results" / "metric_correction_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: result[key] for key in (
        "status", "archives_compared", "correction_scope")}, indent=2))


if __name__ == "__main__":
    main()
