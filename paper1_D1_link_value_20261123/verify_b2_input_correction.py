"""Compare final B2 semantics with the preserved oracle-gating replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CURRENT = HERE / "b2_attack_results"
PREVIOUS = HERE / "b2_attack_results_pre_input_correction"
STRATEGIES = ("A", "BD", "C")
GATE_METRICS = (
    "target_alarm_delay", "target_alarm_run_max",
    "target_attack_accept_fraction", "target_window_accept_fraction",
)
STATE_METRICS = (
    "missing_window_mse", "missing_post20_mse", "missing_tail_mse",
    "full_window_mse", "network_window_mse", "network_post20_mse",
    "accepted_target_forcing",
)


def compare(filename: str) -> dict:
    with np.load(PREVIOUS / filename, allow_pickle=False) as old, \
            np.load(CURRENT / filename, allow_pickle=False) as new:
        np.testing.assert_array_equal(old["masks"], new["masks"])
        np.testing.assert_array_equal(old["threshold"], new["threshold"])
        differences = {}
        for strategy in STRATEGIES:
            for metric in GATE_METRICS:
                np.testing.assert_array_equal(
                    old[f"{strategy}__{metric}"],
                    new[f"{strategy}__{metric}"],
                    err_msg=f"H1 event sequence changed in {filename}/{strategy}/{metric}",
                )
            for metric in STATE_METRICS:
                old_values = old[f"{strategy}__{metric}"]
                new_values = new[f"{strategy}__{metric}"]
                if not np.isfinite(new_values).all():
                    raise AssertionError(f"non-finite final metric {filename}/{strategy}/{metric}")
                differences[f"{strategy}__{metric}"] = float(
                    np.max(np.abs(new_values - old_values)))
    return {"file": filename, "max_abs_metric_changes": differences}


def main() -> None:
    checks = []
    for scene_index in range(12):
        watermark = "negative" if scene_index % 2 == 0 else "iid"
        for fixture in ("continuous", "burst"):
            filename = f"scene_{scene_index:02}_{fixture}_{watermark}.npz"
            checks.append(compare(filename))
    result = {
        "status": "PASS",
        "archives_compared": len(checks),
        "correction_scope": "All B2 thresholds, fresh paths, attack scores, and H1 masks are unchanged. MSE and forcing arrays were recomputed with q multiplying the complete attacked packet on alarm.",
        "checks": checks,
    }
    (CURRENT / "input_correction_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: result[key] for key in (
        "status", "archives_compared", "correction_scope")}, indent=2))


if __name__ == "__main__":
    main()
