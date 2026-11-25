"""Small disposable full-path smoke test; writes no registered stage outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE / "../paper1_sscd_commonrandom_20261122"),
                str(HERE / "../paper1_50node_tv_gate_theory_20261121"),
                str(HERE / "../paper1_common_g50_gain_acceptance_20261121"),
                str(HERE)]

import run_stage1 as runner


def main():
    old, load_model, physical_paths, common_g, _ = runner.load_backend()
    model = load_model("unstable")
    geometry = {int(sender): runner.build_sender_geometry(model, int(sender))
                for sender in np.unique(model.links[:, 1])}
    namespace = runner.SMOKE_NAMESPACE
    calibration = runner.calibrate_model(
        model, physical_paths, geometry, 0, namespace, 64, 16
    )
    thresholds = calibration["thresholds"]
    scenario = {
        "model_index": 0, "model": "unstable", "angle_index": 0,
        "angle": 0., "onset_index": 0, "onset": 50,
        "scenario_index": 0, "name": "smoke_unstable_rot0_onset50",
    }
    result, arrays, meta = runner.evaluate_attack_scenario(
        old, model, scenario, physical_paths, common_g, geometry, thresholds,
        namespace, 16,
    )
    assert set(result) == set(runner.ARMS)
    assert set(arrays) >= {"A_plugin_iid_scores", "D_matched_negative_gates",
                           "E_iid_wrong_key_early3_per_link"}
    for arm, record in result.items():
        assert record["detection"]["early3_per_link"]["n_paths"] == 16
        assert record["losses"]["post20_network_mse"]["n_paths"] == 16
    physical = physical_paths(model, 16, 20261125_8317_9950)
    raw_attack, _ = old.rotated_raw_attack(model, 0.)
    draws, _ = runner.make_crn_draws(
        16, runner.SMOKE_NAMESPACE, 0, 0, 20, 0
    )
    actual_key = runner.matrices_for_pair(draws)["negative_fixed"]
    delta = runner.attack_delta_for_schedule(model, actual_key, raw_attack, 50)
    observed = runner.received_sender_packets(model, physical, delta)
    plugin, _ = runner.score_plugin(model, physical, observed)
    frozen_plugin = common_g(model, physical, delta, gate_on=False).scores
    np.testing.assert_array_equal(plugin[:, runner.SCORE_START:],
                                  frozen_plugin[:, runner.SCORE_START:])
    gates = plugin > float(thresholds["plugin_kl"]["threshold"])
    frozen_world = common_g(model, physical, delta, gate_on=False,
                            gate_override=gates, implementation="aggregate")
    literal_world = common_g(model, physical, delta, gate_on=False,
                             gate_override=gates, implementation="literal")
    np.testing.assert_allclose(frozen_world.post, literal_world.post,
                               rtol=2e-10, atol=2e-10)
    synthetic_score = np.zeros((2, model.T, len(model.links)))
    cutoff_value = float(thresholds["plugin_kl"]["threshold"])
    synthetic_score[:, 65, 0] = cutoff_value + 1.
    synthetic_score[:, 67, 1] = cutoff_value + 1.
    shifted = runner.early_metrics(synthetic_score, cutoff_value, 65)
    assert np.array_equal(shifted["early3_per_link"][:, 0], np.ones(2, dtype=np.uint8))
    assert np.array_equal(shifted["early3_per_link"][:, 1], np.ones(2, dtype=np.uint8))
    assert np.array_equal(shifted["first_alarm_delay_per_link_censored60"][:, 0],
                          np.zeros(2, dtype=np.int16))
    healthy, health_seeds = runner.evaluate_health_holdout(
        model, 0, physical_paths, common_g, geometry, thresholds,
        namespace, 32, 16,
    )
    assert set(healthy) == set(runner.ARMS)
    assert all(len(healthy[arm]["any_alarm"]) == 32 for arm in runner.ARMS)
    output = {
        "status": "PASS_FULL_STAGE1_SMOKE",
        "formal_paths_generated": False,
        "attack_smoke_paths": 16,
        "healthy_smoke_paths": 32,
        "arms": list(runner.ARMS),
        "frozen_plugin_score_max_abs_error_from_common_start": float(np.max(np.abs(plugin[:, runner.SCORE_START:] - frozen_plugin[:, runner.SCORE_START:]))),
        "frozen_common_g_aggregate_literal_max_abs_error": float(np.max(np.abs(frozen_world.post - literal_world.post))),
        "onset65_early_window_alignment": True,
        "thresholds": {name: record for name, record in thresholds.items()},
        "attack_run_seconds_by_arm": {
            arm: result[arm]["score_runtime_seconds"] + result[arm]["common_g_replay_wall_seconds_including_source_plugin_score"]
            for arm in runner.ARMS
        },
        "runtime_note": "The frozen common-G wrapper recomputes the plugin score even with externally supplied gates, so its wall time includes this extra diagnostic score pass.",
        "healthy_seed_words": health_seeds,
        "scenario_metadata": meta,
    }
    target = HERE / "STAGE1_SMOKE.json"
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({"status": output["status"], "report": str(target)}, indent=2))


if __name__ == "__main__":
    main()
