"""Regression checks for the frozen cohorts and two offline counterfactuals."""

import unittest

import numpy as np

from run_diagnosis import evaluate


class A0DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = evaluate()
        cls.models = cls.report["models"]

    def test_frozen_cohorts_and_saturation_costs(self):
        self.assertEqual(self.report["physical_records_added"], 0)
        self.assertEqual(self.report["original_frozen_inputs_verified"], 29)
        for name, n in (("unstable", 670), ("stable", 681)):
            old = self.models[name]["saved_outcomes"]
            self.assertEqual((old["onset"], old["stop_exclusive"], old["path_count"]), (7, 17, n))
            for pct in (0.1, 0.5, 1, 2):
                op = old["methods"][f"anchor@{pct:g}%"]
                self.assertEqual(op["frozen_method"]["operation"], "anchor")
                self.assertAlmostEqual(op["nominal_health"]["relative_pct"], pct, places=9)

    def test_path_level_mse_intervals_are_family_adjusted(self):
        for model in self.models.values():
            old = model["saved_outcomes"]
            methods = old["methods"]
            for name, entry in old["paired_sat_minus_v1a"].items():
                a = entry["active_mse_sat_minus_v1a_family_t95"]
                b = entry["active_mse_sat_minus_v1a_pointwise_t95"]
                self.assertGreater(a["interval"][0], 0)
                self.assertLess(a["interval"][0], b["interval"][0])
                self.assertGreater(a["interval"][1], b["interval"][1])
                self.assertAlmostEqual(a["mean"], methods[name]["active_mse"] - methods["V1a"]["active_mse"])
        stable = self.models["stable"]["saved_outcomes"]["other_onsets_descriptive_sensitivity"]
        late = stable["11"]["active_mse_sat_minus_v1a_pointwise_t95"]["anchor@2%"]
        self.assertLess(late["interval"][0], 0)

    def test_three_squares_and_each_signed_cross_reconstruct_total(self):
        for model in self.models.values():
            terms = model["complete_record_replay"]["decomposition"]
            for stage, count in (("first_packet", 12), ("first_three", 36), ("remaining_seven", 84),
                                 ("whole_support", 120)):
                self.assertEqual(terms[stage]["packets"], count)
                for units in ("decoded_distortion", "current_update_units"):
                    row = terms[stage][units]
                    self.assertEqual(len(row["squares"]), 3)
                    self.assertEqual(len(row["signed_cross"]), 3)
                    self.assertAlmostEqual(sum(row["squares"].values()) + sum(row["signed_cross"].values()),
                                           row["total"], places=11)
            for units in ("decoded_distortion", "current_update_units"):
                first = terms["first_three"][units]["total"]
                late = terms["remaining_seven"][units]["total"]
                self.assertAlmostEqual((3 * first + 7 * late) / 10, terms["whole_support"][units]["total"])

    def test_original_replay_and_label_inactive_healthy_result(self):
        for model in self.models.values():
            replay = model["complete_record_replay"]
            self.assertEqual(replay["path_count"], 12)
            self.assertLess(replay["regression_max_absolute_error"], 1e-10)
            self.assertLess(replay["label_oracle_healthy_equals_v0_max_abs"], 1e-11)
            self.assertEqual(replay["common_bf_threshold"], 20)

    def test_fixed_history_and_propagated_labels_are_scored_independently(self):
        for model in self.models.values():
            replay = model["complete_record_replay"]
            for row in replay["label_diagnostics"].values():
                self.assertGreater(row["max_difference_fixed_vs_propagated_output"], 0.01)
            fixed = replay["fixed_history"]["V1a"]["support_step_mse"]
            actual = replay["closed_loop"]["methods"]["V1a"]["support_mse"]
            self.assertAlmostEqual(fixed, actual, places=13)

    def test_label_risk_relief_preserves_new_discordant_events_and_common_alarms(self):
        for name, original in (("unstable", 5), ("stable", 8)):
            r = self.models[name]["complete_record_replay"]
            events = r["closed_loop"]["methods"]
            for key, count in (("V1a", original), ("anchor@2%", 4),
                               ("activity-label diagnostic", 2), ("true-onset diagnostic", 1)):
                self.assertEqual(events[key]["event_counts"]["before_alarm"], count)
            baseline = r["closed_loop_paths"]["V1a"]
            for label in ("activity-label diagnostic", "true-onset diagnostic"):
                scores = r["closed_loop_paths"][label]
                self.assertEqual([x["tau_a"] for x in baseline], [x["tau_a"] for x in scores])
                self.assertEqual([x["path_id"] for x in baseline], [x["path_id"] for x in scores])
                contrast = r["label_diagnostics"][label]["closed_loop_vs_v1a"]
                self.assertEqual(contrast["only_candidate"], int(name == "unstable"))
                self.assertGreater(contrast["interval"][1], 0)


if __name__ == "__main__":
    unittest.main()
