"""Verify timing and exact certificate on the frozen records."""
import unittest

import numpy as np

from archive_timing import HORIZON, METHODS, OUTPUT, event_partition, load
from pre_alarm_certificate import blind_interval_survival_lower


class PreAlarmChecks(unittest.TestCase):
    def test_strict_order_and_censoring(self):
        d = np.array([7,7,9,HORIZON,HORIZON,11,17])
        a = np.array([8,7,8,HORIZON,9,HORIZON,8])
        n = len(d)
        row = event_partition(d,a,np.full(n,7),np.full(n,17))
        self.assertEqual(row["any_danger_within_horizon"],5)
        self.assertEqual(row["strictly_before_alarm"],2)
        self.assertEqual(row["same_tick_as_alarm"],1)
        self.assertEqual(row["strictly_after_alarm"],2)
        self.assertEqual(row["no_danger_within_horizon"],2)
        self.assertEqual(row["danger_observed_alarm_censored"],1)
        self.assertEqual(row["alarm_observed_danger_censored"],1)
        self.assertEqual(row["both_censored"],1)
        self.assertEqual(row["first_packet_any_danger"],2)
        self.assertEqual(row["first_packet_strictly_before_alarm"],1)
        self.assertEqual(row["first_danger_at_or_after_stop"],1)
        self.assertAlmostEqual(row["alarm_minus_danger_median_if_both_observed"],-.5)

    def test_gaussian_overlap(self):
        self.assertAlmostEqual(blind_interval_survival_lower(0.),.95)
        lower = [blind_interval_survival_lower(delta) for delta in (1.,2.,3.,4.)]
        self.assertTrue(all(x>0 for x in lower))
        self.assertTrue(all(x>y for x,y in zip(lower,lower[1:],strict=False)))

    def test_window_scope_of_exact_certificate(self):
        c = load(OUTPUT/"pre_alarm_certificate.json")
        self.assertEqual(c["physical_episodes_generated"],0)
        self.assertEqual(c["max_stage_penalty_exact"],"3903/156250")
        self.assertEqual(c["healthy_false_alarm_allowance_exact"],"3903/3125000")
        for report in c["models"].values():
            windows = report["windows"]
            B = windows["B_all_32_ticks"]
            self.assertTrue(B["rejects_two_pct"])
            self.assertGreater(B["MSE_lower_down"],report["two_percent_nominal_cap_up"])
            self.assertEqual(report["minimum_eligible_ticks_for_this_certificate"],30)
            for k in ("A_if_single_packet_required_at_each_of_10_ticks",
                      "A_if_single_packet_required_on_union_of_archived_supports"):
                self.assertFalse(windows[k]["rejects_two_pct"])

    def test_freeze_and_probabilities(self):
        doc = load(OUTPUT/"archived_timing.json")
        self.assertEqual(doc["new_trajectories"],0)
        self.assertEqual(doc["live_attacker_runs"],0)
        for report in doc["models"].values():
            self.assertEqual(report["archived_healthy_paths"],2048)
            self.assertEqual(set(report["health"]),set(METHODS))
            for group in report["profiles"].values():
                rows = group["onset_7"]
                for row in rows.values():
                    self.assertGreater(row["strict_pre_fraction_among_danger"],.5)
                    self.assertEqual(row["any_danger_within_horizon"]+row["no_danger_within_horizon"],
                                     row["sample_size"])
                    self.assertEqual(row["first_packet_any_danger"],
                                     sum(row[k] for k in ("first_packet_strictly_before_alarm",
                                            "first_packet_same_tick_as_alarm","first_packet_strictly_after_alarm")))
                self.assertEqual(min(row["strictly_before_alarm"] for row in rows.values()),
                                 rows["anchor@2%"]["strictly_before_alarm"])


if __name__=="__main__":
    unittest.main()
