"""Defensive timing, score, and archived receiver regression tests."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from paired_analysis import paired_risk
from replay_evaluator import HERE, METHODS, evaluate, load_receiver_modules, verify_inputs
from risk_metrics import events_from_times, score_estimates


class TimingTests(unittest.TestCase):
    def test_censored_alarms_do_not_drop_danger(self):
        events = events_from_times(
            np.array([8, 9, 10, 32, 32, 7]),
            np.array([32, 9, 6, 32, 12, 7]),
            np.full(6, 7),
            np.full(6, 17),
            32,
        )
        counts = events.counts()
        self.assertEqual(counts["any_danger"], 4)
        self.assertEqual(counts["before_alarm"], 1)
        self.assertEqual(counts["same_tick"], 2)
        self.assertEqual(counts["after_alarm"], 1)
        self.assertEqual(counts["first_packet"], 1)
        self.assertEqual(counts["first_packet_before_alarm"], 0)
        self.assertEqual(counts["danger_alarm_censored"], 1)
        self.assertEqual(counts["alarm_danger_censored"], 1)
        self.assertEqual(counts["both_censored"], 1)

    def test_thresholds_are_inclusive_and_same_tick_is_not_strict(self):
        nominal = np.zeros((1, 2, 2))
        observed = nominal.copy()
        observed[0, 1, 0] = np.sqrt(0.1)
        row = score_estimates(observed, nominal, nominal,
                              np.log(np.array([[1.0, 20.0]])), [1], [2])
        self.assertEqual(row.events.counts()["same_tick"], 1)
        self.assertEqual(row.events.counts()["before_alarm"], 0)
        self.assertEqual(row.events.counts()["first_packet"], 1)
        self.assertTrue(np.isnan(row.tail_mse[0]))

    def test_true_state_mse_and_stop_tick_in_tail(self):
        truth = np.zeros((1, 4, 2))
        healthy = np.array([[[0.1, 0.0], [0.2, 0.0], [0.2, 0.0], [0.0, 0.1]]])
        observed = np.array([[[0.1, 0.0], [0.3, 0.1], [0.0, 0.2], [0.4, 0.3]]])
        row = score_estimates(observed, healthy, truth,
                              np.log(np.array([[1.0, 20.0, 1.0, 1.0]])), [1], [3])
        np.testing.assert_allclose(row.support_mse, [0.07])
        np.testing.assert_allclose(row.tail_mse, [0.25])
        np.testing.assert_allclose(row.healthy_mse, [0.025])
        np.testing.assert_allclose(row.full_mse, [0.1])
        self.assertEqual(row.events.counts()["first_danger_at_or_after_stop"], 1)
        self.assertEqual(row.events.counts()["after_alarm"], 1)

    def test_inconsistent_episode_times_fail(self):
        with self.assertRaises(ValueError):
            events_from_times([6], [18], [7], [17], 32)


class PairedTests(unittest.TestCase):
    def test_paired_discordance_and_difference_direction(self):
        a = np.array([True, True, False, False])
        b = np.array([False, False, True, False])
        result, reversed_result = paired_risk(a, b), paired_risk(b, a)
        self.assertEqual(result["only_candidate"], 2)
        self.assertEqual(result["only_reference"], 1)
        self.assertEqual(result["difference"], 0.25)
        self.assertLessEqual(result["interval"][0], result["difference"])
        self.assertGreaterEqual(result["interval"][1], result["difference"])
        np.testing.assert_allclose(result["interval"], -np.array(reversed_result["interval"])[::-1])

    def test_identical_zero_observations_keep_interval_uncertainty(self):
        empty_events = np.zeros(12, dtype=bool)
        result = paired_risk(empty_events, empty_events)
        self.assertEqual(result["difference"], 0)
        self.assertLess(result["interval"][0], 0)
        self.assertGreater(result["interval"][1], 0)


class FrozenInputTests(unittest.TestCase):
    def test_changed_input_is_rejected_before_loading_receiver(self):
        files = verify_inputs(HERE.parent)["files"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / next(iter(files))
            path.parent.mkdir(parents=True)
            path.write_bytes(b"wrong snapshot")
            with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                verify_inputs(root)


class ReceiverReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = HERE.parent
        verify_inputs(base)
        receiver, _, _ = load_receiver_modules(base)
        cls.calls = []
        run = receiver.run_receiver

        def inspect_inputs(prepared, observation, table):
            cls.calls.append((len(observation), tuple(item.name for item in table),
                              set(vars(observation)), all(not a.flags.writeable
                                                         for a in vars(observation).values())))
            return run(prepared, observation, table)

        with patch.object(receiver, "run_receiver", side_effect=inspect_inputs):
            cls.report, cls.path_details = evaluate(base)

    def test_receiver_runs_both_models_and_sees_only_recorded_inputs(self):
        self.assertEqual(len(self.calls), 8)
        for count, methods, columns, read_only in self.calls:
            self.assertEqual(count, 12)
            self.assertEqual(methods, METHODS)
            self.assertEqual(columns, {"trusted", "target", "decoder"})
            self.assertTrue(read_only)

    def test_post_bf_and_inference_reproduce_frozen_log(self):
        for model in self.report["models"].values():
            comparisons = model["receiver_replay"]["frozen_log_comparison_max_abs"]
            self.assertEqual(set(comparisons), {"healthy", "A0", "A1_H_mismatch", "A1_C_mismatch"})
            for stage in comparisons.values():
                self.assertLess(max(stage.values()), 1e-10)

    def test_archive_holdout_keeps_all_censored_paths(self):
        expected = {"unstable": (670, 318, 116), "stable": (681, 378, 189)}
        for name, model in self.report["models"].items():
            n, before, censored = expected[name]
            v1a = model["saved_holdout_outcomes"]["A0_fixed_onset"]["methods"]["V1a"]
            self.assertEqual(v1a["n"], n)
            self.assertEqual(v1a["event_counts"]["before_alarm"], before)
            self.assertEqual(v1a["event_counts"]["danger_alarm_censored"], censored)

    def test_true_onset_diagnostics_and_error_decomposition(self):
        for name, model in self.report["models"].items():
            diagnosis = model["receiver_replay"]["cases"]["A0"]["onset_diagnostics"]
            self.assertEqual(diagnosis["phases"]["first_packet"]["packet_count"], 12)
            values = diagnosis["phases"]["whole_support"]
            self.assertAlmostEqual(
                values["decoded_squared_error"],
                values["fit_with_frozen_true_onset_squared"] + values["mixture_gap_squared"]
                + values["mixture_and_true_onset_cross_term"],
                places=12,
            )
            paths = self.path_details["models"][name]["identified_paths"]
            self.assertEqual(len(paths), 12)
            for row in paths:
                self.assertEqual(row["true_onset"], 7)
                self.assertEqual([item["tick"] for item in row["support"]], list(range(7, 17)))


if __name__ == "__main__":
    unittest.main()
