"""Stage-1 paired score-attribution and causal-matched GLRT experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.stats import beta as beta_distribution

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V3 = ROOT / "paper1_sscd_commonrandom_20261122"
THEORY = ROOT / "paper1_50node_tv_gate_theory_20261121"
SOURCE50 = ROOT / "paper1_common_g50_gain_acceptance_20261121"
sys.path[:0] = [str(V3), str(THEORY), str(SOURCE50), str(HERE)]

from calibration_rule import path_fwer_cutoff
from run_crn_development import load_backend
from watermark_keys import (
    attack_delta_for_schedule,
    beta_for_scheme,
    make_key_draws,
    matrices_from_beta,
)
from matched_glrt import (build_sender_geometry, matched_scores, plugin_scores,
                          template_condition_summary)

BASELINE_COMMIT = "8953e1dd9a5e2535ead2698acb6f345a85d4f532"
DEV_NAMESPACE = (20261125, 8317, 1)
CONFIRM_NAMESPACE = (20261125, 8317, 2)
SMOKE_NAMESPACE = (20261125, 8317, 99)
MODELS = ("unstable", "stable")
ANGLES = (0., -45., 45.)
ONSETS = (50, 65)
ARMS = (
    "A_plugin_iid", "B_plugin_negative", "C_matched_iid",
    "D_matched_negative", "E_iid_wrong_key", "F_negative_wrong_key",
)
SCORE_START = 5
ATTACK_DURATION = 60
ALPHA = .05
SPLITS = {
    "development": {"namespace": DEV_NAMESPACE, "paths": 256,
                    "calibration": 512, "health_holdout": 512,
                    "batch_size": 32, "directory": "stage1_development"},
    "confirmation": {"namespace": CONFIRM_NAMESPACE, "paths": 512,
                      "calibration": 4096, "health_holdout": 4096,
                      "batch_size": 32, "directory": "stage1_confirmation"},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_word(namespace, *components):
    return int(np.random.SeedSequence([*namespace, *components]).generate_state(1, dtype=np.uint32)[0])


def score_family_for_arm(arm):
    if arm.startswith(("A_", "B_")):
        return "plugin_kl"
    if arm == "C_matched_iid":
        return "matched_iid"
    if arm == "D_matched_negative":
        return "matched_negative"
    if arm == "E_iid_wrong_key":
        return "mismatched_iid_key"
    return "mismatched_negative_key"


def scheme_for_arm(arm):
    return "iid" if "iid" in arm else "negative_fixed"


def arm_uses_wrong_key(arm):
    return arm.endswith("wrong_key")


def scenario_specs():
    result = []
    for model_index, model in enumerate(MODELS):
        for angle_index, angle in enumerate(ANGLES):
            for onset_index, onset in enumerate(ONSETS):
                scenario_index = angle_index * len(ONSETS) + onset_index
                name = f"{model}__rot{angle:+.0f}__onset{onset}"
                result.append({
                    "model_index": model_index, "model": model,
                    "angle_index": angle_index, "angle": angle,
                    "onset_index": onset_index, "onset": onset,
                    "scenario_index": scenario_index, "name": name,
                })
    return result


def make_crn_draws(count, namespace, model_index, scenario_index,
                   stage, batch_index, wrong_key=False, scheme_index=0):
    key_index = 10 + (20 + scheme_index if wrong_key else 0)
    seeds = [seed_word(namespace, model_index, scenario_index, stage,
                       batch_index, key_index, stream) for stream in range(3)]
    return make_key_draws(count, *seeds), seeds


def matrices_for_pair(draws):
    return {
        "iid": matrices_from_beta(beta_for_scheme(draws, "iid")),
        "negative_fixed": matrices_from_beta(beta_for_scheme(draws, "negative_fixed")),
    }


def build_geometry_caches(model):
    senders = np.unique(model.links[:, 1])
    ordinary = {int(sender): build_sender_geometry(model, int(sender))
                for sender in senders}
    startup = {int(sender): build_sender_geometry(
        model, int(sender), zero_first_report=True
    ) for sender in senders}
    return ordinary, startup


def first_cutoff(values):
    return path_fwer_cutoff(np.asarray(values, dtype=float), ALPHA)


def received_sender_packets(model, physical, decoded_delta):
    """Materialize receiver-visible decoded reports; no latent delta is returned."""
    senders = model.links[:, 1].astype(int)
    if len(np.unique(senders)) != len(senders):
        raise ValueError("Frozen six-link packet interface assumes unique monitored senders.")
    packets = physical.y.copy()
    for edge, sender in enumerate(senders):
        packets[:, :, sender] += decoded_delta[:, :, edge]
    return packets


def score_plugin(model, physical, sender_packets):
    start = time.perf_counter()
    zero_delta = np.zeros((len(physical.x), model.T, len(model.links), 2))
    score = plugin_scores(model, physical, zero_delta, sender_packets=sender_packets)
    score[:, :SCORE_START] = 0.0
    return score, time.perf_counter() - start


def score_glrt(model, physical, sender_packets, matrices, geometry,
               startup_geometry=None):
    start = time.perf_counter()
    score = matched_scores(
        model, physical, sender_packets, matrices, geometry,
        startup_geometry_cache=startup_geometry, start_time=SCORE_START
    )
    return score, time.perf_counter() - start


def score_mismatch_glrt(model, physical, sender_packets, matrices, geometry,
                        startup_geometry=None):
    return score_glrt(model, physical, sender_packets, matrices, geometry,
                      startup_geometry)


def path_max(scores):
    return np.max(scores[:, SCORE_START:], axis=(1, 2))


def calibrate_model(model, physical_paths, geometry, model_index,
                    namespace, count, batch_size, startup_geometry=None):
    maxima = {
        "plugin_kl": [],
        "matched_iid": [],
        "matched_negative": [],
        "mismatched_iid_key": [],
        "mismatched_negative_key": [],
    }
    runtime = {name: 0.0 for name in maxima}
    physical_seeds = []
    roundtrip_max = 0.0
    template_condition = {}
    for batch, first in enumerate(range(0, count, batch_size)):
        n = min(batch_size, count - first)
        pseed = seed_word(namespace, model_index, 0, 0, batch, 1)
        physical_seeds.append(pseed)
        physical = physical_paths(model, n, pseed)
        zero = np.zeros((n, model.T, len(model.links), 2))
        values, elapsed = score_plugin(model, physical, physical.y)
        maxima["plugin_kl"].extend(path_max(values).tolist())
        runtime["plugin_kl"] += elapsed

        draws, _ = make_crn_draws(n, namespace, model_index, 0, 1, batch)
        matrices = matrices_for_pair(draws)
        if batch == 0:
            template_condition = {
                scheme: {
                    "full_packet_windows": template_condition_summary(
                        model, matrices[scheme], geometry
                    ),
                    "startup_zero_packet_window": template_condition_summary(
                        model, matrices[scheme][:, :8], startup_geometry
                    ),
                }
                for scheme in ("iid", "negative_fixed")
            }
        for scheme, family in (("iid", "matched_iid"),
                               ("negative_fixed", "matched_negative")):
            values, elapsed = score_glrt(model, physical, physical.y,
                                         matrices[scheme], geometry, startup_geometry)
            maxima[family].extend(path_max(values).tolist())
            runtime[family] += elapsed
        for scheme, family, scheme_index in (
            ("iid", "mismatched_iid_key", 0),
            ("negative_fixed", "mismatched_negative_key", 1),
        ):
            wrong_draws, _ = make_crn_draws(
                n, namespace, model_index, 0, 2 + scheme_index,
                batch, wrong_key=True, scheme_index=scheme_index,
            )
            wrong_matrix = matrices_for_pair(wrong_draws)[scheme]
            values, elapsed = score_mismatch_glrt(
                model, physical, physical.y, wrong_matrix, geometry, startup_geometry
            )
            maxima[family].extend(path_max(values).tolist())
            runtime[family] += elapsed

        # Verify ideal clean decoding without allowing floating-point roundtrip
        # error into the detector calibration histories.
        beta_pair = {
            scheme: beta_for_scheme(draws, scheme)
            for scheme in ("iid", "negative_fixed")
        }
        for beta in beta_pair.values():
            key = matrices_from_beta(beta)
            encoded = np.linalg.solve(key, physical.y[..., None])[..., 0]
            decoded = np.einsum("btnij,btnj->btni", key, encoded)
            roundtrip_max = max(
                roundtrip_max,
                float(np.max(np.abs(decoded - physical.y)
                             / np.maximum(1., np.abs(physical.y)))),
            )
    return {
        "thresholds": {name: first_cutoff(values) for name, values in maxima.items()},
        "path_maxima": {name: np.asarray(values) for name, values in maxima.items()},
        "physical_seed_words": physical_seeds,
        "score_runtime_seconds": runtime,
        "template_condition_summary": template_condition,
        "clean_roundtrip_max_scaled_error": roundtrip_max,
    }


def early_metrics(scores, threshold, onset):
    alarms = scores > threshold
    stop = onset + ATTACK_DURATION
    early = alarms[:, onset:onset + 3]
    full = alarms[:, onset:stop]
    delay = np.full((len(scores), scores.shape[2]), ATTACK_DURATION, dtype=np.int16)
    hit = full.any(axis=1)
    if hit.any():
        delay[hit] = np.argmax(full.transpose(0, 2, 1)[hit], axis=1)
    return {
        "early1_per_link": alarms[:, onset:onset + 1].any(axis=1).astype(np.uint8),
        "early2_per_link": alarms[:, onset:onset + 2].any(axis=1).astype(np.uint8),
        "early3_per_link": early.any(axis=1).astype(np.uint8),
        "early5_per_link": alarms[:, onset:onset + 5].any(axis=1).astype(np.uint8),
        "early8_per_link": alarms[:, onset:onset + 8].any(axis=1).astype(np.uint8),
        "path_early3_link_fraction": early.any(axis=1).mean(axis=1),
        "path_any_link_early3": early.any(axis=(1, 2)).astype(np.uint8),
        "path_any_link_full_attack": full.any(axis=(1, 2)).astype(np.uint8),
        "path_pre_onset_any_alarm": alarms[:, :onset].any(axis=(1, 2)).astype(np.uint8),
        "no_alarm_fraction_per_link_full_attack": (~hit).mean(axis=1),
        "first_alarm_delay_per_link_censored60": delay,
        "eligible_link_time_alarm_fraction": alarms[:, onset + 7:stop].mean(axis=(1, 2)),
    }


def estimator_losses(model, physical, decoded_delta, scores, threshold,
                     common_g, onset=50):
    gates = scores > threshold
    start = time.perf_counter()
    # Keep the frozen common-G numerical path. The detector has already
    # received and scored decoded packets. The simulator passes its decomposed
    # clean report and decoded additive perturbation only to reproduce the
    # frozen estimator update with its original floating-point operation order.
    world = common_g(
        model, physical, decoded_delta, sender_packets=physical.y,
        gate_on=False, threshold=threshold, gate_override=gates,
    )
    runtime = time.perf_counter() - start
    error = np.sum((world.post - physical.x[:, :, None, :]) ** 2, axis=-1)
    return {
        "gates": gates.astype(np.uint8),
        "losses": {
            "post20_network_mse": error[:, 20:].mean(axis=(1, 2)),
            "post20_affected_mse": error[:, 20:, model.receiver_ids].mean(axis=(1, 2)),
            "attack_affected_mse": error[:, onset:onset+60, model.receiver_ids].mean(axis=(1,2)),
            "attack_network_mse": error[:, onset:onset + 60].mean(axis=(1, 2)),
            "postattack_network_mse": error[:, onset + 60:].mean(axis=(1, 2)),
        },
        "runtime_seconds": runtime,
    }


def evaluate_health_holdout(model, model_index, physical_paths, common_g,
                            geometry, thresholds, namespace, count, batch_size,
                            startup_geometry=None):
    records = {
        arm: {"any_alarm": [], "post20_network_mse": [],
              "post20_affected_mse": [], "max_score": []}
        for arm in ARMS
    }
    physical_seeds = []
    for batch, first in enumerate(range(0, count, batch_size)):
        n = min(batch_size, count - first)
        pseed = seed_word(namespace, model_index, 0, 4, batch, 1)
        physical_seeds.append(pseed)
        physical = physical_paths(model, n, pseed)
        zero = np.zeros((n, model.T, len(model.links), 2))
        draws, _ = make_crn_draws(n, namespace, model_index, 0, 5, batch)
        keys = matrices_for_pair(draws)
        wrong = {}
        for scheme_index, scheme in enumerate(("iid", "negative_fixed")):
            other, _ = make_crn_draws(n, namespace, model_index, 0, 6 + scheme_index,
                                      batch, wrong_key=True, scheme_index=scheme_index)
            wrong[scheme] = matrices_for_pair(other)[scheme]
        score_cache = {}
        score_cache["plugin_kl"], _ = score_plugin(model, physical, physical.y)
        score_cache["matched_iid"], _ = score_glrt(
            model, physical, physical.y, keys["iid"], geometry, startup_geometry
        )
        score_cache["matched_negative"], _ = score_glrt(model, physical, physical.y,
                                                         keys["negative_fixed"], geometry, startup_geometry)
        score_cache["mismatch_iid"], _ = score_mismatch_glrt(
            model, physical, physical.y, wrong["iid"], geometry, startup_geometry
        )
        score_cache["mismatch_negative"], _ = score_mismatch_glrt(
            model, physical, physical.y, wrong["negative_fixed"], geometry, startup_geometry
        )
        for arm in ARMS:
            threshold_family = score_family_for_arm(arm)
            family = threshold_family
            if arm_uses_wrong_key(arm):
                family = "mismatch_iid" if scheme_for_arm(arm) == "iid" else "mismatch_negative"
            score = score_cache[family]
            threshold = float(thresholds[threshold_family]["threshold"])
            estimator = estimator_losses(model, physical, zero, score, threshold,
                                         common_g, onset=50)
            records[arm]["any_alarm"].append(estimator["gates"][:, SCORE_START:].any(axis=(1, 2)).astype(np.uint8))
            records[arm]["post20_network_mse"].append(estimator["losses"]["post20_network_mse"])
            records[arm]["post20_affected_mse"].append(estimator["losses"]["post20_affected_mse"])
            records[arm]["max_score"].append(path_max(score))
    return {
        arm: {key: np.concatenate(values) for key, values in record.items()}
        for arm, record in records.items()
    }, physical_seeds


def evaluate_attack_scenario(old, model, scenario, physical_paths,
                             common_g, geometry, thresholds, namespace, count,
                             startup_geometry=None):
    model_index = scenario["model_index"]
    scenario_id = scenario["scenario_index"]
    physical_seed = seed_word(namespace, model_index, scenario_id, 1, 0)
    physical = physical_paths(model, count, physical_seed)
    draws, common_seed_words = make_crn_draws(
        count, namespace, model_index, scenario_id, 2, 0
    )
    actual_keys = matrices_for_pair(draws)
    wrong_keys = {}
    wrong_seed_words = {}
    for key_index, scheme in enumerate(("iid", "negative_fixed")):
        wrong_draws, words = make_crn_draws(
            count, namespace, model_index, scenario_id, 3 + key_index, 0,
            wrong_key=True, scheme_index=key_index,
        )
        wrong_keys[scheme] = matrices_for_pair(wrong_draws)[scheme]
        wrong_seed_words[scheme] = words
    raw_attack, energy_scale = old.rotated_raw_attack(model, scenario["angle"])
    actual_delta = {
        scheme: attack_delta_for_schedule(model, matrices, raw_attack, scenario["onset"])
        for scheme, matrices in actual_keys.items()
    }
    results = {}
    arrays = {}
    received_packets = {
        scheme: received_sender_packets(model, physical, delta)
        for scheme, delta in actual_delta.items()
    }
    for arm in ARMS:
        actual_scheme = scheme_for_arm(arm)
        scoring_family = score_family_for_arm(arm)
        scoring_matrix = (wrong_keys[actual_scheme] if arm_uses_wrong_key(arm)
                          else actual_keys[actual_scheme])
        delta = actual_delta[actual_scheme]
        observed = received_packets[actual_scheme]
        if scoring_family == "plugin_kl":
            score, score_runtime = score_plugin(model, physical, observed)
        else:
            score_runtime_started = time.perf_counter()
            score = matched_scores(model, physical, observed, scoring_matrix,
                                   geometry, startup_geometry_cache=startup_geometry,
                                   start_time=SCORE_START)
            score_runtime = time.perf_counter() - score_runtime_started
        score[:, :SCORE_START] = 0.0
        threshold_family = scoring_family
        threshold = float(thresholds[threshold_family]["threshold"])
        estimated = estimator_losses(model, physical, delta, score,
                                     threshold, common_g, onset=scenario["onset"])
        detection = early_metrics(score, threshold, scenario["onset"])
        results[arm] = {
            "threshold_family": threshold_family,
            "threshold": threshold,
            "score_runtime_seconds": score_runtime,
            "common_g_replay_wall_seconds_including_source_plugin_score": estimated["runtime_seconds"],
            "detection": {name: _mean_se(values) for name, values in detection.items()},
            "losses": {name: _mean_se(values)
                       for name, values in estimated["losses"].items()},
        }
        prefix = arm
        arrays[f"{prefix}_scores"] = score
        arrays[f"{prefix}_gates"] = estimated["gates"]
        arrays[f"{prefix}_delta"] = delta
        for name, values in estimated["losses"].items():
            arrays[f"{prefix}_{name}"] = values
        for name, values in detection.items():
            arrays[f"{prefix}_{name}"] = values
    metadata = {
        "model": scenario["model"],
        "angle_degrees": scenario["angle"],
        "onset": scenario["onset"],
        "attack_window_half_open": [scenario["onset"], scenario["onset"] + 60],
        "early3_window_half_open": [scenario["onset"], scenario["onset"] + 3],
        "path_count": count,
        "path_ids": [f"stage1_{split_id(namespace)}_{scenario['name']}_{i}"
                     for i in range(count)],
        "seed_words": {
            "physical": physical_seed,
            "actual_shared_radial_initial_transition": common_seed_words,
            "wrong_key_iid": wrong_seed_words["iid"],
            "wrong_key_negative": wrong_seed_words["negative_fixed"],
        },
        "raw_attack_energy": float(np.sum(raw_attack ** 2)),
        "attack_rotation_energy_scale": float(energy_scale),
        "score_thresholds": {family: float(record["threshold"])
                             for family, record in thresholds.items()},
    }
    return results, arrays, metadata


def _mean_se(values):
    x = np.asarray(values, dtype=float)
    path_values = x if x.ndim == 1 else x.reshape(len(x), -1).mean(axis=1)
    return {"mean": float(path_values.mean()),
            "path_se": float(np.std(path_values, ddof=1) / np.sqrt(len(path_values)))
            if len(path_values) > 1 else 0.,
            "n_paths": int(len(path_values))}


def scenario_manifest_record(result, metadata):
    return {
        **metadata,
        "arms": {arm: {
            "score_family": values["threshold_family"],
            "threshold": values["threshold"],
            "score_runtime_seconds": values["score_runtime_seconds"],
            "common_g_replay_wall_seconds_including_source_plugin_score": values[
                "common_g_replay_wall_seconds_including_source_plugin_score"
            ],
            "detection": values["detection"],
            "losses": values["losses"],
        } for arm, values in result.items()},
    }


def _atomic_json(path, value):
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def _atomic_npz(path, arrays):
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def source_hashes():
    return {
        "registration": digest(HERE / "REGISTRATION.json"),
        "audit_legacy.py": digest(HERE / "audit_legacy.py"),
        "audit_interfaces.py": digest(HERE / "audit_interfaces.py"),
        "smoke_stage1.py": digest(HERE / "smoke_stage1.py"),
        "analyze_stage1.py": digest(HERE / "analyze_stage1.py"),
        "01_formula_audit.py": digest(HERE / "01_formula_audit.py"),
        "matched_glrt.py": digest(HERE / "matched_glrt.py"),
        "watermark_keys.py": digest(HERE / "watermark_keys.py"),
        "run_stage1.py": digest(Path(__file__)),
        "frozen_run_common_g_50.py": digest(SOURCE50 / "run_common_g_50.py"),
        "frozen_gain_table.json": digest(SOURCE50 / "gain_table.json"),
        "frozen_calibration_rule.py": digest(SOURCE50 / "calibration_rule.py"),
        "frozen_design_common_g.py": digest(SOURCE50 / "design_common_g.py"),
        "frozen_physical_configuration": digest(SOURCE50 / "physical_configuration.json"),
        "frozen_watson_crn.py": digest(V3 / "watson_crn.py"),
        "frozen_run_crn_development.py": digest(V3 / "run_crn_development.py"),
        "frozen_run_crn_final.py": digest(V3 / "run_crn_final.py"),
        "frozen_replay.py": digest(THEORY / "replay.py"),
        "frozen_source_model.py": digest(THEORY / "source_model.py"),
        "frozen_original_source": digest(
            ROOT / "paper1_original50_source_20261121" / "source"
            / "simulation_revised_detection_metrics_complete_v2.py"
        ),
        "v3_results.md": digest(V3 / "V3_RESULTS.md"),
        "v3_final_results.json": digest(V3 / "crn_final_results.json"),
        "v3_final_path_arrays.npz": digest(V3 / "crn_final_path_arrays.npz"),
        "v3_final_path_manifest.json": digest(V3 / "crn_final_path_manifest.json"),
        "v3_registration.json": digest(V3 / "new_experiment_registration.json"),
        "v3_verification.json": digest(V3 / "verification_v3.json"),
    }


def split_id(namespace):
    return "_".join(str(value) for value in namespace)


def run(split_name: str):
    run_started = time.perf_counter()
    if split_name not in SPLITS:
        raise ValueError(split_name)
    old, load_model, physical_paths, common_g, _ = load_backend()
    branch = subprocess.check_output(
        ["git", "-C", str(ROOT), "branch", "--show-current"], text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if branch != "watermark-early-detection-20261125":
        raise ValueError(f"Run must use the isolated watermark branch, got {branch}.")
    if subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"],
        check=False,
    ).returncode != 0:
        raise ValueError("The frozen baseline commit must be an ancestor of the run commit.")
    phase0 = json.loads((HERE / "PHASE0_REPLAY_AUDIT.json").read_text(encoding="utf-8"))
    interface = json.loads((HERE / "INTERFACE_SMOKE.json").read_text(encoding="utf-8"))
    formula = json.loads((HERE / "FORMULA_AUDIT.json").read_text(encoding="utf-8"))
    stage1_smoke = json.loads((HERE / "STAGE1_SMOKE.json").read_text(encoding="utf-8"))
    if phase0["legacy_two_scenario_replay"]["status"] != "PASS":
        raise ValueError("Phase-0 frozen V3 path regression did not pass.")
    if interface["status"] != "PASS_INTERFACE_AND_COVARIANCE_SMOKE":
        raise ValueError("Interface/covariance smoke did not pass.")
    if formula["status"] != "PASS_INDEPENDENT_FORMULA_AUDIT":
        raise ValueError("The independent watermark-formula audit did not pass.")
    if stage1_smoke["status"] != "PASS_FULL_STAGE1_SMOKE":
        raise ValueError("The full A-F Stage-1 smoke did not pass.")

    split = SPLITS[split_name]
    namespace = tuple(split["namespace"])
    output_dir = HERE / split["directory"]
    source_hash = source_hashes()
    fixture_hashes = old.validate_fixtures()
    identity = {
        "split": split_name,
        "branch": branch,
        "execution_commit": head,
        "seed_namespace": list(namespace),
        "paths_per_attack_scenario": split["paths"],
        "calibration_paths_per_model": split["calibration"],
        "healthy_holdout_paths_per_model": split["health_holdout"],
        "batch_size": split["batch_size"],
        "source_sha256": source_hash,
        "fixture_hashes": fixture_hashes,
        "phase0_sha256": digest(HERE / "PHASE0_REPLAY_AUDIT.json"),
        "interface_smoke_sha256": digest(HERE / "INTERFACE_SMOKE.json"),
        "formula_audit_sha256": digest(HERE / "FORMULA_AUDIT.json"),
        "stage1_smoke_sha256": digest(HERE / "STAGE1_SMOKE.json"),
    }
    if output_dir.exists():
        progress_path = output_dir / "RUN_PROGRESS.json"
        if not progress_path.is_file() or (output_dir / "RUN.json").exists():
            raise FileExistsError(f"Refusing to overwrite existing Stage-1 output {output_dir}")
        manifest = json.loads(progress_path.read_text(encoding="utf-8"))
        if manifest.get("run_identity") != identity:
            raise ValueError("Partial output identity differs from the committed Stage-1 run.")
        for filename, expected in manifest.get("archive_sha256", {}).items():
            if digest(output_dir / filename) != expected:
                raise ValueError(f"Partial archive hash mismatch: {filename}")
    else:
        output_dir.mkdir()
        manifest = {
            "status": "in_progress",
            "split": split_name,
            "baseline_commit": BASELINE_COMMIT,
            "execution_commit": head,
            "branch": branch,
            "seed_namespace": list(namespace),
            "paths_per_attack_scenario": split["paths"],
            "calibration_paths_per_model": split["calibration"],
            "healthy_holdout_paths_per_model": split["health_holdout"],
            "batch_size": split["batch_size"],
            "score_start_time": SCORE_START,
            "threshold_rule": "path maximum across six links and k=5..149; strict score>threshold; one-based rank ceil((n+1)*0.95)",
            "source_sha256": source_hash,
            "fixture_hashes": fixture_hashes,
            "run_identity": identity,
            "runtime": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__},
            "score_families": {
                "plugin_kl": "frozen four-residual plugin Gaussian KL; no watermark matrix input",
                "matched_iid": "causal whitened subspace GLRT with actual iid sender key",
                "matched_negative": "causal whitened subspace GLRT with actual fixed-negative sender key",
                "mismatched_iid_key": "iid-law GLRT template with an independent same-law key and an independently calibrated healthy path-max threshold",
                "mismatched_negative_key": "negative-law GLRT template with an independent same-law key and an independently calibrated healthy path-max threshold",
            },
            "models": {},
            "scenario_specs": scenario_specs(),
            "archive_sha256": {},
            "claim_scope": "known six directed links, two-dimensional message and frozen 50-node common-G estimator",
        }
        _atomic_json(output_dir / "RUN_PROGRESS.json", manifest)

    model_objects = {}
    geometry_objects = {}
    startup_geometry_objects = {}
    for model_index, model_name in enumerate(MODELS):
        model = load_model(model_name)
        geometry, startup_geometry = build_geometry_caches(model)
        model_objects[model_name] = model
        geometry_objects[model_name] = geometry
        startup_geometry_objects[model_name] = startup_geometry
        if model_name in manifest["models"]:
            health_archive = output_dir / f"health_{model_name}.npz"
            expected = manifest["archive_sha256"].get(health_archive.name)
            if not health_archive.exists() or digest(health_archive) != expected:
                raise ValueError(f"Healthy holdout archive is missing or corrupt: {model_name}")
            continue
        calibration_started = time.perf_counter()
        calibration = calibrate_model(
            model, physical_paths, geometry, model_index, namespace,
            split["calibration"], split["batch_size"], startup_geometry,
        )
        thresholds = calibration["thresholds"]
        healthy, health_seeds = evaluate_health_holdout(
            model, model_index, physical_paths, common_g, geometry, thresholds,
            namespace, split["health_holdout"], split["batch_size"],
            startup_geometry,
        )
        calibration_and_health_runtime = time.perf_counter() - calibration_started
        model_record = {
            "geometry": {
                str(sender): {
                    "stationary_full_packet_window": record.diagnostics,
                    "first_score_zero_packet_window": startup_geometry[sender].diagnostics,
                }
                for sender, record in geometry.items()
            },
            "calibration": {
                "thresholds": thresholds,
                "path_id_pattern": f"stage1_cal_{split_id(namespace)}_{model_name}_{{index}}",
                "template_condition_summary": calibration["template_condition_summary"],
                "path_maxima": {name: values.tolist()
                                for name, values in calibration["path_maxima"].items()},
                "path_maxima_sha256": {name: hashlib.sha256(values.tobytes()).hexdigest()
                                       for name, values in calibration["path_maxima"].items()},
                "template_condition_summary": calibration["template_condition_summary"],
                "physical_seed_words": calibration["physical_seed_words"],
                "score_runtime_seconds": calibration["score_runtime_seconds"],
                "calibration_and_healthy_holdout_runtime_seconds": calibration_and_health_runtime,
                "clean_roundtrip_max_scaled_error": calibration["clean_roundtrip_max_scaled_error"],
            },
            "healthy_holdout": {
                "path_count": split["health_holdout"],
                "path_id_pattern": f"stage1_health_{split_id(namespace)}_{model_name}_{{index}}",
                "physical_seed_words": health_seeds,
                "arms": {},
            },
            "attack_scenarios": {},
        }
        health_arrays = {}
        for arm in ARMS:
            family = score_family_for_arm(arm)
            alarms = int(healthy[arm]["any_alarm"].sum())
            n = len(healthy[arm]["any_alarm"])
            model_record["healthy_holdout"]["arms"][arm] = {
                "threshold_family": family,
                "alarm_paths": alarms,
                "FWER": alarms / n,
                "Clopper_Pearson_95": [
                    float(beta_distribution.ppf(.025, alarms, n - alarms + 1)) if alarms else 0.,
                    float(beta_distribution.ppf(.975, alarms + 1, n - alarms)) if alarms < n else 1.,
                ],
                "post20_network_mse_mean": float(healthy[arm]["post20_network_mse"].mean()),
                "post20_affected_mse_mean": float(healthy[arm]["post20_affected_mse"].mean()),
            }
            for metric, values in healthy[arm].items():
                health_arrays[f"{arm}_{metric}"] = values
            health_arrays[f"{arm}_path_row_id"] = np.arange(n, dtype=np.int32)
        health_filename = f"health_{model_name}.npz"
        _atomic_npz(output_dir / health_filename, health_arrays)
        manifest["archive_sha256"][health_filename] = digest(output_dir / health_filename)
        model_record["healthy_holdout"]["archive_sha256"] = manifest["archive_sha256"][health_filename]
        manifest["models"][model_name] = model_record
        _atomic_json(output_dir / "RUN_PROGRESS.json", manifest)
        print(f"completed {split_name} health holdout {model_name}", flush=True)

    for scenario in scenario_specs():
        model_name = scenario["model"]
        if scenario["name"] in manifest["models"][model_name]["attack_scenarios"]:
            filename = f"{scenario['name']}.npz"
            if digest(output_dir / filename) != manifest["archive_sha256"].get(filename):
                raise ValueError(f"Completed scenario archive is corrupt: {filename}")
            continue
        model = model_objects[model_name]
        geometry = geometry_objects[model_name]
        startup_geometry = startup_geometry_objects[model_name]
        thresholds = manifest["models"][model_name]["calibration"]["thresholds"]
        result, arrays, metadata = evaluate_attack_scenario(
            old, model, scenario, physical_paths, common_g, geometry,
            thresholds, namespace, split["paths"], startup_geometry,
        )
        manifest["models"][model_name]["attack_scenarios"][scenario["name"]] = scenario_manifest_record(
            result, metadata
        )
        filename = f"{scenario['name']}.npz"
        _atomic_npz(output_dir / filename, arrays)
        manifest["archive_sha256"][filename] = digest(output_dir / filename)
        _atomic_json(output_dir / "RUN_PROGRESS.json", manifest)
        print(f"completed {split_name} scenario {scenario['name']}", flush=True)

    manifest["status"] = f"complete_{split_name}"
    manifest["archive_sha256"] = {
        path.name: digest(path) for path in sorted(output_dir.glob("*.npz"))
    }
    manifest["total_runtime_seconds"] = time.perf_counter() - run_started
    _atomic_json(output_dir / "RUN.json", manifest)
    (output_dir / "RUN_PROGRESS.json").unlink()
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    args = parser.parse_args()
    run(args.split)


if __name__ == "__main__":
    main()
