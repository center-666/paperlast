"""Independent locked-parameter test for the common-G SSCD development run.

The lock is read-only. This script never reselects a candidate. It evaluates the
pre-registered five-policy family on new calibration and test physical streams,
all twelve registered stress scenarios, and retains the known-six oracle.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
from scipy.stats import beta as beta_dist, t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'paper1_common_g50_gain_acceptance_20261121'))
sys.path.insert(0, str(ROOT / 'paper1_50node_tv_gate_theory_20261121'))
sys.path.append(r'C:\Users\zhongxin\shixidoc\DeepRobotics_M20\lidar_m20\.verify-venv\Lib\site-packages')

from selector_v2 import REFERENCE_COMMIT
from run_development import (rotated_raw_attack, decoded_for_scenario,
                             protocol_windows, SCENARIO_ANGLES, SCENARIO_STARTS)
from run_common_g_50 import load_common_g, physical_paths, run_common_g
from replay import attack_watermarks
from calibration_rule import path_fwer_cutoff

CALIBRATION_PATHS = 512
TEST_PATHS = 512
TEST_SEED = 2026112243


def metrics(world, physical, receiver_ids, threshold, onset, duration=60):
    sq = np.sum((world.post - physical.x[:, :, None, :]) ** 2, axis=-1)
    net = sq.mean(axis=2)
    affected = sq[:, :, receiver_ids].mean(axis=2)
    alarms = world.scores > threshold
    windows = protocol_windows(onset,duration)
    attack_start,stop=windows['attack']
    eligible_start,_=windows['eligible_score']
    eligible = alarms[:, eligible_start:stop]
    if not eligible.shape[1]:
        raise ValueError('No fully attack-eligible score times remain before attack stop.')
    first = np.argmax(eligible, axis=1)
    detected = eligible.any(axis=1)
    censored_delay = np.where(detected, first, eligible.shape[1])
    delay_from_onset = np.where(detected, first + windows['eligible_offset_from_onset'], duration)
    return {
        'network_all150': net.mean(axis=1),
        'network_post20': net[:, 20:].mean(axis=1),
        'network_attack': net[:, attack_start:stop].mean(axis=1),
        'network_post_attack': net[:, windows['post_attack'][0]:windows['post_attack'][1]].mean(axis=1),
        'affected_all150': affected.mean(axis=1),
        'affected_post20': affected[:, 20:].mean(axis=1),
        'affected_attack': affected[:, attack_start:stop].mean(axis=1),
        'affected_post_attack': affected[:, windows['post_attack'][0]:windows['post_attack'][1]].mean(axis=1),
        'eligible_tpr': eligible.mean(axis=(1, 2)),
        'eligible_first_alarm_delay': censored_delay.mean(axis=1),
        'first_alarm_delay_from_attack_onset': delay_from_onset.mean(axis=1),
        'path_any_eligible_alarm': detected.any(axis=1).astype(float),
        'attack_gate_fraction': world.gates[:, attack_start:stop].mean(axis=(1, 2)),
    }


def summarize(values):
    x = np.asarray(values, dtype=float)
    se = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float('nan')
    critical = float(student_t.ppf(.975, len(x) - 1))
    return {'mean': float(x.mean()), 'path_se': se, 'paths': len(x),
            'conditional_95ci_t': [float(x.mean() - critical * se),
                                   float(x.mean() + critical * se)]}


def main():
    dev_manifest = json.loads((HERE / 'development_manifest.json').read_text(encoding='utf-8'))
    lock = json.loads((HERE / 'locked_selection.json').read_text(encoding='utf-8'))
    dev_losses_bytes = (HERE / 'development_losses.json').read_bytes()
    if dev_manifest['split'] != 'development' or lock['status'] != 'DEVELOPMENT_CHOICE_ONLY_NOT_FINAL_TEST':
        raise ValueError('Expected completed, labeled development artifacts.')
    if hashlib.sha256(dev_losses_bytes).hexdigest() != dev_manifest['development_sha256']:
        raise ValueError('Development loss file changed after selection.')
    lock_hash = hashlib.sha256((HERE / 'locked_selection.json').read_bytes()).hexdigest()
    if lock_hash != dev_manifest['lock_sha256']:
        raise ValueError('Locked selection file changed after development.')
    if lock['source_commit'] != REFERENCE_COMMIT or dev_manifest['source_commit'] != REFERENCE_COMMIT:
        raise ValueError('Frozen source commit reference mismatch.')
    if lock['selected'] != 'iid' or lock['theta_locked'] != 0.0:
        raise ValueError('This registered final run is for the current locked iid decision.')
    if TEST_SEED == dev_manifest['seed']:
        raise ValueError('Final test seed must differ from development seed.')
    development = json.loads((HERE / 'development_losses.json').read_text(encoding='utf-8'))
    development_ids = {path_id for scenario in development['scenarios'].values()
                       for path_id in scenario['path_ids']}

    source_fixtures = {
        'gain_table.json': ROOT / 'paper1_common_g50_gain_acceptance_20261121/gain_table.json',
        'physical_configuration.json': ROOT / 'paper1_common_g50_gain_acceptance_20261121/physical_configuration.json',
        'run_common_g_50.py': ROOT / 'paper1_common_g50_gain_acceptance_20261121/run_common_g_50.py',
        'replay.py': ROOT / 'paper1_50node_tv_gate_theory_20261121/replay.py',
    }
    for name, expected in dev_manifest['fixture_sha256'].items():
        if hashlib.sha256(source_fixtures[name].read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source fixture changed: {name}')

    candidate_theta = {name: float(row['theta_used'])
                       for name, row in lock['candidate_audit'].items()
                       if row.get('feasible')}
    expected_candidates = {'neg_boundary', 'neg06', 'iid', 'pos06', 'pos_boundary'}
    if set(candidate_theta) != expected_candidates:
        raise ValueError('The final test candidate family differs from the registered development family.')

    calibration_seed = TEST_SEED + 10_000
    test_seed = TEST_SEED + 20_000
    result = {
        'status': 'FINAL_TEST_COMPLETE_LOCKED_SELECTION_NOT_REOPENED',
        'source_commit_reference': REFERENCE_COMMIT,
        'locked_selection': lock['selected'],
        'locked_theta': lock['theta_locked'],
        'development_manifest_sha256': hashlib.sha256((HERE / 'development_manifest.json').read_bytes()).hexdigest(),
        'locked_selection_sha256': lock_hash,
        'calibration_paths_per_model': CALIBRATION_PATHS,
        'test_paths_per_scenario_and_model': TEST_PATHS,
        'calibration_seed': calibration_seed,
        'test_seed_base': test_seed,
        'registered_angles_degrees': list(SCENARIO_ANGLES),
        'registered_attack_onsets': list(SCENARIO_STARTS),
        'candidates': candidate_theta,
        'threshold_protocol': '512 fresh healthy paths per dynamics; max over k=4..149 and six links; exact one-based rank488; strict score > cutoff',
        'test_protocol': 'Fresh scenario-specific physical paths, paired across fixed pre-registered watermark candidates; no post-test selection.',
        'models': {},
    }
    path_arrays = {}
    for model_index, label in enumerate(['unstable', 'stable']):
        model = load_common_g(label)
        calibration_physical = physical_paths(model, CALIBRATION_PATHS,
                                              calibration_seed + 100_000 * model_index)
        calibration_scores = run_common_g(model, calibration_physical, gate_on=False).scores
        calibration_path_max = np.max(calibration_scores[:, 4:], axis=(1, 2))
        cutoff = path_fwer_cutoff(calibration_path_max, .05)
        threshold = float(cutoff['threshold'])
        receiver_ids = np.asarray(model.receiver_ids, int)
        healthy_seed = test_seed + model_index * 1_000_000 + 99_999
        healthy_physical = physical_paths(model, TEST_PATHS, healthy_seed)
        healthy_world = run_common_g(model, healthy_physical, gate_on=True, threshold=threshold)
        healthy_sq = np.sum((healthy_world.post - healthy_physical.x[:, :, None, :]) ** 2, axis=-1)
        healthy_alarm = healthy_world.gates[:, 4:].any(axis=(1, 2))
        healthy_path_max = np.max(healthy_world.scores[:, 4:], axis=(1, 2))
        healthy_hits = int(healthy_alarm.sum())
        healthy_fwer_ci = [
            float(beta_dist.ppf(.025, healthy_hits, TEST_PATHS - healthy_hits + 1)) if healthy_hits else 0.0,
            float(beta_dist.ppf(.975, healthy_hits + 1, TEST_PATHS - healthy_hits)) if healthy_hits < TEST_PATHS else 1.0,
        ]
        model_result = {
            'threshold': cutoff,
            'independent_healthy_test': {
                'paths': TEST_PATHS, 'seed': healthy_seed,
                'path_any_alarm_count': healthy_hits,
                'empirical_path_fwer': healthy_hits / TEST_PATHS,
                'two_sided_exact_binomial_95ci': healthy_fwer_ci,
                'network_mse_post20': summarize(healthy_sq.mean(axis=2)[:, 20:].mean(axis=1)),
                'affected_mse_post20': summarize(healthy_sq[:, 20:, receiver_ids].mean(axis=2).mean(axis=1)),
                'path_ids': [f'healthy_test_{TEST_SEED}_{label}_{k}' for k in range(TEST_PATHS)],
            },
            'calibration_path_ids': [f'cal_{TEST_SEED}_{label}_{k}' for k in range(CALIBRATION_PATHS)],
            'scenarios': {}}
        path_arrays[f'{label}_calibration_path_max'] = calibration_path_max
        path_arrays[f'{label}_independent_healthy_test_path_max'] = healthy_path_max
        path_arrays[f'{label}_healthy_test__network_mse_post20'] = healthy_sq.mean(axis=2)[:, 20:].mean(axis=1)
        path_arrays[f'{label}_healthy_test__affected_mse_post20'] = healthy_sq[:, 20:, receiver_ids].mean(axis=2).mean(axis=1)
        path_arrays[f'{label}_healthy_test__path_any_alarm'] = healthy_alarm.astype(np.uint8)

        for angle_index, angle in enumerate(SCENARIO_ANGLES):
            raw_attack, attack_scale = rotated_raw_attack(model, angle)
            for onset_index, onset in enumerate(SCENARIO_STARTS):
                scenario = f'{label}__rot{angle:+.0f}__onset{onset}'
                scenario_index = angle_index * len(SCENARIO_STARTS) + onset_index
                scenario_seed = test_seed + model_index * 1_000_000 + scenario_index * 10_000
                physical = physical_paths(model, TEST_PATHS, scenario_seed)
                natural_metrics, path_metrics = {}, {}
                for policy_index, (policy, theta) in enumerate(candidate_theta.items()):
                    delta = decoded_for_scenario(
                        model, TEST_PATHS, theta, raw_attack, onset,
                        radial_seed=scenario_seed + 100,
                        initial_seed=scenario_seed + 200,
                        direction_seed=scenario_seed + 300 + 1000 * policy_index,
                        watermark_generator=attack_watermarks)
                    world = run_common_g(model, physical, delta, gate_on=True, threshold=threshold)
                    values = metrics(world, physical, receiver_ids, threshold, onset)
                    path_metrics[policy] = values
                    natural_metrics[policy] = {key: summarize(arr) for key, arr in values.items()}
                    for metric, arr in values.items():
                        path_arrays[f'{scenario}_{policy}__{metric}'] = np.asarray(arr)

                # Oracle baseline knows the six registered links and drops them
                # from k=1. The attack stream is held at iid only for parity.
                iid_delta = decoded_for_scenario(
                    model, TEST_PATHS, candidate_theta['iid'], raw_attack, onset,
                    radial_seed=scenario_seed + 100,
                    initial_seed=scenario_seed + 200,
                    direction_seed=scenario_seed + 300 + 1000 * list(candidate_theta).index('iid'),
                    watermark_generator=attack_watermarks)
                permanent = np.zeros((TEST_PATHS, 150, 6), dtype=bool)
                permanent[:, 1:, :] = True
                oracle = run_common_g(model, physical, iid_delta,
                                      gate_override=permanent, threshold=threshold)
                oracle_metrics = metrics(oracle, physical, receiver_ids, threshold, onset)
                path_metrics['oracle_static_six'] = oracle_metrics
                for metric, arr in oracle_metrics.items():
                    path_arrays[f'{scenario}_oracle_static_six__{metric}'] = np.asarray(arr)

                paired = {}
                iid = path_metrics['iid']
                for policy in candidate_theta:
                    if policy == 'iid':
                        continue
                    paired[policy] = {metric: summarize(path_metrics[policy][metric] - iid[metric])
                                      for metric in ['network_post20', 'affected_post20',
                                                     'network_attack', 'eligible_tpr']}
                scenario_rec = {
                    'angle_degrees': angle, 'attack_onset': onset,
                    'attack_interval_half_open': [onset, onset + 60],
                    'eligible_score_interval_half_open': [onset + 7, onset + 60],
                    'attack_scale_to_equal_raw_joint_energy': attack_scale,
                    'scenario_seed': scenario_seed,
                    'path_ids': [f'test_{TEST_SEED}_{scenario}_{k}' for k in range(TEST_PATHS)],
                    'candidates': natural_metrics,
                    'paired_minus_iid': paired,
                    'oracle_static_six': {key: summarize(arr) for key, arr in oracle_metrics.items()},
                    'oracle_minus_iid': {key: summarize(oracle_metrics[key] - iid[key])
                                         for key in ['network_post20', 'affected_post20',
                                                     'network_attack', 'eligible_tpr']},
                }
                if len(set(scenario_rec['path_ids'])) != TEST_PATHS or set(scenario_rec['path_ids']) & development_ids:
                    raise ValueError(f'{scenario}: final-test path IDs overlap or are duplicated')
                model_result['scenarios'][scenario] = scenario_rec
                print('Completed final test', scenario, flush=True)
        result['models'][label] = model_result

    paths = HERE / 'final_test_paths.npz'
    np.savez_compressed(paths, **path_arrays)
    manifest = {
        'split': 'final_test', 'development_seed': dev_manifest['seed'],
        'calibration_seed': calibration_seed, 'test_seed_base': test_seed,
        'calibration_paths_per_model': CALIBRATION_PATHS,
        'calibration_maxima_saved': True,
        'test_paths_per_scenario_and_model': TEST_PATHS,
        'independent_healthy_test_paths_per_model': TEST_PATHS,
        'scenario_count': 12, 'candidate_count': len(candidate_theta),
        'path_array_keys': sorted(path_arrays),
        'input_hashes': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in source_fixtures.items()},
        'run_final_evaluation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'locked_selection_sha256': lock_hash,
        'warning': 'The lock remains iid; comparisons with alternatives are held-out descriptive checks, not permission to retune.'
    }
    (HERE / 'final_test_results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n',
                                                  encoding='utf-8', newline='\n')
    (HERE / 'final_test_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
                                                   encoding='utf-8', newline='\n')
    print(json.dumps({'status': result['status'], 'locked_selection': result['locked_selection'],
                      'scenarios': 12, 'test_paths_per_scenario_and_model': TEST_PATHS,
                      'path_array_count': len(path_arrays)}, indent=2))


if __name__ == '__main__':
    main()
