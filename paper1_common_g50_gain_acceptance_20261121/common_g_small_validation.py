"""Independent small closed-loop validation for common-G x SSCD.

This is deliberately a screening experiment, not the paper's final power
study: 512 healthy calibration paths and 128 paired evaluation paths per
dynamics. It uses a single held-out path-max threshold per dynamics, strict
alarm ``score > threshold``, and preserves the known-six static-drop baseline.
"""
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'paper1_50node_tv_gate_theory_20261121'))
sys.path.insert(0, str(HERE.parent / 'paper1_SSCD_review_20261121'))
from run_common_g_50 import load_common_g, physical_paths, run_common_g
from replay import attack_watermarks
from calibration_rule import path_fwer_cutoff
import sscd

CALIBRATION_COUNT = 512
EVALUATION_COUNT = 128
THETA = {'iid': 0.0, 'negative': -1.315236976827179,
         'positive': 1.315236976827179}
BASE_SEED = 2026112201


def summarize(world, physical, threshold):
    err = np.sum((world.post - physical.x[:, :, None]) ** 2, axis=-1)
    net = err.mean(axis=2)
    # Receiver ordering is fixed by the source links; report exact IDs too.
    receiver_ids = np.array([0, 2, 14, 15, 17], dtype=int)
    affected = err[:, :, receiver_ids].mean(axis=2)
    alarm = world.scores > threshold
    full = alarm[:, 57:110]
    return {
        'network_mse_all150': net.mean(axis=1),
        'network_mse_post20': net[:, 20:].mean(axis=1),
        'affected_mse_post20': affected[:, 20:].mean(axis=1),
        'attack_network_mse': net[:, 50:110].mean(axis=1),
        'after_support_network_mse': net[:, 110:].mean(axis=1),
        'after_buffer_recovery_network_mse': net[:, 117:].mean(axis=1),
        'eligible_tpr_full_window': full.mean(axis=(1, 2)),
        'path_alarm_full_window': full.any(axis=(1, 2)),
        'path_alarm_any_decision': alarm[:, 4:].any(axis=(1, 2)),
        'gate_fraction_attack_window': world.gates[:, 50:110].mean(axis=(1, 2)),
    }


def aggregate_metric(values):
    values = np.asarray(values)
    return {'mean': float(values.mean()),
            'path_se': float(values.std(ddof=1) / np.sqrt(len(values))),
            'paths': int(len(values))}


def main():
    result = {
        'status': 'PASS',
        'scope': 'Small held-out common-G screening; no final efficacy claim.',
        'calibration_paths_per_model': CALIBRATION_COUNT,
        'evaluation_paths_per_model': EVALUATION_COUNT,
        'threshold_rule': 'calibration path maximum over k=4..149 and all six links; strict score > threshold',
        'models': {},
    }
    archive = {}
    for mi, label in enumerate(['unstable', 'stable']):
        model = load_common_g(label)
        calibration_physical = physical_paths(model, CALIBRATION_COUNT, BASE_SEED + 10000 * mi)
        calibration_world = run_common_g(model, calibration_physical, gate_on=False)
        calibration_max = np.max(calibration_world.scores[:, 4:], axis=(1, 2))
        cutoff = path_fwer_cutoff(calibration_max, .05)

        physical = physical_paths(model, EVALUATION_COUNT, BASE_SEED + 10000 * mi + 1)
        healthy = run_common_g(model, physical, gate_on=True, threshold=cutoff['threshold'])
        fixed_drop = np.zeros((EVALUATION_COUNT, 150, 6), dtype=bool)
        fixed_drop[:, 1:, :] = True
        static_world = run_common_g(model, physical, gate_override=fixed_drop,
                                    threshold=cutoff['threshold'])
        policies = {}
        path_arrays = {}
        for pi, (policy, theta) in enumerate(THETA.items()):
            delta, beta = attack_watermarks(
                model, EVALUATION_COUNT, theta,
                BASE_SEED + 20000 * mi,
                BASE_SEED + 30000 * mi,
                BASE_SEED + 40000 * mi + 100 * pi)
            gated = run_common_g(model, physical, delta, gate_on=True,
                                 threshold=cutoff['threshold'])
            no_gate = run_common_g(model, physical, delta, gate_on=False)
            gated_metrics = summarize(gated, physical, cutoff['threshold'])
            no_gate_metrics = summarize(no_gate, physical, cutoff['threshold'])
            policies[policy] = {
                'theta': theta,
                'zeta': float(sscd.watson_stats(2, theta)['zeta']),
                'gated': {key: aggregate_metric(value) for key, value in gated_metrics.items()},
                'no_gate': {key: aggregate_metric(value) for key, value in no_gate_metrics.items()},
            }
            path_arrays[policy + '_gated'] = gated_metrics
            path_arrays[policy + '_no_gate'] = no_gate_metrics
            if policy == 'iid':
                static_attack = run_common_g(model, physical, delta,
                                             gate_override=fixed_drop,
                                             threshold=cutoff['threshold'])
                static_attack_metrics = summarize(static_attack, physical,
                                                  cutoff['threshold'])
        static_metrics = summarize(static_world, physical, cutoff['threshold'])
        # Static dropping has the same result for every watermark law, so it is
        # reported once as the explicit known-candidate baseline.
        policies['static_known_six'] = {
            'gated': {key: aggregate_metric(value) for key, value in static_metrics.items()},
            'attacked_with_iid_watermark': {key: aggregate_metric(value)
                                            for key, value in static_attack_metrics.items()},
            'attack_minus_healthy': {key: aggregate_metric(static_attack_metrics[key] - static_metrics[key])
                                     for key in ['network_mse_post20', 'affected_mse_post20',
                                                 'attack_network_mse']},
            'scope': 'Oracle/known-candidate support baseline; all six registered links removed from k=1 onward.'}
        iid_gated = path_arrays['iid_gated']
        iid_nogate = path_arrays['iid_no_gate']
        for policy in ['negative', 'positive']:
            policies[policy]['paired_gated_minus_iid'] = {
                key: aggregate_metric(path_arrays[policy + '_gated'][key] - iid_gated[key])
                for key in ['network_mse_post20', 'affected_mse_post20',
                            'attack_network_mse', 'eligible_tpr_full_window']}
            policies[policy]['paired_no_gate_minus_iid'] = {
                key: aggregate_metric(path_arrays[policy + '_no_gate'][key] - iid_nogate[key])
                for key in ['network_mse_post20', 'affected_mse_post20',
                            'attack_network_mse']}
        healthy_metrics = summarize(healthy, physical, cutoff['threshold'])
        model_result = {
            'calibration': cutoff,
            'calibration_minimum_maximum': float(calibration_max.min()),
            'calibration_maximum_maximum': float(calibration_max.max()),
            'healthy': {key: aggregate_metric(value) for key, value in healthy_metrics.items()},
            'static_known_six': policies.pop('static_known_six'),
            'policies': policies,
            'deterministic_risk_reference': 'common_g_watermark_risk.json',
            'limits': [
                '128 paired paths are a screening sample, not a definitive power study.',
                'The threshold is calibrated independently per dynamics; the rank rule is marginal over calibration and fresh path.',
                'Raw plugin score remains the recovered compatibility statistic; no chi-square replacement.'
            ]
        }
        result['models'][label] = model_result
        archive[label + '_calibration_max'] = calibration_max
        for key, value in path_arrays.items():
            for metric, array in value.items():
                archive[label + '_' + key + '__' + metric] = array
    np.savez_compressed(HERE / 'common_g_small_paths.npz', **archive)
    (HERE / 'common_g_small_results.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
