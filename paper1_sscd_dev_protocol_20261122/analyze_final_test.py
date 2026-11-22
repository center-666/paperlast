"""Stratified paired summaries for the frozen 12-scenario final test."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
RESULT = json.loads((HERE / 'final_test_results.json').read_text(encoding='utf-8'))
ARRAYS = np.load(HERE / 'final_test_paths.npz', allow_pickle=False)


def stratified_difference(model, policy, metric):
    by_scenario = []
    for scenario in RESULT['models'][model]['scenarios']:
        candidate = ARRAYS[f'{scenario}_{policy}__{metric}']
        iid = ARRAYS[f'{scenario}_iid__{metric}']
        by_scenario.append(candidate - iid)
    n_scenarios = len(by_scenario)
    means = np.array([x.mean() for x in by_scenario])
    components = [x.var(ddof=1) / len(x) / n_scenarios**2 for x in by_scenario]
    variance = float(sum(components))
    se = float(np.sqrt(variance))
    df = variance**2 / sum(v**2 / (len(x) - 1) for v, x in zip(components, by_scenario))
    critical = float(student_t.ppf(.975, df))
    return {
        'mean_equal_weight_scenario_difference': float(means.mean()),
        'stratified_path_se': se,
        'satterthwaite_df': float(df),
        'conditional_95ci_t': [float(means.mean() - critical * se),
                               float(means.mean() + critical * se)],
        'scenarios_with_negative_mean': int(np.sum(means < 0)),
        'scenario_count': n_scenarios,
        'paths_per_scenario': len(by_scenario[0]),
        'scenario_means': {name: float(value) for name, value in zip(
            RESULT['models'][model]['scenarios'], means)},
    }


def main():
    out = {
        'status': 'PASS',
        'interpretation': 'Post-lock descriptive comparisons; do not reselect from these final-test outcomes.',
        'locked_selection': RESULT['locked_selection'],
        'models': {},
    }
    for model in ['unstable', 'stable']:
        metrics = {}
        for policy in ['neg_boundary', 'neg06', 'pos06', 'pos_boundary']:
            metrics[policy] = {
                key: stratified_difference(model, policy, key)
                for key in ['network_post20', 'affected_post20', 'network_attack',
                                    'eligible_tpr', 'eligible_first_alarm_delay',
                                    'first_alarm_delay_from_attack_onset']}
        out['models'][model] = {
            'threshold': RESULT['models'][model]['threshold'],
            'independent_healthy_test': RESULT['models'][model]['independent_healthy_test'],
            'candidate_means_equal_weight_scenarios': {
                policy: {key: float(np.mean([
                    RESULT['models'][model]['scenarios'][s]['candidates'][policy][key]['mean']
                    for s in RESULT['models'][model]['scenarios']]))
                         for key in ['network_post20', 'affected_post20', 'eligible_tpr',
                                     'eligible_first_alarm_delay']}
                for policy in ['iid', 'neg_boundary', 'neg06', 'pos06', 'pos_boundary']},
            'paired_minus_iid': metrics,
            'known_six_oracle_equal_weight_scenario_means': {
                key: float(np.mean([RESULT['models'][model]['scenarios'][s]['oracle_static_six'][key]['mean']
                                    for s in RESULT['models'][model]['scenarios']]))
                for key in ['network_post20', 'affected_post20']},
        }
    (HERE / 'final_test_analysis.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({m: {
        p: {k: {'mean': v[k]['mean_equal_weight_scenario_difference'],
                'ci': v[k]['conditional_95ci_t'],
                'sign_scenarios': v[k]['scenarios_with_negative_mean']}
            for k in ['network_post20', 'eligible_tpr']}
        for p, v in d['paired_minus_iid'].items()}
        for m, d in out['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
