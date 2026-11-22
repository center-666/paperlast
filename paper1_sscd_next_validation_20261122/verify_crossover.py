"""Read-back checks for the executed four-world attribution experiment."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    result = json.loads((HERE / 'crossover_results.json').read_text(encoding='utf-8'))
    assert result['kind'] == 'paper1_common_G_50node_counterfactual_gate_ablation'
    assert result['calibration_paths'] == 512 and result['evaluation_paths'] == 128
    assert set(result['models']) == {'unstable', 'stable'}
    checked = 0
    for model in result['models'].values():
        assert np.isfinite(model['cutoff']['threshold'])
        assert model['cutoff']['one_based_rank'] == 488
        for metric, values in model['shapley_counterfactual_attribution'].items():
            total = values['total']['mean']
            direct = values['direct_watermark_at_fixed_gate']['mean']
            gate = values['gate_schedule_effect']['mean']
            assert np.isclose(total, direct + gate, atol=1e-12)
            for term in values.values():
                assert np.isfinite(term['mean']) and np.isfinite(term['paired_path_se'])
            checked += 1
        for policy in ['iid', 'negative']:
            assert model['natural'][policy]['network_post20']['paths'] == 128
            assert model['detection'][policy]['eligible_tpr']['paths'] == 128
    zip_path = HERE / 'source_package.zip'
    package_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    manifest = json.loads((HERE / 'package_manifest.json').read_text(encoding='utf-8'))
    assert package_hash == manifest['source_zip_sha256']
    assert package_hash == 'a9797b3797ebe2e1f7234d3ba3b74fb065163d24fe9f8cd681cf64fa616e50ab'
    arrays = np.load(HERE / 'crossover_paths.npz', allow_pickle=False)
    path_manifest = json.loads((HERE / 'crossover_path_manifest.json').read_text(encoding='utf-8'))
    assert set(arrays.files) == set(path_manifest['array_keys'])
    for model in ['unstable', 'stable']:
        loss_names = ['network_post20', 'network_attack', 'network_post_attack',
                      'affected_post20', 'affected_attack', 'affected_post_attack']
        for world in ['ii', 'ni', 'in', 'nn']:
            for metric in loss_names:
                assert arrays[f'{model}_{world}__{metric}'].shape == (128,)
        for policy in ['iid', 'negative']:
            assert arrays[f'{model}_{policy}__gates'].shape == (128, 150, 6)
            assert arrays[f'{model}_{policy}__masks'].shape == (128, 150)
            assert arrays[f'{model}_{policy}__scores'].shape == (128, 150, 6)
        for metric in loss_names:
            total = arrays[f'{model}_nn__{metric}'] - arrays[f'{model}_ii__{metric}']
            direct = .5 * ((arrays[f'{model}_ni__{metric}'] - arrays[f'{model}_ii__{metric}'])
                           + (arrays[f'{model}_nn__{metric}'] - arrays[f'{model}_in__{metric}']))
            gate = .5 * ((arrays[f'{model}_in__{metric}'] - arrays[f'{model}_ii__{metric}'])
                         + (arrays[f'{model}_nn__{metric}'] - arrays[f'{model}_ni__{metric}']))
            np.testing.assert_allclose(total, direct + gate, rtol=1e-11, atol=1e-11)
            checked += 1
    print(json.dumps({'status': 'PASS', 'models': 2,
                      'shapley_metrics_checked': checked,
                      'path_level_arrays': len(arrays.files),
                      'calibration_paths_per_model': 512,
                      'evaluation_paths_per_model': 128,
                      'source_package_sha256': package_hash}, indent=2))


if __name__ == '__main__':
    main()
