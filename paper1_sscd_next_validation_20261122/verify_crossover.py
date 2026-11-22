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
    print(json.dumps({'status': 'PASS', 'models': 2,
                      'shapley_metrics_checked': checked,
                      'calibration_paths_per_model': 512,
                      'evaluation_paths_per_model': 128,
                      'source_package_sha256': package_hash}, indent=2))


if __name__ == '__main__':
    main()
