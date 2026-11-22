"""Read-back checks for the executed pathwise gate-risk audit."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    result = json.loads((HERE / 'gate_risk_bound_results.json').read_text(encoding='utf-8'))
    arrays = np.load(HERE / 'gate_risk_bound_paths.npz', allow_pickle=False)
    assert result['n_calibration'] == 512 and result['n_paths'] == 64
    assert set(result['models']) == {'unstable', 'stable'}
    checked = 0
    for model, data in result['models'].items():
        assert set(data['policies']) == {'iid', 'negative', 'positive'}
        for policy, values in data['policies'].items():
            assert values['c_spectral_max'] < .1
            assert values['exact_recurrence_max_abs'] < 1e-8
            assert values['paired_identity_max_abs'] < 1e-8
            assert values['J_pair_bias']['mean'] <= values['J_pair_bias_bound']['mean'] + 1e-12
            assert values['J_attacked']['mean'] <= values['J_attacked_bound']['mean'] + 1e-12
            for key in ['J_healthy','J_attacked','J_pair_bias','J_pair_bias_bound',
                        'J_attacked_bound','U_selection','U_attack_forcing',
                        'U_selection_attack_cross','U_total']:
                assert values[key]['n_paths'] == 64
            prefix = model + '_' + policy + '_'
            for key in ['J_healthy','J_attacked','J_pair_bias','U_selection',
                        'U_attack_forcing','U_cross','U_total','bound_pair_bias','bound_attacked']:
                assert arrays[prefix + key].shape == (64,)
            np.testing.assert_allclose(
                arrays[prefix + 'U_selection'] + arrays[prefix + 'U_attack_forcing']
                + arrays[prefix + 'U_cross'], arrays[prefix + 'U_total'], rtol=1e-10, atol=1e-10)
            checked += 1
    source_zip = HERE / 'source_package.zip'
    package = json.loads((HERE / 'package_manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(source_zip.read_bytes()).hexdigest() == package['source_zip_sha256']
    assert package['source_zip_sha256'] == '652eb72f9b68b9cc61afbdb6c0e1040e8bad160da24d0689e9baef5df516bd0d'
    print(json.dumps({'status':'PASS','policies_checked':checked,
                      'path_arrays':len(arrays.files),
                      'source_package_sha256':package['source_zip_sha256']},indent=2))


if __name__ == '__main__':
    main()
