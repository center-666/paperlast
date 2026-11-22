"""Verify both new research folders without launching a simulation.

--refresh-manifests explicitly refreshes the per-folder byte manifests after
all intended changes. Normal invocation verifies them and detects additions.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import re
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FOLDERS = [REPO / 'paper1_50node_tv_gate_theory_20261121', HERE]
EXCLUDED = {'artifact_manifest.json', 'delivery_checks.json'}
CONFIG_SHA = 'c25437077cb73ac2332160e55f6183550bb8fc5318d93bb32c183a11d661b50d'


def digest(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def inventory(folder):
    return {p.relative_to(folder).as_posix(): digest(p)
            for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts
            and p.name not in EXCLUDED}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-manifests', action='store_true')
    args = parser.parse_args()
    cfg_path = HERE / 'physical_configuration.json'
    assert digest(cfg_path)['sha256'] == CONFIG_SHA
    git_cfg = subprocess.check_output([
        'git', '-c', 'safe.directory=' + REPO.as_posix(), '-C', str(REPO),
        'show', '049bdc6:paper1_original50_source_20261121/configuration_snapshot.json'])
    assert hashlib.sha256(git_cfg).hexdigest() == CONFIG_SHA
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    H = np.asarray(cfg['H'])
    diagonal = np.diagonal(H, axis1=1, axis2=2)
    assert np.array_equal(H, np.eye(2)[None] * diagonal[:, None, :])
    assert np.all((diagonal >= 1) & (diagonal < 1.5))
    assert np.array_equal(cfg['Q'], np.eye(2))
    assert np.array_equal(cfg['R'], np.broadcast_to(.5 * np.eye(2), (50, 2, 2)))
    gain_path = HERE / 'gain_table.json'
    gains = json.loads(gain_path.read_text(encoding='utf-8'))
    assert gains['physical_configuration_sha256'] == CONFIG_SHA
    acceptance = json.loads((HERE / 'acceptance_results.json').read_text(encoding='utf-8'))
    assert acceptance['status'] == 'PASS'
    assert acceptance['input_hashes']['gain_table_sha256'] == digest(gain_path)['sha256']
    cert = json.loads((HERE / 'rational_certificate.json').read_text(encoding='utf-8'))
    assert cert['coverage']['exact_rational_checks'] == 228
    assert cert['uses_F_diagnostic'] is False
    for name, row in cert['input_files'].items():
        assert digest(HERE / name) == row
    assert digest(HERE / 'certify_common_g.py')['sha256'] == cert['certificate_runtime']['script_sha256']
    for rowset in cert['certificates']:
        assert rowset['modes'] == 57 and rowset['all_modes_strictly_certified']
        for row in rowset['rows']:
            assert Fraction(row['D11']) > 0 and Fraction(row['D22']) > 0 and Fraction(row['det_D']) > 0
    risk = json.loads((HERE / 'common_g_watermark_risk.json').read_text(encoding='utf-8'))
    assert risk['status'] == 'PASS'
    assert risk['generic_moment_identity_check']['max_absolute_difference'] < 1e-11
    for model in risk['models'].values():
        iid = model['historical_direction_risk']['iid']
        assert model['historical_direction_risk']['negative'] < iid < model['historical_direction_risk']['positive']
    screening = json.loads((HERE / 'common_g_small_results.json').read_text(encoding='utf-8'))
    assert screening['calibration_paths_per_model'] == 512
    assert screening['evaluation_paths_per_model'] == 128
    for model in screening['models'].values():
        assert model['policies']['negative']['paired_gated_minus_iid']['network_mse_post20']['mean'] < 0
        assert model['policies']['positive']['paired_gated_minus_iid']['network_mse_post20']['mean'] > 0
        assert model['static_known_six']['attack_minus_healthy']['network_mse_post20']['mean'] == 0
    paths = np.load(HERE / 'common_g_small_paths.npz', allow_pickle=False)
    assert len(paths.files) > 20 and all(np.all(np.isfinite(paths[k])) for k in paths.files)
    subprocess.run([sys.executable, str(FOLDERS[0] / 'verify_archive.py')], check=True, capture_output=True)
    checked_links = 0
    for folder in FOLDERS:
        for path in folder.glob('*.md'):
            text = path.read_text(encoding='utf-8')
            assert '\ufffd' not in text, path
            for link in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)', text):
                if '://' in link or link.startswith('#'):
                    continue
                target = link.split('#')[0]
                assert (path.parent / target).exists(), (path.name, target)
                checked_links += 1
    files = {}
    for folder in FOLDERS:
        current = inventory(folder)
        manifest_path = folder / 'artifact_manifest.json'
        if args.refresh_manifests:
            manifest_path.write_text(json.dumps({'scope': 'All delivered files except manifests/dynamic delivery_checks and ignored bytecode.',
                                                 'files': current}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert manifest['files'] == current, 'Delivery contents drifted: ' + folder.name
        files[folder.name] = {'file_count': len(current), 'total_bytes': sum(x['bytes'] for x in current.values())}
    report = {'status': 'PASS', 'new_simulations': 0,
              'physical_config_matches_git_049bdc6': True,
              'source_and_assumption_checks': True,
              'rational_checks_verified': 228,
              'acceptance_input_hashes_match': True,
              'legacy_saved_statistics_read_back': True,
              'relative_markdown_links_checked': checked_links,
              'folders': files,
              'limits': 'No common-G SSCD efficacy validation; no conditional-MMSE or floating-instruction certificate.'}
    (HERE / 'delivery_checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
