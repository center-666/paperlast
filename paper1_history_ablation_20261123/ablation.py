"""Registered detector-history ablation on frozen, offline 35->26 packet faults."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.linalg import pinv
from scipy.stats import beta as beta_dist, t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'paper1_critical_complement_20261123'
sys.path.insert(0, str(BASE))
from run_critical import (EDGE, PHYSICAL, RECEIVER, START, STOP, T, beta_from_common,
                          critical_model, digest, make_common_draws, matvec, physical, save_json, summary)

NAMES = ('H4', 'H2', 'H1', 'EWMA')
RIDGE = 1e-6 * np.eye(8)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def residuals(model, paths, error=None, edges=None):
    edges = np.arange(7) if edges is None else np.asarray(edges)
    packets = paths.y[:, :, model.links[edges, 1]].copy()
    packets[:, 0] = 0
    if error is not None:
        packets += error
    windows = np.lib.stride_tricks.sliding_window_view(packets, 5, axis=1)
    windows = windows.transpose(0, 1, 2, 4, 3).reshape(len(packets), T-4, len(edges), 10)
    return np.einsum('edm,btem->bted', model.Z[edges], windows)


def plugin(mean, covariance, inverses, logdet):
    trace = np.einsum('eij,beji->be', inverses, covariance)
    energy = np.einsum('bei,eij,bej->be', mean, inverses, mean)
    sign, log_cov = np.linalg.slogdet(covariance)
    if not np.all(sign > 0):
        raise ArithmeticError('Regularized covariance is not positive definite.')
    return np.maximum(0., .5 * (trace + energy - 8 + logdet - log_cov))


def scores_all(r, reference):
    count, ticks, edges, _ = r.shape
    out = {name: np.zeros((count, T, edges)) for name in NAMES}
    inv = np.stack([pinv(x) for x in reference])
    logdet = np.linalg.slogdet(reference)[1]
    mass, squared_mass = 1., 1.
    ew_mean = r[:, 0].copy()
    scatter = np.zeros((count, edges, 8, 8))
    for j in range(1, ticks):
        now = j + 4
        current = r[:, j]
        for name, length in [('H4', 4), ('H2', 2)]:
            history = r[:, max(0, j-length+1):j+1].transpose(0, 2, 3, 1)
            mean = history.mean(axis=-1)
            centered = history - mean[..., None]
            covariance = np.einsum('bedt,beft->bedf', centered, centered) / (history.shape[-1]-1)
            out[name][:, now] = plugin(mean, covariance + RIDGE, inv, logdet)
        out['H1'][:, now] = .5 * np.einsum('bei,eij,bej->be', current, inv, current)
        prior_mass = .5 * mass
        mass = prior_mass + 1.
        squared_mass = .25 * squared_mass + 1.
        change = current - ew_mean
        ew_mean += change / mass
        scatter *= .5
        scatter += (prior_mass / mass) * change[..., :, None] * change[..., None, :]
        covariance = scatter / (mass - squared_mass/mass)
        out['EWMA'][:, now] = plugin(ew_mean, covariance + RIDGE, inv, logdet)
    return out


def interval(k, n):
    return [float(beta_dist.ppf(.025, k, n-k+1)) if k else 0.,
            float(beta_dist.ppf(.975, k+1, n-k)) if k<n else 1.]


def calibration(model, mi, old, historical_arrays, archive):
    label = model.label
    cal_batches, holdout_batches = [], []
    old_seed = old['calibration']['seed']
    selection = {}
    for stage, target in (('cal', cal_batches), ('hold', holdout_batches)):
        for batch in range(4):
            seed = (84260000 if stage == 'cal' else 85260000) + 10000*mi + batch
            if stage == 'cal' and batch == 0:
                seed = old_seed
            paths = physical(model, 512, seed)
            all_scores = scores_all(residuals(model, paths), model.Theta)
            maxima = np.stack([all_scores[name].max(axis=(1, 2)) for name in NAMES])
            target.append(maxima)
        archive[f'{label}_{stage}_maxima'] = np.stack(target)
    thresholds = np.partition(np.stack(cal_batches), 487, axis=2)[:, :, 487]
    held = np.concatenate(holdout_batches, axis=1)
    np.testing.assert_allclose(cal_batches[0][0], historical_arrays[f'{label}__calibration_maxima'],
                               atol=3e-8, rtol=0)
    np.testing.assert_allclose(thresholds[0, 0], old['calibration']['threshold'], atol=2e-8, rtol=0)
    for i, name in enumerate(NAMES):
        counts = [int((held[i] > u).sum()) for u in thresholds[:, i]]
        selection[name] = {'main_batch': i, 'main_threshold': float(thresholds[i, i]),
                           'threshold_by_batch': thresholds[:, i].tolist(), 'holdout_n': held.shape[1],
                           'holdout_counts_by_batch': counts, 'holdout_ci95': interval(counts[i], held.shape[1])}
    print(label, 'FWER counts out of 2048:', [selection[n]['holdout_counts_by_batch'][i]
          for i, n in enumerate(NAMES)], flush=True)
    return thresholds, selection


def delay(event):
    seen = event.any(axis=-1)
    return np.where(seen, np.argmax(event, axis=-1), event.shape[-1]), ~seen


def longest(gate):
    run = np.zeros(len(gate), int)
    peak = run.copy()
    for bit in gate.T:
        run = (run + 1) * bit
        peak = np.maximum(peak, run)
    return peak


def evaluate_gates(gate, active, last, norms):
    stop_delay, stop_censored = delay(~gate[:, last:])
    detect_delay, detect_censored = delay(gate[:, START:STOP] & active[None, START:STOP])
    full_run = longest(gate[:, 1:])
    words = np.lib.stride_tricks.sliding_window_view(gate[:, 1:], 7, axis=1) @ (1 << np.arange(6,-1,-1))
    values = {'fault_alarm_fraction': gate[:, active].mean(axis=1),
              'fault_accept_fraction': (~gate[:, active]).mean(axis=1),
              'detection_delay': detect_delay, 'any_fault_alarm': ~detect_censored,
              'detection_censored': detect_censored, 'stop_reacq': stop_delay, 'stop_censored': stop_censored,
              'longest_run': full_run, 'longest_run_window': longest(gate[:, START:STOP]),
              'seven_has_accept': full_run <= 6, 'seven_contract': norms[words].max(axis=1) < 1}
    if np.any(~active[START:STOP]):
        gaps = gate[:, START:STOP].reshape(len(gate), 6, 10)[:, :, 6:]
        durations, censored = delay(~gaps)
        values.update(gap_reacq=durations.mean(axis=1), gap_censored=censored.mean(axis=1),
                      gap_accept_fraction=(~gaps).mean(axis=(1,2)))
        for i in range(4):
            values[f'gap_accept_{i+1}'] = (~gaps[:, :, i]).mean(axis=1)
    return values


def node_inputs(model, paths):
    neighbors = model.cfg['graph']['neighbors'][RECEIVER]
    np.testing.assert_array_equal(np.flatnonzero(model.links[:, 0] == RECEIVER), [EDGE])
    return paths.y[:, :, RECEIVER], paths.y[:, :, neighbors].sum(axis=2), paths.y[:, :, model.links[EDGE,1]]


def node_filter(model, inputs, gate, fault, start=1, prior=None):
    local, neighbors, critical = inputs
    state = np.zeros((len(gate), 4)) if prior is None else prior.copy()
    history = np.empty((len(gate), T-start+1, 4))
    history[:, 0] = state
    for k in range(start, T):
        drop = gate[:, k, None]
        mask = gate[:, k].astype(int) * 64
        admitted = neighbors[:, k] - drop * critical[:, k] + (~drop) * fault[:, k]
        state = (matvec(model.F[mask, RECEIVER], state) + matvec(model.K[mask, RECEIVER], local[:, k])
                 + matvec(model.G[mask, RECEIVER], admitted))
        history[:, k-start+1] = state
    return history


def mse(hist, truth):
    return ((hist[..., 2:] - truth[..., 2:])**2).sum(axis=-1).mean(axis=-1)


def tail_intervention(model, inputs, truth, last, fault, actual, clean, sa, sc):
    # Rows select inherited state, columns select subsequent gate schedule.
    aa = mse(sa[:, last:], truth[:, last:])
    cc = mse(sc[:, last:], truth[:, last:])
    ac = mse(node_filter(model, inputs, clean, fault, last, sa[:, last-1])[:, 1:], truth[:, last:])
    ca = mse(node_filter(model, inputs, actual, fault, last, sc[:, last-1])[:, 1:], truth[:, last:])
    inherited = .5*((ac-cc)+(aa-ca))
    scheduling = .5*((ca-cc)+(aa-ac))
    np.testing.assert_allclose(inherited + scheduling, aa-cc, atol=1e-8, rtol=1e-11)
    return dict(tail_aa=aa, tail_ac=ac, tail_ca=ca, tail_cc=cc,
                inherited_state=inherited, subsequent_gate=scheduling)


def run_fixture(model, label, fixture, record, norms, thresholds, reg, old_arrays, archive):
    paths = physical(model, 512, record['physical_seed'])
    inputs = node_inputs(model, paths)
    clean_scores = scores_all(residuals(model, paths), model.Theta)
    active = np.array(record['fixture_active_mask'])
    last = int(np.flatnonzero(active)[-1]+1)
    draws = make_common_draws(512, STOP-START, 1, *record['watermark_seeds'])
    clean_states = {}
    for i,name in enumerate(NAMES):
        cg = clean_scores[name][:, :, EDGE] > thresholds[i,i]
        clean_states[name] = node_filter(model, inputs, cg, np.zeros((512,T,2)))
        archive[f'{label}.{fixture}.clean.{name}.bits'] = np.packbits(
            clean_scores[name] > thresholds[i,i], axis=2, bitorder='little')[:, :, 0]
    reports = {}
    vector = active[START:STOP,None] * np.asarray(reg['fixture_vector'])
    for policy, theta in reg['theta_candidates'].items():
        beta = beta_from_common(draws, theta)[:, :, 0]
        fault = np.zeros((512,T,2))
        fault[:,START:STOP] = .5*vector + beta*np.sum(beta*vector,axis=-1,keepdims=True)
        critical_scores = scores_all(residuals(model, paths, fault[:,:,None], [EDGE]), model.Theta[EDGE:EDGE+1])
        outputs = {}
        reports[policy] = {}
        for i,name in enumerate(NAMES):
            cutoff = thresholds[i,i]
            full_gates = clean_scores[name] > cutoff
            s = critical_scores[name][:,:,0]
            full_gates[:,:,EDGE] = s > cutoff
            gate = full_gates[:,:,EDGE]
            clean_gate = clean_scores[name][:,:,EDGE] > cutoff
            prefix = f'{label}.{fixture}.{policy}.{name}'
            archive[prefix+'.bits'] = np.packbits(full_gates, axis=2, bitorder='little')[:,:,0]
            if name != 'EWMA':
                washout = last + 3 + {'H1':1, 'H2':2, 'H4':4}[name]
                np.testing.assert_allclose(s[:,washout:], clean_scores[name][:,washout:,EDGE], atol=1e-10, rtol=1e-12)
            state = node_filter(model, inputs, gate, fault)
            values = evaluate_gates(gate, active, last, norms)
            values['missing_window_mse'] = mse(state[:,START:STOP], paths.x[:,START:STOP])
            values['missing_post110_mse'] = mse(state[:,STOP:], paths.x[:,STOP:])
            values.update(tail_intervention(model, inputs, paths.x, last, fault, gate, clean_gate,
                                            state, clean_states[name]))
            values['tail_gate_disagreement_fraction'] = (gate[:,last:] != clean_gate[:,last:]).mean(axis=1)
            if name == 'H4':
                old_prefix = f'{label}__{fixture}__{policy}__'
                np.testing.assert_array_equal(gate, old_arrays[old_prefix+'critical_gate'])
                for new,old in [('missing_window_mse','missing_window_mse'), ('missing_post110_mse','missing_tail_mse')]:
                    np.testing.assert_allclose(values[new], old_arrays[old_prefix+old], atol=1e-7, rtol=1e-9)
            for metric,v in values.items():
                archive[prefix+'.'+metric] = v
            outputs[name] = values
            reports[policy][name] = {metric:summary(v.astype(float)) for metric,v in values.items()}
            if fixture == 'burst':
                gaps = s[:,START:STOP].reshape(len(s), 6, 10)[:,:,6:]
                archive[prefix+'.gap_durations'] = delay(gaps <= cutoff)[0].astype(np.uint8)
                archive[prefix+'.sensitivity_gap'] = np.stack([
                    delay(gaps <= u)[0].mean(axis=1) for u in thresholds[:,i]])
        paired = {}
        for name in NAMES[1:]:
            paired[name] = {metric: summary(value.astype(float)-outputs['H4'][metric].astype(float))
                            for metric,value in outputs[name].items()}
            if fixture == 'burst':
                root = f'{label}.{fixture}.{policy}.'
                paired[name]['gap_changes_same_calibration_batch'] = (
                    archive[root+name+'.sensitivity_gap']-archive[root+'H4.sensitivity_gap']).mean(axis=1).tolist()
                gap = paired[name]['gap_reacq']
                paired[name]['gap_one_sided_bonferroni_upper'] = float(gap['mean']+t.ppf(1-.05/3,511)*gap['path_se'])
        reports[policy]['paired_vs_H4'] = paired
    print(label, fixture, 'finished all four detectors and three watermarks', flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if any((output/p).exists() for p in ['results.json','paths.npz']):
        raise FileExistsError('Use a new --output-dir for a new reproduction.')
    base = read(BASE/'critical_results.json')
    reg = read(BASE/'registration.json')
    for p,h in base['sources'].items():
        assert digest(ROOT/p) == h, p
    assert digest(BASE/'run_critical.py') == base['script_sha256']
    assert digest(BASE/'critical_paths.npz') == base['path_archive_sha256']
    report = {'registration_commit':'9acfaca','registration_sha256':digest(HERE/'REGISTRATION.md'),
              'runner_sha256':digest(Path(__file__)), 'historical_arrays_sha256':base['path_archive_sha256'],
              'detector_order':NAMES,'models':{}}
    archive = {}
    with np.load(BASE/'critical_paths.npz', allow_pickle=False) as old:
        for mi,label in enumerate(('unstable','stable')):
            model,_,norms = critical_model(label, read(PHYSICAL))
            thresholds,cal = calibration(model, mi, base['models'][label], old, archive)
            report['models'][label] = {'calibration':cal, 'fixtures':{}}
            for fixture,rec in base['models'][label]['fixtures'].items():
                report['models'][label]['fixtures'][fixture] = run_fixture(
                    model,label,fixture,rec,norms,thresholds,reg,old,archive)
    with TemporaryDirectory(dir=output) as temporary:
        npz = Path(temporary)/'paths.npz'
        result = Path(temporary)/'results.json'
        np.savez_compressed(npz, **archive)
        report['archive_sha256'] = digest(npz)
        report['array_count'] = len(archive)
        save_json(result, report)
        npz.rename(output/npz.name)
        result.rename(output/result.name)
    print('Archive complete', flush=True)


if __name__ == '__main__':
    main()
