"""Paired D0 intervention on the weight assigned to an alarmed packet."""
import argparse
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import t

import sweep_link_weight as d0
from ablation import longest, node_inputs
from run_critical import PHYSICAL, matvec

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'results/D0_20261123_final'
OUTPUT = HERE / 'results/D0_alarm'
REGISTER = HERE / 'D0_alarm_registration.md'
METRICS = ('missing_fault', 'missing_tail', 'missing_end', 'network_fault', 'network_post20',
           'forcing', 'max_window_norm', 'longest_low_run', 'gap_weight', 'gap_accept',
           'fault_weight', 'fault_accept')


def gain_pairs(model):
    bank, global_bounds = [], []
    for q in d0.WEIGHTS:
        tables = d0.design_weighted_gains(model, q)
        K, G, F = [np.stack((unit[0, 26], table[0, 26]))
                   for unit, table in zip((model.K, model.G, model.F), tables)]
        products = []
        for word in itertools.product((0, 1), repeat=7):
            product = np.eye(4)
            for bit in word:
                product = F[bit] @ product
            products.append(np.linalg.norm(product, 2))
        bank.append((K, G, F, np.array(products)))
        global_bounds.append(max(products))
    return bank, global_bounds


def run_node(inputs, truth, fault, alarms, active, q, gains):
    local, neighbor, critical = inputs
    K, G, F, products = gains
    flags = alarms.astype(int)
    weights = np.where(alarms, q, 1.)
    words = np.lib.stride_tricks.sliding_window_view(flags[:, 1:], 7, axis=1) @ (1 << np.arange(6,-1,-1))
    n = len(local)
    state = np.zeros((n, 4))
    square = np.zeros((n, d0.T, 4))
    forcing = np.zeros((n, d0.T))
    for k in range(1, d0.T):
        flag = flags[:, k]
        w = weights[:, k, None]
        received = neighbor[:, k] + (w-1)*critical[:, k] + w*fault[:, k]
        state = matvec(F[flag], state) + matvec(K[flag], local[:, k]) + matvec(G[flag], received)
        square[:, k] = (state - truth[:, k])**2
        forcing[:, k] = (matvec(G[flag, 2:], w*fault[:, k])**2).sum(axis=1)
    stress = slice(d0.START, d0.STOP)
    missing = square[:, :, 2:].sum(axis=2)
    total = square.sum(axis=2)
    result = dict(missing_fault=missing[:, stress].mean(axis=1),
                  missing_tail=missing[:, d0.STOP:].mean(axis=1), missing_end=missing[:, d0.STOP-1],
                  network_fault=total[:, stress].mean(axis=1), network_post20=total[:, 20:].mean(axis=1),
                  forcing=forcing[:, stress].mean(axis=1), max_window_norm=products[words].max(axis=1),
                  longest_low_run=longest(weights[:, 1:] <= .1))
    for label, mask in [('gap', ~active[stress]), ('fault', active[stress])]:
        if np.any(mask):
            subset = weights[:, stress][:, mask]
            result[label+'_weight'] = subset.mean(axis=1)
            result[label+'_accept'] = (subset>0).mean(axis=1)
    return result


def evaluate(model, paths, fault, active, prefix, detector, bank, archive, store, healthy):
    gate_key = (prefix+'__healthy_gates__'+detector if healthy else prefix+'__attack_gates__'+detector)
    hard = (prefix+'__healthy__'+detector if healthy else prefix+'__'+detector+'_weighted')
    fixed = (prefix+'__healthy__all_accept_reference' if healthy else prefix+'__fixed_weight')
    alarms = archive[gate_key][:, :, d0.EDGE]
    inputs = node_inputs(model, paths)
    metrics = [run_node(inputs, paths.x, fault, alarms, active, q, pair)
               for q, pair in zip(d0.WEIGHTS, bank)]
    data = {m:np.stack([row[m] for row in metrics]) for m in metrics[0]}
    for metric, field in [('missing_fault','node26_missing_fault_mse'), ('missing_tail','node26_missing_tail_mse'),
                          ('forcing','accepted_attack_forcing_energy')]:
        for endpoint, baseline in [(0,hard), (-1,fixed)]:
            np.testing.assert_allclose(data[metric][endpoint], archive[baseline+'__'+field][-1],
                                       rtol=2e-8, atol=3e-8)
    for m, field in [('network_fault','network_fault_mse'),('network_post20','network_post20_mse')]:
        node = data[m]
        data[m] = archive[hard+'__'+field][-1] + (node-node[0])/50
    for metric, values in data.items():
        if not np.isfinite(values).all():
            raise ArithmeticError(f'Invalid {prefix} {detector} {metric}')
        store[prefix+'__'+detector+'__'+metric] = values
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(PHYSICAL.read_text(encoding='utf-8'))
    initial = json.loads((SOURCE/'d0_results.json').read_text(encoding='utf-8'))
    assert d0.digest(SOURCE/'d0_paths.npz') == initial['path_archive_sha256']
    report = {'design':'exploratory same-path intervention', 'weights':d0.WEIGHTS.tolist(),
              'd0_paths_sha256':initial['path_archive_sha256'],
              'source_sha256':{p.name:d0.digest(p) for p in [REGISTER, Path(__file__), Path(d0.__file__)]},
              'models':{}, 'primary':{}}
    store = {}
    with np.load(SOURCE/'d0_paths.npz') as archive:
        for mi, label in enumerate(('unstable','stable')):
            model, _, _ = d0.critical_model(label, cfg)
            bank, bounds = gain_pairs(model)
            report['models'][label] = {'seven_step_suprema':bounds}
            for fi, fixture in enumerate(('healthy','continuous','burst')):
                if fixture == 'healthy':
                    paths = d0.physical(model, d0.HOLD_N, d0.ROOT_SEED+mi*100000+2)
                    active = np.zeros(d0.T,bool)
                    faults = {'clean':np.zeros((len(paths.x), d0.T, 2))}
                else:
                    paths, active, delta, _ = d0.attack_paths(model, mi, fi-1, fixture)
                    faults = {wm:d[:, :, d0.EDGE] for wm,d in delta.items()}
                for wm,fault in faults.items():
                    prefix = label if fixture=='healthy' else '__'.join((label,fixture,wm))
                    for detector in d0.DETECTORS:
                        data = evaluate(model, paths, fault, active, prefix, detector, bank, archive,
                                        store, fixture=='healthy')
                        if (label,fixture,wm) != ('unstable','burst','negative'):
                            continue
                        fixed = prefix+'__fixed_weight__'
                        full = archive[fixed+'node26_missing_fault_mse'][-1]
                        healthy_ref = archive[label+'__healthy__all_accept_reference__network_post20_mse'][-1]
                        rows = []
                        for i in range(1,10):
                            upper = []
                            for base in (data['missing_fault'][0],full):
                                diff = data['missing_fault'][i]-base
                                upper.append(float(diff.mean()+t.ppf(1-.05/36,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff))))
                            health_ratio = store[label+'__'+detector+'__network_post20'][i].mean()/healthy_ref.mean()
                            tail_ratio = data['missing_tail'][i].mean()/archive[fixed+'node26_missing_tail_mse'][-1].mean()
                            checks = {'both_mse_bounds':max(upper)<0, 'healthy_mse':health_ratio<=1.01,
                                      'tail':tail_ratio<=1.05, 'forcing':data['forcing'][i].mean()<archive[fixed+'accepted_attack_forcing_energy'][-1].mean(),
                                      'gap_weight':data['gap_weight'][i].mean()>data['gap_weight'][0].mean(),
                                      'fwer':initial['models'][label]['health'][detector]['exact_binomial_ci95'][1]<=.08,
                                      'contraction':np.mean(data['max_window_norm'][i]<1)>=.95}
                            rows.append({'q':float(d0.WEIGHTS[i]),'ucb_vs_hard_all':upper,
                                         'health_ratio':float(health_ratio),'tail_ratio':float(tail_ratio),
                                         'checks':{k:bool(v) for k,v in checks.items()},'pass':all(checks.values())})
                        report['primary'][detector] = rows
                print(label,fixture,flush=True)
    report['region'] = {d:[(a['q'], b['q']) for a,b in zip(rows,rows[1:]) if a['pass'] and b['pass']]
                        for d,rows in report['primary'].items()}
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        destination = Path(tmp)/'results'
        destination.mkdir()
        np.savez_compressed(destination/'paths.npz', **store)
        report['path_sha256'] = d0.digest(destination/'paths.npz')
        d0.save_json(destination/'analysis.json',report)
        destination.rename(args.output)
    print(report['region'])


if __name__ == '__main__':
    main()
