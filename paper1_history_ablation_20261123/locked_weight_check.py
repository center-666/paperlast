"""Independent D0 confirmation of the precommitted H1 alarm weight."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.linalg import pinv
from scipy.stats import t

import alarm_weight_replay as replay
import sweep_link_weight as d0
from ablation import node_inputs, residuals, scores_all
from run_critical import PHYSICAL, critical_model, innovation_aggregate, run_filter

HERE = replay.HERE
BATCH = 128
OUTPUT = HERE/'results/D0_locked'
ROOT_SEED = 3917285401


def seed(model, stage, batch, stream):
    return int(np.random.SeedSequence([ROOT_SEED, model, stage, batch, stream]).generate_state(1,dtype=np.uint64)[0])


def h1(model, paths, fault=None):
    r = residuals(model, paths, fault)
    inv = np.stack([pinv(c) for c in model.Theta])
    score = np.zeros((len(r), d0.T, 7))
    score[:,5:] = .5*np.einsum('btei,eij,btej->bte',r[:,1:],inv,r[:,1:])
    return score


def calibration(model, mi):
    batches = []
    for b in range(16):
        paths = d0.physical(model,BATCH,seed(mi,0,b,0))
        scores = h1(model, paths)
        if b == 0:
            small = type(paths)(x=paths.x[:2],y=paths.y[:2])
            np.testing.assert_allclose(scores[:2],scores_all(residuals(model,small),model.Theta)['H1'],atol=1e-10)
        batches.append(scores.max(axis=(1,2)))
    maxima = np.concatenate(batches)
    threshold, rank = d0.empirical_threshold(maxima)
    assert rank == 1947
    return maxima, threshold


def make_fault(draws, active, theta):
    beta = d0.beta_from_common(draws,theta)[:,:,0,:]
    stress = slice(d0.START, d0.STOP)
    delta = np.zeros((BATCH,d0.T,7,2))
    v = np.array([6.,0.])
    delta[:,stress,d0.EDGE] = (.5*v + beta*(beta@v)[...,None])*active[None,stress,None]
    return delta


def rest_of_network(model, paths, aggregate, mask):
    # The 49 other receivers never use node 26's state or the faulty packet on this edge.
    post,_ = run_filter(model,paths,aggregate,np.zeros((len(paths.x),d0.T,7,2)),mask)
    sq = ((post-paths.x[:,:,None,:])**2).sum(axis=3)
    return {k:sq[:,sl,:26].sum(axis=2).mean(axis=1)+sq[:,sl,27:].sum(axis=2).mean(axis=1)
            for k,sl in [('network_fault',slice(50,110)),('network_post20',slice(20,150))]}


def run_batch(model, mi, stage, batch, bank, threshold, entries):
    paths = d0.physical(model,BATCH,seed(mi,stage,batch,0))
    healthy = stage == 1
    active = (np.arange(d0.T)>=50)&(np.arange(d0.T)<110)&(not healthy)
    if stage == 3:
        active &= (np.arange(d0.T)-50)%10<6
    score0 = h1(model,paths)
    raw_gates = score0>threshold
    aggregate = innovation_aggregate(model,paths)
    rest_gate = rest_of_network(model,paths,aggregate,raw_gates)
    rest_keep = rest_of_network(model,paths,aggregate,np.zeros_like(raw_gates))
    inputs = node_inputs(model,paths)
    draws = None if healthy else d0.make_common_draws(BATCH,60,1,*[seed(mi,stage,batch,i) for i in (1,2,3)])
    stage_name = ('cal','healthy','continuous','burst')[stage]
    for wm,theta in ({'clean':0} if healthy else d0.THETAS).items():
        delta = np.zeros((BATCH,d0.T,7,2)) if healthy else make_fault(draws,active,theta)
        scores = score0 if healthy else h1(model,paths,delta)
        gates = scores > threshold
        prefix = '__'.join((model.label,stage_name,wm))
        entries.setdefault(prefix+'__gates',[]).append(np.packbits(gates,axis=2,bitorder='little')[:,:,0])
        alarms = gates[:,:,d0.EDGE]
        for name, idx, flags, others in [('hard',0,alarms,rest_gate),('locked',8,alarms,rest_gate),
                    ('accept',10,np.zeros_like(alarms),rest_keep),('fixed08',8,np.ones_like(alarms),rest_keep),
                    ('fixed09',9,np.ones_like(alarms),rest_keep)]:
            metrics = replay.run_node(inputs,paths.x,delta[:,:,d0.EDGE],flags,active,d0.WEIGHTS[idx],bank[idx])
            for metric in others:
                metrics[metric] = (others[metric]+metrics[metric])/50
            for metric,value in metrics.items():
                entries.setdefault(prefix+'__'+name+'__'+metric,[]).append(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    destination = parser.parse_args().output
    if destination.exists():
        raise FileExistsError(destination)
    report = {'root_seed':ROOT_SEED,'source_sha256':{p.name:d0.digest(p) for p in
              (Path(__file__),Path(replay.__file__),Path(d0.__file__),HERE/'D0_locked_registration.md')},
              'models':{},'independent_primary':{}}
    entries = {}
    cfg = json.loads(PHYSICAL.read_text(encoding='utf-8'))
    for mi,label in enumerate(('unstable','stable')):
        model,_,_ = critical_model(label,cfg)
        bank,_ = replay.gain_pairs(model)
        maxima, threshold = calibration(model,mi)
        entries[label+'__cal'] = [maxima]
        report['models'][label] = {'threshold':threshold}
        for stage, count in ((1,16),(2,8),(3,8)):
            for batch in range(count):
                run_batch(model,mi,stage,batch,bank,threshold,entries)
            print(label,stage,flush=True)
    data = {k:np.concatenate(parts) for k,parts in entries.items()}
    assert all(np.isfinite(v).all() for v in data.values())
    for label in report['models']:
        hits = int(data[label+'__healthy__clean__gates'].any(axis=1).sum())
        report['models'][label].update(healthy_alarms=hits, healthy_n=2048, fwer_ci=d0.exact_interval(hits,2048))
    primary = report['independent_primary']
    p = 'unstable__burst__negative__'
    target = data[p+'locked__missing_fault']
    for name in ('hard','accept','fixed08','fixed09'):
        diff = target-data[p+name+'__missing_fault']
        primary[name] = d0.summary(diff)
        primary[name]['upper_one_sided_975'] = float(diff.mean()+t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff)))
    def avg(policy,metric,group=p):
        return float(data[group+policy+'__'+metric].mean())
    health = 'unstable__healthy__clean__'
    primary['checks'] = {
        'dual_upper':max(primary[k]['upper_one_sided_975'] for k in ('hard','accept'))<0,
        'healthy_mse':avg('locked','network_post20',health)<=1.01*avg('accept','network_post20',health),
        'tail':avg('locked','missing_tail')<=1.05*avg('accept','missing_tail'),
        'forcing':avg('locked','forcing')<avg('accept','forcing'),
        'gap_weight':avg('locked','gap_weight')>avg('hard','gap_weight'),
        'fwer':report['models']['unstable']['fwer_ci'][1]<=.08,
        'contraction':bool(np.mean(data[p+'locked__max_window_norm']<1)>=.95)}
    with TemporaryDirectory(dir=destination.parent) as tmp:
        folder = Path(tmp)/'result'
        folder.mkdir()
        np.savez_compressed(folder/'paths.npz',**data)
        report['path_sha256']=d0.digest(folder/'paths.npz')
        d0.save_json(folder/'analysis.json',report)
        folder.rename(destination)
    print(primary,flush=True)


if __name__ == '__main__':
    main()
