"""Registered independent-seed C1 and EWMA A/B/C mechanism studies."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from ablation import (HERE, BASE, EDGE, START, STOP, T, critical_model,
                      evaluate_gates, interval, make_common_draws, node_filter,
                      node_inputs, physical, read, residuals, scores_all, summary,
                      tail_intervention)
from run_critical import PHYSICAL, beta_from_common, digest, save_json

ROOT1 = 2026122301
ROOT2 = 2026122401
N_CAL = N_HOLD = 2048
N_TEST = 1024
RANK = 1947
LAMBDA = .5
RECOVERY_RUN = 5
C1_NAMES = ('H4', 'H2', 'H1', 'EWMA-A')
C2_NAMES = ('H4', 'EWMA-A', 'EWMA-B', 'EWMA-C')
WATERMARKS = ('iid', 'negative', 'positive')


def packed(gate):
    return np.packbits(gate, axis=2, bitorder='little')[:, :, 0]


def max_path_score(scores):
    return scores[:, 4:, :].max(axis=(1, 2))


def threshold(values):
    assert values.shape == (2048,)
    return float(np.partition(values, RANK-1)[RANK-1])


def prequential_scores(r, reference, trusted_threshold=None, diagnostics=False):
    """EWMA-B/C predictive score; current residual enters statistics afterward."""
    count, _, edges, dimension = r.shape
    mean = np.zeros((count, edges, dimension))
    covariance = np.broadcast_to(reference[None], (count, edges, dimension, dimension)).copy()
    inverse_reference = np.linalg.inv(reference)
    scores = np.zeros((count, T, edges))
    trusted_score = np.zeros_like(scores)
    update_flags = np.zeros((count, T, edges), dtype=bool)
    streak = np.zeros((count, edges), dtype=np.uint8)
    mean_history, covariance_diagonal = [], []
    for j in range(r.shape[1]):
        current = r[:, j]
        difference = current-mean
        solved = np.linalg.solve(covariance + 1e-6*np.eye(dimension), difference[..., None])[..., 0]
        scores[:, j+4] = .5*np.einsum('bed,bed->be', difference, solved)
        healthy = .5*np.einsum('bed,edh,beh->be', current, inverse_reference, current)
        trusted_score[:, j+4] = healthy
        if trusted_threshold is None:
            eligible = np.ones((count, edges), bool)
        else:
            good = healthy <= trusted_threshold
            streak = np.where(good, np.minimum(streak+1, RECOVERY_RUN), 0)
            eligible = good & (streak >= RECOVERY_RUN)
        update_flags[:, j+4] = eligible
        next_mean = mean + (1-LAMBDA)*difference
        next_covariance = (LAMBDA*covariance + LAMBDA*(1-LAMBDA)
                           * difference[..., :, None]*difference[..., None, :])
        mean = np.where(eligible[..., None], next_mean, mean)
        covariance = np.where(eligible[..., None, None], next_covariance, covariance)
        if diagnostics:
            mean_history.append(mean[:, EDGE].copy())
            covariance_diagonal.append(np.diagonal(covariance[:, EDGE],axis1=-2,axis2=-1).copy())
    details = {'updates':update_flags, 'trusted_score':trusted_score}
    if diagnostics:
        details['mean'] = np.stack(mean_history,axis=1)
        details['covariance_diagonal'] = np.stack(covariance_diagonal,axis=1)
    return scores, details


def detector_scores(r, reference, phase, trusted_threshold=None, diagnostics=False):
    historical = scores_all(r, reference)
    if phase == 'C1':
        return {'H4':historical['H4'], 'H2':historical['H2'],
                'H1':historical['H1'], 'EWMA-A':historical['EWMA']}, {}
    b, b_extra = prequential_scores(r, reference, diagnostics=diagnostics)
    c, c_extra = prequential_scores(r, reference, trusted_threshold, diagnostics=diagnostics)
    scores = {'H4':historical['H4'], 'EWMA-A':historical['EWMA'],
              'EWMA-B':b, 'EWMA-C':c}
    return scores, {'EWMA-B':b_extra, 'EWMA-C':c_extra}


def run_calibration(model, model_index, phase, arrays):
    root = ROOT1 if phase == 'C1' else ROOT2
    trust_threshold = None
    trust_seed = None
    if phase == 'C2':
        trust_seed = root + 200000 + 10000*model_index
        trust_paths = physical(model,N_CAL,trust_seed)
        trust_values = scores_all(residuals(model,trust_paths),model.Theta)['H1']
        trust_maxima = max_path_score(trust_values)
        trust_threshold = threshold(trust_maxima)
        arrays[f'{phase}.{model.label}.trust_reference.maxima'] = trust_maxima
    gate_seed = root + 100000 + 10000*model_index
    cal = physical(model,N_CAL,gate_seed)
    cal_scores, extra = detector_scores(residuals(model,cal),model.Theta,phase,
                                        trust_threshold,diagnostics=(phase=='C2'))
    methods = C1_NAMES if phase == 'C1' else C2_NAMES
    maxima = {name:max_path_score(cal_scores[name]) for name in methods}
    cutoffs = {name:threshold(maxima[name]) for name in methods}
    for name,value in maxima.items():
        arrays[f'{phase}.{model.label}.calibration.{name}.maxima'] = value
    if phase == 'C2':
        for name,value in extra.items():
            arrays[f'{phase}.{model.label}.calibration.{name}.update_bits'] = packed(value['updates'])
    hold_seed = root+(200000 if phase=='C1' else 300000)+10000*model_index
    held = physical(model,N_HOLD,hold_seed)
    held_scores,_ = detector_scores(residuals(model,held),model.Theta,phase,trust_threshold)
    health = {}
    for name,cutoff in cutoffs.items():
        alarm = np.any(held_scores[name][:,4:,:]>cutoff,axis=(1,2))
        arrays[f'{phase}.{model.label}.holdout.{name}.alarm'] = alarm
        k = int(alarm.sum())
        health[name] = {'k':k,'n':len(alarm),'rate':float(alarm.mean()),
                        'exact_ci95':interval(k,len(alarm))}
    record = {'gate_seed':gate_seed,'trusted_reference_seed':trust_seed,
              'holdout_seed':hold_seed,'n_calibration':N_CAL,'n_holdout':N_HOLD,
              'rank_one_based':RANK,'gate_thresholds':cutoffs,'health_fwer':health}
    if phase=='C2':
        record['trusted_reference_threshold'] = trust_threshold
    return cutoffs,trust_threshold,record


def add_metrics(prefix,values,report,arrays):
    report[prefix] = {}
    for name,value in values.items():
        value=np.asarray(value)
        arrays[prefix+'.'+name]=value
        report[prefix][name]=summary(value.astype(float))


def run_fixture(phase,model,label,fixture_index,fixture,cutoffs,trust_threshold,arrays):
    root=ROOT1 if phase=='C1' else ROOT2
    offset=300000 if phase=='C1' else 400000
    seed=root+offset+10000*model.model_index+1000*fixture_index
    paths=physical(model,N_TEST,seed)
    inputs=node_inputs(model,paths)
    active=np.zeros(T,bool)
    active[START:STOP]=True
    if fixture=='burst':
        active[START:STOP]=((np.arange(START,STOP)-START)%10)<6
    last=int(np.flatnonzero(active)[-1]+1)
    clean_raw=residuals(model,paths)
    clean_scores,clean_details=detector_scores(clean_raw,model.Theta,phase,trust_threshold,
                                                diagnostics=(phase=='C2'))
    clean_gates={name:score>cutoffs[name] for name,score in clean_scores.items()}
    clean_states={name:node_filter(model,inputs,gates[:,:,EDGE],np.zeros((N_TEST,T,2)))
                  for name,gates in clean_gates.items()}
    record={'physical_seed':seed,'watermark_seeds':[seed+11,seed+12,seed+13],
            'active_mask':active.tolist(),'detectors':{}}
    watermarks=WATERMARKS if phase=='C1' else ('negative',)
    draws=make_common_draws(N_TEST,STOP-START,1,seed+11,seed+12,seed+13)
    vector=np.tile(np.array([6.,0.]),(STOP-START,1))
    if fixture=='burst':
        vector*=active[START:STOP,None]
    for wm in watermarks:
        theta={'iid':0.,'negative':-1.3152369768257723,
               'positive':1.3152369768257723}[wm]
        beta=beta_from_common(draws,theta)[:,:,0]
        fault=np.zeros((N_TEST,T,2))
        fault[:,START:STOP]=.5*vector+beta*np.sum(beta*vector,axis=-1,keepdims=True)
        fault7=np.zeros((N_TEST,T,7,2))
        fault7[:,:,EDGE]=fault
        attacked_raw=clean_raw.copy()
        attacked_raw[:,:,EDGE:EDGE+1,:]=residuals(model,paths,fault7[:,:,EDGE:EDGE+1],[EDGE])
        attacked_scores,details=detector_scores(attacked_raw,model.Theta,phase,trust_threshold,
                                                diagnostics=(phase=='C2'))
        for name,score in attacked_scores.items():
            gates=score>cutoffs[name]
            gate=gates[:,:,EDGE]
            state=node_filter(model,inputs,gate,fault)
            values=evaluate_gates(gate,active,last,model.window_norms)
            missing=((state[:,:,2:]-paths.x[:,:,2:])**2).sum(axis=-1)
            values['missing_fault_window']=missing[:,START:STOP].mean(axis=1)
            values['missing_after_fault']=missing[:,STOP:].mean(axis=1)
            values['missing_post20']=missing[:,20:].mean(axis=1)
            values.update(tail_intervention(model,inputs,paths.x,last,fault,gate,
                                            clean_gates[name][:,:,EDGE],state,clean_states[name]))
            key=f'{phase}.{label}.{fixture}.{wm}.{name}'
            arrays[key+'.gate_bits']=packed(gates)
            arrays[key+'.critical_score']=score[:,:,EDGE]
            if phase=='C2' and name in details:
                data=details[name]
                arrays[key+'.update_bits']=packed(data['updates'])
                arrays[key+'.trusted_score_critical']=data['trusted_score'][:,:,EDGE]
                arrays[key+'.mean_critical']=data['mean']
                arrays[key+'.covariance_diagonal_critical']=data['covariance_diagonal']
            add_metrics(key,values,record['detectors'],arrays)
    print(phase,label,fixture,'complete',flush=True)
    return record


def run_phase(phase,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    if any((output_dir/name).exists() for name in ('results.json','paths.npz')):
        raise FileExistsError('Use a new output directory for a registered reproduction.')
    cfg=read(PHYSICAL)
    result={'phase':phase,'registration_amendment_sha256':digest(HERE/'REGISTRATION_C1_C2_AMENDMENT.md'),
            'runner_sha256':digest(Path(__file__)),'previous_critical_archive_sha256':digest(BASE/'critical_paths.npz'),
            'seed_root':ROOT1 if phase=='C1' else ROOT2,'models':{}}
    archive={}
    for mi,label in enumerate(('unstable','stable')):
        model,geometry,norms=critical_model(label,cfg)
        model.model_index=mi
        model.window_norms=norms
        cutoffs,trusted,cal_record=run_calibration(model,mi,phase,archive)
        model_record={'geometry':geometry,'calibration':cal_record,'fixtures':{}}
        for fi,fixture in enumerate(('continuous','burst')):
            model_record['fixtures'][fixture]=run_fixture(
                phase,model,label,fi,fixture,cutoffs,trusted,archive)
        result['models'][label]=model_record
    with TemporaryDirectory(dir=output_dir) as staging:
        npz=Path(staging)/'paths.npz'; js=Path(staging)/'results.json'
        np.savez_compressed(npz,**archive)
        result['archive_sha256']=digest(npz); result['array_count']=len(archive)
        save_json(js,result)
        npz.rename(output_dir/'paths.npz'); js.rename(output_dir/'results.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('C1','C2'),required=True)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    run_phase(args.phase,(args.output_dir or HERE/'results'/args.phase).resolve())


if __name__=='__main__':
    main()
