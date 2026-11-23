"""Replay original terminal losses and independently audit the three D0 archives."""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

import alarm_weight_replay as replay
import locked_weight_check as locked
import sweep_link_weight as d0
from ablation import node_inputs
from run_critical import PHYSICAL, matvec

HERE = replay.HERE


def determinant(a):
    if len(a)==1:
        return a[0,0]
    return sum((-1)**j*a[0,j]*determinant(np.delete(a[1:],j,axis=1)) for j in range(len(a)))


def verify_matrices(model, book):
    # Exact rational arithmetic on stored binary64 K,G,H,A and q coefficients.
    rational = np.vectorize(Fraction.from_float,otypes=[object])
    bound = Fraction(31,50)
    minors = {}
    for i in (8,10):
        q = Fraction.from_float(float(d0.WEIGHTS[i]))
        K,G,_ = book[i]
        H = rational(model.H)
        C = H[16]+H[28]+q*H[35]
        F = (np.eye(4,dtype=int)-rational(K[0,26])@H[26]-rational(G[0,26])@C)@rational(model.A)
        D = bound**2*np.eye(4,dtype=int)-F.T@F
        values = [determinant(D[:k,:k]) for k in range(1,5)]
        assert min(values)>0
        minors[str(i/10)] = {'leading_minors':[str(x) for x in values],
                              'K':K[0,26].tolist(),'G':G[0,26].tolist()}
    return minors


def terminal(model, inputs, truth, fault, weights, bank):
    local, neighbors, packet = inputs
    state = np.zeros((len(truth),4))
    total = np.zeros(len(state))
    K,G = bank
    for k in range(1,110):
        idx = weights[:,k]
        alpha = idx[:,None]/10
        p = state@model.A.T
        j35 = packet[:,k]+fault[:,k]-matvec(model.H[35],p)
        remaining = neighbors[:,k]-packet[:,k]-matvec(model.H[16]+model.H[28],p)
        state = p+matvec(K[idx],local[:,k]-matvec(model.H[26],p))+matvec(G[idx],remaining+alpha*j35)
        end = ((state[:,2:]-truth[:,k,2:])**2).sum(axis=1)
        if k>=50:
            total += end
    return end,total/60


def original_endpoints(original):
    data, models = {},{}
    cfg = json.loads(PHYSICAL.read_text(encoding='utf-8'))
    for mi,label in enumerate(('unstable','stable')):
        model,_,_=d0.critical_model(label,cfg)
        full = [d0.design_weighted_gains(model,q) for q in d0.WEIGHTS]
        bank = tuple(np.stack([tables[b][0,26] for tables in full]) for b in (0,1))
        models[label] = {'q08_unit_certificate':verify_matrices(model,full),
                         'gain_table_sha256':[hashlib.sha256(b''.join(a.astype('<f8').tobytes() for a in tables)).hexdigest()
                                              for tables in full]}
        for fi,fixture in enumerate(('healthy','continuous','burst')):
            if fi==0:
                paths=d0.physical(model,512,d0.ROOT_SEED+mi*100000+2)
                active=np.zeros(150,bool)
                deltas={'clean':np.zeros((512,150,2))}
            else:
                paths,active,all_delta,_=d0.attack_paths(model,mi,fi-1,fixture)
                deltas={wm:v[:,:,6] for wm,v in all_delta.items()}
            inputs=node_inputs(model,paths)
            for wm,fault in deltas.items():
                prefix=label+'__healthy' if fi==0 else '__'.join((label,fixture,wm))
                for policy in (('all_accept_reference','H4','H1') if fi==0 else
                               ('fixed_weight','H4_weighted','H1_weighted','known_timing')):
                    root=prefix+'__'+policy
                    if policy[:2] in ('H4','H1'):
                        key = label+'__healthy_gates__' if fi==0 else prefix+'__attack_gates__'
                        kept=~original[key+policy[:2]][:,:,6]
                    elif policy=='known_timing':
                        kept=np.broadcast_to(~active,(512,150))
                    else:
                        kept=np.ones((512,150),bool)
                    target=original[root+'__node26_missing_fault_mse']
                    rows=[]
                    for j,index in enumerate([10] if policy=='known_timing' else range(11)):
                        end,mean=terminal(model,inputs,paths.x,fault,kept*index,bank)
                        np.testing.assert_allclose(mean,target[j],rtol=1e-8,atol=1e-8)
                        rows.append(end)
                    data[root]=np.stack(rows)
    return data,models


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results/D0_audit')
    dest=parser.parse_args().output
    if dest.exists():
        raise FileExistsError(dest)
    report={}
    for p,meta,file in [(replay.SOURCE,'d0_results.json','d0_paths.npz'),
                        (replay.OUTPUT,'analysis.json','paths.npz'),(locked.OUTPUT,'analysis.json','paths.npz')]:
        r=json.loads((p/meta).read_text(encoding='utf-8'))
        assert d0.digest(p/file)==r.get('path_archive_sha256',r.get('path_sha256'))
        for source,h in r.get('source_hashes',r.get('source_sha256',{})).items():
            assert d0.digest((d0.ROOT if '/' in source else HERE)/source)==h
        with np.load(p/file) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
            report[p.name]={'array_count':len(z.files),'hash':d0.digest(p/file)}
            if p==replay.SOURCE:
                ends,models=original_endpoints(z)
                checked=0
                for label,m in r['models'].items():
                    for fixture,sc in m['attacks'].items():
                        for wm,w in sc.items():
                            for pol,rec in w['policies'].items():
                                for metric,rows in rec.items():
                                    if metric=='paired_comparisons':
                                        continue
                                    vals=z['__'.join((label,fixture,wm,pol,metric))]
                                    np.testing.assert_allclose(vals.mean(axis=1),[v['mean'] for v in rows])
                                    np.testing.assert_allclose(vals.std(axis=1,ddof=1)/np.sqrt(vals.shape[1]),
                                                               [v['path_se'] for v in rows])
                                    checked+=len(rows)
                report['original_fault_summary_rows_checked']=checked
            if p==locked.OUTPUT:
                for label,m in r['models'].items():
                    assert d0.empirical_threshold(z[label+'__cal'])[0]==m['threshold']
                    assert z[label+'__healthy__clean__gates'].any(axis=1).sum()==m['healthy_alarms']
                w='unstable__burst__negative__'
                for base in ('accept','hard','fixed08','fixed09'):
                    delta=z[w+'locked__missing_fault']-z[w+base+'__missing_fault']
                    assert np.isclose(delta.mean(),r['independent_primary'][base]['mean'])
    dest.mkdir()
    np.savez_compressed(dest/'original_missing_end.npz',**ends)
    report['end_archive_sha256']=d0.digest(dest/'original_missing_end.npz')
    report['models']=models
    d0.save_json(dest/'verification.json',report)
    print('Verified',len(ends),'endpoint arrays')


if __name__=='__main__':
    main()
