"""Standard full-history Gaussian GLRT when the frozen initial prior is known.

This is an audit supplement to the registered nuisance-invariant five-packet
test, not a new detector or a replacement of the registered experiment.
"""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
import numpy as np
from scipy.linalg import cholesky,solve_triangular,eigh
import measurement_model as mm
from batch_power_and_attacks import secrets,P1,p3,power
H=Path(__file__).resolve().parent

def full_info(m,aux=None,M=None):
    end=11
    rows=[(t,j,c) for t in range(end+1) for j in (1,0,2) for c in (0,1)]
    if aux:
        rows += [(t,3,aux['coordinate']) for t in range(end+1)
                 if t%5==aux['phase'] and t+aux['delay']<=end]
    r=mm.raw_matrices(m,rows,aux)
    Sigma=r['cov']+r['O']@np.eye(2)@r['O'].T
    D=mm.attack_map(m,rows,7,M)
    U=solve_triangular(cholesky(Sigma,lower=True),D,lower=True)
    G=U.T@U
    # Independent dense inverse identity; no call to the parity-whitening score.
    assert np.max(np.abs(G-D.T@np.linalg.solve(Sigma,D)))<1e-8
    return G

def main():
    results={};mats={}
    old=json.loads((H/'batch_power_results.json').read_text('utf8'))
    for name in ('unstable','stable'):
        m=mm.ref.load_model(name);a,S,W=P1(m);ag,Sg,_=P1(m,True)
        G=full_info(m);base=old['models'][name]['schemes']['B1']
        G5=np.load(H/'batch_matrices.npz')[name+'__B1__G'][0]
        assert eigh(G-G5,eigvals_only=True)[0]>-1e-8
        schemes={'B1_prior_history':(G[None],S,a)}
        for c in (0,1):
            X=full_info(m,mm.schedule(coordinate=c))
            assert eigh(X-G,eigvals_only=True)[0]>-1e-8
            schemes['B3_prior_history_coord'+str(c)]=(X[None],S,a)
        GB2=np.stack([full_info(m,M=M) for M in secrets(m)])
        schemes['B2_prior_history']=(GB2,Sg,ag)
        rows={}
        for key,(Z,V,at) in schemes.items():
            rows[key]={'P3':p3(Z,2,V,W,key.startswith('B2')),
                       'Pd_at_P1':power(Z,2,np.array(at['xi'])),
                       'min_expected_ncp_E10':float(10*eigh(Z.mean(0),W,eigvals_only=True)[0])}
            mats[name+'__'+key+'__G']=Z
        results[name]={'schemes':rows,'registered_B1_P3':base['P3']['Pd'],
                       'min_eigenvalue_Gprior_minus_Gregistered':float(eigh(G-G5,eigvals_only=True)[0])}
    out={'date':'2026-11-16','audit_supplement':True,
         'purpose':'do not call five-packet nuisance-invariant GLRT globally optimal when the known Gaussian prior and raw history are available',
         'information':'local1, raw neighbors0/2 at times0..11; known x0~N(0,I); auxiliary periodic times0,5,10 when included',
         'batch_FA':.05,'attack_start':7,'known_start_single_batch_only':True,
         'models':results,'new_detector_or_G2':False,'new_physical_samples':0,
         'secret_average_is_diagnostic_not_certified':True,
         'unknown_start_online_full_history_optimality_not_claimed':True}
    (H/'prior_baseline_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8')
    np.savez_compressed(H/'prior_baseline_matrices.npz',**mats)
    print(json.dumps(results,ensure_ascii=False))
if __name__=='__main__':main()
