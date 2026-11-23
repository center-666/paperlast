"""Self-contained 2D gate/estimation toy (NOT the source 50-node experiment).

Quantifies how much of the candidate-minus-iid paired standard error can be
removed by sharing Watson transition innovations across two candidate laws.
All physical noise, attack energy, radial variables and first direction are
identical across the two experimental designs. It never modifies live attacks.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from watson_crn import make_common_draws, beta_from_common, watermark_attack_from_beta

HERE=Path(__file__).resolve().parent
THETA=-1.3152369768257723

def risk(beta,noise,e0,raw,A,gate_threshold):
    B,L,edges,_=beta.shape
    assert edges==1
    dot=np.einsum('btef,tf->bte',beta,raw)
    delta=(.5*raw[None,:,None,:]+beta*dot[...,None])[:,:,0,:]
    recv=delta+noise
    res=np.zeros_like(recv)
    res[:,1:]=recv[:,1:]-recv[:,:-1]@A.T
    gate=np.sum(res*res,axis=-1)>gate_threshold
    b=np.zeros((B,L,2))
    for k in range(1,L):
        b[:,k]=.08*b[:,k-1]+.2*(~gate[:,k])[:,None]*delta[:,k]
    return np.mean(np.sum((e0+b)**2,axis=-1),axis=1)


def run(n=4096,length=60,reps=8,seed=20261122):
    A=np.array([[1.025,.015],[.0,1.014]])
    d=np.array([.9,-.7])
    raw=np.stack([np.linalg.matrix_power(A,k)@d for k in range(length)])
    raw*=np.sqrt(60./np.sum(raw**2))
    sigma=.3
    threshold=float(chi2.ppf(.95,2)*sigma*sigma*(2+.05))
    rs=np.random.default_rng(seed)
    rows=[]
    for i in range(reps):
        base=seed+100_000*i
        common=make_common_draws(n,length,1,base+1,base+2,base+3)
        alt=make_common_draws(n,length,1,base+1,base+2,base+4)
        physical=rs.normal(size=(n,length,2))*sigma
        healthy=rs.normal(size=(n,length,2))*.15
        iid=beta_from_common(common,0)
        neg_c=beta_from_common(common,THETA)
        neg_ind=beta_from_common(alt,THETA)
        vals={key:risk(beta,physical,healthy,raw,A,threshold)
              for key,beta in [('iid',iid),('negative_crn',neg_c),('negative_independent',neg_ind)]}
        dcrn=vals['negative_crn']-vals['iid']
        dind=vals['negative_independent']-vals['iid']
        rows.append({'rep':i,'mean_delta_crn':float(dcrn.mean()),
                     'mean_delta_independent':float(dind.mean()),
                     'se_delta_crn':float(dcrn.std(ddof=1)/np.sqrt(n)),
                     'se_delta_independent':float(dind.std(ddof=1)/np.sqrt(n)),
                     'corr_iid_neg_crn':float(np.corrcoef(vals['iid'],vals['negative_crn'])[0,1]),
                     'corr_iid_neg_ind':float(np.corrcoef(vals['iid'],vals['negative_independent'])[0,1])})
        print(f'rep {i}: paired SE CRN {rows[-1]["se_delta_crn"]:.7f} vs independent {rows[-1]["se_delta_independent"]:.7f}',flush=True)
    x=np.array([r['se_delta_crn'] for r in rows]);y=np.array([r['se_delta_independent'] for r in rows])
    out={'kind':'SYNTHETIC_MODEL_ONLY_NOT_50NODE',
         'n_paths_per_rep':n,'reps':reps,'seed':seed,'theta_negative':THETA,
         'rng':'common physical, same r and initial direction; CRN shares quantile U and sign branch; independent comparison draws distinct U and branch',
         'average_SE_CRN':float(x.mean()),'average_SE_independent':float(y.mean()),
         'SE_ratio_CRN_to_independent':float(x.mean()/y.mean()),
         'equivalent_sample_size_ratio_independent_over_CRN':float((y.mean()/x.mean())**2),
         'mean_delta_CRN':float(np.mean([r['mean_delta_crn'] for r in rows])),
         'mean_delta_independent':float(np.mean([r['mean_delta_independent'] for r in rows])),
         'replicates':rows}
    (HERE/'synthetic_variance_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('replicates','rng')},indent=2))
    return out

if __name__=='__main__':run()
