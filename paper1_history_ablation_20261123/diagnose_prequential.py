"""Replay C2 calibration streams to inspect adaptive Mahalanobis scaling."""

import json
from pathlib import Path
import numpy as np

from ablation import critical_model, physical, read, residuals
from experiment_c import N_CAL, RANK, ROOT2, prequential_scores
from run_critical import PHYSICAL


def main():
    config=read(PHYSICAL)
    frozen=read(Path(__file__).resolve().parent/'results/C2/results.json')
    report={'status':'PASS','scope':'Replay of registered C2 calibration streams only. No new sample.',
            'models':{}}
    for mi,label in enumerate(('unstable','stable')):
        model,_,_=critical_model(label,config)
        seed=ROOT2+100000+10000*mi
        paths=physical(model,N_CAL,seed)
        r=residuals(model,paths)
        count,_,edges,dimension=r.shape
        mean=np.zeros((count,edges,dimension))
        covariance=np.broadcast_to(model.Theta[None],(count,edges,dimension,dimension)).copy()
        minimum=[]
        for j in range(r.shape[1]):
            if j>0:
                minimum.append(np.linalg.eigvalsh(covariance).min(axis=(1,2)))
            delta=r[:,j]-mean
            mean+=.5*delta
            covariance=.5*covariance+.25*delta[..., :,None]*delta[...,None,:]
        scores,_=prequential_scores(r,model.Theta)
        maxima=scores[:,4:,:].max(axis=(1,2))
        expected=frozen['models'][label]['calibration']['gate_thresholds']['EWMA-B']
        np.testing.assert_equal(np.partition(maxima,RANK-1)[RANK-1],expected)
        min_eigs=np.concatenate(minimum)
        report['models'][label]={
            'calibration_seed':seed,'prequential_threshold':float(expected),
            'path_max_quantiles':np.quantile(maxima,[.5,.9,.95,.99,1.]).tolist(),
            'minimum_predictive_covariance_eigenvalue_quantiles':np.quantile(min_eigs,[.001,.01,.1,.5,1.]).tolist(),
            'single_score_quantiles':np.quantile(scores[:,5:,:],[.5,.9,.99,.999,1.]).tolist(),
            'EWMA_effective_sample_size_at_lambda_0_5':3,
            'residual_dimension':8,'ridge':1e-6,
        }
    path=Path(__file__).resolve().parent/'prequential_scale_diagnostic.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
