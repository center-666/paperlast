"""Exploratory heldout BSCW checks after model audit; not original 10-tap validation."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bscw_source',HERE/'source_snapshots/BSCW/risk_frontier.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
sigma=np.array([1.,.5,1/1.04**2,.58/1.12**2])
W=np.diag(1/sigma);b=np.array([0.,1.,0.,1.]);P=1.
p=float(b@W@b);l0=(P/(1+P*p))*b@W
h0=P/(1+P*p);target=1.02*h0
E=np.diag([1+1.04**2,1+1.12**2]);Ei=np.diag(1/np.sqrt(np.diag(E)))


def prepare(B,joint):
    A=B[:,:,1:]
    precision=W-np.outer(W@b,b@W)/(1/P+p if joint else p)
    C=np.einsum('nri,rs,nsj->nij',A,precision,A)
    ev,V=np.linalg.eigh((C+C.transpose(0,2,1))/2)
    v=np.einsum('r,nri->ni',l0,A)
    ve=np.einsum('ni,nij->nj',v,V)
    return np.maximum(ev,0),V,ve


def assess(prepared,regularizer):
    ev,V,ve=prepared
    excess=np.sum(ve**2*ev/(ev+regularizer)**2,axis=1)
    g=np.einsum('nj,nij->ni',ve*regularizer/(ev+regularizer),V)
    S=g.T@g/len(g)
    health=h0+float(excess.mean())
    return dict(health=health,health_relative_increment=health/h0-1,
                risk_fixed=health+10*float(np.linalg.eigvalsh(Ei@S@Ei)[-1]),
                risk_pair_oracle=health+10*float(np.trace(Ei@S@Ei)))


def run():
    out=[]
    samplers=[('iid',source.sample_iid),('bscw035',lambda n,s:source.sample_bscw(n,s,.35)),
              ('LC_TCW',source.sample_full_lctcw)]
    for index,(name,sampler) in enumerate(samplers):
        for rep in range(3):
            cal=source.build_block(sampler(50000,202611250000+1000*index+rep))
            test=source.build_block(sampler(50000,202611260000+1000*index+rep))
            for joint in (False,True):
                prepared=prepare(cal,joint)
                hi=1.
                while assess(prepared,hi)['health']>target:
                    hi*=2
                reg=brentq(lambda x:assess(prepared,x)['health']-target,1e-6,hi)
                row=dict(scheme=name,family='joint_prior' if joint else 'contrast',replicate=rep,
                         regularizer=reg,calibration=assess(prepared,reg),evaluation=assess(prepare(test,joint),reg))
                out.append(row)
    result=dict(scope='Exploratory two-tap diagnostic, 3 repetitions, independent calibration and evaluation; E=10 uses I+A^TA, not original ten-tap E=10.',
                sigma_diag=sigma.tolist(),state_prior=P,baseline_health=h0,calibration_health_target=target,
                energy_gram=E.tolist(),rows=out)
    (HERE/'bscw_holdout_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('bscw_holdout_results.json written')


if __name__=='__main__':
    run()
