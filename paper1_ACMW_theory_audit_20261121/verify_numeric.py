"""Independent direct-matrix verification of risk identities, no output overwrite."""
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bscw_check',HERE/'bscw_holdout.py')
check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)
rows=[]
for joint in (False,True):
    B=check.source.build_block(check.source.sample_bscw(1000,202611270000,.35))
    A=B[:,:,1:];b=check.b;W=check.W;p=check.p;P=check.P;l0=check.l0
    precision=W-np.outer(W@b,b@W)/(1/P+p if joint else p)
    regularizer=17.
    C=np.einsum('nri,rs,nsj->nij',A,precision,A)
    R=np.linalg.solve(C+regularizer*np.eye(2),A.transpose(0,2,1)@precision)
    v=np.einsum('r,nri->ni',l0,A)
    L=l0-np.einsum('ni,nir->nr',v,R)
    health=P*(L@b-1)**2+np.sum(L**2*check.sigma,axis=1)
    g=np.einsum('nr,nri->ni',L,A)
    S=g.T@g/len(g)
    direct_fixed=float(health.mean())+10*np.linalg.eigvalsh(check.Ei@S@check.Ei)[-1]
    prepared=check.assess(check.prepare(B,joint),regularizer)
    np.testing.assert_allclose(health.mean(),prepared['health'],rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(direct_fixed,prepared['risk_fixed'],rtol=2e-12,atol=2e-12)
    rows.append(dict(joint_prior=joint,absolute_health_error=abs(float(health.mean())-prepared['health']),
                     absolute_fixed_risk_error=abs(float(direct_fixed)-prepared['risk_fixed'])))
result=json.loads((HERE/'review_results.json').read_text())
for row in result['information']:
    assert max(row['quadrature'].values())-min(row['quadrature'].values())<1e-9
    assert row['pure_same_zeta']['mi_nats']<=row['mixed']['mi_nats']+1e-12
for case in ('aligned','rotation'):
    values=[r for r in result['geometry']['exact'] if r['case']==case]
    delta=values[1]['dg']-values[0]['dg']
    assert delta>0 if case=='aligned' else delta<0
verification=dict(direct_risk_checks=rows,information_quadrature='passed',adverse_geometry='passed')
(HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
print(json.dumps(verification,indent=2))
