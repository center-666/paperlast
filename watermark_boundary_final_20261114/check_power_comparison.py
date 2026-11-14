"""Independent finite identity checks for the numerical comparison.

The noise map is rebuilt from primitive state/noise equations rather than the
block covariance formula. Full 1024 whole-sign damage paths independently
check the analytic terminal moment. No old research scripts are run.
"""
from pathlib import Path
import json,itertools
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import block_diag,eigh
from scipy.stats import ncx2

H=Path(__file__).resolve().parent
source=H.parent/'original_estimator_gate_20261114/five_node_aggregate_model.json'
model=json.loads(source.read_text(encoding='utf-8'))
result=json.loads((H/'baseline_comparison.json').read_text(encoding='utf-8'))
checks=[]
for name,m in model['models'].items():
    A=np.array(m['A']);F=np.array(m['nodes'][1]['posterior_F']);G=np.array(m['nodes'][1]['G'])
    # x0 + four process draws + five measurement draws, each two-dimensional.
    Y=np.zeros((10,20));x=np.zeros((2,20));x[:,:2]=np.eye(2)
    for t in range(5):
        if t:
            x=A@x;x[:,2+2*(t-1):2+2*t]+=np.eye(2)
        Y[2*t:2*t+2]=x
        Y[2*t:2*t+2,10+2*t:12+2*t]+=np.eye(2)
    Z=np.zeros((8,10))
    for t in range(4):Z[2*t:2*t+2,2*t:2*t+2]=-A;Z[2*t:2*t+2,2*t+2:2*t+4]=np.eye(2)
    N=Z@Y;assert np.max(abs(N[:,:2]))<1e-14
    noise=N@block_diag(np.eye(10),.5*np.eye(10))@N.T
    theta=np.zeros((8,8))
    for j in range(4):
        theta[2*j:2*j+2,2*j:2*j+2]=1.5*np.eye(2)+.5*A@A.T
        if j<3:
            theta[2*j:2*j+2,2*j+2:2*j+4]=-.5*A.T
            theta[2*j+2:2*j+4,2*j:2*j+2]=-.5*A
    assert np.max(abs(noise-theta))<1e-14
    acc=np.zeros((2,2))
    As=[np.linalg.matrix_power(A,t)for t in range(10)]
    for signs in itertools.product([-1.,1.],repeat=10):
        B=np.zeros((2,2))
        for t in range(10):B=F@B+signs[t]*G@As[t]
        acc+=B.T@B/1024
    expected=np.array(result['models'][name]['schemes']['whole_sign']['S'])
    assert np.max(abs(acc-expected))<1e-12
    whole_error=float(np.max(abs(acc-expected)))
    # Matrix operator identities on several anisotropic symmetric inputs.
    errors=[]
    hs,ws=hermgauss(3)
    for X in [G.T@G,np.array([[2.,.7],[.7,1.]]),A.T@A]:
        co=np.zeros((2,2))
        for a,b in itertools.product([-1.,1.],repeat=2):
            S=np.diag([a,b]);co+=S.T@X@S/4
        assert np.max(abs(co-np.diag(np.diag(X))))<1e-12
        haar=np.zeros((2,2));gm=np.zeros((2,2))
        for th in 2*np.pi*np.arange(64)/64:
            R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
            haar+=R.T@X@R/64
        for i,x in enumerate(hs):
            for j,y in enumerate(hs):
                beta=np.sqrt(2)*np.array([x,y]);M=np.eye(2)+np.outer(beta,beta)
                gm+=ws[i]*ws[j]/np.pi*(M.T@X@M)
        e=max(np.max(abs(haar-np.trace(X)/2*np.eye(2))),np.max(abs(gm-(5*X+np.trace(X)*np.eye(2)))))
        assert e<1e-12;errors.append(float(e))
    detrows=[]
    for scheme in ['identity','public_scalar_alternation','public_coordinate_alternation','public_R90_supplement']:
        r=result['models'][name]['schemes'][scheme]
        W=np.array(r['W']);Q=np.array(r['expected_ncp_matrix'])
        lam=10*eigh(Q,W,eigvals_only=True)[0]
        closed=float(ncx2.sf(result['threshold'],8,max(0,lam)))
        numer=r['probes'][-1]['E10_min_power']['Pd']
        assert abs(closed-numer)<1e-9
        detrows.append(dict(scheme=scheme,closed_form_min_power=closed,angular_search= numer,error=abs(closed-numer)))
    checks.append(dict(model=name,primitive_noise_covariance_error=float(np.max(abs(noise-theta))),
        whole_sign_all_1024_damage_error=whole_error,matrix_operator_errors=errors,
        deterministic_fixed_word_global_E10_min_power_check=detrows))

# Add plug-in-at-mean diagnostics without overwriting the research data.
plugin=[]
for name,m in result['models'].items():
    for scheme,r in m['schemes'].items():
        xi=np.array(r['probes'][-1]['E10_min_power']['initial_state']);Q=np.array(r['expected_ncp_matrix'])
        lam=float(xi@Q@xi)
        plugin.append(dict(model=name,scheme=scheme,saved_actual_minimizer=xi.tolist(),
            expected_ncp_at_saved_direction=lam,actual_integrated_power=r['probes'][-1]['E10_min_power']['Pd'],
            incorrect_mean_ncp_plugin_power=float(ncx2.sf(result['threshold'],8,max(0,lam)))))
out=dict(date='2026-11-14',checks=checks,mean_ncp_plugin_is_not_power=plugin,
         physical_noise_samples=0,old_scripts_run=0,new_watermark_design=False)
(H/'power_comparison_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(checks,indent=2))
