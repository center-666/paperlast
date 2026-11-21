"""Registered SSCD review checks and supplemental-model design experiments."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
from scipy.special import i0e, i1e
from scipy.stats import chi2
import sscd
from continuous_design import (_residual_covariance_polynomials,state_risk_polynomials,
    screen_continuum_2d,angle_minimum,_poly_value,_dg)

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def stats_checks():
    original=load_file('sscd_original',HERE/'source_snapshot/sscd.py')
    failures=[]
    for z in (-.99999,.99999):
        try:
            original.theta_from_zeta(2,z)
            failures.append(dict(zeta=z,error=None))
        except Exception as e:
            failures.append(dict(zeta=z,error=type(e).__name__,message=str(e)))
    rows=[]
    for m in (2,3,4,6,10):
        for theta in (-20.,-5.,-1.,0.,1.,5.,20.):
            vals=[sscd.watson_stats(m,theta,order=n) for n in (128,256,512)]
            err=max(abs(x[k]-vals[-1][k]) for x in vals[:-1] for k in ('zeta','mi_nats'))
            inv=sscd.theta_from_zeta(m,vals[1]['zeta'])
            assert err<1e-8
            assert abs(inv-theta)<1e-8
            if m==2:
                np.testing.assert_allclose(vals[1]['zeta'],i1e(theta/2)/i0e(theta/2),atol=1e-12)
            rows.append(dict(m=m,theta=theta,zeta=vals[1]['zeta'],mi=vals[1]['mi_nats'],
                             quadrature_max_error=err,inverse_error=abs(inv-theta)))
    # Show the signed-proof sign error with Gamma=Gamma_theta: true I>0,
    # while the reversed signs in the source proof would give -I.
    s=sscd.watson_stats(2,1.)
    proof=dict(theta=1.,correct_I=s['mi_nats'],source_rhs_when_D_zero=-s['mi_nats'])
    demo=np.array([[np.sqrt(5.),0.],[np.sqrt(5.),0.]])
    try:
        original.optimize_signed_zeta(2,[(demo,np.hstack([-np.eye(2),np.eye(2)]),3*np.eye(2))],
                                      lambda z:np.eye(2),np.eye(2),10.,.1,10.,.5)
    except Exception as e:
        default_failure=dict(error=type(e).__name__,message=str(e))
    else:raise AssertionError('expected source default endpoint bug was not reproduced')
    return dict(source_endpoint_failures=failures,source_default_optimizer_failure=default_failure,
                signed_proof_check=proof,compact_stats=rows)


def controlled(name,rotation):
    a=1.04;A=a*rotation;H=np.eye(2);T=[np.eye(2),A]
    G=.25*np.eye(2);F=.5*A
    Ks=[np.vstack([G,F@G]),np.vstack([np.zeros((2,2)),G])]
    return dict(name=name,m=2,A=A,H=H,F=F,G=G,T=T,Tdet=T,K=Ks,W=np.eye(4)/2,
                E=sum(t.T@t for t in T),Z=np.hstack([-A,np.eye(2)]),
                nominal=(2+a*a)*np.eye(2),horizon=2,origin='synthetic analytic gain/loss pair')


def fixtures():
    out=[controlled('aligned',np.eye(2)),controlled('rotation',np.array([[0.,-1.],[1.,0.]]))]
    reference=load_file('prior_review_checks',BASE/'paper1_ACMW_theory_audit_20261121/review_checks.py')
    common,models=reference.load_models()
    for model in models:
        A,H=model.A,model.H[0];m=H.shape[0];n=len(A);F=model.F(1);G=model.G[1]
        T=[H@np.linalg.matrix_power(A,t) for t in range(10)]
        Ks=[]
        for lag in range(10):
            K=np.zeros((32*n,m))
            for time in range(7+lag,32):
                K[time*n:(time+1)*n]=np.linalg.matrix_power(F,time-7-lag)@G
            Ks.append(K)
        _,Z,_,nominal=common.parity_model(A,H,model.Q,model.R[0],4)
        out.append(dict(name=model.name,m=m,A=A,H=H,F=F,G=G,T=T,Tdet=T[:5],K=Ks,W=np.eye(32*n)/32,
                        E=sum(t.T@t for t in T),Z=Z,nominal=nominal,horizon=32,
                        origin='existing frozen supplementary model; original 50-node inputs not supplied'))
    return out


def precompute(f):
    det=_residual_covariance_polynomials(f['Tdet'],f['Z'],f['nominal'],f['E'],10.,.5)
    risk=state_risk_polynomials(f['T'],f['K'],f['W'],.5,.5)
    ev,U=np.linalg.eigh(f['E']);Ei=(U/np.sqrt(ev))@U.T
    budget=float(10*np.linalg.eigvalsh(Ei@risk[0]@Ei)[-1])
    # Independent scalar-risk implementation from the supplied prototype.
    for z in (-.2,0.,.2):
        direct=sscd.common_g_risk_matrix(f['T'],f['K'],f['W'],z,.5,.5)
        np.testing.assert_allclose(_poly_value(risk,z),direct,rtol=2e-11,atol=2e-12)
    return det,risk,budget


def design(f,det,risk,budget,additional=()):
    return screen_continuum_2d(det,risk,f['E'],10.,.1*(len(f['Tdet'])-1),budget,len(f['Tdet']),
                               angle_points=256,zeta_tolerance=.0005,evaluation_cap=1500,
                               measurement_dimension=f['m'],additional_detector_coeff=additional,
                               directionwise_tolerance=0.)


def exact_direction(f,phi):
    ev,U=np.linalg.eigh(f['E']);Ei=(U/np.sqrt(ev))@U.T
    x=np.sqrt(10.)*Ei@np.array([np.cos(phi),np.sin(phi)])
    return x,np.stack([T@x for T in f['Tdet']])


def detection_check(f,design_result,index):
    """Calibrated single-window power at a frozen worst-proxy direction."""
    dim=f['Z'].shape[0];rng=np.random.default_rng(202611280000+index)
    calibration=rng.normal(size=(100000,dim))
    cutoff=float(np.quantile(np.sum(calibration**2,axis=1),.95))
    analytic_cutoff=float(chi2.ppf(.95,dim))
    chol=np.linalg.cholesky(f['nominal']);Zwhite=np.linalg.solve(chol,f['Z'])
    # Freeze direction before drawing heldout watermark/physical-noise paths.
    phi=design_result['best']['inner']['parts'][0]['grid_phi']
    xi,d=exact_direction(f,phi)
    schemes={'iid':0.,'negative_control':-1.,'positive_control':1.,
             'selected':design_result['best']['theta']}
    rows=[]
    for batch in range(20):
        noise=np.random.default_rng(202611290000+1000*index+batch).normal(size=(5000,dim))
        for sidx,(name,theta) in enumerate(schemes.items()):
            wm=np.random.default_rng(202611300000+10000*index+100*sidx+batch)
            beta=sscd.sample_signed_watermark_pairs(wm,5000,len(d),f['m'],theta)
            delta=.5*d[None]+beta*np.einsum('ntm,tm->nt',beta,d)[:,:,None]
            shift=delta.reshape(5000,-1)@Zwhite.T
            power=float(np.mean(np.sum((noise+shift)**2,axis=1)>cutoff))
            rows.append(dict(batch=batch,scheme=name,theta=theta,power=power,
                             false_positive=float(np.mean(np.sum(noise**2,axis=1)>cutoff))))
    summary=[]
    for name in schemes:
        p=np.array([r['power'] for r in rows if r['scheme']==name])
        iid=np.array([r['power'] for r in rows if r['scheme']=='iid'])
        summary.append(dict(scheme=name,theta=schemes[name],power=float(p.mean()),
                            power_batch_se=float(p.std(ddof=1)/np.sqrt(20)),
                            paired_change=float((p-iid).mean()),paired_change_se=float((p-iid).std(ddof=1)/np.sqrt(20))))
    return dict(scope='one internal window; independent healthy calibration; frozen worst-proxy grid direction, not worst detection power over all directions',
                calibration_n=100000,test_n=100000,threshold=cutoff,analytic_threshold=analytic_cutoff,
                test_false_positive=float(np.mean([r['false_positive'] for r in rows])),
                parameter=xi.tolist(),full_attack_coded_energy=float(xi@f['E']@xi),summary=summary,batches=rows)


def run():
    stats=stats_checks();all_fixtures=fixtures();out=[];saved=[]
    for idx,f in enumerate(all_fixtures):
        det,risk,budget=precompute(f)
        result=design(f,det,risk,budget)
        chosen=result['best'];dense=angle_minimum(det,chosen['zeta'],2048)
        assert chosen['inner']['lower']-1e-10<=dense['upper']<=chosen['inner']['upper']+1e-10
        out.append(dict(name=f['name'],origin=f['origin'],m=f['m'],design=result,
                        dense_angle_check=dense,independent_detection=detection_check(f,result,idx)))
        saved.append((f,det,risk,budget))
    f,det,risk,budget=saved[0]
    conflicting=design(f,det,risk,budget,additional=(saved[1][1],))
    assert abs(conflicting['best']['zeta'])<1e-10
    # Exercise the corrected generic finite-template API separately, without
    # pretending its finite templates cover the continuum of attack directions.
    templates=[]
    for phi in np.arange(8)*np.pi/8:
        _,d=exact_direction(f,phi)
        templates.append((d,f['Z'],f['nominal']))
    finite=sscd.optimize_signed_zeta(f['m'],templates,lambda z:_poly_value(risk,z),
                                     f['E'],10.,.1,budget,.5)
    # Explicitly check the computational domain and invalid-budget behavior.
    rejected=[]
    try:sscd.theta_from_zeta(2,.99999)
    except ValueError as e:rejected.append(dict(test='outside compact image',message=str(e)))
    else:raise AssertionError('must reject unsupported endpoint before root solving')
    try:sscd.optimize_signed_zeta(f['m'],templates,lambda z:_poly_value(risk,z),f['E'],10.,.1,budget/2,.5,grid_size=9)
    except ValueError as e:rejected.append(dict(test='infeasible IID budget',message=str(e)))
    else:raise AssertionError('must not offer an infeasible IID fallback')
    result=dict(scope='Paper1 SSCD code review; generic m finite-template core plus p=2 bounded continuum prototype; supplementary fixtures only',
                stats=stats,fixtures=out,conflicting_family=conflicting,finite_template_api=finite,
                explicit_rejections=rejected)
    (HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False,
        default=lambda o:o.item() if isinstance(o,np.generic) else str(o))+'\n',encoding='utf-8')
    print('results.json written')


if __name__=='__main__':run()
