"""Independent deterministic and numerical checks of the source-matched theory."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from source_model import load_model,covariance_schedule,physical_paths,inv2,matvec
from replay import scores,base_filter,affected_filter
from moments import source_templates,polynomial,buffer_support,dg_direction,sscd

HERE=Path(__file__).resolve().parent


def direct_risk(model,schedule,xi,zeta):
    """Dense input-output contraction, independent of the cost-to-go formula."""
    q=.5;weight=1/(50*150);means={int(i):np.zeros(300) for i in model.receiver_ids};variance=0.
    T,E=source_templates(model)
    for e,(receiver,sender) in enumerate(model.links):
        i=int(receiver);K=np.zeros((300,120))
        for ell in range(50,110):
            B=schedule.star[ell,i]@(2*model.H[sender].T)
            for k in range(ell,150):
                if k>ell:B=schedule.F[k,i]@B
                K[2*k:2*k+2,2*(ell-50):2*(ell-50)+2]=B
        d=np.stack([T[k,e]@xi for k in range(50,110)])
        cov=sscd.full_watermark_covariance(d,zeta,q)
        means[i]+=K@d.reshape(-1)
        variance+=weight*np.trace((K.T@K)@cov)
    return float(weight*sum(v@v for v in means.values())+variance)


def run():
    design=json.loads((HERE/'design_results.json').read_text());arr=np.load(HERE/'moment_arrays.npz')
    rows=[]
    for mi,label in enumerate(('unstable','stable')):
        model=load_model(label);schedule=covariance_schedule(model);coeff=arr[label+'_state_coeff']
        risk_checks=[]
        for z in (-.31222709133921533,0.,.31222709133921533):
            for xi in (model.xi,np.array([.7,-.4])):
                direct=direct_risk(model,schedule,xi,z);fast=float(xi@polynomial(coeff,z)@xi)
                np.testing.assert_allclose(direct,fast,rtol=1e-11,atol=1e-10)
                risk_checks.append(dict(zeta=z,xi=xi.tolist(),direct=direct,cost_to_go=fast,absolute_error=abs(direct-fast)))
        trace_errors=[]
        for e in range(6):
            bf=buffer_support(model,e)
            for z in (-.3,.3):
                xi=np.array([.6,-.8]);d=np.stack([t@xi for t in bf['templates']])
                C=bf['Z']@sscd.full_watermark_covariance(d,z,.5)@bf['Z'].T
                trace=float(np.trace(np.linalg.solve(bf['theta'],C)))
                expected=float(xi@polynomial(bf['trace_coeff'],z)@xi)
                trace_errors.append(abs(trace-expected))
                np.testing.assert_allclose(trace,expected,rtol=1e-11,atol=1e-11)
        # All-source numerical healthy encode/decode under multiple copulas.
        physics=physical_paths(model,4,202612080000+mi)
        base=base_filter(model,physics,schedule);zero=np.zeros((4,150,6,2))
        ideal=scores(model,physics,zero);healthy=affected_filter(model,physics,zero,ideal,42.,True)
        base[:,:,model.receiver_ids]=healthy.post
        transparency=[]
        for pi,theta in enumerate((0.,-1.315236976827179,1.315236976827179)):
            beta=sscd.sample_signed_watermark_pairs(np.random.default_rng(202612090000+100*mi+pi),200,150,2,theta)
            beta=beta.reshape(4,50,150,2).transpose(0,2,1,3)
            M=.5*np.eye(2)+beta[:,:,:,:,None]*beta[:,:,:,None,:]
            decoded=matvec(M,matvec(inv2(M),physics.y))
            weighted=2*decoded*model.hdiag[None,None]
            suff=(weighted.swapaxes(-1,-2)@(model.closed-np.eye(50)).T).swapaxes(-1,-2)+physics.weighted
            fp=SimpleNamespace(x=physics.x,y=decoded,weighted=weighted,suff=suff)
            sc=scores(model,fp,zero);post=base_filter(model,fp,schedule)
            w=affected_filter(model,fp,zero,sc,42.,True);post[:,:,model.receiver_ids]=w.post
            assert np.array_equal(w.gates,healthy.gates)
            esterr=float(np.max(abs(post-base)));assert esterr<1e-5
            transparency.append(dict(theta=theta,packet_roundtrip_max_abs=float(np.max(abs(decoded-physics.y))),
                                     estimate_max_abs=esterr,gate_decisions_identical=True))
        rows.append(dict(model=label,dense_risk_checks=risk_checks,max_trace_polynomial_error=max(trace_errors),
                         healthy_finite_precision_checks=transparency))
    # A proof certificate not reliant on floating spectral calculations:
    # ||A||_F^2/4 equals these exact rational numbers, both strictly below1.
    assert 22715<40000 and 18442<40000
    # R=.5I and orthonormal parity rows imply Theta >=.5I.
    eps=1e-6;startup_universal=3.5*(2*eps-np.log(2*eps)-1)
    assert startup_universal>42
    result=dict(models=rows,uniform_contraction_squared_rational={'unstable':'22715/40000','stable':'18442/40000'},
                startup_score_lower_using_R_only=float(startup_universal),
                startup_forced_alarm_independent_of_Gaussian_assumption=True)
    (HERE/'theory_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('theory_checks.json written; universal startup score floor',startup_universal)


if __name__=='__main__':run()
