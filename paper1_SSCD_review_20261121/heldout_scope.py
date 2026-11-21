"""Independent direction-profile and start-time checks of frozen SSCD policies."""
from pathlib import Path
import json
import numpy as np
import sscd
from continuous_design import state_risk_polynomials,_poly_value
from run_checks import fixtures,exact_direction

HERE=Path(__file__).resolve().parent


def run():
    result=json.loads((HERE/'results.json').read_text())
    out=[]
    for idx,(f,r) in enumerate(zip(fixtures(),result['fixtures'])):
        nominal=f['nominal'];chol=np.linalg.cholesky(nominal);Zwhite=np.linalg.solve(chol,f['Z'])
        threshold=r['independent_detection']['threshold'];theta=r['design']['best']['theta']
        # Place every angle halfway between points of the actual 256-point
        # optimization grid. A half-shift of a 32-point grid would alias onto it.
        angles=(8*np.arange(32)+.5)*np.pi/256
        assert np.all(abs(angles/(np.pi/256)-np.rint(angles/(np.pi/256)))>.49)
        d=np.stack([exact_direction(f,phi)[1] for phi in angles])
        noise=np.random.default_rng(202612010000+idx).normal(size=(3000,len(chol)))
        profiles={}
        for scheme,th in [('iid',0.),('selected',theta)]:
            rng=np.random.default_rng(202612020000+idx*100+(scheme=='selected'))
            beta=sscd.sample_signed_watermark_pairs(rng,3000,len(d[0]),f['m'],th)
            dot=np.einsum('ntm,atm->ant',beta,d)
            delta=.5*d[:,None]+beta[None]*dot[:,:,:,None]
            shifts=delta.reshape(32,3000,-1)@Zwhite.T
            score=np.sum((shifts+noise[None])**2,axis=2)
            counts=np.sum(score>threshold,axis=1)
            profiles[scheme]=dict(successes=counts.tolist(),power=(counts/3000).tolist(),
                                   min_empirical_power=float(counts.min()/3000))
        onset_rows=[]
        phase_metrics={}
        time_metrics=[]
        if f['horizon']==32:
            ev,U=np.linalg.eigh(f['E']);Ei=(U/np.sqrt(ev))@U.T
            for onset in range(23):
                K=[];G=f['G'];F=f['F']
                for lag in range(10):
                    k=np.zeros_like(f['K'][0])
                    for time in range(onset+lag,32):
                        k[time*2:(time+1)*2]=np.linalg.matrix_power(F,time-onset-lag)@G
                    K.append(k)
                coeff=state_risk_polynomials(f['T'],K,f['W'])
                q0=coeff[0];q1=_poly_value(coeff,r['design']['best']['zeta'])
                onset_rows.append(dict(onset=onset,
                     iid_worst=float(10*np.linalg.eigvalsh(Ei@q0@Ei)[-1]),
                     selected_worst=float(10*np.linalg.eigvalsh(Ei@q1@Ei)[-1]),
                     directionwise_change=float(10*np.linalg.eigvalsh(Ei@(q1-q0)@Ei)[-1])))
            z=r['design']['best']['zeta']
            for label,start,stop in [('attack',7,17),('after_support',17,32)]:
                weights=np.zeros(64);weights[2*start:2*stop]=1/(stop-start)
                coeff=state_risk_polynomials(f['T'],f['K'],np.diag(weights))
                q0=coeff[0];q1=_poly_value(coeff,z)
                phase_metrics[label]=dict(iid_worst=float(10*np.linalg.eigvalsh(Ei@q0@Ei)[-1]),
                                         selected_worst=float(10*np.linalg.eigvalsh(Ei@q1@Ei)[-1]))
            for time in range(32):
                K=[k[2*time:2*(time+1)] for k in f['K']]
                coeff=state_risk_polynomials(f['T'],K,np.eye(2))
                q0=coeff[0];q1=_poly_value(coeff,z)
                time_metrics.append(dict(time=time,iid_worst=float(10*np.linalg.eigvalsh(Ei@q0@Ei)[-1]),
                     selected_worst=float(10*np.linalg.eigvalsh(Ei@q1@Ei)[-1]),
                     directional_change=float(10*np.linalg.eigvalsh(Ei@(q1-q0)@Ei)[-1])))
        out.append(dict(name=f['name'],frozen_theta=theta,direction_angles=angles.tolist(),profiles=profiles,
                        samples_per_direction=3000,onset_state_checks=onset_rows,
                        phase_state_checks=phase_metrics,pointwise_state_checks=time_metrics))
    obj=dict(scope='Finite heldout direction profile and exact-moment onset sensitivity; does not certify worst true detection probability.',fixtures=out)
    (HERE/'heldout_results.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    print('heldout_results.json written')


if __name__=='__main__':run()
