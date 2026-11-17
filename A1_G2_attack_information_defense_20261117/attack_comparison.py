"""Small OLD-input comparison; no new physical noise and no expanded angle grid."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_laguerre
H=Path(__file__).resolve().parent;P=H.parent
sys.path.insert(0,str(P/'A1_information_aware_attack_audit_20261116'))
import attack_models as am
from screen_attack_classes import detectors as original_detectors
CFG=json.loads((H/'registration.json').read_text('utf8'))
def phi(F,r,t):
    X=np.eye(2)
    for k in range(t+1,r+1):X=F[k]@X
    return X
def maps(g):
    F=np.array(g['F']);G=np.array(g['G'])
    return [np.vstack([np.zeros((2,2)) if r<t else phi(F,r,t)@G[t] for r in range(7,32)]) for t in range(7,17)]
def expected_gram(g):
    Ms=maps(g);R=np.block([[a.T@b for b in Ms] for a in Ms])
    for i,a in enumerate(Ms):
        C=a.T@a;R[2*i:2*i+2,2*i:2*i+2]=1.25*C+.25*np.trace(C)*np.eye(2)
    return R,Ms
def target_covariance(g,k,t):
    return phi(np.array(g['F']),k,t)@np.array(g['target_cov'])[t]
def policies(m,g):
    R,B=expected_gram(g);ev,V=eigh(R);v=V[:,-1];v*=1 if v[0]>=0 else -1
    ell={}
    for obs in (6,7):
        row=np.zeros(2)
        for j,t in enumerate(range(7,17)):
            for r in range(t,32):
                row+=v[2*j:2*j+2]@np.array(g['G'])[t].T@phi(np.array(g['F']),r,t).T@target_covariance(g,r,obs)
        ell[obs]=row
    plan=np.sqrt(10)*(1-1e-12)*v.reshape(10,2)
    out=[dict(id='OL_spectral'+sgn,family='OL',kind='plan',plan=(sign*plan).tolist()) for sgn,sign in [('plus',1),('minus',-1)]]
    for family,obs in [('H',6),('C',7)]:
        for sign in (1,-1):out.append(dict(id=family+'_signed'+str(sign),family=family,kind='signed',plan=plan.tolist(),obs=obs,ell=(sign*ell[obs]).tolist()))
    C=np.array(g['G'])[7].T@np.array(g['G'])[7];Q=1.25*C+.25*np.trace(C)*np.eye(2)
    _,U=eigh(Q);packet=np.sqrt(10)*(1-1e-12)*U[:,-1]
    for sign in (1,-1):out.append(dict(id='OL_pulse'+str(sign),family='OL',kind='pulse',packet=(sign*packet).tolist()))
    Hs=[m.H[0]@np.linalg.matrix_power(m.A,t) for t in range(10)]
    W=sum(a.T@a for a in Hs);S=[]
    for i in range(25):
        C0=sum(B[t][2*i:2*i+2]@Hs[t] for t in range(10))
        S0=C0.T@C0
        for t in range(10):
            C=B[t][2*i:2*i+2].T@B[t][2*i:2*i+2]
            S0+=.25*Hs[t].T@(C+np.trace(C)*np.eye(2))@Hs[t]
        S.append(S0)
    a0cum=eigh(sum(S),W)[1][:,-1]*np.sqrt(10)*(1-1e-12)
    peak=max([(eigh(S0,W)[0][-1],i) for i,S0 in enumerate(S)])
    a0peak=eigh(S[peak[1]],W)[1][:,-1]*np.sqrt(10)*(1-1e-12)
    for name,xi in [('cum',a0cum),('peak',a0peak)]:
        for sign in (1,-1):out.append(dict(id='A0_'+name+str(sign),family='A0',kind='plan',plan=[(sign*h@xi).tolist() for h in Hs]))
    for j,a in enumerate(np.linspace(0,2*np.pi,12,endpoint=False)):
        u=np.array([np.cos(a),np.sin(a)]);scale=np.sqrt(.1/max(u@s@u for s in S));xi=u*scale
        if xi@W@xi<=10*(1-1e-12):out.append(dict(id='A0_P3ray'+str(j),family='A0',kind='plan',plan=[(h@xi).tolist() for h in Hs]))
    out += [dict(id='C_perp_current_pulse',family='C',kind='perp'),
            dict(id='C_feasible_reflection',family='C',kind='reflect')]
    return out,dict(OL_global_expected_increment=float(10*ev[-1]),
      A0_cumulative_expected_increment=float(10*eigh(sum(S),W)[0][-1]),
      OL_unit_waveform=v.tolist(),sign_ell={str(k):x.tolist() for k,x in ell.items()},
      waveform_parameters_recomputed_for_gain=True)
def actions(q,p):
    a=np.zeros_like(q);n=len(a)
    if p['kind']=='plan':a[:,7:17]=np.array(p['plan'])
    elif p['kind']=='signed':
        sg=np.where(q[:,p['obs']]@np.array(p['ell'])>=0,1.,-1.)
        a[:,7:17]=sg[:,None,None]*np.array(p['plan'])
    elif p['kind']=='pulse':a[:,7]=np.array(p['packet'])
    elif p['kind']=='perp':
        z=q[:,7][:,::-1]*np.array([-1.,1.]);norm=np.linalg.norm(z,axis=1)
        a[:,7]=np.sqrt(10)*(1-1e-12)*z/np.maximum(norm[:,None],1e-300)
    elif p['kind']=='reflect':
        p0=dict(xi=[0.,0.],theta=-2.,family='A1-C',kind='feasible_reflection')
        a=am.causal_policy(q,np.eye(2),np.eye(2),p0)[0]
    else:raise ValueError(p)
    assert np.einsum('ntm,ntm->n',a,a).max()<=10
    return a
def healthy_error(m,bank,g):
    K=np.array(g['K']);G=np.array(g['G']);F=np.array(g['F'])
    x=np.zeros((len(bank['x']),2));out=[]
    for k in range(32):
        y=bank['y'][:,k];x=x@F[k].T+y[:,1]@K[k].T+(y[:,0]+y[:,2])@G[k].T
        out.append(x-bank['x'][:,k])
    return np.stack(out,axis=1)
def evaluate(m,bank,g,p,dets):
    a=actions(bank['q'],p);delta=np.einsum('ntij,ntj->nti',bank['M'],a)
    e=healthy_error(m,bank,g);F=np.array(g['F']);G=np.array(g['G']);b=np.zeros_like(a);prev=np.zeros((len(a),2))
    for k in range(32):prev=prev@F[k].T+delta[:,k]@G[k].T;b[:,k]=prev
    bias=np.sum(b*b,axis=2);cross=2*np.sum(e*b,axis=2);actual=np.sum((e+b)**2,axis=2)
    bad=bias>=.1;tD=np.where(bad.any(1),bad.argmax(1),1000)
    windows=np.stack([delta[:,k-4:k+1].reshape(len(a),-1) for k in range(4,32)],1)
    metrics=dict(cumulative_MSE=float(actual[:,7:].sum(1).mean()),
      cumulative_increment=float((bias+cross)[:,7:].sum(1).mean()),bias_component=float(bias[:,7:].sum(1).mean()),
      cross_component=float(cross[:,7:].sum(1).mean()),danger_rate=float((tD<1000).mean()),
      energy_max=float(np.sum(a*a,axis=(1,2)).max()),detectors={})
    for lab,d in dets.items():
        r=d['clean']+np.einsum('dw,ntw->ntd',d['target'],windows)
        alarm=np.sum(r*r,axis=2)>float(CFG['conservative_thresholds'][lab])
        tA=np.where(alarm.any(1),np.arange(4,32)[alarm.argmax(1)],1000)
        metrics['detectors'][lab]=dict(event_alarm_rate=float(alarm.any(1).mean()),
          risk=float((tD<tA).mean()),risk_count=int((tD<tA).sum()),sample_size=len(a),
          pre_alarm_count=int((tA<7).sum()))
    return metrics
def perp_quadratic(m,g):
    R,B=expected_gram(g);C=B[0].T@B[0]
    Px=np.eye(2)
    for k in range(7):Px=m.A@Px@m.A.T+m.Q
    Py=Px+m.R[0];r,w=roots_laguerre(64);theta=(np.arange(128)+.5)*np.pi/128
    u=np.stack([np.cos(theta),np.sin(theta)],1);outer=np.einsum('ni,nj->nij',u,u)
    M=.5*np.eye(2)+r[:,None,None,None]*outer[None]
    S=2*np.eye(2)-2*r[:,None,None,None]/(.5+r[:,None,None,None])*outer[None]
    V=np.einsum('rnij,jk,rnkl->rnil',S,Py,S);sd=np.sqrt(np.linalg.det(V));tr=np.trace(V,axis1=-2,axis2=-1)
    direction=(V+sd[:,:,None,None]*np.eye(2))/(tr+2*sd)[:,:,None,None]
    J=np.array([[0.,-1.],[1.,0.]]);adir=np.einsum('ij,rnjk,lk->rnil',J,direction,J)
    C_M=np.einsum('rnij,jk,rnkl->rnil',M,C,M)
    return float(10*np.sum(w[:,None]*np.einsum('rnij,rnji->rn',C_M,adir))/128)
def main():
    designs=json.loads((H/'gain_designs.json').read_text('utf8'))['models'];results={}
    selections={};alltraining={}
    # Both model selections are fixed before any validation metrics are read.
    for name,ds in designs.items():
        m=am.ref.load_model(name);bank=am.old_sample_bank(m,range(16));dets=original_detectors(m,bank);rr={};ss={}
        for label,g in ds.items():
            ps,analytic=policies(m,g);rows={p['id']:dict(policy=p,metrics=evaluate(m,bank,g,p,dets)) for p in ps}
            pick={}
            for family in ('A0','OL','H','C'):
                allowed={'A0'} if family=='A0' else ({'A0','OL'} if family=='OL' else ({'A0','OL','H'} if family=='H' else {'A0','OL','H','C'}))
                pool=[v for v in rows.values() if v['policy']['family'] in allowed]
                pick[family+'/MSE']=max(pool,key=lambda r:r['metrics']['cumulative_MSE'])['policy']['id']
                for d in ('temporal','joint_GLRT'):
                    pick[family+'/'+d]=max(pool,key=lambda r:r['metrics']['detectors'][d]['risk'])['policy']['id']
            rr[label]=dict(rows=rows,analytic=analytic);ss[label]=pick
        alltraining[name]=rr;selections[name]=ss
    rec=dict(date='2026-11-17',selection=selections,training_indices=list(range(16)),validation_not_read=True)
    fp=H/'selection_before_validation.json'
    if fp.exists():assert json.loads(fp.read_text('utf8'))==rec
    else:fp.write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n','utf8')
    for name,ds in designs.items():
        m=am.ref.load_model(name);bank=am.old_sample_bank(m,range(16,64));dets=original_detectors(m,bank);out={}
        for label,g in ds.items():
            ps,analytic=policies(m,g);pool={p['id']:p for p in ps}
            wanted=set(selections[name][label].values())|{'C_perp_current_pulse','H_signed1','C_signed1','OL_spectralplus'}
            out[label]=dict(analytic=analytic,perpendicular_pulse_expected_bias_increment=perp_quadratic(m,g),
              selected=selections[name][label],validation={p:dict(policy=pool[p],metrics=evaluate(m,bank,g,pool[p],dets)) for p in sorted(wanted)})
        results[name]=out
        print(json.dumps({name:{k:dict(OL=r['analytic']['OL_global_expected_increment'],perp_bias=r['perpendicular_pulse_expected_bias_increment'],
            C_MSE=r['validation'][r['selected']['C/MSE']]['metrics']['cumulative_MSE'],
            C_risk=r['validation'][r['selected']['C/joint_GLRT']]['metrics']['detectors']['joint_GLRT']['risk']) for k,r in out.items()}}),flush=True)
    (H/'attack_comparison.json').write_text(json.dumps(dict(date='2026-11-17',models=results,
      new_physical_samples=0,finite_family_not_class_optimum=True,independent_new_holdout=False),ensure_ascii=False,indent=2)+'\n','utf8')
    (H/'training_metrics.json').write_text(json.dumps(alltraining,ensure_ascii=False,indent=2)+'\n','utf8')
if __name__=='__main__':main()
