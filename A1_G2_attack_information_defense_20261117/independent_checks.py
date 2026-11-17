from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
from scipy.linalg import null_space,block_diag
import attack_comparison as ac
from gain_models import ref
from risk_bounds import residual_info
H=Path(__file__).resolve().parent;P=H.parent
sys.path.insert(0,str(P/'A1_information_aware_attack_audit_20261116'))
from attacker_observer_bounds import primitive_maps
def main():
    designs=json.loads((H/'gain_designs.json').read_text('utf8'))['models']
    errors=dict(trajectory=0.,covariance=0.,constraint=0.,local_optimality=0.,detector_info=0.,causality=0.)
    updates=0;energymax=0.;overlaps={}
    for name,ds in designs.items():
        m=ref.load_model(name);bank=ac.am.old_sample_bank(m,[16,33,63]);det=ac.original_detectors(m,bank)
        X,Y,_,_=primitive_maps(m)
        for lab,d in det.items():
            D,_=residual_info(m,lab);T=d['target'][:,-2:]
            errors['detector_info']=max(errors['detector_info'],float(np.max(np.abs(T.T@T-np.array(D,float)))))
        vs=[]
        for end in (8,9):
            coeff=np.concatenate([Y[t,j] for t in range(end-4,end+1) for j in (1,0,2)],axis=0)
            vs.append(det['joint_GLRT']['V']@coeff)
        overlaps[name]=float(np.linalg.norm(vs[0]@vs[1].T));assert overlaps[name]>1e-6
        for label,g in ds.items():
            K=np.array(g['K']);G=np.array(g['G']);F=np.array(g['F'])
            pred=np.zeros((2,Y.shape[-1]));error=[]
            for k in range(32):
                pred=m.A@pred
                est=pred+K[k]@(Y[k,1]-m.H[1]@pred)+G[k]@(Y[k,0]+Y[k,2]-(m.H[0]+m.H[2])@pred)
                error.append(est-X[k]);pred=est
            error=np.array(error)
            for k in range(32):
                errors['covariance']=max(errors['covariance'],float(np.max(np.abs(error[k]@Y[k,0].T-np.array(g['target_cov'][k])))))
            ps,_=ac.policies(m,g)
            for p in (ps[0],next(p for p in ps if p['kind']=='perp')):
                a=ac.actions(bank['q'],p);energymax=max(energymax,float(np.sum(a*a,axis=(1,2)).max()))
                q=bank['q'].copy();q[:,9:]+=100
                errors['causality']=max(errors['causality'],float(np.max(np.abs(ac.actions(q,p)[:,:9]-a[:,:9]))))
                delta=np.einsum('ntij,ntj->nti',bank['M'],a)
                for i in range(3):
                    h=np.zeros(2);z=np.zeros(2);b=np.zeros(2)
                    for k in range(32):
                        _,h=ref.common_step(h,bank['y'][i,k,1],[bank['y'][i,k,0],bank['y'][i,k,2]],
                          m.A,m.H[1],[m.H[0],m.H[2]],K[k],G[k])
                        _,z=ref.common_step(z,bank['y'][i,k,1],[bank['y'][i,k,0]+delta[i,k],bank['y'][i,k,2]],
                          m.A,m.H[1],[m.H[0],m.H[2]],K[k],G[k])
                        b=F[k]@b+G[k]@delta[i,k]
                        errors['trajectory']=max(errors['trajectory'],float(np.max(np.abs(z-h-b))))
                        updates+=1
            if label=='raw_orthogonal':
                Hc=np.vstack([m.H[1],m.H[0]+m.H[2]]);R=block_diag(m.R[1],m.R[0]+m.R[2])
                for k in (0,7,31):
                    V=np.eye(2) if k==0 else m.A@np.array(g['P'][k-1])@m.A.T+m.Q
                    D=-np.eye(2) if k==0 else m.A@np.array(g['C'][k-1])@m.A.T-m.Q
                    S=Hc@V@Hc.T+R;B=-Hc@D+np.vstack([np.zeros((2,2)),m.R[0]])
                    L=np.hstack([K[k],G[k]]);Z=np.array([[1.,2.],[3.,1.]])@null_space(B.T).T/100
                    errors['constraint']=max(errors['constraint'],float(np.linalg.norm(D+L@B)))
                    def cov(W):J=np.eye(2)-W@Hc;return J@V@J.T+W@R@W.T
                    errors['local_optimality']=max(errors['local_optimality'],float(np.max(np.abs(cov(L+Z)-cov(L)-Z@S@Z.T))))
    assert max(errors.values())<1e-8 and energymax<=10
    out=dict(date='2026-11-17',errors=errors,independent_target_updates=updates,
      maximum_executed_energy=energymax,joint_overlap_covariance_norms=overlaps,
      no_independent_window_assumption=True,attack_API_inputs='coded q plus public plan only',
      no_new_physical_samples=True,original_reference_unchanged=True)
    (H/'independent_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps(out))
if __name__=='__main__':main()
