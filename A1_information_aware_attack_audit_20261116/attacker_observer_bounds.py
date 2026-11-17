"""Analytic second moments and observer uncertainty; no hidden-state oracle."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
import attack_models as am
H=Path(__file__).resolve().parent
def primitive_maps(m,T=32):
    dim=2+2*(T-1)+3*2*T
    X=np.zeros((T,2,dim));Y=np.zeros((T,3,2,dim));X[0,:,:2]=np.eye(2)
    for t in range(T):
        if t:
            X[t]=m.A@X[t-1];X[t,:,2+2*(t-1):4+2*(t-1)]=np.linalg.cholesky(m.Q)
        for j in range(3):
            Y[t,j]=m.H[j]@X[t]
            off=2*T+(t*3+j)*2;Y[t,j,:,off:off+2]+=np.linalg.cholesky(m.R[j])
    post=np.zeros((2,dim));prior_errors=[];post_errors=[]
    for t in range(T):
        prior=m.A@post;prior_errors.append(prior-X[t])
        post=m.J(1)@prior+m.K[1]@Y[t,1]+m.G[1]@(Y[t,0]+Y[t,2])
        post_errors.append(post-X[t])
    return X,Y,np.array(prior_errors),np.array(post_errors)
def observer(m,k,current=False):
    X,Y,E,Ep=primitive_maps(m);ids=range(k+int(current));mom=am.inverse_moments()
    a=mom['mean'];cX=mom['coef_X'];ct=mom['coef_trace'];N=len(ids)
    Qcov=np.zeros((2*N,2*N));clean=np.zeros_like(Qcov);cross=np.zeros((2,2*N))
    for i in ids:
        cross[:,2*i:2*i+2]=a*E[k]@Y[i,0].T
        for j in ids:
            R=Y[i,0]@Y[j,0].T;clean[2*i:2*i+2,2*j:2*j+2]=R
            Qcov[2*i:2*i+2,2*j:2*j+2]=a*a*R if i!=j else cX*R+ct*np.trace(R)*np.eye(2)
    Pee=E[k]@E[k].T;L=np.linalg.solve(Qcov,cross.T).T;P=Pee-L@cross.T
    cross_clean=cross/a;genie=Pee-cross_clean@np.linalg.solve(clean,cross_clean.T)
    U=m.K[1]@m.R[1]@m.K[1].T+m.G[1]@m.R[2]@m.G[1].T
    independent=np.zeros((2,2))
    for r in range(k):independent+=np.linalg.matrix_power(m.F(1),r)@U@np.linalg.matrix_power(m.F(1).T,r)
    independent=m.A@independent@m.A.T
    if not current:independent+=m.Q
    assert np.linalg.eigvalsh(P-genie).min()>-1e-9
    assert np.linalg.eigvalsh(genie-independent).min()>-1e-9
    return dict(k=k,information='q0..k inclusive' if current else 'q0..k-1',
      prior_error_covariance=Pee.tolist(),linear_attacker_error_covariance=P.tolist(),
      clean_packet_genie_error_covariance=genie.tolist(),unseen_noise_lower_bound=independent.tolist(),
      linear_estimator_matrix=L.tolist(),trace_prior=float(np.trace(Pee)),
      trace_linear=float(np.trace(P)),trace_genie=float(np.trace(genie)),
      trace_unseen_lower_bound=float(np.trace(independent)),
      Bayesian_MMSE_exact=False,applicability='clean pre-attack prediction at k=Ta; no attacked Gaussian recursive observer is claimed')
def main():
    results={}
    for name in ('unstable','stable'):
        m=am.ref.load_model(name)
        results[name]={lab:observer(m,7,current) for lab,current in [('A1-H',False),('A1-C',True)]}
    out=dict(date='2026-11-17',models=results,inverse_watermark_moments=am.inverse_moments(),
      new_physical_paths=0,secret_is_not_given_to_attacker=True,
      uncertainty_after_attack='unseen local/healthy-neighbor measurement-noise lower bound remains; exact Bayesian encrypted observer after feedback not solved')
    (H/'observer_bounds.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps({n:{lab:{k:r[k] for k in ('trace_prior','trace_linear','trace_genie','trace_unseen_lower_bound')} for lab,r in z.items()} for n,z in results.items()}))
if __name__=='__main__':main()
