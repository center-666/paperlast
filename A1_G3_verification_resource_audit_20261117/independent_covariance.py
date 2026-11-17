"""Primitive-noise map independent of the covariance recurrence."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
from protocol import ref
H=Path(__file__).resolve().parent
def main():
    cert=json.loads((H/'boundary_certificates.json').read_text('utf8'))['models'];rows={}
    for name in ('unstable','stable'):
        m=ref.load_model(name);dim=32*2+32*5*2
        variances=np.r_[np.ones(32*2),np.full(32*5*2,.5)]
        state=np.zeros((2,dim));state[:,:2]=np.eye(2);hat=np.zeros_like(state)
        for k in range(8):
            maps=[]
            for j in range(5):
                Y=m.H[j]@state
                off=64+(k*5+j)*2;Y[:,off:off+2]+=np.eye(2);maps.append(Y)
            prior=m.A@hat;nu=maps[0]-m.H[0]@prior
            clean=prior+m.K[1]@(maps[1]-m.H[1]@prior)+m.G[1]@(nu+maps[2]-m.H[2]@prior)
            if k==7:
                reject=prior+m.K[1]@(maps[1]-m.H[1]@prior)+m.G[1]@(maps[2]-m.H[2]@prior)
                B=reject-clean;E0=clean-state;Ea=reject-state
                covariance=(B*variances)@B.T
                from fractions import Fraction
                expected=np.array([[float(Fraction(v)) for v in r] for r in cert[name]['rejection_covariance_at7_exact']])
                cov_error=float(np.max(abs(covariance-expected)))
                cm=cert[name]['first_rejection_actual_MSE']
                healthy=float(np.sum(E0*E0*variances));attacked=float(np.sum(Ea*Ea*variances))
                err=max(cov_error,abs(healthy-cm['healthy_display']),abs(attacked-cm['rejected_display']))
                assert err<1e-12
                rows[name]=dict(covariance_max_error=cov_error,healthy_MSE=healthy,rejected_MSE=attacked,
                  aggregate_max_error=err,primitive_dimension=dim,shared_state_and_noise=True)
            hat=clean
            if k<7:
                state=m.A@state;off=2+2*k;state[:,off:off+2]+=np.eye(2)
    (H/'independent_covariance.json').write_text(json.dumps(dict(executed='2026-11-18',models=rows,
      no_random_sampling=True,independent_primitive_map=True),indent=2)+'\n','utf8')
    print(json.dumps(rows))
if __name__=='__main__':main()
