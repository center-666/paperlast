"""Independent audit of the fixed reference, using different algebraic routes.

No new physical realization, old simulation import, or detector development.
The H0=I difference-residual covariance is derived without parity_model's W.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import sympy as sp
from common_g_reference import load_model,parity_model,legacy_stacked_step

HERE=Path(__file__).resolve().parent
def rational(x):return sp.Matrix([[sp.Rational(str(z)) for z in row] for row in x])
def array(x):return np.array(x.tolist(),dtype=float)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    model_data=json.loads((HERE/'inputs/five_node_aggregate_model.json').read_text('utf8'))
    saved=np.load(HERE/'test_results.npz');primary=json.loads((HERE/'test_results.json').read_text('utf8'))
    rows=[]
    for name in ('unstable','stable'):
        model=load_model(name);A=rational(model_data['models'][name]['A'])
        Q=rational(model_data['Q']);R=rational(model_data['R'][0]);n=2;s=4
        assert rational(model_data['H'][0])==sp.eye(2)
        # e_r=y_(r+1)-A y_r=w_r+v_(r+1)-A v_r, r=0..s-1.
        Zdiff=sp.zeros(s*n,(s+1)*n)
        Vdiff=sp.zeros(s*n,s*n)
        for r in range(s):
            Zdiff[r*n:(r+1)*n,r*n:(r+1)*n]=-A
            Zdiff[r*n:(r+1)*n,(r+1)*n:(r+2)*n]=sp.eye(n)
            Vdiff[r*n:(r+1)*n,r*n:(r+1)*n]=Q+R+A*R*A.T
            if r<s-1:
                Vdiff[r*n:(r+1)*n,(r+1)*n:(r+2)*n]=-R*A.T
                Vdiff[(r+1)*n:(r+2)*n,r*n:(r+1)*n]=-A*R
        O=sp.Matrix.vstack(*(A**r for r in range(s+1)))
        assert Zdiff*O==sp.zeros(s*n,n)
        assert Zdiff.rank()==s*n
        _,Zsvd,_,theta=parity_model(model.A,model.H[0],model.Q,model.R[0],s)
        Zd=array(Zdiff);Td=Zsvd@Zd.T@np.linalg.inv(Zd@Zd.T)
        basis_error=float(np.max(np.abs(Zsvd-Td@Zd)))
        cov_error=float(np.max(np.abs(theta-Td@array(Vdiff)@Td.T)))
        assert basis_error<1e-12 and cov_error<1e-11
        # Check raw vs compressed information loss by exact rational arithmetic.
        Hi=rational(model_data['H'][1]);H0=rational(model_data['H'][0]);H2=rational(model_data['H'][2]);Ri=rational(model_data['R'][1]);R0=rational(model_data['R'][0]);R2=rational(model_data['R'][2])
        raw=Hi.T*Ri.inv()*Hi+H0.T*R0.inv()*H0+H2.T*R2.inv()*H2
        compressed=Hi.T*Ri.inv()*Hi+(H0+H2).T*(R0+R2).inv()*(H0+H2)
        loss=raw-compressed
        assert loss==sp.diag(sp.Rational(1,25),sp.Rational(4,25))
        Pprior=np.array(model_data['models'][name]['nodes'][1]['prior_covariance'])
        Pprevious=np.array(model_data['models'][name]['nodes'][1]['posterior_covariance'])
        previous=np.array([.3,-.2]);ys=[np.array([.4,-.1]),np.array([.7,.2]),np.array([-.2,.5])]
        Praw=np.linalg.inv(np.linalg.inv(Pprior)+array(raw))
        b=np.linalg.solve(Pprior,model.A@previous)
        for H,Ri,y in zip([model.H[1],model.H[0],model.H[2]],[model.R[1],model.R[0],model.R[2]],ys):
            b+=H.T@np.linalg.solve(Ri,y)
        bayes=Praw@b
        _,legacy,_,_=legacy_stacked_step(model,1,previous,Pprevious,ys[0],ys[1:])
        legacy_error=float(np.max(np.abs(legacy-bayes)))
        assert legacy_error<1e-12
        # Recompute all saved nominal/attacked outputs via expanded affine form,
        # never common_step, run_common, Model.J or bias_recursion.
        y=saved[name+'__physical__y'];attack=saved[name+'__attack'];T=len(y)
        result_errors={}
        for scheme in ('identity','Gaussian'):
            M=saved[name+'__'+scheme+'__M']
            for attacked in (False,True):
                estimates=np.zeros((5,2));outputs=[]
                for k in range(T):
                    next_est=[]
                    for i in range(5):
                        hs=sum((model.H[j] for j in model.neighbors[i]),np.zeros((2,2)))
                        Fi=(np.eye(2)-model.K[i]@model.H[i]-model.G[i]@hs)@model.A
                        yh=np.zeros(2)
                        for j in model.neighbors[i]:
                            packet=y[k,j].copy()
                            if attacked and (i,j)==(1,0):packet+=M[k,j]@attack[k]
                            yh+=packet
                        next_est.append(Fi@estimates[i]+model.K[i]@y[k,i]+model.G[i]@yh)
                    estimates=np.stack(next_est);outputs.append(estimates.copy())
                key='posterior_attacked' if attacked else 'posterior_clean'
                err=float(np.max(np.abs(np.stack(outputs)-saved[name+'__'+scheme+'__'+key])))
                assert err<1e-10
                result_errors[scheme+'__'+key]=err
        rows.append({'model':name,'difference_parity_nullspace_exact':True,'difference_covariance_rational_entries':[[str(z) for z in row] for row in Vdiff.tolist()],
            'SVD_vs_difference_basis_error':basis_error,'independent_covariance_error':cov_error,
            'raw_vs_compressed_information_loss_exact':[['1/25','0'],['0','4/25']],
            'legacy_gain_update_vs_Bayesian_precision_form_error':legacy_error,
            'saved_trajectory_vs_expanded_affine_update_errors':result_errors})
    out={'date':'2026-11-15','status':'fixed-model algebra and indexing passed; not external peer review',
        'input_hashes':{n:sha(HERE/n) for n in ('common_g_reference.py','test_reference.py','inputs/five_node_aggregate_model.json','test_results.npz')},
        'primary_tests_at_review':primary['passed'],'models':rows,'new_physical_samples':0,'MC':0,
        'limits':['Fixed common-G gains, no gating or loss; not original 50-node reproduction','No empirical FA/Pd or novelty claims','This review uses a different residual basis and expanded affine updates, not merely the primary test runner']}
    (HERE/'code_review_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps({'models':len(rows),'covariance_check_errors':[x['independent_covariance_error'] for x in rows],
        'max_affine_replay_error':max(max(x['saved_trajectory_vs_expanded_affine_update_errors'].values()) for x in rows),'new_samples':0},ensure_ascii=False))
if __name__=='__main__':main()
