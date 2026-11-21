"""Independent moment, geometry and fixed common-G checks for Paper 1 revision.

No online attacker or physical system interface. All inputs are synthetic or
the existing supplementary frozen models. Uses NumPy/SciPy; inputs untouched.
"""
from pathlib import Path
import json
import importlib.util
import sys
import hashlib
import numpy as np
from scipy.linalg import block_diag, null_space, solve_discrete_are
from scipy.special import hyp1f1, roots_jacobi, roots_legendre, beta as beta_fn
from scipy.optimize import brentq
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
QW, LAM = 0.5, 0.5


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def constants(m, kappa, rho=0.7, order=128):
    z = hyp1f1(0.5, m/2, -kappa)
    tilted_mean = hyp1f1(1.5, m/2+1, -kappa)/(m*z)
    cosine = (1-rho)/m+rho*tilted_mean
    eigen = (m*cosine-1)/(m-1)
    x, w = roots_jacobi(order, (m-3)/2, -0.5)
    w = w/w.sum()
    g = 1-rho+rho*np.exp(-kappa*(x+1)/2)/z
    mi = float(w @ (g*np.log(g)))
    return dict(m=m, kappa=float(kappa), rho=rho, cosine=float(cosine),
                zeta=float(eigen), mi_nats=mi, density_floor=float(1-rho+rho*np.exp(-kappa)/z))


def covariance_blocks(d, eigen):
    length, m = d.shape
    out = np.empty((length*m,length*m))
    for i in range(length):
        for j in range(length):
            if i == j:
                val = QW**2*(d[i]@d[i]*np.eye(m)+np.outer(d[i],d[i]))
            else:
                eta = QW**2*m/(m+2)*eigen**abs(j-i)
                val = eta*((d[i]@d[j])*np.eye(m)+np.outer(d[j],d[i])-2/m*np.outer(d[i],d[j]))
            out[i*m:(i+1)*m,j*m:(j+1)*m] = val
    np.testing.assert_allclose(out,out.T,atol=1e-13)
    assert np.linalg.eigvalsh(out)[0] > -1e-10
    return out


def proxy(theta, cov, mean=None):
    val = 0.5*(np.trace(np.linalg.solve(theta,cov))
               +np.linalg.slogdet(theta)[1]-np.linalg.slogdet(theta+cov)[1])
    if mean is not None:
        val += 0.5*mean@np.linalg.solve(theta,mean)
    return float(val)


def directions(rng, size, m):
    v = rng.normal(size=(size,m))
    return v/np.linalg.norm(v,axis=1)[:,None]


def sample_matrices(rng, size, length, m, kappa, rho):
    u = directions(rng,size,m)
    matrices = []
    for t in range(length):
        if t:
            new = directions(rng,size,m)
            todo = np.flatnonzero(rng.random(size)<rho) if kappa else np.array([],int)
            while len(todo):
                proposal = directions(rng,len(todo),m)
                accept = rng.random(len(todo)) < np.exp(-kappa*np.sum(proposal*u[todo],axis=1)**2)
                new[todo[accept]] = proposal[accept]
                todo = todo[~accept]
            u = new
        radii2 = QW*rng.chisquare(m,size=size)
        matrices.append(LAM*np.eye(m)+radii2[:,None,None]*u[:,:,None]*u[:,None,:])
    return np.stack(matrices,axis=1)


def information():
    rows = []
    for m in (2,3,4,6,10):
        mixed = constants(m,2,0.7)
        match = brentq(lambda k: constants(m,k,1)['cosine']-mixed['cosine'],0,2)
        pure = constants(m,match,1)
        assert pure['mi_nats'] <= mixed['mi_nats']+1e-12
        np.testing.assert_allclose(pure['zeta'],mixed['zeta'],atol=1e-12)
        quadratures = {str(n): constants(m,2,.7,n)['mi_nats'] for n in (64,128,256)}
        x,w = roots_legendre(600);t=(x+1)/2
        density=t**(-.5)*(1-t)**((m-3)/2)/beta_fn(.5,(m-1)/2)
        z=hyp1f1(.5,m/2,-2);g=.3+.7*np.exp(-2*t)/z
        original=float(np.sum(w/2*density*g*np.log(g)))
        rows.append(dict(m=m,mixed=mixed,pure_same_zeta=pure,quadrature=quadratures,
                         original_legendre_600=original,old_minus_jacobi=original-mixed['mi_nats']))
    return rows


def geometry():
    a, energy, m = 1.04,10.0,2
    j = np.array([[0.,-1.],[1.,0.]])
    amp = np.sqrt(energy/(1+a*a))
    metrics = []
    for label, rotation in (("aligned",np.eye(2)),("rotation",j)):
        A=a*rotation
        d=np.stack([amp*np.array([1.,0.]),A@(amp*np.array([1.,0.]))])
        z=np.concatenate([-A,np.eye(2)],axis=1)
        theta=(2+a*a)*np.eye(2)
        for k in (0.,2.):
            eig=constants(2,k,.7)['zeta']
            c=z@covariance_blocks(d,eig)@z.T
            expected = a*a*amp*amp*QW**2*(np.diag([4-eig,2-eig]) if label=="aligned" else np.diag([2+eig,4+eig]))
            np.testing.assert_allclose(c,expected,rtol=1e-13,atol=1e-13)
            metrics.append(dict(case=label,kappa=k,zeta=eig,covariance=c.tolist(),theta=theta.tolist(),dg=proxy(theta,c)))
    batches=[]
    for label, rotation in (("aligned",np.eye(2)),("rotation",j)):
        A=a*rotation;z=np.concatenate([-A,np.eye(2)],axis=1);theta=(2+a*a)*np.eye(2)
        d=np.stack([amp*np.array([1.,0.]),A@(amp*np.array([1.,0.]))])
        for batch in range(20):
            common_noise=np.random.default_rng(202611219000+batch).normal(size=(5000,2))*np.sqrt(2+a*a)
            for k in (0.,2.):
                rng=np.random.default_rng(202611219100+1000*(label=="rotation")+100*int(k)+batch)
                M=sample_matrices(rng,5000,2,2,k,.7)
                transfer=np.einsum('ntij,tj->nti',M,d).reshape(5000,4)@z.T
                eig=constants(2,k,.7)['zeta'];exact=z@covariance_blocks(d,eig)@z.T
                estimate=(transfer.T@transfer)/len(transfer)
                cutoff=chi2.ppf(.95,2)
                fpr=float(np.mean(np.sum(common_noise**2,axis=1)/(2+a*a)>cutoff))
                power=float(np.mean(np.sum((transfer+common_noise)**2,axis=1)/(2+a*a)>cutoff))
                batches.append(dict(case=label,kappa=k,batch=batch,fpr=fpr,power=power,
                                    covariance_estimate=estimate.tolist(),covariance_exact=exact.tolist(),
                                    transfer_mean=transfer.mean(axis=0).tolist()))
    summaries=[]
    for label in ("aligned","rotation"):
        for k in (0.,2.):
            rows=[row for row in batches if row['case']==label and row['kappa']==k]
            avg=np.mean([row['covariance_estimate'] for row in rows],axis=0)
            exact=np.array(rows[0]['covariance_exact'])
            sem=np.std([row['covariance_estimate'] for row in rows],axis=0,ddof=1)/np.sqrt(20)
            powers=np.array([row['power'] for row in rows])
            paired_power=np.array([row['power'] for row in rows])-np.array(
                [row['power'] for row in batches if row['case']==label and row['kappa']==0.])
            summaries.append(dict(case=label,kappa=k,samples=100000,
                                  max_cov_abs_error=float(np.max(np.abs(avg-exact))),
                                  max_cov_standard_errors=float(np.max(np.abs(avg-exact)/np.maximum(sem,1e-15))),
                                  average_covariance=avg.tolist(),power=float(powers.mean()),
                                  power_batch_se=float(powers.std(ddof=1)/np.sqrt(20)),
                                  paired_power_change=float(paired_power.mean()),
                                  paired_power_change_batch_se=float(paired_power.std(ddof=1)/np.sqrt(20)),
                                  fpr=float(np.mean([row['fpr'] for row in rows]))))
    return dict(exact=metrics,monte_carlo=summaries,batches=batches)


def load_models():
    spec=importlib.util.spec_from_file_location("common_g_review",BASE/'common_g_reference_revision_20261115/common_g_reference.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    names=("unstable","stable")
    models=[mod.load_model(n) for n in names]
    reg=json.loads((BASE/'B1_three_hypothesis_feasibility_20261120/registration.json').read_text(encoding='utf-8'))
    p=reg['regional_model'];A=np.array(p['A']);H=[np.array(h) for h in p['H']];R=np.array(p['R_each']);Q=np.array(p['Q'])
    C=np.vstack([H[1],H[0]+H[2]]);V=block_diag(R,2*R)
    prior=solve_discrete_are(A.T,C.T,Q,V)
    gain=np.linalg.solve(C@prior@C.T+V,C@prior).T
    from types import SimpleNamespace
    models.append(SimpleNamespace(name="regional",A=A,H=H,Q=Q,R=[R,R,R],K=[None,gain[:,:2]],
                                  G=[None,gain[:,2:]],neighbors=[[],[0,2],[]],
                                  F=lambda i:(np.eye(2)-gain@C)@A))
    return mod,models


def healthy_risk(model):
    n=len(model.A);i=1
    K,G=model.K[i],model.G[i]
    sumH=sum((model.H[j] for j in model.neighbors[i]),np.zeros_like(model.H[i]))
    J=np.eye(n)-K@model.H[i]-G@sumH
    Rsum=sum((model.R[j] for j in model.neighbors[i]),np.zeros_like(model.R[i]))
    noise=K@model.R[i]@K.T+G@Rsum@G.T
    F=J@model.A
    P=J@np.eye(n)@J.T+noise
    losses=[np.trace(P)]
    for _ in range(31):
        P=F@P@F.T+J@model.Q@J.T+noise
        losses.append(np.trace(P))
    return float(np.mean(losses))


def state_bridge():
    mod,models=load_models()
    rows=[]
    for model in models:
        A,H=model.A,model.H[0];n=len(A);m=H.shape[0];F=model.F(1);G=model.G[1]
        nominal=healthy_risk(model)
        support=np.vstack([H@np.linalg.matrix_power(A,t) for t in range(10)])
        vector=np.array([-1.89,.42]);xi=vector*np.sqrt(10/np.linalg.norm(support@vector)**2)
        d=(support@xi).reshape(10,m)
        K=np.zeros((32*n,10*m))
        for time in range(32):
            for lag in range(10):
                if 7+lag<=time:
                    K[time*n:(time+1)*n,lag*m:(lag+1)*m]=np.linalg.matrix_power(F,time-7-lag)@G
        # Fixed internal five-packet window. It is not an all-window detector.
        obs,Z,_,theta=mod.parity_model(A,H,model.Q if hasattr(model,'Q') else np.array([[.05,.015],[.015,.08]]),
                                      model.R[0] if hasattr(model,'R') else .5*np.eye(m),4)
        for k in (0.,.5,1.,2.,4.):
            eig=constants(m,k,.7)['zeta']
            cov=covariance_blocks(d,eig)
            output_cov=K@cov@K.T
            output_mean=K@((LAM+QW)*d.reshape(-1))
            extra=(np.trace(output_cov)+output_mean@output_mean)/32
            parity_cov=Z@cov[:5*m,:5*m]@Z.T
            parity_mean=Z@((LAM+QW)*d[:5].reshape(-1))
            # Cross-check covariance contraction by direct summation for state risk.
            direct=0.
            for i in range(10):
                for j in range(10):
                    direct+=np.trace(K[:,i*m:(i+1)*m]@cov[i*m:(i+1)*m,j*m:(j+1)*m]@K[:,j*m:(j+1)*m].T)
            np.testing.assert_allclose(direct,np.trace(output_cov),rtol=1e-12)
            rows.append(dict(model=model.name,kappa=k,zeta=eig,mi=constants(m,k,.7)['mi_nats'],
                             coded_energy=float(np.sum(d*d)),common_g_extra_mse=float(extra),
                             mean_bias_mse=float(output_mean@output_mean/32),
                             variance_mse=float(np.trace(output_cov)/32),
                             healthy_mse=nominal,attacked_total_mse=float(nominal+extra),
                             internal_window_dg=proxy(theta,parity_cov,parity_mean),
                             parity_mean_norm=float(np.linalg.norm(parity_mean))))
    return rows


def parity_energy_conditions():
    mod,models=load_models();out=[]
    for model in models:
        O,Z,_,theta=mod.parity_model(model.A,model.H[0],model.Q,model.R[0],4)
        m=model.H[0].shape[0]
        block=[Z[:,ell*m:(ell+1)*m] for ell in range(5)]
        weights=[float(np.trace(np.linalg.solve(theta,x@x.T))) for x in block]
        out.append(dict(model=model.name,residual_dimension=int(Z.shape[0]),
                        rank_observability=int(np.linalg.matrix_rank(O)),
                        block_trace_weights=weights,c_z=min(weights),
                        positive_all_blocks=bool(min(weights)>0)))
    return out


def state_grams_and_checks():
    mod,models=load_models()
    output=[]
    for model in models:
        A,H=model.A,model.H[0];n=len(A);m=H.shape[0];F=model.F(1);G=model.G[1]
        support=np.vstack([H@np.linalg.matrix_power(A,t) for t in range(10)])
        energy_gram=support.T@support
        ev,V=np.linalg.eigh(energy_gram)
        E_inv_sqrt=(V/np.sqrt(ev))@V.T
        K=np.zeros((32*n,10*m))
        for t in range(32):
            for lag in range(10):
                if t>=7+lag:
                    K[t*n:(t+1)*n,lag*m:(lag+1)*m]=np.linalg.matrix_power(F,t-7-lag)@G
        def risk_for(x,eigen):
            d=(support@x).reshape(10,m)
            covariance=covariance_blocks(d,eigen)
            mu=K@((LAM+QW)*d.reshape(-1))
            return float((np.trace(K@covariance@K.T)+mu@mu)/32)
        grams=[]
        for k in (0.,.5,1.,2.,4.):
            eigen=constants(m,k,.7)['zeta']
            Q=np.zeros((n,n))
            for i in range(n):
                Q[i,i]=risk_for(np.eye(n)[i],eigen)
            for i in range(n):
                for j in range(i):
                    x=np.eye(n)[i]+np.eye(n)[j]
                    Q[i,j]=Q[j,i]=.5*(risk_for(x,eigen)-Q[i,i]-Q[j,j])
            grams.append((k,Q))
        base=grams[0][1]
        for k,Q in grams:
            output.append(dict(model=model.name,kappa=k,Q=Q.tolist(),coded_energy_gram=energy_gram.tolist(),
                               worst_energy10_extra_mse=float(10*np.linalg.eigvalsh(E_inv_sqrt@Q@E_inv_sqrt)[-1]),
                               all_direction_difference_eigenvalues=np.linalg.eigvalsh(E_inv_sqrt@(Q-base)@E_inv_sqrt).tolist()))
    return output


def multitime_validation():
    records=[]
    for m in (2,3,4):
        length=5
        # Fixed ordinary vectors and projection weights, independent of watermarks.
        d=np.arange(1,length*m+1,dtype=float).reshape(length,m)
        d/=np.linalg.norm(d)
        eig=constants(m,2,.7)['zeta']
        exact=covariance_blocks(d,eig)
        batch_cov=[];batch_cos=[]
        for b in range(20):
            rng=np.random.default_rng(202611220000+100*m+b)
            mats=sample_matrices(rng,5000,length,m,2,.7)
            transfer=np.einsum('ntij,tj->nti',mats,d).reshape(5000,length*m)
            centered=transfer-(LAM+QW)*d.reshape(-1)
            batch_cov.append(centered.T@centered/len(centered))
            outer=mats-LAM*np.eye(m)
            proj=outer/np.trace(outer,axis1=2,axis2=3)[:,:,None,None]
            batch_cos.append([float(np.einsum('nij,nji->n',proj[:,0],proj[:,h]).mean()) for h in range(1,length)])
        estimates=np.asarray(batch_cov)
        average=estimates.mean(axis=0)
        lag0=float(np.linalg.norm(np.diag(average-exact))/np.linalg.norm(np.diag(exact)))
        # All covariance entries checked jointly via a descriptive Frobenius ratio.
        records.append(dict(m=m,samples=100000,lag1to4_cosine_means=np.mean(batch_cos,axis=0).tolist(),
                            lag1to4_cosine_exact=[1/m+(1-1/m)*eig**h for h in range(1,length)],
                            covariance_relative_frobenius_error=float(np.linalg.norm(average-exact)/np.linalg.norm(exact)),
                            diagonal_relative_error=lag0,
                            cosine_batch_se=(np.std(batch_cos,axis=0,ddof=1)/np.sqrt(20)).tolist()))
    return records


def compare_original():
    source=json.loads((HERE/'source_snapshots/ACMW/results.json').read_text())
    rerun=json.loads((HERE/'reproduction/angular_coupled_watermark_results_recomputed.json').read_text())
    rows=[]
    for s,r in zip(source['results'],rerun['results']):
        old=s['mean_gaussian_equivalent_KL']
        new=r['means']
        rows.append(dict(n=s['n'],m=s['m'],supplied_iid=old['iid'],rerun_iid=new[0],
                         supplied_kappa2=old['kappa_2'],rerun_kappa2=new[3],
                         rerun_relative_gain=float(new[3]/new[0]-1),
                         rerun_fraction_positive=r['paired_differences']['2.0']['fraction_positive'],
                         absolute_kappa2_table_difference=float(new[3]-old['kappa_2'])))
    return dict(note="Exact supplied code was run unchanged; table values are not exactly reproduced. RNG stream/version provenance is not fully recorded in supplied table.",rows=rows)


def physical_state_check():
    mod,models=load_models()
    output=[]
    for model_number,model in enumerate(models):
        A=model.A;F=model.F(1);K=model.K[1];G=model.G[1]
        n=len(A);m=model.H[0].shape[0]
        support=np.vstack([model.H[0]@np.linalg.matrix_power(A,t) for t in range(10)])
        vector=np.array([-1.89,.42]);xi=vector*np.sqrt(10/np.linalg.norm(support@vector)**2)
        d=(support@xi).reshape(10,m)
        sumH=sum((model.H[j] for j in model.neighbors[1]),np.zeros_like(model.H[1]))
        sumR=sum((model.R[j] for j in model.neighbors[1]),np.zeros_like(model.R[1]))
        Qchol=np.linalg.cholesky(model.Q)
        Rchol=np.linalg.cholesky(model.R[1]);S_chol=np.linalg.cholesky(sumR)
        batches=[]
        roundtrip=0.
        for batch in range(20):
            size=5000
            physical=np.random.default_rng(202611230000+100*model_number+batch)
            x=physical.normal(size=(size,n));health=np.zeros((size,n))
            bias={k:np.zeros((size,n)) for k in (0.,2.)}
            sums={k:np.zeros(size) for k in (0.,2.)};healthy_sums=np.zeros(size)
            transfers={}
            for k in (0.,2.):
                rng=np.random.default_rng(202611240000+1000*model_number+100*int(k)+batch)
                matrices=sample_matrices(rng,size,10,m,k,.7)
                transfers[k]=np.einsum('ntij,tj->nti',matrices,d)
                # Check ideal encoding/decoding without feeding truth to a receiver.
                packet=np.ones((size,m))
                encoded=np.linalg.solve(matrices[:,0],packet[...,None])[...,0]
                decoded=np.einsum('nij,nj->ni',matrices[:,0],encoded)
                roundtrip=max(roundtrip,float(np.max(abs(decoded-packet))))
            for time in range(32):
                if time:
                    x=x@A.T+physical.normal(size=(size,n))@Qchol.T
                local=x@model.H[1].T+physical.normal(size=(size,m))@Rchol.T
                neighbors=x@sumH.T+physical.normal(size=(size,m))@S_chol.T
                health=health@F.T+local@K.T+neighbors@G.T
                healthy_sums+=np.sum((health-x)**2,axis=1)/32
                for k in (0.,2.):
                    bias[k]=bias[k]@F.T
                    if 7<=time<17:
                        bias[k]+=transfers[k][:,time-7]@G.T
                    sums[k]+=np.sum((health+bias[k]-x)**2,axis=1)/32
            batches.append(dict(batch=batch,healthy=float(healthy_sums.mean()),
                                iid=float(sums[0.].mean()),coupled=float(sums[2.].mean()),
                                paired_difference=float((sums[2.]-sums[0.]).mean())))
        pairs=np.array([b['paired_difference'] for b in batches])
        output.append(dict(model=model.name,physical_paths=100000,
                           healthy=float(np.mean([b['healthy'] for b in batches])),
                           iid_attacked=float(np.mean([b['iid'] for b in batches])),
                           coupled_attacked=float(np.mean([b['coupled'] for b in batches])),
                           paired_difference=float(pairs.mean()),paired_batch_se=float(pairs.std(ddof=1)/np.sqrt(20)),
                           encoding_roundtrip_max_abs=roundtrip,batches=batches))
    return output


def main():
    data=dict(scope="Paper1 reviewer verification; original supplementary models, not original 50-node validation.",
              information=information(),geometry=geometry(),state_bridge=state_bridge(),
              parity_energy=parity_energy_conditions(),
              state_grams=state_grams_and_checks(),multitime_validation=multitime_validation(),
              source_table_comparison=compare_original(),physical_state=physical_state_check())
    write('review_results.json',data)
    print('review_results.json written')


if __name__=='__main__':
    main()
