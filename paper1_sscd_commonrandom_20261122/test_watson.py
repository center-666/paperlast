"""Distributional and regression checks for exact-target-law Watson CRN.

The tests compare laws, NOT individual samples. All output is synthetic.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
from watson_crn import (von_mises_icdf_signed,von_mises_cdf_signed,
    theoretical_zeta_mi,make_common_draws,beta_from_common,watermark_attack_from_beta)
from watson_nd import (tilted_beta_cdf,tilted_beta_icdf,zeta_mi,
                       make_nd_draws,beta_nd_from_common)

HERE=Path(__file__).resolve().parent


def legacy_unit_directions(n,T,E,theta,seed):
    """Replicate original accepted-normal proposals, without attack payload."""
    rng=np.random.default_rng(seed)
    v=rng.normal(size=(n,E,2));v/=np.linalg.norm(v,axis=-1,keepdims=True)
    out=np.empty((n,T,E,2));out[:,0]=v
    for k in range(1,T):
        old=v.reshape(-1,2).copy()
        prop=rng.normal(size=old.shape);prop/=np.linalg.norm(prop,axis=-1,keepdims=True)
        pending=np.arange(len(old))
        while len(pending):
            t=np.sum(prop[pending]*old[pending],axis=1)**2
            accept=rng.random(len(pending))<np.exp(theta*t-max(theta,0.))
            old[pending[accept]]=prop[pending[accept]]
            pending=pending[~accept]
            if len(pending):
                p=rng.normal(size=(len(pending),2))
                p/=np.linalg.norm(p,axis=1,keepdims=True)
                prop[pending]=p
        v=old.reshape(n,E,2)
        out[:,k]=v
    return out


def test_2d_exact_law():
    n=24000;T=5;E=1
    draw=make_common_draws(n,T,E,5,6,7)
    output={}
    for theta in (-1.3152369768257723,0.,1.3152369768257723):
        beta=beta_from_common(draw,theta)
        u=beta/np.linalg.norm(beta,axis=-1,keepdims=True)
        orig=legacy_unit_directions(n,T,E,theta,12345)
        n0=np.sum(u[:,:-1]*u[:,1:],axis=-1)**2
        n1=np.sum(orig[:,:-1]*orig[:,1:],axis=-1)**2
        zeta,mi=theoretical_zeta_mi(theta)
        obs_zeta=2*np.mean(n0)-1
        ks=ks_2samp(n0.ravel(),n1.ravel(),method='asymp')
        assert abs(obs_zeta-zeta)<.015,(theta,obs_zeta,zeta)
        assert ks.statistic<.02,(theta,ks)
        # Test exact decoder M: rank-one construction & byte-level comparison
        attack=np.random.default_rng(90).normal(size=(T,E,2))
        delta=watermark_attack_from_beta(attack,beta,T=12,offset=3)
        decoded=delta[:,3:3+T]
        explicit=.5*attack[None]+np.einsum('btem,tem->bte',beta,attack)[...,None]*beta
        np.testing.assert_allclose(decoded,explicit,atol=2e-14,rtol=1e-14)
        cov=np.cov(beta.reshape(-1,2).T)
        assert np.max(np.abs(cov-.5*np.eye(2)))<.018,(theta,cov)
        output[f'{theta:+.7f}']={'empirical_zeta':float(obs_zeta),'theory_zeta':zeta,
            'theory_MI':mi,'KS_vs_original_sampler':float(ks.statistic),
            'Gaussian_beta_covariance':cov.tolist()}
    q=np.r_[0.,1e-13,np.linspace(0,1,10001,endpoint=False),np.nextafter(1.,0.)]
    for theta in (-2.,-1.3152369768257723,-.6,0.,.6,1.3152369768257723,2.):
        p=von_mises_icdf_signed(q,theta)
        max_err=np.max(np.abs(von_mises_cdf_signed(p,theta/2)-q))
        assert max_err<5e-13,(theta,max_err)
    return output


def test_nd_law():
    n=16000;T=4
    output={}
    for m in [3,4,6]:
        draw=make_nd_draws(n,T,1,m,.5,seed=100+m)
        for theta in [-1.3152369768257723,0.,1.3152369768257723]:
            beta=beta_nd_from_common(draw,theta)
            norms=np.linalg.norm(beta,axis=-1)
            u=beta/norms[...,None]
            observed=np.mean(np.sum(u[:,:-1]*u[:,1:],axis=-1)**2)
            ezeta,mi=zeta_mi(theta,m)
            predicted=(1+(m-1)*ezeta)/m
            cov=np.cov(beta.reshape(-1,m).T)
            assert abs(observed-predicted)<.014,(m,theta,observed,predicted)
            assert np.max(np.abs(cov-.5*np.eye(m)))<.028,(m,theta,cov)
            output[f'm{m}_theta{theta:+.4f}']={'predicted_E_cos2':float(predicted),
                'observed_E_cos2':float(observed),'MI_nats':mi,
                'max_beta_cov_error':float(np.max(np.abs(cov-.5*np.eye(m))))}
    # Equal |theta| is NOT equal MI for m>2.
    assert abs(zeta_mi(1.3152369768257723,4)[1]-zeta_mi(-1.3152369768257723,4)[1])>.02
    return output


def main():
    a=test_2d_exact_law();b=test_nd_law()
    result={'status':'PASS','claims':'SYNTHETIC_MARGINAL_AND_SAMPLING_CHECKS_ONLY',
            'm2':a,'higher_dimensional':b}
    p=HERE/'sampling_validation.json'
    p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','2D_cases':len(a),'ND_cases':len(b),
                      'output':str(p)},indent=2))

if __name__=='__main__':main()
