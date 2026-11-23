"""Multidimensional (m>=2) exact-target-law Watson Markov sampler via tilted-Beta quantiles.

The theory is dimension-independent, but this is a numerical sampler with an
explicitly validated |theta|<=6 range. It is a research extension; the frozen
original 50-node estimator/message dimensions remain m=2.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import betainc,betaincinv,hyp1f1,betaln


def tilted_beta_cdf(z,theta,m=4,terms=60):
    """CDF of Z=(u·u')^2 under signed Watson transition on S^(m-1)."""
    x=np.asarray(z,float)
    if m<2 or abs(theta)>6.0 or np.any((x<0)|(x>1)):
        raise ValueError('m>=2, 0<=z<=1, |theta|<=6.0 required')
    a=.5;b=(m-1)/2.; c=a+b
    if theta==0:return betainc(a,b,x)
    coeff=1.;out=np.zeros_like(x)
    for j in range(terms):
        out+=coeff*betainc(a+j,b,x)
        coeff*=theta*(a+j)/(c+j)/(j+1)
    return out/hyp1f1(a,c,theta)


def tilted_beta_icdf(u,theta,m=4,terms=60,steps=72):
    q=np.asarray(u,float)
    if np.any(~np.isfinite(q)) or np.any((q<0)|(q>=1)):
        raise ValueError('u must be in [0,1)')
    if m<2 or abs(theta)>6.0:
        raise ValueError('validated domain m>=2 and |theta|<=6.0')
    a=.5;b=(m-1)/2.;c=a+b
    if theta==0:return betaincinv(a,b,q)
    xx=betaincinv(a,b,q)
    lo=np.zeros_like(q);hi=np.ones_like(q)
    z=xx.copy()
    logn=np.log(hyp1f1(a,c,theta));logB=betaln(a,b)
    for _ in range(steps):
        err=tilted_beta_cdf(z,theta,m,terms)-q
        lo=np.where(err<0,z,lo)
        hi=np.where(err>0,z,hi)
        zz=np.clip(z,1e-300,1.-1e-14)
        logpdf=(a-1)*np.log(zz)+(b-1)*np.log1p(-zz)-logB+theta*zz-logn
        pdf=np.exp(np.clip(logpdf,-700,700))
        nxt=z-err/pdf
        z=np.where(np.abs(err)<2e-12,z,
                   np.where((nxt>lo)&(nxt<hi),nxt,.5*(lo+hi)))
    err=tilted_beta_cdf(z,theta,m,terms)-q
    bad=np.abs(err)>2e-10
    if np.any(bad):
        # For m=2 and a positive extreme tilt, the upper-tail Beta density is
        # singular. Some target quantiles lie between adjacent float64 z values.
        # Bisection then picks the representable endpoint with the smaller CDF
        # residual rather than failing despite converging to machine precision.
        blo=np.zeros_like(q);bhi=np.ones_like(q)
        for _ in range(80):
            mid=blo+(bhi-blo)*.5
            cm=tilted_beta_cdf(mid,theta,m,terms)
            blo=np.where(cm<q,mid,blo)
            bhi=np.where(cm>=q,mid,bhi)
        elo=np.abs(tilted_beta_cdf(blo,theta,m,terms)-q)
        ehi=np.abs(tilted_beta_cdf(bhi,theta,m,terms)-q)
        endpoint=np.where(elo<=ehi,blo,bhi)
        z=np.where(bad,endpoint,z)
        err=tilted_beta_cdf(z,theta,m,terms)-q
    # Near z=1, float64 spacing bounds the absolute CDF accuracy for the m=2
    # positive-extreme-tilt endpoint. Away from that narrow tail, residuals are
    # around 1e-12 as checked independently in test_nd_endpoints.py.
    if np.max(np.abs(err))>2e-8:
        raise ArithmeticError(f'tilted Beta inverse CDF residual too high: {np.max(np.abs(err))}')
    return z


def expected_cosine2(theta,m):
    a=.5;c=m/2.;norm=hyp1f1(a,c,theta)
    return (a/c)*hyp1f1(a+1,c+1,theta)/norm


def zeta_mi(theta,m):
    ex=expected_cosine2(theta,m)
    zeta=(m*ex-1)/(m-1)
    mi=theta*ex-np.log(hyp1f1(.5,m/2.,theta))
    return float(zeta),float(mi)


@dataclass
class NdDraws:
    radius:np.ndarray        # [B,L,E] q chi^2_m
    initial:np.ndarray       # [B,E,m] unit sphere
    quantiles:np.ndarray     # [B,L-1,E]
    signs:np.ndarray         # [B,L-1,E] +/-1
    ortho_gaussian:np.ndarray # [B,L-1,E,m]


def make_nd_draws(count,length,edges,m,q,seed):
    if m<2 or min(count,length,edges)<=0 or q<=0:raise ValueError('positive dimensions, m>=2')
    # Separate named RNG streams from one SeedSequence; seed independent of theta.
    streams=np.random.SeedSequence(seed).spawn(5)
    r=q*np.random.default_rng(streams[0]).chisquare(m,(count,length,edges))
    init=np.random.default_rng(streams[1]).normal(size=(count,edges,m))
    init/=np.linalg.norm(init,axis=-1,keepdims=True)
    us=np.random.default_rng(streams[2]).random((count,length-1,edges))
    signs=2*np.random.default_rng(streams[3]).integers(0,2,(count,length-1,edges))-1
    orth=np.random.default_rng(streams[4]).normal(size=(count,length-1,edges,m))
    return NdDraws(r,init,us,signs,orth)


def beta_nd_from_common(draws,theta):
    B,L,E=draws.radius.shape;m=draws.initial.shape[-1]
    zz=tilted_beta_icdf(draws.quantiles,theta,m)
    u=np.empty((B,L,E,m));u[:,0]=draws.initial
    for t in range(1,L):
        cur=u[:,t-1]
        g=draws.ortho_gaussian[:,t-1]
        ortho=g-np.sum(g*cur,axis=-1,keepdims=True)*cur
        norm=np.linalg.norm(ortho,axis=-1,keepdims=True)
        if np.any(norm<=1e-14): raise ArithmeticError('degenerate orthogonal Gaussian projection')
        ortho/=norm
        z=zz[:,t-1]
        u[:,t]=(draws.signs[:,t-1]*np.sqrt(z))[...,None]*cur+np.sqrt(1-z)[...,None]*ortho
    return np.sqrt(draws.radius)[...,None]*u
