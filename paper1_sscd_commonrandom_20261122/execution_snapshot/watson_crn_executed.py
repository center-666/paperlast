"""Exact-law m=2 Watson chain using numerical von-Mises quantile inversion.

Statistical target is precisely the same signed Watson transition kernel as
legacy replay.attack_watermarks, not a new watermark security protocol.
The common random numbers are *offline experiment coupling only*.

For unit vectors u=(cos a,sin a), p(a'|a) proportional to
  exp(theta * cos(a'-a)**2)
  = exp(theta/2) * exp((theta/2)*cos(2*(a'-a))).
Let d=2(a'-a) mod 2pi. Then d follows a signed-parameter von Mises
law f(d) = exp(k*cos(d))/(2pi I0(k)), k=theta/2. The two lifts
 a'=a+d/2 and a'=a+d/2+pi have equal conditional probability.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from scipy.special import iv, i0, i1

TAU=2*np.pi


def von_mises_cdf_signed(phi, kappa, terms=18):
    """Fourier CDF from angle 0 to phi, signed kappa, phi in [0,2pi]."""
    x=np.asarray(phi,dtype=np.float64)
    k=float(kappa)
    if abs(k)>1.05:
        raise ValueError('Validated range |theta|<=2.1; extend and re-test Fourier tail before use')
    # Sum_{j>=1} I_j(k) sin(j phi)/(pi*j*I_0(k))
    ans=x/TAU
    norm=i0(k)
    for j in range(1,terms+1):
        ans=ans+(iv(j,k)/(np.pi*j*norm))*np.sin(j*x)
    return ans


def von_mises_icdf_signed(u,theta,iterations=10,terms=18):
    """Deterministic inverse CDF; shares u across candidate theta values.

    Newton steps safeguarded by enclosing bisection. The returned angles have
    the desired law up to finite Fourier truncation and floating point error.
    """
    q=np.asarray(u,dtype=np.float64)
    if np.any(~np.isfinite(q)) or np.any((q<0)|(q>=1)):
        raise ValueError('u must be finite in [0,1)')
    k=.5*float(theta)
    if not np.isfinite(k) or abs(k)>1.05:
        raise ValueError('Validated range |theta|<=2.1')
    if k==0.:return TAU*q
    lo=np.zeros_like(q);hi=np.full_like(q,TAU)
    x=TAU*q
    invnorm=1/(TAU*i0(k))
    for _ in range(iterations):
        err=von_mises_cdf_signed(x,k,terms=terms)-q
        lo=np.where(err<0,x,lo)
        hi=np.where(err>0,x,hi)
        density=np.exp(k*np.cos(x))*invnorm
        guess=x-err/density
        x=np.where(np.abs(err)<2e-14,x,
                   np.where((guess>lo)&(guess<hi),guess,.5*(lo+hi)))
    err=von_mises_cdf_signed(x,k,terms=terms)-q
    if np.max(np.abs(err))>5e-13:
        raise ArithmeticError(f'inverse-CDF residual too large: {np.max(np.abs(err))}')
    return x


def theoretical_zeta_mi(theta):
    k=float(theta)/2
    zeta=i1(k)/i0(k)
    return float(zeta),float(k*zeta-np.log(i0(k)))


@dataclass
class CommonDraws:
    radial_chi2_half: np.ndarray  # [B,L,E] = 0.5 chi^2_2
    initial_angle: np.ndarray     # [B,E] uniform [0,2pi)
    quantile_u: np.ndarray        # [B,L-1,E] independent uniforms
    branch: np.ndarray            # [B,L-1,E] independent fair bits


def make_common_draws(count,length,edges,radial_seed,initial_seed,direction_seed):
    if min(count,length,edges)<=0:raise ValueError('positive shapes required')
    radial=.5*np.random.default_rng(radial_seed).chisquare(2,size=(count,length,edges))
    rng0=np.random.default_rng(initial_seed)
    # isotropic 2D Gaussian initial direction, same as original sampler
    v=rng0.normal(size=(count,edges,2))
    a=np.arctan2(v[...,1],v[...,0])%TAU
    rg=np.random.default_rng(direction_seed)
    q=rg.random(size=(count,length-1,edges))
    branch=rg.integers(0,2,size=(count,length-1,edges),dtype=np.int8)
    return CommonDraws(radial,a,q,branch)


def beta_from_common(draws,theta):
    B,L,E=np.shape(draws.radial_chi2_half)
    angle=np.empty((B,L,E),dtype=np.float64)
    angle[:,0]=draws.initial_angle
    phase=von_mises_icdf_signed(draws.quantile_u,theta)
    for k in range(1,L):
        angle[:,k]=(angle[:,k-1]+.5*phase[:,k-1]+np.pi*draws.branch[:,k-1])%TAU
    direction=np.stack([np.cos(angle),np.sin(angle)],axis=-1)
    return np.sqrt(draws.radial_chi2_half)[...,None]*direction


def watermark_attack_from_beta(attack,beta,T=150,offset=50):
    """Keep original decoded attack delta = .5*attack + beta*(beta·attack)."""
    attack=np.asarray(attack,float)
    if attack.ndim!=3 or attack.shape[1:]!=beta.shape[2:] or attack.shape[0]!=beta.shape[1]:
        raise ValueError('expected attack [length,edges,2] matching beta [paths,length,edges,2]')
    dot=np.einsum('btem,tem->bte',beta,attack)
    decoded=.5*attack[None]+beta*dot[...,None]
    out=np.zeros((beta.shape[0],T,beta.shape[2],2))
    out[:,offset:offset+len(attack)]=decoded
    return out


def attack_watermarks_coupled(model,count,theta,radial_seed,initial_seed,direction_seed):
    """Drop-in statistical equivalent to replay.attack_watermarks.

    The caller MUST use the same direction_seed for every candidate theta in
    the same physical scenario. Different scenarios should use independent seeds.
    """
    length=60;E=len(model.links)
    d=make_common_draws(count,length,E,radial_seed,initial_seed,direction_seed)
    beta=beta_from_common(d,theta)
    delta=watermark_attack_from_beta(model.attack[50:110],beta,model.T,50)
    return delta,beta
