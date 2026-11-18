"""Upper envelopes on paired displacement, in receiver observables and coded energy."""
import numpy as np
from scipy.linalg import block_diag, solve_triangular

from energy_inputs import TrustedConditioning
from spectral_influence import coefficients


def row_norm(blocks):
    row = blocks.transpose(1,0,2).reshape(blocks.shape[1],-1)
    return float(np.linalg.norm(row,ord=2))


def certificate(p, trusted, observed, watermarks, energy, radius, clip_threshold):
    """All arguments are receiver history, its healthy model, and a declared energy cap."""
    T = p.settings.horizon
    G = p.model.G[p.settings.receiver]
    F = p.model.F(p.settings.receiver)
    if energy<0:
        raise ValueError("The declared coded energy cap must be nonnegative.")
    estimate = TrustedConditioning(p)
    gain = np.linalg.norm(G,ord=2)
    steps = []
    named = ("V0","V1a","V1.2a","V1.4","anchor@2%","innovation-clip@2%","trusted-ledger@2%")
    if energy==0:
        return {key:np.zeros(T) for key in named}
    e = np.sqrt(energy)
    mixture_changes, correction_pairs, whitened_influence, nominal_factors = [],[],[],[]
    prior_factor = p.parameter_prior_cholesky
    for k,prefix in enumerate(p.prefixes):
        B = coefficients(p,k)
        currentM = watermarks[k]
        raw = row_norm(B["common_G"] @ watermarks[:k+1])*e
        full = row_norm(B["full_nominal_conditional"] @ watermarks[:k+1])*e
        U = observed[:k+1].reshape(-1)-prefix.target_from_trusted @ trusted[:k+1].reshape(-1)
        scaled = solve_triangular(prefix.target_cholesky,U,lower=True)
        delta_white = np.linalg.norm(solve_triangular(prefix.target_cholesky,
                                        block_diag(*watermarks[:k+1]),lower=True),ord=2)*e
        start = max(0,k+1-p.settings.template_duration)
        gain_bound = .5*gain*max(np.linalg.norm(currentM @ p.template[l] @ prior_factor,ord=2)
                                 for l in range(k-start+1))
        size = float(np.linalg.norm(scaled))
        mixture_changes.append(gain_bound*(1.+2*(size+delta_white)**2)*delta_white)
        yh = estimate.at(trusted[None],k).mean_y[0]
        deviation = np.linalg.norm(G @ (observed[k]-yh))
        null_deviation_upper = deviation+np.linalg.norm(G @ currentM,ord=2)*e
        correction_pairs.append(max(gain_bound*size,deviation)
                                +max(gain_bound*(size+delta_white),null_deviation_upper))
        L = estimate.lower_y[k]
        whitened_influence.append(float(np.linalg.norm(solve_triangular(L,currentM,lower=True),ord=2)))
        nominal_factors.append(G @ L)
        memory = [np.linalg.matrix_power(F,k-t) for t in range(k+1)]
        sizes = np.array([np.linalg.norm(m,ord=2) for m in memory])
        gm = [np.linalg.norm(a @ b,ord=2) for a,b in zip(memory,nominal_factors,strict=True)]
        clip = min(e*float(np.linalg.norm(np.array(gm)*whitened_influence)),
                   2*clip_threshold*sum(gm))
        mixture = raw+sizes @ mixture_changes
        capped = raw+sizes @ correction_pairs
        steps.append((raw,mixture,capped,capped,min(full,2*radius),clip,capped))
    return {key:column for key,column in zip(named,np.asarray(steps).T,strict=True)}
