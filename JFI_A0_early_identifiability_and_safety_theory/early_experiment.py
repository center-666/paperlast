"""Prefix discrimination for the archived A0 law. Only summary metrics are exported."""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit, ndtr, ndtri, roots_hermitenorm


LOG2 = np.log(2.0)
STARTS = (3, 7, 11)


def gaussian_errors(distance):
    """Equal-prior errors for a signed shift and for even decision rules."""
    d = np.asarray(distance, dtype=float)
    x = d**2 / 2
    threshold = np.divide(x + np.log1p(np.sqrt(-np.expm1(-2*x))), d,
                          out=np.ones_like(d), where=d > 0)
    signed = ndtr(-d / 2)
    even = ndtr(-threshold) + 0.5*(ndtr(threshold - d) - ndtr(-threshold - d))
    return signed, even


def normal_quadrature(order):
    nodes, weights = roots_hermitenorm(order)
    return nodes, weights / np.sqrt(2*np.pi)


def signed_error_with_prior(length, active_prior):
    offset = np.log((1-active_prior)/active_prior)
    threshold = length/2 + np.divide(offset, length, out=np.zeros_like(length), where=length > 0)
    error = (1-active_prior)*ndtr(-threshold) + active_prior*ndtr(threshold-length)
    return np.where(length > 0, error, min(active_prior, 1-active_prior))


def mixture_risk(means, order, active_prior=.5):
    """Equal mixture weights. Integrate the scalar conditional LR boundary."""
    lengths = np.linalg.norm(means, axis=-1)
    if means.ndim != 3 or means.shape[1] != 2 or np.any(lengths <= 0):
        raise ValueError("Two nonzero conditional mean shifts are required.")
    dot = np.sum(means[:, 0]*means[:, 1], axis=1)
    cosine = np.clip(dot / (lengths[:, 0]*lengths[:, 1]), -1., 1.)
    axial = lengths * np.sqrt((1+cosine[:, None]) / 2)
    transverse = lengths * np.sqrt((1-cosine[:, None]) / 2) * np.array([1, -1])
    if np.any(axial < 1e-10):
        raise ValueError("The conditional boundary formula requires a common positive axis.")
    nodes, weights = normal_quadrature(order)
    points = nodes[None, None, :] + np.column_stack((np.zeros(len(means)), transverse))[..., None]
    log_threshold = np.log((1-active_prior)/active_prior)
    constants = (lengths[:, None, None, :]**2/2 + log_threshold
                 - points[..., None]*transverse[:, None, None, :])
    slopes = axial[:, None, None, :]
    lower = np.min(constants/slopes, axis=-1)
    upper = np.min((constants+LOG2)/slopes, axis=-1)
    for _ in range(52):
        middle = (lower+upper)/2
        log_lr = np.logaddexp.reduce(slopes*middle[..., None]-constants, axis=-1)-LOG2
        lower = np.where(log_lr < 0, middle, lower)
        upper = np.where(log_lr >= 0, middle, upper)
    boundary = (lower+upper)/2
    false = ndtr(-boundary[:, 0]) @ weights
    miss = (ndtr(boundary[:, 1:] - axial[:, :, None]) @ weights).mean(axis=1)
    return (1-active_prior)*false + active_prior*miss


def mixture_information(means, order):
    """Start-label KL loss from the one-dimensional Gaussian component ratio."""
    nodes, weights = normal_quadrature(order)
    gap = np.linalg.norm(means[:, 0]-means[:, 1], axis=1)
    label_information = LOG2 - np.logaddexp(0., gap[:, None]*nodes - gap[:, None]**2/2) @ weights
    label_information = np.clip(label_information, 0., LOG2)
    labelled_kl = np.mean(np.sum(means**2, axis=-1), axis=1)/2
    return labelled_kl - label_information, label_information, labelled_kl


def prefix_coded_terms(saved_profile, start, k):
    """Reindex the fixed archived profile for a likelihood calculation only."""
    length = min(k+1-start, len(saved_profile))
    terms = np.zeros((k+1, saved_profile.shape[1]))
    if length > 0:
        terms[start:start+length] = saved_profile[:length]
    return terms


def whitened_shift(cholesky, matrices, terms):
    decoded = np.einsum("ntij,tj->nti", matrices, terms)
    return solve_triangular(cholesky, decoded.reshape(len(matrices), -1).T, lower=True).T


def mean_distance_squared(cholesky, terms, lam, q):
    """Exact Gaussian-watermark moment using the full conditional precision."""
    size, width = terms.shape
    base = solve_triangular(cholesky, terms.ravel(), lower=True)
    whitening = solve_triangular(cholesky, np.eye(size*width), lower=True)
    covariance_part = 0.0
    for k, a in enumerate(terms):
        block = whitening[:, k*width:(k+1)*width]
        covariance_part += q**2*(np.dot(a, a)*np.sum(block**2) + np.sum((block @ a)**2))
    return float((lam+q)**2*np.dot(base, base)), float(covariance_part)


def average(values):
    return float(np.mean(values))


def prefix_report(p, watermark, profile, decoder, v0_offsets):
    """Compare known onset 7 and the equal active-onset mixture over 3 and 7."""
    lam, q = watermark["lambda"], watermark["q"]
    output = {}
    observed_danger = np.sum(v0_offsets[:, 7:10]**2, axis=-1) >= .1
    first = np.min(np.where(observed_danger, np.arange(3), 3), axis=1)
    distances = []
    for k in (7, 8, 9):
        candidate_terms = [prefix_coded_terms(profile, s, k) for s in STARTS[:2]]
        chol = p.prefixes[k].target_cholesky
        means = np.stack([whitened_shift(chol, decoder[:, :k+1], sig)
                          for sig in candidate_terms], axis=1)
        d7 = np.linalg.norm(means[:, 1], axis=1)
        distances.append(d7)
        known, even = gaussian_errors(d7)
        mean_part, random_part = mean_distance_squared(chol, candidate_terms[1], lam, q)
        lengths = np.linalg.norm(means, axis=-1)
        both = gaussian_errors(lengths)[0].mean(axis=1)

        mix_lo_order, mix = (mixture_risk(means, order) for order in (64, 128))
        occupied_prior = 1/3
        mixed_with_prior = mixture_risk(means, 128, occupied_prior)
        labelled_with_prior = signed_error_with_prior(lengths, occupied_prior).mean(axis=1)
        mix_kl, loss_kl, labelled = mixture_information(means, 128)
        kl_low_order = mixture_information(means, 64)[0]
        kl_risk_floor = (1 - np.sqrt(-np.expm1(-np.maximum(mix_kl, 0.))))/2
        np.testing.assert_array_less(both, mix + 2e-10)
        np.testing.assert_array_less(labelled_with_prior, mixed_with_prior + 2e-10)
        np.testing.assert_array_less(np.zeros_like(loss_kl), loss_kl + 1e-12)
        cov = chol @ chol.T
        block_diag = np.zeros_like(cov)
        for t in range(k+1):
            sub = slice(t*p.m, (t+1)*p.m)
            block_diag[sub, sub] = cov[sub, sub]
        output[str(k-6)] = {
            "k": k,
            "conditioned_decoder_records": len(decoder),
            "conditional_covariance_off_block_frobenius": float(np.linalg.norm(cov-block_diag)),
            "fixed_start_7_equal_prior": {
                "bayes_error_saved_decoder_average": average(known),
                "tv_saved_decoder_average": average(1-2*known),
                "best_even_decision_error_saved_decoder_average": average(even),
                "even_rule_penalty": average(even-known),
                "kl_saved_decoder_average": average(d7**2/2),
                "population_kl_mean_shift": mean_part/2,
                "population_kl_decoder_variation": random_part/2,
                "population_error_jensen_lower": float(ndtr(-np.sqrt(mean_part+random_part)/2)),
                "constant_mean_decoder_bayes_error": float(ndtr(-np.sqrt(mean_part)/2)),
            },
            "active_start_3_or_7_equal_prior": {
                "labelless_bayes_error_decoder_average": average(mix),
                "labelless_tv_decoder_average": average(1-2*mix),
                "label_known_bayes_error_lower": average(both),
                "error_paid_for_hiding_label": average(mix-both),
                "kl_labelled_average": average(labelled),
                "kl_mixture_average": average(mix_kl),
                "conditional_label_information_nats_average": average(loss_kl),
                "bh_error_lower_decoder_average": average(kl_risk_floor),
                "risk_64_128_max_abs": float(np.max(np.abs(mix-mix_lo_order))),
                "kl_64_128_max_abs": float(np.max(np.abs(mix_kl-kl_low_order))),
            },
            "balanced_event_cohort_active_prior_one_third": {
                "labelless_bayes_error_decoder_average": average(mixed_with_prior),
                "label_known_bayes_error_lower": average(labelled_with_prior),
            },
        }
    distances = np.column_stack(distances)
    first_distance = distances[np.arange(len(decoder)), np.minimum(first, 2)]
    conditioned_no_alarm_floor = ndtr(ndtri(1-p.settings.event_false_alarm)-first_distance)
    for i, row in enumerate(output.values()):
        row["v0_before_bf_declaration_within_prefix"] = {
            "dangerous_decoder_records_by_this_prefix": int(np.sum(first <= i)),
            "empirical_decoder_law_pre_alarm_lower_bound": average(
                (first <= i)*conditioned_no_alarm_floor
            ),
        }
    return output


def posterior_checks(p, trace, attacked_observations, profile):
    """Reweight frozen prefix likelihoods for a sensitivity audit, with no estimator updates."""
    horizon = p.settings.horizon
    p_null = p.settings.no_event_prior
    weight = trace["weights"]
    active_prior = (1-p_null)*np.minimum(np.arange(horizon)+1, p.settings.template_duration)/horizon
    partitions = np.empty((*weight.shape[:2], 4))
    for k in range(horizon):
        begin = max(0, k+1-p.settings.template_duration)
        partitions[:, k] = np.column_stack((
            weight[:, k, 0],
            weight[:, k, k+2:].sum(axis=1),
            weight[:, k, 1+begin:k+2].sum(axis=1),
            weight[:, k, 1:1+begin].sum(axis=1),
        ))
    np.testing.assert_allclose(partitions.sum(axis=-1), 1, atol=3e-13)
    np.testing.assert_allclose(partitions[:, :, 2], trace["p_active"], atol=3e-13)
    tables = {}
    for k in (7,8,9):
        candidate_weights = weight[:, k, 1+np.array(STARTS)]
        odds = candidate_weights / weight[:, k, :1] * p_null*horizon/(1-p_null)
        actual_prefix = p.prefixes[k]
        obs = attacked_observations
        target = obs.target[:, :k+1].reshape(len(obs), -1)
        trusted = obs.trusted[:, :k+1].reshape(len(obs), -1)
        residual = (target - trusted @ actual_prefix.target_from_trusted.T)
        z = solve_triangular(actual_prefix.target_cholesky, residual.T, lower=True).T
        signs = []
        for s in STARTS[:2]:
            terms = prefix_coded_terms(profile, s, k)
            w = whitened_shift(actual_prefix.target_cholesky, obs.decoder[:, :k+1], terms)
            signs.append(np.sum(w*z, axis=1) - np.sum(w**2, axis=1)/2)
        logsum = np.logaddexp(*signs)
        partial = odds[:, :2].sum(axis=1)
        tables[str(k-6)] = {
            "number_of_complete_a0_paths": len(obs),
            "p_active_working_prior": float(active_prior[k]),
            "p_active_original_mean": average(partitions[:, k, 2]),
            "p_null_original_mean": average(partitions[:, k, 0]),
            "p_future_original_mean": average(partitions[:, k, 1]),
            "p_true_start_original_mean": average(weight[:, k, 8]),
            "p_active_reweight_start_3_7_11_same_gaussian_amplitude": average(partial/(4+partial)),
            "p_active_known_start_7_same_gaussian_amplitude": average(odds[:, 1]/(1+odds[:, 1])),
            "p_active_fixed_amplitude_and_start_3_7_11_offline_reference": average(expit(logsum-np.log(4.))),
        }
    return {
        "max_partition_residual": float(np.max(np.abs(partitions.sum(axis=-1)-1))),
        "working_active_prior_when_ten_onsets_are_active": float(active_prior[9]),
        "early_posterior_sensitivity": tables,
    }
