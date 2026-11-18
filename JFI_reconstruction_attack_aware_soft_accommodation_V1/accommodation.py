"""Receiver-side finite-episode inference and common-G accommodation.

At update k, the only observed inputs are the local, healthy-neighbor and
target samples and the shared decoder matrix for that timestamp.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import logsumexp
from scipy.stats import chi2


@dataclass(frozen=True)
class Settings:
    horizon: int
    template_duration: int
    coded_energy_reference: float
    receiver: int
    target: int
    healthy: int
    no_event_prior: float
    nominal_one_step_allowance: float
    uncertainty_standard_deviations: float
    event_false_alarm: float

    @classmethod
    def from_registration(cls, record):
        return cls(**{key: record[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Prefix:
    target_from_trusted: np.ndarray
    state_from_trusted: np.ndarray
    target_cholesky: np.ndarray
    state_from_white: np.ndarray
    conditional_state_cov: np.ndarray


class PreparedModel:
    """Append-only prefix likelihoods, using one common process-noise law."""

    def __init__(self, model, settings):
        self.model = model
        self.settings = settings
        self.n = model.A.shape[0]
        self.m = model.H[settings.target].shape[0]
        if set(model.neighbors[settings.receiver]) != {settings.target, settings.healthy}:
            raise ValueError("The declared receiver must have the target and healthy neighbors.")
        if (not 1 <= settings.template_duration <= settings.horizon
                or settings.coded_energy_reference <= 0
                or not 0 < settings.no_event_prior < 1
                or not 0 < settings.event_false_alarm < 1
                or settings.nominal_one_step_allowance < 0
                or settings.uncertainty_standard_deviations < 0):
            raise ValueError("Invalid finite-episode design parameters.")

        self.prefixes = []
        self._prepare_prefixes()
        self.template = np.stack([
            model.H[settings.target] @ np.linalg.matrix_power(model.A, lag)
            for lag in range(settings.template_duration)
        ])
        rows = self.template.reshape(-1, self.n)
        energy = rows.T @ rows
        if np.linalg.matrix_rank(energy) != self.n:
            raise ValueError("This implementation requires a full-rank supported template.")
        parameter_prior = (settings.coded_energy_reference / self.n) * np.linalg.solve(
            energy, np.eye(self.n)
        )
        self.parameter_prior_cholesky = np.linalg.cholesky(parameter_prior)
        self.active_test_count = sum(
            min(k + 1, settings.template_duration) for k in range(settings.horizon)
        )

    def _prepare_prefixes(self):
        model, s = self.model, self.settings
        n, m, t = self.n, self.m, s.horizon
        dimension = t * (n + 3 * m)
        x = np.zeros((n, dimension))
        x[:, :n] = np.eye(n)
        states, local_rows, healthy_rows, target_rows = [], [], [], []
        measurement_sqrt = [
            np.linalg.cholesky(model.R[j]) for j in (s.receiver, s.healthy, s.target)
        ]
        process_sqrt = np.linalg.cholesky(model.Q)
        previous_reference = np.zeros_like(x)
        reference_error, reference_prior, reference_posterior = [], [], []
        k_gain, g_gain = model.K[s.receiver], model.G[s.receiver]
        j_ref = np.eye(n) - k_gain @ model.H[s.receiver] - g_gain @ model.H[s.healthy]
        for k in range(t):
            states.append(x)
            rows = []
            for i, j in enumerate((s.receiver, s.healthy, s.target)):
                obs = model.H[j] @ x
                start = t * n + (3 * k + i) * m
                obs[:, start:start + m] = measurement_sqrt[i]
                rows.append(obs)
            local, healthy, target = rows
            local_rows.append(local)
            healthy_rows.append(healthy)
            target_rows.append(target)
            prior = model.A @ previous_reference
            reference_prior.append(prior)
            reference_error.append(prior - x)
            previous_reference = j_ref @ prior + k_gain @ local + g_gain @ healthy
            reference_posterior.append(previous_reference)
            if k + 1 < t:
                x = model.A @ x
                x[:, (k + 1) * n:(k + 2) * n] = process_sqrt

        self.state_maps = np.stack(states)
        self.trusted_maps = np.stack([np.vstack(z) for z in zip(local_rows, healthy_rows)])
        self.target_maps = np.stack(target_rows)
        self.reference_prior_maps = np.stack(reference_prior)
        self.reference_post_maps = np.stack(reference_posterior)
        e_minus = np.stack(reference_error)
        self.reference_error_covariance = e_minus @ e_minus.swapaxes(1, 2)
        self.reference_target_cross = np.einsum("knd,tmd->ktnm", e_minus, self.target_maps)

        for k in range(t):
            u = self.trusted_maps[:k + 1].reshape(-1, dimension)
            q = self.target_maps[:k + 1].reshape(-1, dimension)
            x = self.state_maps[k]
            chol_u = np.linalg.cholesky(u @ u.T)
            q_gain = cho_solve((chol_u, True), u @ q.T).T
            x_gain = cho_solve((chol_u, True), u @ x.T).T
            q_res = q - q_gain @ u
            x_res = x - x_gain @ u
            chol_q = np.linalg.cholesky(q_res @ q_res.T)
            white_q_map = solve_triangular(chol_q, q_res, lower=True)
            x_on_white = x_res @ white_q_map.T
            x_clean = x_res - x_on_white @ white_q_map
            self.prefixes.append(Prefix(q_gain, x_gain, chol_q, x_on_white, x_clean @ x_clean.T))

    def nominal_mse(self, neighbor_scale=1.0, target_accepted=True):
        model, s = self.model, self.settings
        ids = list(model.neighbors[s.receiver]) if target_accepted else [s.healthy]
        k = model.K[s.receiver]
        g = neighbor_scale * model.G[s.receiver]
        h_sum = sum(model.H[j] for j in ids)
        r_sum = sum(model.R[j] for j in ids)
        jac = np.eye(self.n) - k @ model.H[s.receiver] - g @ h_sum
        noise = k @ model.R[s.receiver] @ k.T + g @ r_sum @ g.T
        prior = np.eye(self.n)
        trace = []
        for _ in range(s.horizon):
            posterior = jac @ prior @ jac.T + noise
            trace.append(np.trace(posterior))
            prior = model.A @ posterior @ model.A.T + model.Q
        return np.array(trace)


@dataclass(frozen=True)
class TemplateFit:
    log_bf: float
    delta: np.ndarray
    delta_cov: np.ndarray
    state_shift: np.ndarray
    state_shift_cov: np.ndarray
    gls_delta: np.ndarray
    matched_pvalue: float
    rank: int


@dataclass(frozen=True)
class Evidence:
    timestamp: int
    reference_prior: np.ndarray
    reference_posterior: np.ndarray
    reference_residual: np.ndarray
    conditional_residual: np.ndarray
    weights: np.ndarray
    delta: np.ndarray
    delta_cov: np.ndarray
    nominal_state_mean: np.ndarray
    nominal_state_cov: np.ndarray
    mixture_state_mean: np.ndarray
    state_mean_uncertainty: np.ndarray
    posterior_state_cov: np.ndarray
    log_event_bf: float
    mixture_alarm: bool
    matched_alarm: bool
    matched_start: int
    fits: list


class OnlineEvidence:
    def __init__(self, prepared, common_step):
        self.prepared = prepared
        self.common_step = common_step
        self.k = 0
        self.reference = np.zeros(prepared.n)
        t, m = prepared.settings.horizon, prepared.m
        self.trusted = np.zeros((t, 2 * m))
        self.target = np.zeros((t, m))
        self.decoders = np.zeros((t, m, m))

    def _fit(self, start, prefix, white):
        p, k = self.prepared, self.k
        stop = min(k + 1, start + p.settings.template_duration)
        design = np.zeros((k + 1, p.m, p.n))
        design[start:stop] = self.decoders[start:stop] @ p.template[:stop - start]
        design = design.reshape(-1, p.n)
        scaled = solve_triangular(
            prefix.target_cholesky, design @ p.parameter_prior_cholesky, lower=True
        )
        left, singular, _ = np.linalg.svd(scaled, full_matrices=False)
        coefficients = left.T @ white
        squared = singular**2
        shrink = squared / (1 + squared)
        mean = left @ (shrink * coefficients)
        spread = left * (singular / np.sqrt(1 + squared))
        delta_map = prefix.target_cholesky[-p.m:]
        delta_sqrt = delta_map @ spread
        state_sqrt = prefix.state_from_white @ spread
        rank = int(np.count_nonzero(singular > np.finfo(float).eps * max(scaled.shape) * singular[0]))
        projection = left[:, :rank] @ coefficients[:rank]
        return TemplateFit(
            log_bf=float(0.5 * (shrink @ coefficients**2 - np.sum(np.log1p(squared)))),
            delta=delta_map @ mean,
            delta_cov=delta_sqrt @ delta_sqrt.T,
            state_shift=prefix.state_from_white @ mean,
            state_shift_cov=state_sqrt @ state_sqrt.T,
            gls_delta=delta_map @ projection,
            matched_pvalue=float(chi2.sf(coefficients[:rank] @ coefficients[:rank], rank)) if rank else 1.0,
            rank=rank,
        )

    def observe(self, local, healthy, target, decoder):
        p, s, k = self.prepared, self.prepared.settings, self.k
        if k >= s.horizon:
            raise ValueError("Configured finite episode already ended.")
        local, healthy, target = [np.asarray(value, dtype=float) for value in (local, healthy, target)]
        decoder = np.asarray(decoder, dtype=float)
        if any(value.shape != (p.m,) for value in (local, healthy, target)) or decoder.shape != (p.m, p.m):
            raise ValueError("Unexpected measurement or decoder shape.")
        self.trusted[k] = np.r_[local, healthy]
        self.target[k] = target
        self.decoders[k] = decoder
        if not all(np.all(np.isfinite(v)) for v in (self.trusted[k], self.target[k], self.decoders[k])):
            raise ValueError("This protocol requires a finite synchronized packet triple.")
        model = p.model
        prior, reference_post = self.common_step(
            self.reference, local, [healthy], model.A, model.H[s.receiver],
            [model.H[s.healthy]], model.K[s.receiver], model.G[s.receiver]
        )

        prefix = p.prefixes[k]
        trusted = self.trusted[:k + 1].reshape(-1)
        nominal_target = prefix.target_from_trusted @ trusted
        innovation = self.target[:k + 1].reshape(-1) - nominal_target
        white = solve_triangular(prefix.target_cholesky, innovation, lower=True)
        nominal_state = prefix.state_from_trusted @ trusted + prefix.state_from_white @ white
        fits = [self._fit(start, prefix, white) for start in range(k + 1)]
        log_bfs = np.zeros(s.horizon)
        log_bfs[:k + 1] = [fit.log_bf for fit in fits]
        log_event_bf = float(logsumexp(log_bfs) - np.log(s.horizon))
        log_weights = np.r_[
            np.log(s.no_event_prior), np.log((1 - s.no_event_prior) / s.horizon) + log_bfs
        ]
        weights = np.exp(log_weights - logsumexp(log_weights))
        active_weights = weights[1:k + 2]
        blank_weight = weights[0] + weights[k + 2:].sum()
        delta = sum(w * f.delta for w, f in zip(active_weights, fits))
        shift = sum(w * f.state_shift for w, f in zip(active_weights, fits))
        delta_cov = blank_weight * np.outer(delta, delta)
        state_spread = blank_weight * np.outer(shift, shift)
        for w, fit in zip(active_weights, fits):
            dd = fit.delta - delta
            dx = fit.state_shift - shift
            delta_cov += w * (fit.delta_cov + np.outer(dd, dd))
            state_spread += w * (fit.state_shift_cov + np.outer(dx, dx))

        first_active = max(0, k + 1 - s.template_duration)
        matched_start = min(range(first_active, k + 1), key=lambda start: fits[start].matched_pvalue)
        matched_alarm = fits[matched_start].matched_pvalue <= s.event_false_alarm / p.active_test_count
        evidence = Evidence(
            k, prior, reference_post, target - model.H[s.target] @ prior,
            innovation[-p.m:].copy(), weights, delta, delta_cov, nominal_state,
            prefix.conditional_state_cov, nominal_state - shift, state_spread,
            prefix.conditional_state_cov + state_spread, log_event_bf,
            bool(log_event_bf >= -np.log(s.event_false_alarm)), bool(matched_alarm), matched_start, fits
        )
        self.reference = reference_post
        self.k += 1
        return evidence


class CommonGPath:
    """A single actual closed-loop trajectory, with one shared neighbor gain."""

    def __init__(self, prepared, common_step, neighbor_scale=1.0):
        self.prepared = prepared
        self.common_step = common_step
        self.neighbor_scale = neighbor_scale
        self.posterior = np.zeros(prepared.n)

    def preview(self, local, healthy, target, *, accept_target=True):
        p = self.prepared
        model, s = p.model, p.settings
        measurements, matrices = [healthy], [model.H[s.healthy]]
        if accept_target:
            measurements.append(target)
            matrices.append(model.H[s.target])
        return self.common_step(
            self.posterior, local, measurements, model.A, model.H[s.receiver],
            matrices, model.K[s.receiver], self.neighbor_scale * model.G[s.receiver]
        )[1]

    def update(self, local, healthy, target, *, accept_target=True):
        self.posterior = self.preview(local, healthy, target, accept_target=accept_target)
        return self.posterior


@dataclass(frozen=True)
class WeightDecision:
    alpha: float
    risk_without_correction: float
    nominal_risk_increase: float
    robust_model_risk_increase: float
    posterior_mean_weight: float = 1.0
    uncertainty_limited_weight: float = 1.0
    nominal_weight_limit: float = 1.0


def choose_soft_weight(uncorrected, evidence, gain, settings):
    direction = gain @ evidence.delta
    squared = float(direction @ direction)
    nominal_error = uncorrected - evidence.nominal_state_mean
    nominal_risk = float(nominal_error @ nominal_error + np.trace(evidence.nominal_state_cov))
    if squared <= 1e-24:
        return WeightDecision(0.0, nominal_risk, 0.0, 0.0, 0.0, 0.0)

    nominal_linear = float(direction @ nominal_error)
    mixture_linear = float(direction @ (uncorrected - evidence.mixture_state_mean))
    sigma = np.sqrt(max(0.0, direction @ evidence.state_mean_uncertainty @ direction))
    conservative_linear = mixture_linear - settings.uncertainty_standard_deviations * sigma
    budget = settings.nominal_one_step_allowance * nominal_risk
    root = np.sqrt(nominal_linear**2 + squared * budget)
    if nominal_linear < 0:
        safe_alpha = budget / (root - nominal_linear)
    else:
        safe_alpha = (nominal_linear + root) / squared
    mean_weight = float(np.clip(mixture_linear / squared, 0.0, 1.0))
    uncertainty_weight = float(np.clip(conservative_linear / squared, 0.0, 1.0))
    nominal_limit = float(min(1.0, safe_alpha))
    alpha = min(uncertainty_weight, nominal_limit)
    return WeightDecision(
        alpha, nominal_risk, squared * alpha**2 - 2 * alpha * nominal_linear,
        squared * alpha**2 - 2 * alpha * conservative_linear,
        mean_weight, uncertainty_weight, nominal_limit
    )


@dataclass(frozen=True)
class Update:
    posterior: np.ndarray
    evidence: Evidence
    weight: WeightDecision


class OnlineSoftCompensator:
    """Public API. Call step once per received packet triple."""

    def __init__(self, prepared, common_step, full_compensation=False):
        self.observer = OnlineEvidence(prepared, common_step)
        self.path = CommonGPath(prepared, common_step)
        self.full_compensation = full_compensation

    def step(self, local, healthy, target, decoder):
        evidence = self.observer.observe(local, healthy, target, decoder)
        p = self.observer.prepared
        uncorrected = self.path.preview(local, healthy, target)
        if self.full_compensation:
            decision = WeightDecision(1.0, float("nan"), float("nan"), float("nan"))
        else:
            decision = choose_soft_weight(uncorrected, evidence, p.model.G[p.settings.receiver], p.settings)
        corrected = target - decision.alpha * evidence.delta
        posterior = self.path.update(local, healthy, corrected)
        return Update(posterior, evidence, decision)
