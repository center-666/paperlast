"""Recompute defensive covariance and norm certificates from the frozen model.

The inputs are saved model coefficients and fixed availability cases. The
script evaluates estimator sensitivity and probability bounds. It has no
packet interface, waveform search, or policy selection.
"""

import hashlib
import json
import sys
from math import factorial, isqrt
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REFERENCE = PARENT / "common_g_reference_revision_20261115"
MODEL_FILE = REFERENCE / "inputs" / "five_node_aggregate_model.json"
G3_REGISTRATION = PARENT / "A1_G3_verification_resource_audit_20261117" / "registration.json"
G3_CERTIFICATE = G3_REGISTRATION.with_name("boundary_certificates.json")
CONFIG_FILE = HERE / "registration.json"
ONE = sp.Integer(1)
ZERO = sp.Integer(0)
HALF = sp.Rational(1, 2)
I2 = sp.eye(2)
SCALE = 10**12


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_float=sp.Rational)


def digest(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def sqrt_bounds(value):
    """Outward rational enclosure of a nonnegative square root."""
    value = sp.Rational(value)
    assert value >= 0
    n = isqrt(int(value.p) * SCALE**2 // int(value.q))
    lower, upper = sp.Rational(n, SCALE), sp.Rational(n + 1, SCALE)
    assert lower**2 <= value < upper**2
    return lower, upper


def spectral_bounds(matrix):
    """Eigenvalue enclosures for a symmetric 2 by 2 matrix."""
    assert matrix.shape == (2, 2) and matrix == matrix.T
    trace = sp.trace(matrix)
    _, high = sqrt_bounds((matrix[0, 0] - matrix[1, 1])**2 + 4 * matrix[0, 1]**2)
    return (trace - high) / 2, (trace + high) / 2


def positive_margins(matrix):
    """Sylvester's criterion with exact rational entries."""
    assert matrix == matrix.T
    head, determinant = matrix[0, 0], matrix.det()
    assert head > 0 and determinant > 0
    return {"leading_minor": str(head), "determinant": str(determinant)}


def number(value):
    return {"exact": str(value), "display": float(value)}


def interval(lower, upper):
    lo = sp.floor(lower * SCALE) / SCALE
    hi = sp.ceiling(upper * SCALE) / SCALE
    return {"lower": number(lo), "upper": number(hi)}


def doubled_normal_tail(variance, danger):
    """Outward enclosure of Pr(X^2 >= danger) for X ~ N(0,variance)."""
    z2 = danger / (2 * variance)
    assert 0 < z2 < 1
    # Check the pi bounds by Machin's identity and alternating arctan sums.
    def arctan_sum(d, last):
        return sum((-1)**i * sp.Rational(1, d)**(2 * i + 1) / (2 * i + 1)
                   for i in range(last + 1))

    pi_lower = 16 * arctan_sum(5, 13) - 4 * arctan_sum(239, 12)
    pi_upper = 16 * arctan_sum(5, 12) - 4 * arctan_sum(239, 13)
    factor_lower = 2 * sqrt_bounds(z2 / pi_upper)[0]
    factor_upper = 2 * sqrt_bounds(z2 / pi_lower)[1]

    def erf_sum(last):
        return sum((-z2)**i / (factorial(i) * (2 * i + 1)) for i in range(last + 1))

    erf_lower, erf_upper = factor_lower * erf_sum(13), factor_upper * erf_sum(12)
    return ONE - erf_upper, ONE - erf_lower


def covariance_update(prior, jacobian, noise):
    return jacobian * prior * jacobian.T + noise


def watermark_second_moment(matrix):
    # Gaussian fourth moment for beta ~ N(0, I/2), M = I/2 + beta beta^T.
    return sp.Rational(5, 4) * matrix + sp.trace(matrix) * I2 / 4


def identity_checks(plant, local, target, healthy, gain_local, gain_shared):
    """Check the pathwise difference by subtracting the two original updates."""
    def variables(prefix):
        return sp.Matrix(sp.symbols(f"{prefix}_0:2"))

    initial, displacement = variables("h"), variables("b")
    y_local, y_target, y_healthy, eta = [
        variables(name) for name in ("y1", "y0", "y2", "eta")
    ]
    prior_healthy = plant * initial
    prior_fault = plant * (initial + displacement)

    def step(prior, m, perturbation):
        healthy_innovation = y_healthy - healthy * prior
        target_innovation = y_target + perturbation - target * prior
        return prior + gain_local * (y_local - local * prior) + gain_shared * (
            healthy_innovation + m * target_innovation
        )

    for m in (0, 1):
        transition = (I2 - gain_local * local - gain_shared * (healthy + m * target)) * plant
        nominal = step(prior_healthy, 1, sp.zeros(2, 1))
        difference = step(prior_fault, m, eta) - nominal
        formula = (transition * displacement + m * gain_shared * eta
                   - (1 - m) * gain_shared * (y_target - target * prior_healthy))
        assert (difference - formula).applyfunc(sp.expand) == sp.zeros(2, 1)
    return 2


def envelope(innovation_covariances, health_maximum, energy, count, burst, constants):
    """Uniform second-moment bound with possibly dependent data and masks."""
    length = len(innovation_covariances)
    accepted, missing, g_upper = constants
    transition_bound_squared = ZERO
    for gap in range(length):
        max_losses = min(count, gap - gap // (burst + 1))
        transition_bound_squared += accepted**(2 * (gap - max_losses)) * missing**(2 * max_losses)
    first = sum(ONE / j for j in range(1, length + 1))
    second = sum(ONE / j**2 for j in range(1, length + 1))
    max_watermark_second = (HALF + first)**2 + second
    encoded_input_second = energy * max_watermark_second

    traces = [sp.trace(c) for c in innovation_covariances]
    fourth_sum = sum(sp.trace(c)**2 + 2 * sp.trace(c * c) for c in innovation_covariances)
    missing_second = min(sum(traces), sqrt_bounds(count * fourth_sum)[1]) if count else ZERO
    composite = (sqrt_bounds(encoded_input_second)[1] if energy else ZERO) + (
        sqrt_bounds(missing_second)[1] if count else ZERO
    )
    bias_rms = g_upper * sqrt_bounds(transition_bound_squared)[1] * composite
    actual_upper = (sqrt_bounds(health_maximum)[1] + bias_rms)**2
    return {
        "input_energy_cap": number(energy),
        "missing_count_cap": count,
        "max_consecutive_missing": burst,
        "probability_independence_required_for_masks": False,
        "period_peak_bias_rms_upper_b0_zero": number(bias_rms),
        "actual_pointwise_MSE_upper_b0_zero": number(actual_upper),
        "transition_squared_weight_sum_upper": number(transition_bound_squared),
        "accepted_decoded_disturbance_energy_expectation_upper": number(encoded_input_second),
        "omitted_healthy_innovations_energy_expectation_upper": number(missing_second),
    }


def evaluate(model, data, config, prior_g3):
    receiver = config["receiver"]
    source = config["source"]
    healthy_id = config["healthy_neighbor"]
    assert set(data["neighbors"][receiver]) == {source, healthy_id}
    local, target, healthy = (sp.Matrix(data["H"][j]) for j in (receiver, source, healthy_id))
    r_local, r_target, r_healthy = (sp.Matrix(data["R"][j]) for j in (receiver, source, healthy_id))
    plant = sp.Matrix(model["A"])
    process_noise = sp.Matrix(data["Q"])
    weights = model["nodes"][receiver]
    k_gain, g_gain = sp.Matrix(weights["K"]), sp.Matrix(weights["G"])
    jacobian_0 = I2 - k_gain * local - g_gain * healthy
    jacobian_1 = jacobian_0 - g_gain * target
    f0, f1 = jacobian_0 * plant, jacobian_1 * plant
    noise_0 = k_gain * r_local * k_gain.T + g_gain * r_healthy * g_gain.T
    noise_1 = noise_0 + g_gain * r_target * g_gain.T

    bounds = config["certification_bounds"]
    rho_1 = sp.Rational(bounds["accepted_transition_norm"])
    rho_0 = sp.Rational(bounds["missing_transition_norm"])
    gain_upper = sp.Rational(bounds["common_gain_norm"])
    difference_lower = sp.Rational(bounds["retained_energy_difference_lower"])
    difference = f0.T * f0 - f1.T * f1
    signs = {
        "F0_norm_squared_upper": positive_margins(rho_0**2 * I2 - f0.T * f0),
        "F1_norm_squared_upper": positive_margins(rho_1**2 * I2 - f1.T * f1),
        "G_norm_squared_upper": positive_margins(gain_upper**2 * I2 - g_gain.T * g_gain),
        "retained_energy_difference_lower": positive_margins(difference - difference_lower * I2),
    }
    assert g_gain.det() != 0
    identities = identity_checks(plant, local, target, healthy, k_gain, g_gain)

    nominal_prior = I2
    posteriors, priors = [], []
    for _ in range(config["time_horizon"]):
        priors.append(nominal_prior)
        posterior = covariance_update(nominal_prior, jacobian_1, noise_1)
        posteriors.append(posterior)
        nominal_prior = plant * posterior * plant.T + process_noise
    session_start = config["first_episode_time"]
    session_end = session_start + config["episode_length"]
    comparison = config["reference_fixture"]["comparison_deadline"]
    comparison_prior = priors[comparison]
    nominal_post = posteriors[comparison]
    missing_post = covariance_update(comparison_prior, jacobian_0, noise_0)
    loss_bias_cov = g_gain * (target * comparison_prior * target.T + r_target) * g_gain.T
    nominal_cross = -jacobian_1 * comparison_prior * target.T + g_gain * r_target
    mse_increment = sp.trace(missing_post - nominal_post)
    assert mse_increment == sp.trace(loss_bias_cov) - 2 * sp.trace(nominal_cross * g_gain.T)

    # Mode sensitivity per unit of energy, given an independent inherited
    # distortion component. No direction, waveform or policy is recovered.
    retention_A = watermark_second_moment(g_gain.T * f1.T * f1 * g_gain)
    retention_C = watermark_second_moment(g_gain.T * f0.T * f0 * g_gain)
    retention_increment = retention_C - retention_A
    coefficients = {}
    for label, matrix in [
        ("A_complete_measurement", retention_A),
        ("C_missing_measurement", retention_C),
        ("C_minus_A_excess", retention_increment),
    ]:
        lower, upper = spectral_bounds(matrix)
        assert lower > 0
        coefficients[label] = interval(lower, upper)

    # Separate first-unavailable fixture. It is prior to any corruption.
    danger = sp.Rational(config["paired_estimate_danger_squared"])
    missing_prior = priors[session_start]
    missing_first_cov = g_gain * (target * missing_prior * target.T + r_target) * g_gain.T
    coordinate_event = doubled_normal_tail(missing_first_cov[0, 0], danger)
    alpha = sp.Rational(1, 20)
    first_monitor = 4
    earlier_windows = session_start - first_monitor
    possible_windows = config["time_horizon"] - first_monitor
    risk_lower = coordinate_event[0] - earlier_windows * alpha / possible_windows
    assert risk_lower > 0
    # G3 had certified this coordinate marginal with independent Arb arithmetic.
    g3_tail = prior_g3["drop_coordinate_danger_probability"]
    assert coordinate_event[0] <= sp.Rational(g3_tail["upper"])
    assert coordinate_event[1] >= sp.Rational(g3_tail["lower"])

    energy = sp.Rational(config["coded_energy_upper"])
    count = config["reference_fixture"]["outage_cap"]
    burst = config["reference_fixture"]["consecutive_outage_cap"]
    innovation_covs = [target * prior * target.T + r_target for prior in priors[session_start:session_end]]
    health_maximum = max(map(sp.trace, posteriors))
    constants = (rho_1, rho_0, gain_upper)
    envelopes = {
        "A": envelope(innovation_covs, health_maximum, energy, 0, 0, constants),
        "B": envelope(innovation_covs, health_maximum, ZERO, count, burst, constants),
        "C": envelope(innovation_covs, health_maximum, energy, count, burst, constants),
    }
    return {
        "pathwise_symbolic_mask_checks": identities,
        "exact_positive_margins": signs,
        "two_update_comparison": {
            "deadline": comparison,
            "clean_actual_MSE": float(sp.trace(nominal_post)),
            "B_only_missing_actual_MSE": float(sp.trace(missing_post)),
            "B_only_missing_expected_squared_displacement": float(sp.trace(loss_bias_cov)),
            "B_true_MSE_increment_vs_complete": float(mse_increment),
            "B_cross_term_actual_MSE": float(-2 * sp.trace(nominal_cross * g_gain.T)),
            "retention_coefficients_per_unit_coded_energy": coefficients,
        },
        "first_unavailable_statistical_alarm": {
            "coordinate_variance_exact": str(missing_first_cov[0, 0]),
            "coordinate_danger_probability_enclosure": interval(*coordinate_event),
            "earlier_possible_window_ends": earlier_windows,
            "B_risk_lower_bound_both_detectors": number(sp.floor(risk_lower * SCALE) / SCALE),
            "G3_independent_interval_overlap": True,
            "pure_B_risk_with_availability_alarm": "0",
        },
        "uniform_second_moment_envelopes": envelopes,
    }


def main():
    config, data, prior_g3 = read(CONFIG_FILE), read(MODEL_FILE), read(G3_CERTIFICATE)
    g3_config = read(G3_REGISTRATION)
    assert config["time_horizon"] == g3_config["T"]
    assert config["first_episode_time"] == g3_config["Ta"]
    assert config["episode_length"] == g3_config["L"]
    assert sp.Rational(config["coded_energy_upper"]) == g3_config["E"]
    assert sp.Rational(config["paired_estimate_danger_squared"]) == g3_config["D"]
    assert len(g3_config["window_ends"]) == 28 and g3_config["event_FA"] == sp.Rational(1, 20)
    models = {
        name: evaluate(data["models"][name], data, config, prior_g3["models"][name])
        for name in config["models"]
    }
    inputs = [MODEL_FILE, REFERENCE / "common_g_reference.py", CONFIG_FILE,
              G3_REGISTRATION, G3_CERTIFICATE]
    result = {
        "baseline": config["baseline"],
        "models": models,
        "inputs": {path.relative_to(PARENT).as_posix(): digest(path) for path in inputs},
        "dependencies": {"python": sys.version.split()[0], "sympy": sp.__version__},
        "input_hash_normalization": "UTF-8 with LF newlines",
        "numeric_scope": "exact frozen decimal matrices; outward rational Gaussian and spectral enclosures; floats for display",
        "new_physical_samples": 0,
    }
    (HERE / "boundary_evidence.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for label, evidence in models.items():
        result = evidence["two_update_comparison"]
        gap = result["retention_coefficients_per_unit_coded_energy"]["C_minus_A_excess"]
        risk = evidence["first_unavailable_statistical_alarm"]["B_risk_lower_bound_both_detectors"]
        print(f"{label}: sensitivity excess [{gap['lower']['display']:.8f}, "
              f"{gap['upper']['display']:.8f}], missing-only risk lower {risk['display']:.9f}")


if __name__ == "__main__":
    main()
