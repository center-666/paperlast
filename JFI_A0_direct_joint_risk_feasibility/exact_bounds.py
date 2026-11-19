"""Exact inequalities for the fixed-template Gaussian early-risk certificate."""

from fractions import Fraction
from math import factorial, isqrt
import json

import sympy as sp


START = 7
LAST = 9
DANGER = Fraction(1, 10)
RADIUS_FOR_QUANTITATIVE_LOWER = Fraction(17, 50)
GRID_DEN = 40
EXP_STEPS = 128


def fraction(value):
    value = sp.Rational(value)
    return Fraction(int(value.p), int(value.q))


def matrix(rows):
    return sp.Matrix([[sp.Rational(x) for x in row] for row in rows])


def positive_2(mat):
    return mat.shape == (2, 2) and mat == mat.T and mat[0, 0] > 0 and mat.det() > 0


def sqrt_upper(x):
    scale = 10**7
    x = fraction(x) * scale**2
    return Fraction(isqrt(x.numerator // x.denominator) + 1, scale)


def decimal_bound(value, places, upper=False):
    value = fraction(value) * 10**places
    rounded = (-(-value.numerator//value.denominator) if upper
               else value.numerator//value.denominator)
    return float(Fraction(rounded, 10**places))


def exp_negative_upper(x):
    return 1 / sum(x**j / factorial(j) for j in range(14))


def exp_negative_lower(x):
    return (1-x/EXP_STEPS)**EXP_STEPS


def exact_stages(path, model, registration):
    source = json.loads(path.read_text(encoding="utf-8"), parse_float=str)
    system = source["models"][model]
    target, local, neighbor = (registration[x] for x in ("target", "receiver", "healthy"))
    h = [matrix(row) for row in source["H"]]
    r = [matrix(row) for row in source["R"]]
    a, q = matrix(system["A"]), matrix(source["Q"])
    g = matrix(system["nodes"][local]["G"])
    gain = matrix(system["nodes"][local]["K"])
    if len(source["neighbors"][local]) != 2 or set(source["neighbors"][local]) != {target, neighbor}:
        raise ValueError("Unexpected common-G neighbor set.")
    identity = sp.eye(2)
    correction = identity-gain*h[local]-g*(h[target]+h[neighbor])
    fixed_f = correction*a
    common_noise = gain*r[local]*gain.T+g*(r[target]+r[neighbor])*g.T
    info_t = h[local].T*r[local].inv()*h[local]+h[neighbor].T*r[neighbor].inv()*h[neighbor]
    info_y = h[target].T*r[target].inv()*h[target]
    trust_prior = full_prior = common_prior = identity
    full_trace = gap_trace = common_trace = sp.S(0)
    stages = []
    horizon = registration["horizon"]
    for k in range(horizon):
        pt = (trust_prior.inv()+info_t).inv()
        ph = (full_prior.inv()+info_t).inv()
        pf = (ph.inv()+info_y).inv()
        p0 = correction*common_prior*correction.T+common_noise
        full_trace += sp.trace(pf)
        gap_trace += sp.trace(pt-pf)
        common_trace += sp.trace(p0)
        if START <= k <= LAST:
            assert positive_2(ph) and positive_2(pt-pf) and h[target].det() != 0
            conditional = h[target]*ph*h[target].T+r[target]
            ky = ph*h[target].T*conditional.inv()
            stages.append({
                "trusted": pt,
                "before_target": ph,
                "full": pf,
                "target_innovation": conditional,
                "gain": ky,
                "state_transition": pf*full_prior.inv()*a,
                "target_given_trusted": h[target]*pt*h[target].T+r[target],
            })
        trust_prior = a*pt*a.T+q
        full_prior = a*pf*a.T+q
        common_prior = a*p0*a.T+q
    return stages, g, fixed_f, tuple(fraction(x/horizon) for x in (full_trace,gap_trace,common_trace))


def first_packet_tests(row, g, first, factor_squared):
    ky, v, ph, pt, pf = (row[k] for k in
                          ("gain", "target_innovation", "before_target", "trusted", "full"))
    d_h = fraction((first.T*ky.T*ky*first)[0])*factor_squared
    d_z = fraction((first.T*v.inv()*first)[0])*factor_squared
    d_g = fraction((first.T*g.T*g*first)[0])*factor_squared
    # Exact strict margins at the saturation zero/positive-risk boundary.
    assert d_h/4 < DANGER and DANGER*d_z < 4*d_h
    assert d_z/4 < Fraction(25, 4)
    assert d_g > Fraction(21,100)**2*d_z
    assert positive_2(sp.Rational(9,40)**2*sp.eye(2)-g*row["target_given_trusted"]*g.T)
    assert Fraction(5,2)*(Fraction(21,100)-Fraction(3,10)*Fraction(9,40)) > Fraction(317,1000)
    # Exact variance and gain envelopes for the quantitative slab bound.
    assert positive_2(sp.Rational(31,1000)**2*sp.eye(2)-(pt-ph))
    assert positive_2(sp.Rational(1,5)**2*sp.eye(2)-(ph-pf))
    return {
        "state_gap_at_lambda_squared_upper": decimal_bound(d_h/4, 10, upper=True),
        "whitened_length_squared_at_state_threshold_upper": decimal_bound(DANGER*d_z/d_h, 9, upper=True),
        "history_gap_standard_deviation_upper": 0.031,
        "conditional_full_mean_map_norm_upper": 0.2,
        "v1a_midpoint_displacement_strict_lower": 0.35625,
    }


def coefficient_intervals(gram):
    """Return exact integer intervals for a quadratic in u,v."""
    scale = 10**12
    terms = [fraction(x)*scale for x in (gram[0,0], 2*gram[0,1], gram[1,1])]
    return [(t.numerator//t.denominator, -(-t.numerator//t.denominator)) for t in terms], scale


def product_interval(one, two):
    values = [x*y for x in one for y in two]
    return min(values), max(values)


def quadratic_interval(bounds, u, v):
    ul, uh = u
    vl, vh = v
    ss = [(ul**2, uh**2), product_interval(u, v),
          (0 if vl <= 0 <= vh else min(vl**2, vh**2), max(vl**2, vh**2))]
    total = [product_interval(b, s) for b,s in zip(bounds, ss, strict=True)]
    return sum(x[0] for x in total), sum(x[1] for x in total)


def first_packet_slab_certificate(row, g, first, factor_squared):
    rot = sp.Matrix([[0,-1],[1,0]])
    basis = first.row_join(rot*first)
    state_columns = row["gain"]*basis
    state, scale = coefficient_intervals(factor_squared*state_columns.T*state_columns)
    v0, _ = coefficient_intervals(factor_squared*basis.T*g.T*g*basis)
    white, _ = coefficient_intervals(factor_squared*basis.T*row["target_innovation"].inv()*basis)
    den = GRID_DEN
    coef_den = 2*den**2
    divisor = scale * coef_den**2
    numerator = 0
    numerator_small = 0
    numerator_v0 = 0
    accepted = 0
    accepted_small = 0
    base = EXP_STEPS*den**2
    # A fixed grid, reflected under (x,y)->(-x,-y). Integrate only proven cells.
    for i in range(2*den):
        u = (den**2+2*i*i, den**2+2*(i+1)**2)
        for j in range(-3*den//2, 3*den//2):
            v = product_interval((i, i+1), (2*j, 2*(j+1)))
            _, z_hi = quadratic_interval(white, u, v)
            if 100*z_hi > 441*divisor:
                continue
            max_radius2_numerator = (i+1)**2+max(j*j, (j+1)**2)
            density_numerator = (base-max_radius2_numerator)**EXP_STEPS
            h_lo, h_hi = quadratic_interval(state, u, v)
            if 10*h_lo > divisor and 25*h_hi <= 4*divisor:
                numerator += density_numerator
                accepted += 1
                if 10000*h_hi <= 1089*divisor:
                    numerator_small += density_numerator
                    accepted_small += 1
            if 10*quadratic_interval(v0, u, v)[0] > divisor:
                numerator_v0 += density_numerator
    mass_scale = Fraction(7,22)*Fraction(2,den**2*base**EXP_STEPS)
    def certificate_for_box(mass, cells, radius, state_max, history_max, white_jitter):
        history = 1-exp_negative_upper((history_max/Fraction(31,1000))**2/2)-Fraction(1,20)-Fraction(1,4)
        exponent = (Fraction(21,20)+white_jitter)**2/2
        innovation = white_jitter**2/2*exp_negative_lower(exponent)
        lower = mass*history*innovation
        assert lower > 0 and exponent < 1
        assert history_max+white_jitter/5+state_max/2 < radius
        return {
            "radius_minimum": float(radius),
            "grid_accepted_cells_before_sign_reflection": cells,
            "decoder_probability_lower_down": decimal_bound(mass, 10),
            "history_no_alarm_and_bf_guard_probability_lower_down": decimal_bound(history, 10),
            "current_innovation_probability_lower_down": decimal_bound(innovation, 10),
            "strict_first_packet_risk_lower_down": decimal_bound(lower, 11),
        }

    # Mills' bound certifies that the 95% standard-normal quantile exceeds 1.5.
    assert exp_negative_lower(Fraction(9,8))*Fraction(6,13)/sqrt_upper(Fraction(44,7)) > Fraction(1,20)
    assert numerator_v0 > 0
    return {
        "saturation_radius_at_least_0p34": certificate_for_box(
            mass_scale*numerator, accepted, RADIUS_FOR_QUANTITATIVE_LOWER, Fraction(2,5),
            Fraction(65,1000), Fraction(7,20)),
        "saturation_radius_at_least_0p24": certificate_for_box(
            mass_scale*numerator_small, accepted_small, Fraction(6,25), Fraction(33,100),
            Fraction(55,1000), Fraction(7,100)),
        "v0_decoder_probability_lower_down": decimal_bound(mass_scale*numerator_v0, 10),
        "v0_first_packet_risk_lower_down": decimal_bound(mass_scale*numerator_v0/4, 11),
    }


def propagation_matrices(stages, g, fixed_f):
    saturation, v0 = [], []
    for age, row in enumerate(stages):
        propagation = sp.eye(2)
        row_maps = [row["gain"]]
        for j in range(age-1, -1, -1):
            propagation = propagation*stages[j+1]["state_transition"]
            row_maps.append(propagation*stages[j]["gain"])
        saturation.append(row_maps[::-1])
        v0.append([fixed_f**(age-j)*g for j in range(age+1)])
    return saturation, v0


def bounded_decoder_upper(maps, profile, factor_squared):
    """Uniform exclusion of danger on a positive-probability small-decoder set."""
    caps = []
    for age, gains in enumerate(maps):
        terms = list(zip(gains, profile[:age+1], strict=True))
        mean = sum((b*a/2 for b,a in terms), sp.zeros(2,1))
        broad = sum(sqrt_upper((b.T*b).trace())*sqrt_upper(factor_squared*(a.T*a)[0])
                    for b,a in terms)
        budget = (Fraction(316227,10**6)-sqrt_upper(factor_squared*(mean.T*mean)[0]))/broad
        assert budget > 0
        caps.append(budget)
    answer = {}
    for j in range(3):
        cap = min(caps[:j+1])
        bound = cap.numerator*1000//cap.denominator
        allowed = Fraction(bound,1000)
        safe = (1-exp_negative_upper(allowed))**(j+1)
        answer[str(j+1)] = {
            "squared_beta_norm_cap": float(allowed),
            "unconditional_risk_upper_up": decimal_bound(1-safe, 9, upper=True),
        }
    return answer


def ellipse_constants(stages, maps, profile, factor_squared):
    sigma = Fraction(4,25)
    eye = sp.eye(2)
    coefficient = Fraction(0)
    bounds = {}
    for age, (row, gains) in enumerate(zip(stages, maps, strict=True)):
        assert positive_2(row["trusted"]-row["full"]-sigma**2*eye)
        terms = list(zip(gains, profile[:age+1], strict=True))
        # With lambda=q=1/2, the decoder mean is I and Cov(Ma)=(|a|²I+aaᵀ)/4.
        mean = sum((b*a for b,a in terms), sp.zeros(2,1))
        covariance = sum((b*((a.T*a)[0]*eye+a*a.T)*b.T/4 for b,a in terms), sp.zeros(2))
        second_moment = factor_squared*((mean.T*mean)[0]+sp.trace(covariance))
        coefficient += Fraction(4,5)/sigma*sqrt_upper(second_moment)
        bounds[str(age+1)] = {
            "state_shift_second_moment_upper_up": decimal_bound(second_moment, 9, upper=True),
            "coefficient_upper_up": decimal_bound(coefficient, 7, upper=True),
        }
    return bounds


def health_budget_test(health):
    """Lower-bound J(0.24) by a rotational Jensen bound and rational tail sums."""
    full, gap, common_g = health
    sigma = Fraction(91,500)
    r = Fraction(6,25)
    assert gap/2 > sigma**2
    subdivisions = 64
    excess = Fraction(0)
    for j in range(1,6*subdivisions+1):
        x = r/sigma+Fraction(j,subdivisions)
        excess += sigma**2*Fraction(2*j-1,subdivisions**2)*exp_negative_lower(x*x/2)
    cost = full+excess
    cap = Fraction(51,50)*common_g
    assert cost > cap
    return {
        "radius": float(r),
        "nominal_32_round_mse_lower_down": decimal_bound(cost, 10),
        "nominal_2pct_cap_upper_up": decimal_bound(cap, 10, upper=True),
        "margin_lower_down": decimal_bound(cost-cap, 10),
        "isotropic_standard_deviation_lower": float(sigma),
        "at_most_2pct_implies_radius_strictly_above_0p24": True,
    }


def certificate(path, model, registration, fixed_profile, normalization_squared=Fraction(1)):
    stages, g, fixed_f, health = exact_stages(path, model, registration)
    profile = [sp.Matrix([sp.Rational(x) for x in row]) for row in fixed_profile]
    first = stages[0]
    sat_maps, v0_maps = propagation_matrices(stages, g, fixed_f)
    slab = first_packet_slab_certificate(first,g,profile[0],normalization_squared)
    return {
        "checked_first_packet_conditions": first_packet_tests(first,g,profile[0],normalization_squared),
        "first_packet_probability_certificate": slab,
        "saturation_upper_bound_all_radii": bounded_decoder_upper(sat_maps,profile,normalization_squared),
        "saturation_ellipse_upper_bound": ellipse_constants(stages,sat_maps,profile,normalization_squared),
        "v0_upper_bound": bounded_decoder_upper(v0_maps,profile,normalization_squared),
        "health_budget_radius_certificate": health_budget_test(health),
    }
