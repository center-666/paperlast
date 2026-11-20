"""Exact rational checks of new B1 formulas; no simulated physical paths."""

from fractions import Fraction as F
from math import factorial
from pathlib import Path
import json


def add(p, q):
    out = dict(p)
    for key, value in q.items():
        out[key] = out.get(key, F(0)) + value
    return {key: value for key, value in out.items() if value}


def scale(p, a):
    return {key: value * a for key, value in p.items() if value * a}


def mul(p, q):
    out = {}
    for kp, vp in p.items():
        for kq, vq in q.items():
            key = tuple(a + b for a, b in zip(kp, kq))
            out[key] = out.get(key, F(0)) + vp * vq
    return {key: value for key, value in out.items() if value}


def moment(p, radial_scale):
    result = F(0)
    for (c, s, x), value in p.items():
        # All polynomials below have even total angular degree.
        assert (c + s) % 2 == 0
        if c % 2 or s % 2:
            continue
        a, b = c // 2, s // 2
        angular = F(factorial(2 * a) * factorial(2 * b),
                    4 ** (a + b) * factorial(a) * factorial(b) * factorial(a + b))
        radial = factorial(x) * (2 * radial_scale) ** x
        result += value * angular * radial
    return result


ONE = {(0, 0, 0): F(1)}
C4 = {(4, 0, 0): F(1), (2, 2, 0): F(-6), (0, 4, 0): F(1)}
S4 = {(3, 1, 0): F(4), (1, 3, 0): F(-4)}


def quadratic_response(lam, qw, d, isotropic, u, w):
    c_projection = {(2, 0, 1): d[0], (1, 1, 1): d[1]}
    s_projection = {(1, 1, 1): d[0], (0, 2, 1): d[1]}
    z0 = add(scale(ONE, lam * d[0]), c_projection)
    z1 = add(scale(ONE, lam * d[1]), s_projection)
    polynomial = add(add(scale(mul(z0, z0), isotropic + u),
                         scale(mul(z0, z1), 2 * w)),
                     scale(mul(z1, z1), isotropic - u))
    actual = [moment(mul(polynomial, harmonic), qw) for harmonic in (C4, S4)]
    e_x2_over_8 = qw ** 2
    expected = [e_x2_over_8 * (u * (d[0] ** 2 - d[1] ** 2) - 2 * w * d[0] * d[1]),
                e_x2_over_8 * (w * (d[0] ** 2 - d[1] ** 2) + 2 * u * d[0] * d[1])]
    assert actual == expected
    assert sum(z * z for z in actual) == qw ** 4 * (u * u + w * w) * sum(z * z for z in d) ** 2
    return polynomial, actual


def main():
    checks = []
    for d, u, w in [((F(2, 3), F(-5, 4)), F(3, 7), F(-2, 5)),
                    ((F(1), F(0)), F(0), F(1)),
                    ((F(-2), F(3)), F(-4), F(2))]:
        _, response = quadratic_response(F(1, 2), F(1, 2), d, F(5), u, w)
        checks.append({"check": "fourth_harmonic_and_norm", "response": list(map(str, response))})

    # Compare independent product expansion against the proposed quadratic form.
    eta0, eta1, eta2 = F(2, 7), F(-1, 13), F(3, 29)
    h0 = F(5, 3)
    a, b = (F(2), F(-1, 2)), (F(4, 5), F(-2))
    sinv = ((F(2), F(1, 3)), (F(1, 3), F(3)))
    kbk = ((F(5), F(-2, 3)), (F(-2, 3), F(4)))
    for t in [(F(1), F(2)), (F(-2, 3), F(7, 5)), (F(0), F(3))]:
        at = sum(a[i] * t[i] for i in range(2))
        bt = sum(b[i] * t[i] for i in range(2))
        tq = lambda mat: sum(t[i] * mat[i][j] * t[j] for i in range(2) for j in range(2))
        independent = eta0 * tq(kbk) - 4 * eta1 * at * bt + h0 * eta1 * tq(sinv) + 2 * h0 * eta2 * at ** 2
        qtask = [[eta0 * kbk[i][j] + eta1 * h0 * sinv[i][j]
                  - 2 * eta1 * (a[i] * b[j] + b[i] * a[j])
                  + 2 * eta2 * h0 * a[i] * a[j] for j in range(2)] for i in range(2)]
        assert independent == tq(qtask)
    checks.append({"check": "Q_task_product_expansion", "status": "exact"})

    isotropic, _ = quadratic_response(F(1, 2), F(1, 2), (F(1), F(0)), F(1), F(0), F(0))
    assert moment(mul(isotropic, C4), F(1, 2)) == 0
    assert moment(mul(isotropic, S4), F(1, 2)) == 0
    beta2 = F(8) * F(1, 2) ** 4 + F(48) * F(1, 2) ** 4 + F(96) * F(1, 2) ** 4
    assert beta2 == F(19, 2)
    assert moment(mul(mul(isotropic, isotropic), C4), F(1, 2)) == beta2 / 4
    checks.append({"check": "equal_H_positive_example", "E_beta_squared_unit_direction": str(beta2),
                   "fourth_order_gain_coefficient_pi_half_g_sigma2_one": str(beta2 / 64)})

    # Posterior mass identity and operational frozen-weight estimator baseline.
    priors = [F(1, 3), F(2, 3)]
    probabilities = [F(1, 4), F(3, 4)]
    posteriors = [[F(2, 3), F(1, 3)], [F(2, 9), F(7, 9)]]
    targets = [[F(1), F(-2)], [F(3), F(1)]]
    mean = lambda weights: [sum(weights[i] * targets[i][j] for i in range(2)) for j in range(2)]
    norm2 = lambda z: sum(x * x for x in z)
    prior_mean = mean(priors)
    ambiguity_before = sum(priors[i] * norm2([targets[i][j] - prior_mean[j] for j in range(2)]) for i in range(2))
    ambiguity_after = F(0)
    information = F(0)
    for mass, posterior in zip(probabilities, posteriors):
        posterior_mean = mean(posterior)
        ambiguity_after += mass * sum(posterior[i] * norm2([targets[i][j] - posterior_mean[j] for j in range(2)]) for i in range(2))
        information += mass * norm2([posterior_mean[j] - prior_mean[j] for j in range(2)])
    assert all(sum(mass * posterior[i] for mass, posterior in zip(probabilities, posteriors)) == priors[i] for i in range(2))
    assert ambiguity_before == ambiguity_after + information
    checks.append({"check": "finite_hypothesis_total_variance", "before": str(ambiguity_before),
                   "after": str(ambiguity_after), "information": str(information)})

    result = {"arithmetic": "Python standard-library exact Fractions", "physical_paths_generated": 0,
              "scope": "Algebra checks only; no frozen-system or estimator-performance claim", "checks": checks}
    target = Path(__file__).with_name("algebra_checks.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
