"""Check three exact same-resource comparisons in a one-observation Gaussian model."""

import json
from pathlib import Path

import sympy as sp


def compare(beta, cross):
    """Zero-mean healthy target covariance is I; the scalar nuisance has variance one."""
    eye = sp.eye(2)
    M = eye + beta * beta.T
    C = sp.Matrix([cross])
    P = sp.Rational(1, 2)
    assert M.trace() == 3 and M.det() == 2
    assert P - (C*C.T)[0] > 0  # The declared physical covariance is positive definite.
    d = M[:, 0]
    information = (d.T*d)[0]
    prior_covariance = d*d.T
    P_delta = 1 / (1 + information)
    H = (C*d)[0]**2
    U = H * P_delta
    posterior_gain = C * (eye+prior_covariance).inv()
    floor = P - (C*C.T)[0]
    work_cost = P - (posterior_gain*C.T)[0]
    health_cost = floor + ((posterior_gain-C)*(posterior_gain-C).T)[0]
    assert work_cost == floor + U
    return dict(encoding_energy=sp.trace(M.inv()*eye*M.inv().T), min_decoder_scale=1,
                max_decoder_scale=2, conditional_information=information,
                nuisance_parameter_variance=P_delta, immediate_G_effect_second_moment=H,
                projected_posterior_correction_error=U, model_fused_mse=work_cost,
                healthy_mse_of_the_same_fusion_rule=health_cost)


def main():
    unit = sp.sqrt(2)/2
    ex = [
        ("information_increases_and_mse_worsens",
         [sp.Matrix([0, 1]), sp.Matrix([1, 0])], [sp.Rational(1, 2), 0]),
        ("same_information_different_mse",
         [sp.Matrix([unit, unit]), sp.Matrix([unit, -unit])],
         [sp.Rational(1, 4), sp.Rational(1, 4)]),
        ("same_immediate_effect_and_smaller_posterior_error",
         [sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5)]),
          sp.Matrix([sp.Rational(4, 5), sp.Rational(3, 5)])],
         [0, sp.Rational(1, 2)]),
    ]
    results = {}
    for label, parameters, C in ex:
        p, q = (compare(beta, C) for beta in parameters)
        assert p["encoding_energy"] == q["encoding_energy"] == sp.Rational(5, 4)
        if label == ex[0][0]:
            assert p["conditional_information"] < q["conditional_information"]
            assert p["nuisance_parameter_variance"] > q["nuisance_parameter_variance"]
            assert p["model_fused_mse"] < q["model_fused_mse"]
        elif label == ex[1][0]:
            assert p["conditional_information"] == q["conditional_information"]
            assert p["model_fused_mse"] > q["model_fused_mse"]
        else:
            assert p["immediate_G_effect_second_moment"] == q["immediate_G_effect_second_moment"]
            assert p["projected_posterior_correction_error"] > q["projected_posterior_correction_error"]
            assert p["model_fused_mse"] > q["model_fused_mse"]
            assert p["healthy_mse_of_the_same_fusion_rule"] > q["healthy_mse_of_the_same_fusion_rule"]
        results[label] = [{k:str(v) for k, v in row.items()} for row in (p, q)]
    path = Path(__file__).resolve().with_name("exact_gaussian_results.json")
    path.write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8", newline="\n")
    print("Three exact Gaussian comparisons verified. No sampled observations.")


if __name__ == "__main__":
    main()
