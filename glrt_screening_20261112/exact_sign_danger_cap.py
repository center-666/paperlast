"""Exact integer danger-word cap in the frozen decimal rational model.

No detector is needed for this cap. The JSON's saved F/g decimals are treated
as exact rational inputs, including the tiny cross-coordinate entries. This
does not certify the rounding error between those inputs and the DARE model.
No old file is modified, no physical/noise samples are generated.
"""
from fractions import Fraction
from hashlib import sha256
from itertools import product
from math import lcm
from pathlib import Path
import json


HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "checks/水印收束_理论与映射核验_20261105.json"
OUTPUT = HERE / "exact_sign_danger_cap.json"


def main():
    raw = INPUT.read_bytes()
    source = json.loads(raw.decode("utf-8-sig"))
    F = [[Fraction(str(x)) for x in row] for row in source["F"]]
    g = [Fraction(str(x)) for x in source["g"]]
    W = [[Fraction(str(x)) for x in row] for row in source["W10"]]
    assert len(F) == len(g) == 4 and all(len(row) == 4 for row in F)
    assert W == [[Fraction(10), Fraction(9, 2)],
                 [Fraction(9, 2), Fraction(57, 20)]]
    assert W[0][0] > 0 and W[0][0]*W[1][1]-W[0][1]**2 > 0

    # F=F_int/d and g=g_int/d share this exact integer denominator.
    d = 1
    for value in sum(F, []) + g:
        d = lcm(d, value.denominator)
    F_int = [[int(value*d) for value in row] for row in F]
    g_int = [int(value*d) for value in g]
    assert all(Fraction(F_int[i][j], d) == F[i][j]
               for i in range(4) for j in range(4))
    assert all(Fraction(g_int[i], d) == g[i] for i in range(4))

    words = list(product([-1, 1], repeat=10))
    numerators = [[[0, 0] for _ in range(4)] for _ in words]
    d_power = 1
    by_k = []
    danger_union = set()
    strict_safe_checks = 0
    nonstrict_checks = 0
    zero_first_minor = 0
    zero_determinant = 0

    # B_{k+1}(epsilon) maps xi to the open-loop counterfactual displacement.
    # With H_k=[10,k]/10, B_{k+1}=I_{k+1}/(10*d**(k+1)).
    for k in range(13):
        for word_index, word in enumerate(words):
            previous = numerators[word_index]
            numerators[word_index] = [
                [sum(F_int[i][j]*previous[j][ell] for j in range(4))
                 + (word[k]*g_int[i]*[10, k][ell]*d_power if k < 10 else 0)
                 for ell in range(2)] for i in range(4)]
        d_power *= d
        denominator = 10*d_power
        denominator_sq = denominator**2
        capable = []

        for word_index, matrix in enumerate(numerators):
            h00 = sum(row[0]**2 for row in matrix)
            h01 = sum(row[0]*row[1] for row in matrix)
            h11 = sum(row[1]**2 for row in matrix)

            # D=1/10, E=10. No xi on the W-energy ball can reach D iff
            # (D/E)*W - B^T B is positive definite. Multiply by the positive
            # scalar 2000*denominator^2. Since 20W=[[200,90],[90,57]],
            # this yields the following symmetric integer 2x2 matrix.
            m00 = 200*denominator_sq - 2000*h00
            m01 = 90*denominator_sq - 2000*h01
            m11 = 57*denominator_sq - 2000*h11
            determinant = m00*m11 - m01*m01
            zero_first_minor += int(m00 == 0)
            zero_determinant += int(determinant == 0)
            if m00 > 0 and determinant > 0:
                strict_safe_checks += 1
            else:
                # A non-positive-definite comparison matrix implies the
                # generalized maximum damage is at least D on the energy ball.
                capable.append(word_index)
                danger_union.add(word_index)
                nonstrict_checks += 1
        by_k.append({"k": k, "danger_capable_word_count": len(capable),
                     "danger_capable_word_indices": capable})

    expected = [0, 0, 0, 0, 0, 0, 16, 8, 8, 48, 18, 2, 0]
    assert [row["danger_capable_word_count"] for row in by_k] == expected
    assert len(danger_union) == 64
    assert strict_safe_checks + nonstrict_checks == 13*1024
    assert zero_first_minor == zero_determinant == 0
    bound = Fraction(len(danger_union), len(words))
    assert bound == Fraction(1, 16)

    result = {
        "date": "2026-11-12",
        "method": "Python integer arithmetic; exact rational interpretation of frozen F/g decimals",
        "input_file": str(INPUT.relative_to(HERE.parent)),
        "input_sha256": sha256(raw).hexdigest(),
        "energy_budget": "10",
        "danger_threshold": "1/10",
        "attack": "a_t=(p+t*v/10) for t=0,...,9, zero otherwise",
        "attack_timing": "xi fixed before secret sign word; cannot adapt xi to a word",
        "counterfactual_displacement": "b_{k+1}=F*b_k+g*epsilon_k*H_k*xi; b_0=0",
        "monitor_k": list(range(13)),
        "word_order": "itertools.product([-1,1], repeat=10)",
        "word_count": len(words),
        "common_denominator": str(d),
        "integer_recurrence": "I_{k+1}=F_int*I_k+epsilon_k*g_int*[10,k]*d^k; injection=0 for k>=10",
        "comparison_integer_matrix": "[[200,90],[90,57]]*den^2 - 2000*I^T*I, den=10*d^(k+1)",
        "positive_definiteness_test": "m00>0 and m00*m11-m01^2>0 (Sylvester)",
        "W": [[str(x) for x in row] for row in W],
        "word_time_checks": strict_safe_checks+nonstrict_checks,
        "strict_safe_word_time_checks": strict_safe_checks,
        "danger_capable_word_time_checks": nonstrict_checks,
        "zero_first_minor_checks": zero_first_minor,
        "zero_determinant_checks": zero_determinant,
        "per_k": by_k,
        "danger_capable_word_union_count": len(danger_union),
        "danger_capable_word_union_indices": sorted(danger_union),
        "uniform_danger_probability_upper": str(bound),
        "uniform_danger_before_alarm_risk_upper": str(bound),
        "proof": [
            "For each excluded word every k has 0.01W-B_k^TB_k positive definite, so all xi with xi^TWxi<=10 have ||B_k xi||^2<0.1.",
            "Only 64 of 1024 uniformly distributed words can have any dangerous xi at any monitored k; the same xi cannot make more words dangerous.",
            "Danger-before-alarm is a subset of the event that any danger occurs, for every detector; hence its risk is at most 64/1024 uniformly over xi."
        ],
        "scope": "frozen decimal rational model, not a validated interval DARE/physical-parameter certificate",
        "exact_worst_risk_computed": False,
        "GLRT_gain_certified_by_this_cap": False,
        "physical_interval_certification": False,
        "physical_trajectories": 0,
        "noise_samples": 0,
        "old_files_modified": 0,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"word_time_checks": result["word_time_checks"],
                      "danger_word_union": len(danger_union),
                      "uniform_risk_cap": str(bound),
                      "scope": result["scope"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
