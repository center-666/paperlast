"""G1 citation-scope checks, not a detector or a reproduction of old scripts.

Run from any directory with Python + SymPy. Only writes in this new task dir.
"""
from pathlib import Path
import hashlib
import json
import sys
import sympy as s

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    # Exact scalar example using the observer in Olfat Eq. (3), rather than
    # treating its stationary residuals as independent observations.
    A = s.Rational(1, 2)
    B = C = s.Integer(1)
    K = -s.Rational(1, 10)
    L = -s.Rational(1, 5)
    Q = R = s.Integer(1)
    ac = A + B * K
    ao = A + L * C
    p = (Q + L**2 * R) / (1 - ao**2)
    residual_var = C**2 * p + R
    # delta_next = (A+LC)delta - w - L z; residual=C delta-z.
    lag_one = C**2 * ao * p + C * L * R
    assert (ac, ao, p, lag_one) == (
        s.Rational(2, 5), s.Rational(3, 10),
        s.Rational(104, 91), s.Rational(1, 7),
    )
    assert abs(ac) < 1 and abs(ao) < 1 and C * ac * B != 0
    ell = 2
    wishart_variance = 2 * ell * residual_var**2
    actual_variance = wishart_variance + 4 * (ell - 1) * lag_one**2
    assert actual_variance > wishart_variance
    if '--math-only' in sys.argv:
        print(json.dumps({'lag_one':str(lag_one),'variance_excess':str(actual_variance-wishart_variance),'exact_assertions_passed':True}))
        return

    definitions = [
        {
            "key": "FPN22",
            "title": "Innovation-based stealthy attack against distributed state estimation over sensor networks",
            "path": ROOT / "参考文献/02_期刊与会议论文/FPN22_2023_Paper_Niu2023_InnovationStealthyAttack.pdf",
            "source": "https://www.sciencedirect.com/science/article/pii/S0005109823001140",
            "doi": "10.1016/j.automatica.2023.110962",
            "pdf_pages": 9,
            "read_pages": [2, 3, 4, 5, 6],
            "read_scope": "Equations (1)-(15), Assumptions 1-3, Theorems 1-4 and key proof steps; not all simulation source data.",
        },
        {
            "key": "CNA01",
            "title": "Distributed Kalman Filtering: When to Share Measurements",
            "path": HERE / "literature_work/Greiff2022_MERL_TR2022-158.pdf",
            "source": "https://www.merl.com/publications/docs/TR2022-158.pdf",
            "doi": "10.1109/CDC51059.2022.9993404",
            "pdf_pages": 8,
            "read_pages": [3, 4, 5, 6, 7, 8],
            "read_scope": "Six body pages, pp.5399-5404; first two container pages are institutional cover/blank. Proposition 1 and Algorithm 1 reviewed; not a source-code replication.",
        },
        {
            "key": "CNA02",
            "title": "Covariance-Robust Dynamic Watermarking",
            "path": HERE / "literature_work/Olfat2020_arxiv_official.pdf",
            "source": "https://arxiv.org/pdf/2003.13908",
            "arxiv": "2003.13908v1",
            "pdf_pages": 11,
            "read_pages": [1, 2, 3, 4, 5, 6, 7],
            "read_scope": "Current official v1 freshly downloaded, SHA identical to prior local file. Body/reference pages 1-7; four external-PNG placeholders at pp.8-11. Eqs.2-24, Lemmas1-5, Theorem1 reviewed with explicit statistical applicability limits.",
        },
    ]
    sources = []
    for row in definitions:
        row = dict(row)
        path = row.pop("path")
        raw = path.read_bytes()
        row["local_path"] = str(path)
        row["sha256"] = hashlib.sha256(raw).hexdigest()
        row["bytes"] = len(raw)
        sources.append(row)
    output = {
        "purpose": "Citation applicability audit only; no G2 algorithm development",
        "independent_exact_observer_counterexample": {
            "A": str(A), "B": str(B), "C": str(C),
            "K": str(K), "L": str(L), "Q": str(Q), "R": str(R),
            "A_plus_BK": str(ac), "A_plus_LC": str(ao),
            "k_prime": 1, "steady_delta_variance": str(p),
            "residual_variance": str(residual_var),
            "residual_lag_one_covariance": str(lag_one),
            "ell": ell,
            "iid_Wishart_diagonal_sum_variance": str(wishart_variance),
            "actual_Gaussian_diagonal_sum_variance": str(actual_variance),
            "variance_excess": str(actual_variance - wishart_variance),
            "scope": "Refutes automatic iid-Wishart interpretation for arbitrary stabilizing observer; does not establish or refute every possible dependent-data asymptotic theorem.",
        },
        "sources": sources,
    }
    destination = HERE / "literature_one_sources.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"source_count": len(sources), "lag_one": str(lag_one),
                      "variance_excess": str(actual_variance-wishart_variance)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
