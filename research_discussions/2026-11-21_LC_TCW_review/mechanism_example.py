"""Noiseless worked examples for explaining the reviewed two-frame equations.

These are algebraic illustrations, not physical trajectories or performance evidence.
Python 3.11+, standard library only.
"""

import json
import math
from pathlib import Path
from statistics import NormalDist


def illustrate():
    normal = NormalDist()
    quantile = lambda u: 0.5 * normal.inv_cdf((1 + u) / 2) ** 2
    b0, b1 = math.sqrt(quantile(0.25)), math.sqrt(quantile(0.75))
    c0, c1 = 0.5 + b0**2, 0.5 + b1**2
    state, nuisance = 2.0, 3.0
    y0, y1 = state + c0 * nuisance, state + c1 * nuisance
    scalar = dict(c0=c0, c1=c1, state=state, nuisance=nuisance, Y0=y0, Y1=y1,
                  recovered_nuisance=(y1-y0)/(c1-c0),
                  recovered_state=(c1*y0-c0*y1)/(c1-c0))
    branches = []
    for p0, p1 in ((0.76, 0.78), (0.500001, 0.500004)):
        q0, q1 = math.sqrt(p0-0.5)*b0, -math.sqrt(p1-0.5)*b1
        r = 14/13
        xi1, xi2 = 0.3, -0.4
        u0, v0 = p0*xi1+q0*xi2, state+q0*xi1+c0*xi2
        u1, v1 = p1*xi1+r*q1*xi2, state+(q1/r)*xi1+c1*xi2
        first_coefficient = r*q1-(p1/p0)*q0
        h0, h1 = 0.5+0.5*b0**2/p0, 0.5+0.5*b1**2/p1
        if abs(first_coefficient) >= 0.15:
            branch = "first_difference"
            recovered_xi2 = (u1-(p1/p0)*u0)/first_coefficient
        else:
            branch = "schur_difference"
            w0, w1 = v0-(q0/p0)*u0, v1-(q1/(r*p1))*u1
            recovered_xi2 = (w1-w0)/(h1-h0)
        recovered_xi1 = (u0-q0*recovered_xi2)/p0
        recovered_state = v0-q0*recovered_xi1-c0*recovered_xi2
        branches.append(dict(p0=p0, p1=p1, D=first_coefficient, h_difference=h1-h0,
                             branch=branch, target=[state,xi1,xi2],
                             recovered=[recovered_state,recovered_xi1,recovered_xi2]))
    return dict(scope="Noiseless algebra only; no system simulation or risk comparison.",
                scalar=scalar, two_coordinate_examples=branches)


if __name__ == "__main__":
    result = illustrate()
    output = Path(__file__).with_name("mechanism_example.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
