"""Rational conditional covariances and certified Gaussian health-cost integration."""

from fractions import Fraction
import json

from flint import acb, arb, ctx, fmpq, fmpz
import numpy as np
import sympy as sp


ctx.prec = 96


def real(x):
    """Preserve rational (and stored binary64) inputs."""
    if isinstance(x, (arb, acb)):
        return x
    if isinstance(x, (float, np.floating)):
        x = Fraction(float(x))
    if isinstance(x, sp.Rational):
        return arb(fmpq(fmpz(int(x.p)),fmpz(int(x.q))))
    if isinstance(x, Fraction):
        return arb(fmpq(x.numerator,x.denominator))
    return arb(str(x))


def outward(x):
    assert x.is_finite()
    lo = float(np.nextafter(float(x.lower()), -np.inf))
    hi = float(np.nextafter(float(x.upper()), np.inf))
    assert arb(lo) <= x.lower() and arb(hi) >= x.upper()
    return lo, hi


def exact_matrix(rows):
    return sp.Matrix([[sp.Rational(k) for k in row] for row in rows])


def pos2(matrix):
    return matrix == matrix.T and matrix[0,0] > 0 and matrix.det() > 0


def eigen_pair(covariance):
    a = real(covariance[0,0])
    d = real(covariance[1,1])
    b = real(covariance[0,1])
    gap = ((a-d)**2+4*b*b).sqrt()
    return ((a+d-gap)/2, (a+d+gap)/2)


def cholesky2(rows):
    a = real(rows[0][0]).sqrt()
    b = real(rows[1][0])/a
    c = (real(rows[1][1])-b*b).sqrt()
    return [[a, arb(0)], [b, c]]


class HealthCost:
    def __init__(self, path, name, spec):
        raw = json.loads(path.read_text(encoding="utf-8"), parse_float=str)
        model = raw["models"][name]
        a, q = exact_matrix(model["A"]), exact_matrix(raw["Q"])
        h, r = [[exact_matrix(value) for value in raw[key]] for key in ("H", "R")]
        i, j, l = (spec[tag] for tag in ("receiver", "target", "healthy"))
        k, g = (exact_matrix(model["nodes"][i][tag]) for tag in ("K","G"))
        J = sp.eye(2)-k*h[i]-g*(h[j]+h[l])
        F = J*a
        info_trust = h[i].T*r[i].inv()*h[i]+h[l].T*r[l].inv()*h[l]
        info_y = h[j].T*r[j].inv()*h[j]
        noise = k*r[i]*k.T+g*(r[j]+r[l])*g.T
        assert pos2(sp.Rational(".12")**2*sp.eye(2)-F.T*F)
        assert pos2(sp.Rational(".271")**2*sp.eye(2)-g.T*g)
        assert (sp.Rational(".055")+sp.Rational(".271")*sp.Rational(".831")/sp.Rational(".88"))**2 < sp.Rational(".1")
        assert sp.Rational(".831")**2 > sp.Rational(".69")
        assert sp.Rational(".055")**2 > sp.Rational(".003")
        pt = pf = p0 = sp.eye(2)
        tr_pf = tr_p0 = sp.S(0)
        self.eigenvalues = []
        self.stages = []
        self.horizon = spec["horizon"]
        for _ in range(self.horizon):
            trust = (pt.inv()+info_trust).inv()
            half = (pf.inv()+info_trust).inv()
            full = (half.inv()+info_y).inv()
            common = J*p0*J.T+noise
            v = h[j]*half*h[j].T+r[j]
            ky = half*h[j].T*v.inv()
            gap = trust-full
            assert pos2(gap)
            assert 0 <= sp.trace(common-full) < sp.Rational(".003")
            assert pos2(sp.Rational(".69")*sp.eye(2)-(h[j]*trust*h[j].T+r[j]))
            self.eigenvalues.append(eigen_pair(gap))
            self.stages.append((gap, trust-half, v, ky))
            tr_pf += sp.trace(full)
            tr_p0 += sp.trace(common)
            pt, pf, p0 = [a*x*a.T+q for x in (trust, full, common)]
        self.full = real(tr_pf/self.horizon)
        self.v0 = real(tr_p0/self.horizon)

    def at(self, r):
        """Integrate the radial moment over normalized angles with Arb."""
        r = real(r)
        assert r >= 0
        radius_squared = r*r
        pi = arb.pi()
        normalizer = (2*pi).sqrt()
        sqrt2 = arb(2).sqrt()
        total = arb(0)
        for low, high in self.eigenvalues:
            def integrand(x, analytic):
                v = low+(high-low)*(x*pi/2).sin()**2
                root = v.sqrt(analytic=analytic)
                return 2*v*(-radius_squared/(2*v)).exp()-r*normalizer*root*(r/(sqrt2*root)).erfc()
            part = acb.integral(integrand, 0, 1, abs_tol=arb("1e-21"),
                                rel_tol=arb("1e-20"), eval_limit=8000)
            assert part.is_finite() and part.imag.contains(0)
            total += part.real
        ans = self.full+total/self.horizon
        assert ans.rad() < arb("1e-14")
        return ans

    def clipping_bias(self, clip, duration):
        return self.horizon*(duration+1)*(-5*real(clip)).exp()/5
