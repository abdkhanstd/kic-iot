# Author: abdkhan
"""Interval arithmetic with second-order forward differentiation, vectorised
over a batch of boxes. Used by certify_env.py to *prove* upper bounds on the
envelopes G (per-realization sensitivity) and L (slope Lipschitz constant)
of the simulated plants, instead of estimating them from samples.

An `I` holds elementwise enclosures [lo, hi] (numpy arrays). A `Jet` holds an
enclosure of a function's value, its gradient with respect to the two
actuators, and its Hessian (h00, h01, h11), each an `I`. Every operation
returns an enclosure of the exact result for every point of the box, so the
final bounds are rigorous up to floating-point rounding, which we cover by
widening each result by a relative 1e-12 (far below the margins we certify).
"""
from __future__ import annotations

import math

import numpy as np

_EPS = 1e-12


def _pad(lo, hi):
    w = _EPS * (np.abs(lo) + np.abs(hi) + 1.0)
    return lo - w, hi + w


class I:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        lo = np.asarray(lo, dtype=float)
        hi = lo if hi is None else np.asarray(hi, dtype=float)
        self.lo, self.hi = np.broadcast_arrays(lo, hi)

    @staticmethod
    def c(x, like):
        z = np.full_like(like.lo, float(x))
        return I(z, z)

    def __add__(self, o):
        if isinstance(o, I):
            return I(*_pad(self.lo + o.lo, self.hi + o.hi))
        return I(*_pad(self.lo + o, self.hi + o))

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, o):
        return self + (-o if isinstance(o, I) else -o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, I):
            p = np.stack([self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi])
            return I(*_pad(p.min(0), p.max(0)))
        o = float(o)
        a, b = self.lo * o, self.hi * o
        return I(*_pad(np.minimum(a, b), np.maximum(a, b)))

    __rmul__ = __mul__

    def sq(self):
        a, b = self.lo ** 2, self.hi ** 2
        lo = np.where((self.lo <= 0) & (self.hi >= 0), 0.0, np.minimum(a, b))
        return I(*_pad(lo, np.maximum(a, b)))

    def recip(self):
        assert np.all(self.lo > 0), "reciprocal of an interval containing 0"
        return I(*_pad(1.0 / self.hi, 1.0 / self.lo))

    def __truediv__(self, o):
        if isinstance(o, I):
            return self * o.recip()
        return self * (1.0 / float(o))

    def __rtruediv__(self, o):
        return self.recip() * float(o)

    def absmax(self):
        return np.maximum(np.abs(self.lo), np.abs(self.hi))

    def width(self):
        return self.hi - self.lo


def hull(*vals):
    v = np.stack(vals)
    return v.min(0), v.max(0)


# ---------------------------------------------------------------- scalar functions
# Each returns enclosures of (f, f', f'') over an interval x.

def f_sigmoid(x: I):
    s = lambda z: 1.0 / (1.0 + np.exp(-z))
    d1 = lambda z: s(z) * (1 - s(z))
    d2 = lambda z: d1(z) * (1 - 2 * s(z))
    v = I(*_pad(s(x.lo), s(x.hi)))
    # f' is unimodal with its maximum 1/4 at 0
    lo1 = np.minimum(d1(x.lo), d1(x.hi))
    hi1 = np.where((x.lo <= 0) & (x.hi >= 0), 0.25, np.maximum(d1(x.lo), d1(x.hi)))
    # f'' is odd with extrema at -+z*, z* = ln(2 + sqrt 3)
    zs = math.log(2 + math.sqrt(3))
    cand = [d2(x.lo), d2(x.hi)]
    lo2, hi2 = hull(*cand)
    inm = (x.lo <= -zs) & (x.hi >= -zs)
    inp = (x.lo <= zs) & (x.hi >= zs)
    hi2 = np.where(inm, np.maximum(hi2, d2(-zs)), hi2)
    lo2 = np.where(inp, np.minimum(lo2, d2(zs)), lo2)
    return v, I(*_pad(lo1, hi1)), I(*_pad(lo2, hi2))


def f_exp(x: I):
    v = I(*_pad(np.exp(x.lo), np.exp(x.hi)))
    return v, v, v


def f_log(x: I):
    assert np.all(x.lo > 0)
    v = I(*_pad(np.log(x.lo), np.log(x.hi)))
    d1 = I(*_pad(1 / x.hi, 1 / x.lo))
    d2 = I(*_pad(-1 / x.lo ** 2, -1 / x.hi ** 2))
    return v, d1, d2


def f_recip(x: I):
    assert np.all(x.lo > 0)
    v = I(*_pad(1 / x.hi, 1 / x.lo))
    d1 = I(*_pad(-1 / x.lo ** 2, -1 / x.hi ** 2))
    d2 = I(*_pad(2 / x.hi ** 3, 2 / x.lo ** 3))
    return v, d1, d2


def f_smoothclip(sigma: float):
    """a -> E[clip(a + e, 0, 1)], e ~ N(0, sigma^2): the measured value of a
    channel clipped to [0, 1] after additive Gaussian noise, averaged over
    the noise. f' = P(0 < a + e < 1) in [0, 1];
    f'' = (phi(a / s) - phi((1 - a) / s)) / s, maximal at a = 0, minimal at 1."""
    from scipy.stats import norm

    def f(a):
        # E clip(a+e,0,1) = E max(a+e,0) - E max(a+e-1,0)
        g = lambda m: m * norm.cdf(m / sigma) + sigma * norm.pdf(m / sigma)
        return g(a) - g(a - 1.0)

    d1 = lambda a: norm.cdf((1 - a) / sigma) - norm.cdf(-a / sigma)
    d2 = lambda a: (norm.pdf(a / sigma) - norm.pdf((1 - a) / sigma)) / sigma

    def enc(x: I):
        v = I(*_pad(f(x.lo), f(x.hi)))                    # f is increasing
        c1 = [d1(x.lo), d1(x.hi)]
        lo1, hi1 = hull(*c1)
        mid = (x.lo <= 0.5) & (x.hi >= 0.5)               # f' is maximal at 1/2
        hi1 = np.where(mid, np.maximum(hi1, d1(0.5)), hi1)
        lo2, hi2 = hull(d2(x.lo), d2(x.hi))
        hi2 = np.where((x.lo <= 0) & (x.hi >= 0), np.maximum(hi2, d2(0.0)), hi2)
        lo2 = np.where((x.lo <= 1) & (x.hi >= 1), np.minimum(lo2, d2(1.0)), lo2)
        return v, I(*_pad(lo1, hi1)), I(*_pad(lo2, hi2))
    return enc


# ---------------------------------------------------------------- jets

class Jet:
    """Enclosure of a C^2 function of the two actuators (u0, u1) over a box:
    value v, gradient g = (g0, g1), Hessian (h00, h01, h11)."""
    __slots__ = ("v", "g", "h")

    def __init__(self, v: I, g, h):
        self.v, self.g, self.h = v, list(g), list(h)

    @staticmethod
    def var(x: I, k: int):
        z = I.c(0.0, x)
        one = I.c(1.0, x)
        return Jet(x, [one if k == 0 else z, one if k == 1 else z], [z, z, z])

    @staticmethod
    def const(x: I):
        z = I.c(0.0, x)
        return Jet(x, [z, z], [z, z, z])

    def __add__(self, o):
        if isinstance(o, Jet):
            return Jet(self.v + o.v, [a + b for a, b in zip(self.g, o.g)],
                       [a + b for a, b in zip(self.h, o.h)])
        return Jet(self.v + o, self.g, self.h)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, [-a for a in self.g], [-a for a in self.h])

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, Jet):
            a, b = self, o
            v = a.v * b.v
            g = [a.v * b.g[i] + b.v * a.g[i] for i in range(2)]
            h00 = a.v * b.h[0] + b.v * a.h[0] + 2.0 * (a.g[0] * b.g[0])
            h11 = a.v * b.h[2] + b.v * a.h[2] + 2.0 * (a.g[1] * b.g[1])
            h01 = a.v * b.h[1] + b.v * a.h[1] + a.g[0] * b.g[1] + a.g[1] * b.g[0]
            return Jet(v, g, [h00, h01, h11])
        if isinstance(o, I):
            return self * Jet.const(o)
        return Jet(self.v * o, [x * o for x in self.g], [x * o for x in self.h])

    __rmul__ = __mul__

    def apply(self, enc):
        f, d1, d2 = enc(self.v)
        g = [d1 * self.g[0], d1 * self.g[1]]
        h00 = d2 * self.g[0].sq() + d1 * self.h[0]
        h11 = d2 * self.g[1].sq() + d1 * self.h[2]
        h01 = d2 * (self.g[0] * self.g[1]) + d1 * self.h[1]
        return Jet(f, g, [h00, h01, h11])

    def __truediv__(self, o):
        if isinstance(o, Jet):
            return self * o.apply(f_recip)
        return self * (1.0 / float(o))

    def clamp(self, lo=-np.inf, hi=np.inf):
        """Intersect the value enclosure with a range known to contain the
        exact value (e.g. a probability in [0, 1]); derivatives unchanged."""
        v = I(np.maximum(self.v.lo, lo), np.minimum(self.v.hi, hi))
        v.lo = np.minimum(v.lo, v.hi)
        return Jet(v, self.g, self.h)

    def hess_row_norm(self, j: int):
        """Upper bound of the Euclidean norm of Hessian row j."""
        d = self.h[0] if j == 0 else self.h[2]
        return np.sqrt(d.absmax() ** 2 + self.h[1].absmax() ** 2)


# ---------------------------------------------------------------- univariate rational primitives

class RationalPrimitive:
    """Exact enclosures of g, g', g'' over intervals of r >= 0 for a rational
    function g = num / den without poles on [0, inf): each derivative's range
    on [a, b] is attained at a, b or at an interior root of the next
    derivative, and those roots are precomputed (sympy numerators, numpy
    roots). Replacing the naive interval evaluation of a ratio of polynomials
    by these enclosures removes the dependency problem inside the primitive."""

    def __init__(self, num_coeffs, den_coeffs, rmax=1e3):
        import sympy as sp
        r = sp.symbols("r", nonnegative=True)
        g = sp.Poly(list(reversed(num_coeffs)), r).as_expr() / sp.Poly(list(reversed(den_coeffs)), r).as_expr()
        self.fs, self.crit = [], []
        d = g
        for k in range(3):
            self.fs.append(sp.lambdify(r, d, "numpy"))
            nxt = sp.together(sp.diff(d, r))
            numer = sp.Poly(sp.numer(nxt), r)
            roots = np.roots([float(c) for c in numer.all_coeffs()]) if numer.degree() > 0 else np.array([])
            real = sorted(float(z.real) for z in roots if abs(z.imag) < 1e-9 and 0 <= z.real <= rmax)
            self.crit.append(np.array(real))
            d = nxt

    def enc(self, x: I):
        out = []
        for f, cr in zip(self.fs, self.crit):
            lo = np.minimum(f(x.lo), f(x.hi))
            hi = np.maximum(f(x.lo), f(x.hi))
            for c in cr:
                inside = (x.lo <= c) & (x.hi >= c)
                v = f(c)
                lo = np.where(inside, np.minimum(lo, v), lo)
                hi = np.where(inside, np.maximum(hi, v), hi)
            out.append(I(*_pad(lo, hi)))
        return tuple(out)


# M/M/1/3 in the utilization r: blocking, busy, T(1-P)*mu and T*mu
MM1K3_P = RationalPrimitive([0, 0, 0, 1], [1, 1, 1, 1])
MM1K3_B = RationalPrimitive([0, 1, 1, 1], [1, 1, 1, 1])
MM1K3_K = RationalPrimitive([1, 2, 3], [1, 1, 1, 1])
MM1K3_H = RationalPrimitive([1, 2, 3], [1, 1, 1])
