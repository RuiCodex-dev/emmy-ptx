"""Elementwise math: one Emmy op name -> a short PTX sequence on f32 registers.

Accuracy choices, because every problem is graded against numpy within a
tolerance and the pointwise ones with ``atol = rtol = 0``:

* ``add`` / ``sub`` / ``mul`` carry an explicit ``.rn``.  Without a rounding
  modifier ptxas is allowed to contract a ``mul`` followed by an ``add`` into
  one ``fma``, which rounds once instead of twice and no longer matches numpy
  bit for bit.  ``.rn`` pins the IR's semantics: one IR op, one rounding.
* ``divide``, ``reciprocal`` and ``sqrt`` use the IEEE-rounded forms
  (``div.rn``, ``rcp.rn``, ``sqrt.rn``), never ``.approx``.
* Transcendentals use the hardware ``ex2.approx`` / ``lg2.approx`` units
  (about 2 ulp) through the identities exp(x) = 2^(x log2 e) and
  ln(x) = log2(x) ln 2.  ``tanh`` is built from ``exp`` rather than the
  ``tanh.approx`` instruction (about 2^-11 relative error, too coarse here),
  and ``erf`` uses Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7).

Every function returns a *fresh* register: callers may bind it to an IR
name, and a register shared between two names would be clobbered the next
time either is reassigned inside a loop.
"""

from __future__ import annotations

import math

from .emitter import Emitter, f32_hex
from .values import F, IntOps, Val

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453

# Abramowitz & Stegun 7.1.26
_ERF_P = 0.3275911
_ERF_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)

_FCMP = {"greater": "gt", "greater_equal": "ge", "less": "lt", "less_equal": "le",
         "equal": "eq", "not_equal": "neu"}


class FloatOps:
    def __init__(self, em: Emitter, ints: IntOps) -> None:
        self.em = em
        self.ints = ints

    # ---- primitives --------------------------------------------------------
    def r(self, a: Val) -> str:
        return self.ints.as_reg(self.ints.to_f32(a))

    def t(self, a: Val) -> str:
        return self.ints.to_f32(a).text()

    def un(self, opcode: str, a: Val) -> Val:
        d = self.em.reg("f")
        self.em.ins(opcode, d, self.r(a))
        return F(d)

    def bin(self, opcode: str, a: Val, b: Val, commutative: bool = False) -> Val:
        a, b = self.ints.to_f32(a), self.ints.to_f32(b)
        if a.is_const and commutative and not b.is_const:
            a, b = b, a
        d = self.em.reg("f")
        self.em.ins(opcode, d, self.r(a), self.t(b))
        return F(d)

    def fma(self, a: Val, b: Val, c: Val) -> Val:
        d = self.em.reg("f")
        self.em.ins("fma.rn.f32", d, self.r(a), self.t(b), self.t(c))
        return F(d)

    def add(self, a, b):
        return self.bin("add.rn.f32", a, b, True)

    def sub(self, a, b):
        return self.bin("sub.rn.f32", a, b)

    def mul(self, a, b):
        return self.bin("mul.rn.f32", a, b, True)

    def div(self, a, b):
        return self.bin("div.rn.f32", a, b)

    def rcp(self, a):
        return self.un("rcp.rn.f32", a)

    def neg(self, a):
        return self.un("neg.f32", a)

    def abs(self, a):
        return self.un("abs.f32", a)

    def max(self, a, b):
        return self.bin("max.f32", a, b, True)

    def min(self, a, b):
        return self.bin("min.f32", a, b, True)

    def exp(self, a: Val) -> Val:
        return self.un("ex2.approx.f32", self.mul(a, F(LOG2E)))

    def exp2(self, a: Val) -> Val:
        return self.un("ex2.approx.f32", a)

    def log(self, a: Val) -> Val:
        return self.mul(self.un("lg2.approx.f32", a), F(LN2))

    def log2(self, a: Val) -> Val:
        return self.un("lg2.approx.f32", a)

    def sqrt(self, a):
        return self.un("sqrt.rn.f32", a)

    def sigmoid(self, a: Val) -> Val:
        # 1 / (1 + e^-x); e^-x overflows to inf for very negative x and rcp(inf) = 0
        return self.rcp(self.add(self.exp(self.neg(a)), F(1.0)))

    def tanh(self, a: Val) -> Val:
        # tanh|x| = (1 - e^(-2|x|)) / (1 + e^(-2|x|)); e^(-2|x|) <= 1 never overflows
        t = self.exp(self.mul(self.abs(a), F(-2.0)))
        q = self.div(self.sub(F(1.0), t), self.add(t, F(1.0)))
        d = self.em.reg("f")
        self.em.ins("copysign.f32", d, self.r(a), q.v)  # sign of the input onto |result|
        return F(d)

    def erf(self, a: Val) -> Val:
        x = self.abs(a)
        t = self.rcp(self.fma(x, F(_ERF_P), F(1.0)))
        a1, a2, a3, a4, a5 = _ERF_A
        poly = self.fma(t, F(a5), F(a4))           # Horner: ((((a5 t + a4) t + a3) t + a2) t + a1) t
        poly = self.fma(poly, t, F(a3))
        poly = self.fma(poly, t, F(a2))
        poly = self.fma(poly, t, F(a1))
        poly = self.mul(poly, t)
        e = self.exp(self.neg(self.mul(x, x)))
        y = self.sub(F(1.0), self.mul(poly, e))
        d = self.em.reg("f")
        self.em.ins("copysign.f32", d, self.r(a), y.v)
        return F(d)

    def softplus(self, a: Val) -> Val:
        # max(x, 0) + log(1 + e^-|x|): exact limits on both tails, no overflow
        return self.add(self.max(a, F(0.0)), self.log(self.add(self.exp(self.neg(self.abs(a))), F(1.0))))

    def power(self, a: Val, b: Val) -> Val:
        if b.is_const:
            n = float(b.v)
            if n == 2.0:
                return self.mul(a, a)
            if n == 1.0:
                return self.mul(a, F(1.0))
            if n == 0.5:
                return self.sqrt(a)
            if n == -1.0:
                return self.rcp(a)
        # |a|^b, then the sign for a negative base with an odd integer exponent
        mag = self.exp2(self.mul(b, self.log2(self.abs(a))))
        bi = self.em.reg("f")
        self.em.ins("cvt.rzi.f32.f32", bi, self.r(b))
        half = self.mul(F(bi), F(0.5))
        hi = self.em.reg("f")
        self.em.ins("cvt.rzi.f32.f32", hi, half.v)
        p_int, p_odd, p_neg, p = (self.em.reg("p") for _ in range(4))
        self.em.ins("setp.eq.f32", p_int, bi, self.r(b))
        self.em.ins("setp.neu.f32", p_odd, hi, half.v)
        self.em.ins("setp.lt.f32", p_neg, self.r(a), f32_hex(0.0))
        self.em.ins("and.pred", p, p_int, p_odd)
        self.em.ins("and.pred", p, p, p_neg)
        d = self.em.reg("f")
        self.em.ins("neg.f32", d, mag.v)
        self.em.ins("selp.f32", d, d, mag.v, p)
        return F(d)

    def cvt_round(self, mode: str, a: Val) -> Val:
        return self.un(f"cvt.{mode}.f32.f32", a)

    # ---- dispatch -------------------------------------------------------------
    def apply(self, name: str, args: list[Val]) -> Val:
        """Lower one elementwise op.  Integer and predicate operands take the
        integer / logic path; everything else is computed in f32."""
        if args and all(a.kind == "i" for a in args) and name in _INT_OPS:
            return _INT_OPS[name](self.ints, *args)
        if args and all(a.kind == "p" for a in args) and name in _PRED_OPS:
            return _PRED_OPS[name](self.ints, *args)
        if name in _FCMP:
            a, b = (self.ints.to_f32(x) for x in args)
            if a.is_const:
                a, b = b, a
                cc = {"gt": "lt", "ge": "le", "lt": "gt", "le": "ge"}.get(_FCMP[name], _FCMP[name])
            else:
                cc = _FCMP[name]
            p = self.em.reg("p")
            self.em.ins(f"setp.{cc}.f32", p, self.r(a), self.t(b))
            return Val("p", p)
        if name == "where":
            c, a, b = args
            if c.kind != "p":
                c = self.apply("not_equal", [c, F(0.0)])
            a, b = self.ints.to_f32(a), self.ints.to_f32(b)
            if c.is_const:
                return self.mul(a if c.v else b, F(1.0))
            d = self.em.reg("f")
            self.em.ins("selp.f32", d, self.t(a), self.t(b), c.v)
            return F(d)
        fn = _UNARY.get(name)
        if fn is not None:
            (a,) = args
            return fn(self, a)
        fn = _BINARY.get(name)
        if fn is not None:
            a, b = args
            return fn(self, a, b)
        raise NotImplementedError(f"elementwise op {name!r} is not supported by this compiler")


def _gelu(f: FloatOps, a: Val) -> Val:
    # 0.5 x (1 + erf(x / sqrt 2))
    e = f.erf(f.mul(a, F(1.0 / math.sqrt(2.0))))
    return f.mul(f.mul(a, F(0.5)), f.add(e, F(1.0)))


def _gelu_tanh(f: FloatOps, a: Val) -> Val:
    # 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
    x3 = f.mul(f.mul(a, a), a)
    inner = f.mul(f.fma(x3, F(0.044715), a), F(math.sqrt(2.0 / math.pi)))
    return f.mul(f.mul(a, F(0.5)), f.add(f.tanh(inner), F(1.0)))


def _sign(f: FloatOps, a: Val) -> Val:
    pos = f.apply("greater", [a, F(0.0)])
    neg = f.apply("less", [a, F(0.0)])
    return f.sub(f.ints.to_f32(pos), f.ints.to_f32(neg))


_UNARY = {
    "negative": FloatOps.neg,
    "abs": FloatOps.abs,
    "absolute": FloatOps.abs,
    "fabs": FloatOps.abs,
    "reciprocal": FloatOps.rcp,
    "sqrt": FloatOps.sqrt,
    "rsqrt": lambda f, a: f.rcp(f.sqrt(a)),
    "exp": FloatOps.exp,
    "exp_fast": FloatOps.exp,
    "exp2": FloatOps.exp2,
    "expm1": lambda f, a: f.sub(f.exp(a), F(1.0)),
    "log": FloatOps.log,
    "log2": FloatOps.log2,
    "log10": lambda f, a: f.mul(f.log2(a), F(math.log10(2.0))),
    "log1p": lambda f, a: f.log(f.add(a, F(1.0))),
    "square": lambda f, a: f.mul(a, a),
    "relu": lambda f, a: f.max(a, F(0.0)),
    "sigmoid": FloatOps.sigmoid,
    "silu": lambda f, a: f.mul(a, f.sigmoid(a)),
    "tanh": FloatOps.tanh,
    "softplus": FloatOps.softplus,
    "erf": FloatOps.erf,
    "gelu": _gelu,
    "gelu_tanh": _gelu_tanh,
    "sin": lambda f, a: f.un("sin.approx.f32", a),
    "cos": lambda f, a: f.un("cos.approx.f32", a),
    "floor": lambda f, a: f.cvt_round("rmi", a),
    "ceil": lambda f, a: f.cvt_round("rpi", a),
    "trunc": lambda f, a: f.cvt_round("rzi", a),
    "rint": lambda f, a: f.cvt_round("rni", a),
    "round": lambda f, a: f.cvt_round("rni", a),
    "sign": _sign,
    "copy": lambda f, a: f.mul(a, F(1.0)),
    "pad": lambda f, a: f.mul(a, F(1.0)),
    "positive": lambda f, a: f.mul(a, F(1.0)),
}

_BINARY = {
    "add": FloatOps.add,
    "subtract": FloatOps.sub,
    "multiply": FloatOps.mul,
    "divide": FloatOps.div,
    "true_divide": FloatOps.div,
    "maximum": FloatOps.max,
    "fmax": FloatOps.max,
    "minimum": FloatOps.min,
    "fmin": FloatOps.min,
    "power": FloatOps.power,
}

_INT_OPS = {
    "add": IntOps.add,
    "subtract": IntOps.sub,
    "multiply": IntOps.mul,
    "floor_divide": IntOps.div,
    "maximum": lambda ints, a, b: ints.select(ints.cmp("gt", a, b), a, b),
    "minimum": IntOps.min_,
}

_PRED_OPS = {
    "logical_and": IntOps.land,
    "logical_or": IntOps.lor,
    "logical_not": IntOps.lnot,
    "bitwise_and": IntOps.land,
    "bitwise_or": IntOps.lor,
}
