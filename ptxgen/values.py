"""Compile-time values and the integer / predicate arithmetic on them.

A ``Val`` is either a constant known while compiling or a virtual register.
Keeping constants symbolic until something forces a register is the whole
optimisation story of this compiler: index expressions such as
``((a0 * 2) + 0)`` or an unrolled loop's ``(_ks / 64 % 3) * 64 + _ki`` fold
to fewer instructions, and an index that folds completely becomes an
immediate address offset (``[%rd3+12]``) instead of a multiply-add chain.

Integer semantics are C's (what Emmy's IR means): ``/`` truncates toward
zero and ``%`` takes the sign of the dividend.  Division and remainder by a
power of two become a shift / mask only when the dividend is provably
non-negative, which the code generator tracks per register (thread ids,
axis indices and products/sums of them are non-negative).
"""

from __future__ import annotations

from dataclasses import dataclass

from .emitter import Emitter, f32_hex


@dataclass(frozen=True)
class Val:
    kind: str     # "i" = int32, "f" = f32, "p" = predicate
    v: object     # a Python int / float / bool constant, or a register name

    @property
    def is_const(self) -> bool:
        return not isinstance(self.v, str)

    def text(self) -> str:
        """The operand as PTX source."""
        if not self.is_const:
            return self.v
        if self.kind == "f":
            return f32_hex(self.v)
        if self.kind == "p":
            raise ValueError("a constant predicate has no operand form; materialise it first")
        return str(int(self.v))


def I(v) -> Val:  # noqa: E743 - short constructor names read well in the lowering code
    return Val("i", v)


def F(v) -> Val:
    return Val("f", v)


def c_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def c_rem(a: int, b: int) -> int:
    return a - b * c_div(a, b)


def _pow2(n: int) -> int | None:
    return n.bit_length() - 1 if n > 0 and n & (n - 1) == 0 else None


class IntOps:
    """int32 / predicate arithmetic with constant folding."""

    def __init__(self, em: Emitter) -> None:
        self.em = em
        self.nonneg: set[str] = set()   # registers proven >= 0

    # ---- helpers -------------------------------------------------------------
    def is_nonneg(self, a: Val) -> bool:
        return (a.is_const and a.v >= 0) or (not a.is_const and a.v in self.nonneg)

    def as_reg(self, a: Val) -> str:
        """Force a value into a register (some operand slots refuse immediates)."""
        if not a.is_const:
            return a.v
        if a.kind == "p":
            t, r = self.em.reg("r"), self.em.reg("p")
            self.em.ins("mov.u32", t, "1" if a.v else "0")
            self.em.ins("setp.ne.s32", r, t, "0")
            return r
        r = self.em.reg("f" if a.kind == "f" else "r")
        self.em.ins("mov.f32" if a.kind == "f" else "mov.u32", r, a.text())
        if a.kind == "i" and a.v >= 0:
            self.nonneg.add(r)
        return r

    def _bin(self, opcode: str, a: Val, b: Val, nonneg: bool, commutative: bool) -> Val:
        if a.is_const and commutative:
            a, b = b, a
        d = self.em.reg("r")
        self.em.ins(opcode, d, self.as_reg(a), b.text())
        if nonneg:
            self.nonneg.add(d)
        return I(d)

    # ---- arithmetic ------------------------------------------------------------
    def add(self, a: Val, b: Val) -> Val:
        if a.is_const and b.is_const:
            return I(int(a.v) + int(b.v))
        if a.is_const and a.v == 0:
            return b
        if b.is_const and b.v == 0:
            return a
        return self._bin("add.s32", a, b, self.is_nonneg(a) and self.is_nonneg(b), True)

    def sub(self, a: Val, b: Val) -> Val:
        if a.is_const and b.is_const:
            return I(int(a.v) - int(b.v))
        if b.is_const and b.v == 0:
            return a
        if a == b:
            return I(0)
        return self._bin("sub.s32", a, b, False, False)

    def mul(self, a: Val, b: Val) -> Val:
        if a.is_const and b.is_const:
            return I(int(a.v) * int(b.v))
        if (a.is_const and a.v == 0) or (b.is_const and b.v == 0):
            return I(0)
        if a.is_const and a.v == 1:
            return b
        if b.is_const and b.v == 1:
            return a
        return self._bin("mul.lo.s32", a, b, self.is_nonneg(a) and self.is_nonneg(b), True)

    def div(self, a: Val, b: Val) -> Val:
        if b.is_const and b.v == 0:
            raise ZeroDivisionError("integer division by a literal zero in the IR")
        if a.is_const and b.is_const:
            return I(c_div(int(a.v), int(b.v)))
        if b.is_const and b.v == 1:
            return a
        if a.is_const and a.v == 0:
            return I(0)
        nn = self.is_nonneg(a) and self.is_nonneg(b)
        if b.is_const and nn and (k := _pow2(int(b.v))) is not None:
            return self._bin("shr.u32", a, I(k), True, False)
        return self._bin("div.s32", a, b, nn, False)

    def rem(self, a: Val, b: Val) -> Val:
        if a.is_const and b.is_const:
            return I(c_rem(int(a.v), int(b.v)))
        if b.is_const and b.v == 1:
            return I(0)
        if a.is_const and a.v == 0:
            return I(0)
        nn = self.is_nonneg(a) and self.is_nonneg(b)
        if b.is_const and nn and _pow2(int(b.v)) is not None:
            return self._bin("and.b32", a, I(int(b.v) - 1), True, False)
        return self._bin("rem.s32", a, b, nn, False)

    def bitop(self, op: str, a: Val, b: Val) -> Val:
        if a.is_const and b.is_const:
            x, y = int(a.v), int(b.v)
            return I({"&": x & y, "|": x | y, "^": x ^ y, "<<": x << y, ">>": x >> y}[op])
        opcode = {"&": "and.b32", "|": "or.b32", "^": "xor.b32", "<<": "shl.b32", ">>": "shr.s32"}[op]
        nn = self.is_nonneg(a) and (op != "<<") and self.is_nonneg(b)
        return self._bin(opcode, a, b, nn, op in "&|^")

    def min_(self, a: Val, b: Val) -> Val:
        if a.is_const and b.is_const:
            return I(min(int(a.v), int(b.v)))
        return self._bin("min.s32", a, b, self.is_nonneg(a) and self.is_nonneg(b), True)

    # ---- comparisons and logic -------------------------------------------------
    _SWAP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}
    _PY = {"lt": lambda x, y: x < y, "le": lambda x, y: x <= y, "gt": lambda x, y: x > y,
           "ge": lambda x, y: x >= y, "eq": lambda x, y: x == y, "ne": lambda x, y: x != y}

    def cmp(self, cc: str, a: Val, b: Val, ty: str = "s32") -> Val:
        if a.is_const and b.is_const:
            return Val("p", bool(self._PY[cc](a.v, b.v)))
        if a.is_const:
            a, b, cc = b, a, self._SWAP[cc]
        p = self.em.reg("p")
        self.em.ins(f"setp.{cc}.{ty}", p, a.v, b.text())
        return Val("p", p)

    def land(self, a: Val, b: Val) -> Val:
        for x, y in ((a, b), (b, a)):
            if x.is_const:
                return y if x.v else Val("p", False)
        p = self.em.reg("p")
        self.em.ins("and.pred", p, a.v, b.v)
        return Val("p", p)

    def lor(self, a: Val, b: Val) -> Val:
        for x, y in ((a, b), (b, a)):
            if x.is_const:
                return Val("p", True) if x.v else y
        p = self.em.reg("p")
        self.em.ins("or.pred", p, a.v, b.v)
        return Val("p", p)

    def lnot(self, a: Val) -> Val:
        if a.is_const:
            return Val("p", not a.v)
        p = self.em.reg("p")
        self.em.ins("not.pred", p, a.v)
        return Val("p", p)

    def select(self, c: Val, a: Val, b: Val) -> Val:
        """``c ? a : b`` for two values of the same kind."""
        if c.is_const:
            return a if c.v else b
        if a == b:
            return a
        kind = a.kind
        if kind == "p":
            p = self.em.reg("p")
            t, f = self.as_reg(a), self.as_reg(b)
            # p = (c & a) | (!c & b)
            self.em.ins("and.pred", p, c.v, t)
            q = self.em.reg("p")
            self.em.ins("and.pred", q, f"!{c.v}", f)
            self.em.ins("or.pred", p, p, q)
            return Val("p", p)
        d = self.em.reg("f" if kind == "f" else "r")
        ty = "f32" if kind == "f" else "s32"
        self.em.ins(f"selp.{ty}", d, a.text(), b.text(), c.v)
        if kind == "i" and self.is_nonneg(a) and self.is_nonneg(b):
            self.nonneg.add(d)
        return Val(kind, d)

    def to_f32(self, a: Val) -> Val:
        if a.kind == "f":
            return a
        if a.is_const:
            return F(float(a.v))
        d = self.em.reg("f")
        if a.kind == "p":
            self.em.ins("selp.f32", d, f32_hex(1.0), f32_hex(0.0), a.v)
        else:
            self.em.ins("cvt.rn.f32.s32", d, a.v)
        return F(d)

    def to_i32(self, a: Val) -> Val:
        if a.kind == "i":
            return a
        if a.is_const:
            return I(int(a.v))
        d = self.em.reg("r")
        if a.kind == "p":
            self.em.ins("selp.s32", d, "1", "0", a.v)
        else:
            self.em.ins("cvt.rzi.s32.f32", d, a.v)
        return I(d)
