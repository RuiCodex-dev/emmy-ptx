"""Compile-time values and the integer / predicate arithmetic on them.

A ``Val`` is a constant known while compiling, a virtual register, or (for
integers) an *affine form*: ``c0 + c1*r1 + c2*r2 + ...`` over registers.

Keeping integer index arithmetic symbolic is what makes the generated code
small.  Emmy's indices are full of structure that folds away:

* ``((a1_b * 64) + (a1_u * 2) + 1) - (a1_b * 64)`` cancels to ``a1_u*2 + 1``;
* inside an unrolled loop ``_ki`` is a constant, so a whole column of loads
  ``_b_smem[(stage*64 + _ki) * 64 + a1_u*2 + c]`` shares one variable part
  ``stage*4096 + a1_u*2`` and differs only by a constant;
* that constant becomes an immediate address offset, ``[%r7+260]``.

An affine form is turned into instructions only when an instruction needs it
(``materialise``), and every materialisation and every opaque operation
(multiply of two registers, division, remainder) is common-subexpression
cached.  The cache is scoped like the generated code: entries created inside
a loop body or an ``if`` arm are dropped when the region ends, so a cached
register is only reused where its definition dominates the use.

Integer semantics are C's (what Emmy's IR means): ``/`` truncates toward
zero and ``%`` takes the sign of the dividend.  Division and remainder by a
power of two become a shift / mask only when the dividend is provably
non-negative, which is tracked per register (thread ids, axis indices and
sums/products of them are non-negative).
"""

from __future__ import annotations

from dataclasses import dataclass

from .emitter import Emitter, f32_hex


@dataclass(frozen=True)
class Lin:
    """Affine integer form: ``const + sum(coeff * register)``."""
    terms: tuple[tuple[str, int], ...]   # sorted by register name, no zero coefficients
    const: int = 0

    @staticmethod
    def of(terms: dict[str, int], const: int) -> "Lin":
        return Lin(tuple(sorted((r, c) for r, c in terms.items() if c)), const)


@dataclass(frozen=True)
class Val:
    kind: str     # "i" = int32, "f" = f32, "p" = predicate
    v: object     # Python constant | register name | Lin (integers only)

    @property
    def is_const(self) -> bool:
        return not isinstance(self.v, (str, Lin))

    @property
    def is_reg(self) -> bool:
        return isinstance(self.v, str)

    def text(self) -> str:
        """The operand as PTX source (registers and constants only)."""
        if isinstance(self.v, Lin):
            raise ValueError("affine value must be materialised before use as an operand")
        if isinstance(self.v, str):
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
    """int32 / predicate arithmetic: constant folding, affine forms, scoped CSE."""

    def __init__(self, em: Emitter) -> None:
        self.em = em
        self.nonneg: set[str] = set()      # registers proven >= 0
        self.scopes: list[dict] = [{}]     # CSE: key -> register, innermost scope last

    # ---- CSE scopes ------------------------------------------------------------
    def push(self) -> None:
        self.scopes.append({})

    def pop(self) -> None:
        self.scopes.pop()

    def forget_all(self) -> None:
        """A register some cache entry may depend on was overwritten: drop everything."""
        for s in self.scopes:
            s.clear()

    def _cached(self, key):
        for s in reversed(self.scopes):
            if key in s:
                return s[key]
        return None

    def _remember(self, key, reg: str) -> None:
        self.scopes[-1][key] = reg

    # ---- affine views ------------------------------------------------------------
    @staticmethod
    def lin(a: Val) -> Lin:
        if isinstance(a.v, Lin):
            return a.v
        if isinstance(a.v, str):
            return Lin(((a.v, 1),), 0)
        return Lin((), int(a.v))

    @staticmethod
    def mk(lin: Lin) -> Val:
        if not lin.terms:
            return I(lin.const)
        if lin.const == 0 and len(lin.terms) == 1 and lin.terms[0][1] == 1:
            return I(lin.terms[0][0])
        return I(lin)

    def is_nonneg(self, a: Val) -> bool:
        if a.kind != "i":
            return False
        lin = self.lin(a)
        return lin.const >= 0 and all(c > 0 and r in self.nonneg for r, c in lin.terms)

    # ---- materialisation ---------------------------------------------------------
    def as_reg(self, a: Val) -> str:
        """Force a value into a register (instructions need register operands)."""
        if a.is_reg:
            return a.v
        if a.kind == "i" and isinstance(a.v, Lin):
            return self._materialise(a.v)
        if a.kind == "p":
            t, r = self.em.reg("r"), self.em.reg("p")
            self.em.ins("mov.u32", t, "1" if a.v else "0")
            self.em.ins("setp.ne.s32", r, t, "0")
            return r
        key = ("const", a.kind, a.v)
        r = self._cached(key)
        if r is None:
            r = self.em.reg("f" if a.kind == "f" else "r")
            self.em.ins("mov.f32" if a.kind == "f" else "mov.u32", r, a.text())
            if a.kind == "i" and a.v >= 0:
                self.nonneg.add(r)
            self._remember(key, r)
        return r

    def operand(self, a: Val) -> str:
        """Register or immediate text, materialising an affine form if needed."""
        if a.kind == "i" and isinstance(a.v, Lin):
            return self._materialise(a.v)
        return a.text()

    def _materialise(self, lin: Lin) -> str:
        key = ("lin", lin)
        r = self._cached(key)
        if r is not None:
            return r
        nn = lin.const >= 0 and all(c > 0 and t in self.nonneg for t, c in lin.terms)
        if lin.const != 0:
            # variable part first: it is a CSE candidate of its own, since several
            # constant offsets of one base are the common case
            acc = self._materialise(Lin(lin.terms, 0))
            r = self.em.reg("r")
            self.em.ins("add.s32", r, acc, str(lin.const))
        else:
            acc = None
            for reg, c in lin.terms:   # c1*r1 (+ c2*r2 ...), one mad per extra term
                if acc is None and c == 1:
                    acc = reg
                    continue
                d = self.em.reg("r")
                if acc is None:
                    self.em.ins("mul.lo.s32", d, reg, str(c))
                elif c == 1:
                    self.em.ins("add.s32", d, acc, reg)
                else:
                    self.em.ins("mad.lo.s32", d, reg, str(c), acc)
                acc = d
            r = acc
        if nn:
            self.nonneg.add(r)
        self._remember(key, r)
        return r

    def sym_reg(self, symbol: str) -> str:
        """A register holding a shared-memory symbol's address (cached)."""
        key = ("sym", symbol)
        r = self._cached(key)
        if r is None:
            r = self.em.reg("r")
            self.em.ins("mov.u32", r, symbol)
            self.nonneg.add(r)
            self._remember(key, r)
        return r

    def global_addr(self, base: str, index: Lin, esize: int) -> str:
        """64-bit ``base + index * esize`` (cached): one mul.wide + add per distinct index."""
        key = ("gaddr", base, index, esize)
        r = self._cached(key)
        if r is None:
            wkey = ("wide", index, esize)          # the byte offset is shared by every buffer
            wide = self._cached(wkey)
            if wide is None:
                idx = self.mk(index)
                ireg = self.as_reg(idx)
                wide = self.em.reg("rd")
                self.em.ins(f"mul.wide.{'u32' if self.is_nonneg(idx) else 's32'}", wide, ireg, str(esize))
                self._remember(wkey, wide)
            r = self.em.reg("rd")
            self.em.ins("add.s64", r, base, wide)
            self._remember(key, r)
        return r

    def _op(self, opcode: str, a: Val, b: Val, nonneg: bool, commutative: bool) -> Val:
        """One opaque integer instruction, CSE-cached by (opcode, operands)."""
        if a.is_const and commutative:
            a, b = b, a
        ra = self.as_reg(a)
        rb = self.operand(b)
        key = ("op", opcode, ra, rb)
        d = self._cached(key)
        if d is None:
            d = self.em.reg("r")
            self.em.ins(opcode, d, ra, rb)
            if nonneg:
                self.nonneg.add(d)
            self._remember(key, d)
        return I(d)

    # ---- arithmetic ---------------------------------------------------------------
    def add(self, a: Val, b: Val) -> Val:
        la, lb = self.lin(a), self.lin(b)
        terms = dict(la.terms)
        for r, c in lb.terms:
            terms[r] = terms.get(r, 0) + c
        return self.mk(Lin.of(terms, la.const + lb.const))

    def sub(self, a: Val, b: Val) -> Val:
        return self.add(a, self.scale(b, -1))

    def scale(self, a: Val, k: int) -> Val:
        la = self.lin(a)
        return self.mk(Lin.of({r: c * k for r, c in la.terms}, la.const * k))

    def mul(self, a: Val, b: Val) -> Val:
        if b.is_const:
            return self.scale(a, int(b.v))
        if a.is_const:
            return self.scale(b, int(a.v))
        return self._op("mul.lo.s32", a, b, self.is_nonneg(a) and self.is_nonneg(b), True)

    def div(self, a: Val, b: Val) -> Val:
        if b.is_const and b.v == 0:
            raise ZeroDivisionError("integer division by a literal zero in the IR")
        if a.is_const and b.is_const:
            return I(c_div(int(a.v), int(b.v)))
        if b.is_const:
            d = int(b.v)
            if d == 1:
                return a
            la = self.lin(a)
            # every term and the constant divisible by d: exact, sign-agnostic
            if la.const % d == 0 and all(c % d == 0 for _, c in la.terms):
                return self.mk(Lin.of({r: c // d for r, c in la.terms}, la.const // d))
        if a.is_const and a.v == 0:
            return I(0)
        nn = self.is_nonneg(a) and self.is_nonneg(b)
        if b.is_const and nn and (k := _pow2(int(b.v))) is not None:
            return self._op("shr.u32", a, I(k), True, False)
        return self._op("div.s32", a, b, nn, False)

    def rem(self, a: Val, b: Val) -> Val:
        if a.is_const and b.is_const:
            return I(c_rem(int(a.v), int(b.v)))
        if b.is_const and b.v == 1:
            return I(0)
        if a.is_const and a.v == 0:
            return I(0)
        nn = self.is_nonneg(a) and self.is_nonneg(b)
        if b.is_const and nn:
            d = int(b.v)
            la = self.lin(a)
            # (d*k*x + c) % d == c % d when everything is non-negative
            if all(c % d == 0 for _, c in la.terms):
                return I(la.const % d)
            if _pow2(d) is not None:
                return self._op("and.b32", a, I(d - 1), True, False)
        return self._op("rem.s32", a, b, nn, False)

    def bitop(self, op: str, a: Val, b: Val) -> Val:
        if a.is_const and b.is_const:
            x, y = int(a.v), int(b.v)
            return I({"&": x & y, "|": x | y, "^": x ^ y, "<<": x << y, ">>": x >> y}[op])
        if op == "<<" and b.is_const:
            return self.scale(a, 1 << int(b.v))
        opcode = {"&": "and.b32", "|": "or.b32", "^": "xor.b32", "<<": "shl.b32", ">>": "shr.s32"}[op]
        nn = self.is_nonneg(a) and self.is_nonneg(b)
        return self._op(opcode, a, b, nn, op in "&|^")

    def min_(self, a: Val, b: Val) -> Val:
        if a.is_const and b.is_const:
            return I(min(int(a.v), int(b.v)))
        return self._op("min.s32", a, b, self.is_nonneg(a) and self.is_nonneg(b), True)

    # ---- comparisons and logic ----------------------------------------------------
    _SWAP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}
    _PY = {"lt": lambda x, y: x < y, "le": lambda x, y: x <= y, "gt": lambda x, y: x > y,
           "ge": lambda x, y: x >= y, "eq": lambda x, y: x == y, "ne": lambda x, y: x != y}

    def cmp(self, cc: str, a: Val, b: Val, ty: str = "s32") -> Val:
        if a.is_const and b.is_const:
            return Val("p", bool(self._PY[cc](a.v, b.v)))
        if a.is_const:
            a, b, cc = b, a, self._SWAP[cc]
        p = self.em.reg("p")
        self.em.ins(f"setp.{cc}.{ty}", p, self.as_reg(a), self.operand(b))
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
            t, f = self.as_reg(a), self.as_reg(b)
            p, q = self.em.reg("p"), self.em.reg("p")
            self.em.ins("and.pred", p, c.v, t)          # p = (c & a) | (!c & b)
            self.em.ins("and.pred", q, f"!{c.v}", f)
            self.em.ins("or.pred", p, p, q)
            return Val("p", p)
        d = self.em.reg("f" if kind == "f" else "r")
        ty = "f32" if kind == "f" else "s32"
        self.em.ins(f"selp.{ty}", d, self.operand(a), self.operand(b), c.v)
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
            self.em.ins("cvt.rn.f32.s32", d, self.as_reg(a))
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
