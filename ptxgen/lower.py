"""Lowering: walk the Kernel IR statement tree and emit PTX.

The model is deliberately close to the C the IR describes.  Every IR name
(``in0``, ``acc0``, ``v4__gn__c0_0`` ...) owns one virtual register for its
whole lifetime and an assignment to a name that already exists is a ``mov``
into that register.  That gives loop-carried values (accumulators, the online
softmax state) the right semantics without building SSA phis: the loop body
is emitted once, and every iteration reads and writes the same registers.
``ptxas`` converts the result to SSA and allocates the physical registers.

Only two kinds of names are bound to compile-time constants instead of
registers: ``Let`` literals and the induction variable of a loop that is
unrolled here.  Those never change inside the region that reads them, so
folding them is safe and turns unrolled shared-memory indices into immediate
address offsets.
"""

from __future__ import annotations

import math

from .emitter import Emitter, f32_hex
from .ops import FloatOps
from .values import F, I, IntOps, Lin, Val

ESIZE = {"f32": 4, "float": 4, "i32": 4, "int": 4}

# Fully unroll a static loop marked ``unroll`` only while the copies stay
# small: ptxas gets 30 s per kernel on the runner and code size is not free.
UNROLL_BUDGET = 6000   # trip count x statements in the body

_INT_BINOPS = {"+": IntOps.add, "-": IntOps.sub, "*": IntOps.mul, "/": IntOps.div,
               "//": IntOps.div, "%": IntOps.rem}
_CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
_BUILTINS = {"threadIdx.x": "%tid.x", "threadIdx.y": "%tid.y", "threadIdx.z": "%tid.z",
             "blockIdx.x": "%ctaid.x", "blockIdx.y": "%ctaid.y", "blockIdx.z": "%ctaid.z",
             "blockDim.x": "%ntid.x", "blockDim.y": "%ntid.y", "blockDim.z": "%ntid.z",
             "gridDim.x": "%nctaid.x", "gridDim.y": "%nctaid.y", "gridDim.z": "%nctaid.z"}


# statement fields that DEFINE names rather than read them (ignored when counting uses)
_DEF_FIELDS = {"Assign": ("name",), "Let": ("name",), "Init": ("name",), "Load": ("names",),
               "Select": ("name",)}


class CompileError(Exception):
    pass


def dtype_name(dt) -> str:
    if dt is None:
        return "f32"
    return getattr(dt, "name", dt)


def kind_of(dt) -> str:
    n = dtype_name(dt)
    if n in ("bool",):
        return "p"
    if n in ("i32", "int", "i64", "u32"):
        return "i"
    if n in ("f32", "float"):
        return "f"
    raise CompileError(f"value dtype {n!r} is not supported (this compiler handles f32, i32 and bool values)")


def _tname(node) -> str:
    return type(node).__name__


def static_extent(axis) -> int:
    e = axis.extent
    if hasattr(e, "is_static") and not e.is_static:
        raise CompileError(f"axis {axis.name!r} has a symbolic extent; only static shapes are supported")
    return int(e.as_static()) if hasattr(e, "as_static") else int(e)


class Buffer:
    def __init__(self, name: str, shape: tuple[int, ...], dtype: str, space: str) -> None:
        self.name = name
        self.shape = tuple(int(d) for d in shape)
        self.dtype = dtype
        self.space = space            # "global" or "shared"
        self.base = ""                # global: 64-bit base register; shared: symbol
        self.base_off = 0             # shared buffers carved from the dynamic pool: byte offset
        self.readonly = False
        if dtype not in ESIZE:
            raise CompileError(f"buffer {name!r} has dtype {dtype!r}; only 32-bit buffers are supported")
        self.esize = ESIZE[dtype]


class CodeGen:
    def __init__(self, buffers: dict[str, Buffer], launch_block: int | None) -> None:
        self.em = Emitter()
        self.ints = IntOps(self.em)
        self.fl = FloatOps(self.em, self.ints)
        self.buffers = buffers
        self.launch_block = launch_block
        self.env: dict[str, Val] = {}
        self.owner: dict[str, str] = {}   # register -> the IR name that owns it
        self.uses: dict[str, int] = {}    # IR name -> number of references (see count_uses)

    def count_uses(self, stmts) -> None:
        """Count every reference to every name, conservatively: any string field
        or ``Var`` anywhere in a statement counts, except a statement's own
        definition target.  Over-counting only ever disables a fusion."""
        import dataclasses

        def visit(x, skip=None):
            if isinstance(x, str):
                if x != skip:
                    self.uses[x] = self.uses.get(x, 0) + 1
            elif isinstance(x, (list, tuple)):
                for y in x:
                    visit(y)
            elif dataclasses.is_dataclass(x) and not isinstance(x, type):
                defines = _DEF_FIELDS.get(_tname(x), ())
                for f in dataclasses.fields(x):
                    if f.name not in defines:
                        visit(getattr(x, f.name))
            elif hasattr(x, "__iter__") and not isinstance(x, (bytes, dict)):
                for y in x:
                    visit(y)

        visit(stmts)

    # ======================================================================
    # names
    # ======================================================================
    def lookup(self, name: str) -> Val:
        v = self.env.get(name)
        if v is None:
            raise CompileError(f"IR name {name!r} is read before it is defined")
        return v

    def define(self, name: str, val: Val) -> None:
        """Bind an IR name.  An existing name keeps its register (mov into it);
        a new name takes over ``val``'s register if nobody owns it yet."""
        cur = self.env.get(name)
        if cur is not None and cur.is_reg and self.owner.get(cur.v) == name:
            if cur.kind != val.kind:
                val = self.ints.to_f32(val) if cur.kind == "f" else self.ints.to_i32(val)
            if val != cur:
                self._mov(cur, val)
                if cur.kind == "i":
                    self.ints.forget_all()   # cached index math may have read the old value
            return
        if val.kind == "i" and not val.is_const and not val.is_reg:
            val = I(self.ints.as_reg(val))   # names hold registers, never affine forms
        if val.is_reg and val.v not in self.owner:
            self.owner[val.v] = name
            self.env[name] = val
            return
        reg = self.em.reg({"f": "f", "i": "r", "p": "p"}[val.kind])
        self.owner[reg] = name
        out = Val(val.kind, reg)
        self._mov(out, val)
        if val.kind == "i" and self.ints.is_nonneg(val):
            self.ints.nonneg.add(reg)
        self.env[name] = out

    def bind_const(self, name: str, val: Val) -> None:
        self.env[name] = val

    def _mov(self, dst: Val, src: Val) -> None:
        if dst.kind == "f":
            self.em.ins("mov.f32", dst.v, src.text())
        elif dst.kind == "i":
            self.em.ins("mov.u32", dst.v, self.ints.operand(src))
        else:
            self.em.ins("mov.pred", dst.v, self.ints.as_reg(src))

    # ======================================================================
    # expressions (indices, predicates, loop bounds)
    # ======================================================================
    def expr(self, e) -> Val:
        k = _tname(e)
        if k == "Literal":
            v = e.value
            if isinstance(v, bool):
                return Val("p", v)
            if isinstance(v, float) or str(e.dtype) in ("float", "f32", "double", "f64"):
                return F(float(v))
            return I(int(v))
        if k == "Var":
            return self.lookup(e.name)
        if k == "Builtin":
            sreg = _BUILTINS.get(e.name)
            if sreg is None:
                raise CompileError(f"unknown builtin {e.name!r}")
            r = self.em.reg("r")
            self.em.ins("mov.u32", r, sreg)
            self.ints.nonneg.add(r)
            return I(r)
        if k == "BinaryExpr":
            return self.binexpr(e.op, self.expr(e.left), self.expr(e.right))
        if k == "TernaryExpr":
            c = self.as_pred(self.expr(e.cond))
            if c.is_const:                         # fold: only the taken side is emitted
                return self.expr(e.if_true if c.v else e.if_false)
            a, b = self.expr(e.if_true), self.expr(e.if_false)
            if a.kind != b.kind:
                a, b = self.ints.to_f32(a), self.ints.to_f32(b)
            return self.ints.select(c, a, b)
        if k == "CastExpr":
            v = self.expr(e.expr)
            target = kind_of(e.dtype)
            if target == "f":
                return self.ints.to_f32(v)
            if target == "i":
                return self.ints.to_i32(v)
            return self.as_pred(v)
        if k == "FuncCallExpr":
            return self.fl.apply(e.name, [self.expr(a) for a in e.args])
        if k == "FlatIndex":
            return self.flat_index(e.buffer, e.index)
        raise CompileError(f"unsupported expression node {k}")

    def as_pred(self, v: Val) -> Val:
        if v.kind == "p":
            return v
        if v.kind == "i":
            return self.ints.cmp("ne", v, I(0))
        return self.fl.apply("not_equal", [v, F(0.0)])

    def binexpr(self, op: str, a: Val, b: Val) -> Val:
        if op in ("&&", "||"):
            a, b = self.as_pred(a), self.as_pred(b)
            return self.ints.land(a, b) if op == "&&" else self.ints.lor(a, b)
        if a.kind == "p" and b.kind == "p" and op in ("&", "|", "^", "==", "!="):
            if op == "&":
                return self.ints.land(a, b)
            if op == "|":
                return self.ints.lor(a, b)
            a, b = self.ints.to_i32(a), self.ints.to_i32(b)
        if a.kind == "f" or b.kind == "f":
            a, b = self.ints.to_f32(a), self.ints.to_f32(b)
            if op in _CMP:
                name = {"lt": "less", "le": "less_equal", "gt": "greater", "ge": "greater_equal",
                        "eq": "equal", "ne": "not_equal"}[_CMP[op]]
                return self.fl.apply(name, [a, b])
            fop = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}.get(op)
            if fop is None:
                raise CompileError(f"float operator {op!r} in an index expression")
            return self.fl.apply(fop, [a, b])
        a, b = self.ints.to_i32(a), self.ints.to_i32(b)
        if op in _CMP:
            return self.ints.cmp(_CMP[op], a, b)
        if op in _INT_BINOPS:
            return _INT_BINOPS[op](self.ints, a, b)
        if op in ("&", "|", "^", "<<", ">>"):
            return self.ints.bitop(op, a, b)
        raise CompileError(f"unsupported operator {op!r}")

    # ======================================================================
    # memory
    # ======================================================================
    def buffer(self, name: str) -> Buffer:
        buf = self.buffers.get(name)
        if buf is None:
            raise CompileError(f"unknown buffer {name!r}")
        return buf

    def flatten(self, buf: Buffer, coords: list[Val]) -> Val:
        """Row-major element offset of a coordinate tuple in ``buf``."""
        if not coords:
            return I(0)
        if len(coords) == 1 or len(coords) != len(buf.shape):
            if len(coords) != 1:
                raise CompileError(f"{len(coords)}-d index into {buf.name}{list(buf.shape)}")
            return coords[0]
        flat = coords[0]
        for c, d in zip(coords[1:], buf.shape[1:]):
            flat = self.ints.add(self.ints.mul(flat, I(d)), c)
        return flat

    def flat_index(self, name: str, index) -> Val:
        buf = self.buffer(name)
        return self.flatten(buf, [self.ints.to_i32(self.expr(e)) for e in index])

    def address(self, buf: Buffer, flat: Val, extra_bytes: int = 0) -> str:
        """A PTX address operand for element ``flat`` of ``buf`` (+ a byte offset).

        The index is split into its variable part, which goes into a (cached)
        base register, and its constant part, which becomes the immediate
        offset: ``x[a0*4 + 3]`` is ``[%rd9+12]`` with ``%rd9 = x + a0*16``."""
        lin = self.ints.lin(flat)
        off = lin.const * buf.esize + extra_bytes + buf.base_off
        var = Lin(lin.terms, 0)
        if not var.terms:
            base = buf.base
        elif buf.space == "shared":
            sym = self.ints.sym_reg(buf.base)
            base = self.ints.as_reg(self.ints.add(I(sym), self.ints.scale(self.ints.mk(var), buf.esize)))
        else:
            base = self.ints.global_addr(buf.base, var, buf.esize)
        if off > 0:
            return f"[{base}+{off}]"
        return f"[{base}-{-off}]" if off < 0 else f"[{base}]"

    def swizzle(self, mode: str, flat: Val) -> Val:
        """Emmy's software smem swizzle: e ^ (((e >> shift) & mask) << 3)."""
        if not mode or mode == "NONE":
            return flat
        base, _, shift_override = mode.partition("@")
        table = {"B128": (6, 0x7), "B64": (6, 0x3), "B32": (6, 0x1)}
        if base not in table:
            raise CompileError(f"unsupported swizzle mode {mode!r}")
        shift, mask = table[base]
        if shift_override:
            shift = int(shift_override)
        sel = self.ints.bitop("<<", self.ints.bitop("&", self.ints.bitop(">>", flat, I(shift)), I(mask)), I(3))
        return self.ints.bitop("^", flat, sel)

    @staticmethod
    def _vec(n: int) -> str:
        return {1: "", 2: ".v2", 4: ".v4"}[n]

    # ======================================================================
    # statements
    # ======================================================================
    def body(self, stmts) -> None:
        stmts = list(stmts)
        i = 0
        while i < len(stmts):
            s = stmts[i]
            if i + 1 < len(stmts) and self._fuse_mul_accum(s, stmts[i + 1]):
                i += 2
                continue
            handler = getattr(self, "s_" + _tname(s), None)
            if handler is None:
                raise CompileError(f"statement {_tname(s)} is not supported by this compiler")
            handler(s)
            i += 1

    def _fuse_mul_accum(self, s, nxt) -> bool:
        """``t = a * b; acc <- add(acc, t)`` -> ``acc = fma(a, b, acc)``.

        Only for an Accum (a reduction, always graded with a tolerance because
        summation order is free) and only when the product is used nowhere
        else, so pointwise kernels keep numpy's two-rounding semantics."""
        if not (_tname(s) == "Assign" and _tname(nxt) == "Accum"):
            return False
        if s.op.name != "multiply" or nxt.op.name != "add" or nxt.value != s.name:
            return False
        if self.uses.get(s.name, 0) != 1 or len(s.args) != 2:
            return False
        a, b = (self.lookup(x) for x in s.args)
        if a.kind != "f" or b.kind != "f" or nxt.name not in self.env:
            return False
        base = self.lookup(nxt.base) if nxt.base else self.lookup(nxt.name)
        self.define(nxt.name, self.fl.fma(a, b, base))
        return True

    # ---- launch geometry -------------------------------------------------------
    def s_Tile(self, s) -> None:
        extents = [static_extent(a) for a in s.axes]
        n = math.prod(extents) if extents else 1
        if n > 2**31 - 1:
            raise CompileError("iteration space exceeds int32")
        em, ints = self.em, self.ints
        em.comment(f"Tile[{', '.join(a.name for a in s.axes)}] N={n}"
                   + (f" block_threads={s.block_threads}" if s.block_threads else ""))
        tid, cta = self.env["%tid"], self.env["%ctaid"]
        bt = s.block_threads
        if s.aux_threads:
            # warp-specialised producer band: cells are decoded modulo block_threads
            gid = ints.add(ints.mul(cta, I(bt)), ints.rem(tid, I(bt)))
        else:
            block = self.launch_block or bt
            if block:
                gid = ints.add(ints.mul(cta, I(block)), tid)
            else:
                nt = em.reg("r")
                em.ins("mov.u32", nt, "%ntid.x")
                ints.nonneg.add(nt)
                gid = ints.add(ints.mul(cta, I(nt)), tid)
        cells = bt if bt is not None else 256
        guard = bool(s.aux_threads) or n % cells != 0 or (
            self.launch_block is not None and bt is None and n % self.launch_block != 0)
        end = em.label("tile_end")
        if guard:
            p = ints.cmp("ge", gid, I(n))
            em.ins("bra", end, pred=p.v, comment=f"threads past the {n} cells do nothing")
        strides = [1] * len(extents)
        acc = 1
        for i in range(len(extents) - 1, -1, -1):
            strides[i] = acc
            acc *= extents[i]
        raster = self._raster(s, extents)
        for i, ax in enumerate(s.axes):
            if raster is not None and i in raster:
                continue
            q = gid if strides[i] == 1 else ints.div(gid, I(strides[i]))
            val = q if i == 0 else ints.rem(q, I(extents[i]))
            self.define(ax.name, val)
        if raster is not None:
            self._raster_decode(s, extents, strides, gid, raster)
        self.body(s.body)
        if guard:
            em.place(end)

    def _raster(self, s, extents):
        """Indices of the two block axes that grouped CTA rasterisation remaps,
        or None when the tile uses the plain row-major order."""
        if not (s.raster_group and s.raster_axes):
            return None
        names = [a.name for a in s.axes]
        if s.raster_axes[0] not in names or s.raster_axes[1] not in names:
            return None
        mi, ni = names.index(s.raster_axes[0]), names.index(s.raster_axes[1])
        if ni != mi + 1:
            return None
        gi = mi if s.raster_orient == "m" else ni
        if extents[gi] <= 1 or min(s.raster_group, extents[gi]) <= 1:
            return None
        return (mi, ni)

    def _raster_decode(self, s, extents, strides, gid, raster) -> None:
        """Grouped ("swizzled") CTA order: G tiles of the grouped axis run fastest
        inside each launch stripe so consecutive CTAs reuse one operand slab in L2."""
        ints = self.ints
        mi, ni = raster
        gi, oi = (mi, ni) if s.raster_orient == "m" else (ni, mi)
        g = min(s.raster_group, extents[gi])
        ea, eb = extents[gi], extents[oi]
        sub = gid if strides[ni] == 1 else ints.div(gid, I(strides[ni]))
        if mi != 0:
            sub = ints.rem(sub, I(ea * eb))
        if ea % g == 0:
            a = ints.add(ints.mul(ints.div(sub, I(g * eb)), I(g)), ints.rem(sub, I(g)))
            b = ints.div(ints.rem(sub, I(g * eb)), I(g))
        else:
            grp = ints.div(sub, I(g * eb))
            size = ints.min_(I(g), ints.sub(I(ea), ints.mul(grp, I(g))))
            inner = ints.rem(sub, I(g * eb))
            a = ints.add(ints.mul(grp, I(g)), ints.rem(inner, size))
            b = ints.div(inner, size)
        self.define(s.axes[gi].name, a)
        self.define(s.axes[oi].name, b)

    # ---- scalar compute -------------------------------------------------------
    def s_Load(self, s) -> None:
        buf = self.buffer(s.input)
        flat = self.flat_index(s.input, s.index)
        kind = kind_of(s.dtype) if s.dtype is not None else "f"
        if kind != "f" or buf.dtype not in ("f32", "float"):
            raise CompileError(f"load of {dtype_name(s.dtype)} from {buf.name} ({buf.dtype}) is not supported")
        names = list(s.names)
        regs = [self.em.reg("f") for _ in names]
        space = "shared" if buf.space == "shared" else ("global.nc" if buf.readonly else "global")
        groups = [names] if len(names) in (1, 2, 4) else [[nm] for nm in names]
        pos = 0
        for grp in groups:
            r = regs[pos:pos + len(grp)]
            addr = self.address(buf, flat, pos * buf.esize)
            dst = r[0] if len(r) == 1 else "{" + ", ".join(r) + "}"
            self.em.ins(f"ld.{space}{self._vec(len(r))}.f32", dst, addr,
                        comment=f"{', '.join(grp)} = {buf.name}[...]")
            pos += len(grp)
        for nm, r in zip(names, regs):
            self.define(nm, F(r))

    def s_Assign(self, s) -> None:
        args = [self.lookup(a) if isinstance(a, str) else self.expr(a) for a in s.args]
        out = self.fl.apply(s.op.name, args)
        if s.dtype is not None and out.kind != kind_of(s.dtype):
            want = kind_of(s.dtype)
            out = self.ints.to_f32(out) if want == "f" else (self.ints.to_i32(out) if want == "i" else self.as_pred(out))
        self.define(s.name, out)

    def s_Select(self, s) -> None:
        """First branch whose coordinate predicate holds; built back to front as
        a chain of ``selp`` so the earliest true predicate wins."""
        result = None
        for br in reversed(s.branches):
            v = self.ints.to_f32(self.lookup(br.value))
            c = self.as_pred(self.expr(br.select))
            result = v if result is None else self.ints.select(c, v, result)
        self.define(s.name, result)

    def s_Let(self, s) -> None:
        v = self.expr(s.value)
        if s.dtype is not None:
            want = kind_of(s.dtype)
            if want == "f":
                v = self.ints.to_f32(v)
            elif want == "i":
                v = self.ints.to_i32(v)
        if v.is_const:
            self.bind_const(s.name, v)   # a pure literal: fold it into every use
        else:
            self.define(s.name, v)

    def s_Accum(self, s) -> None:
        if s.name not in self.env:
            raise CompileError(f"accumulator {s.name!r} folded before it was seeded")
        base = self.lookup(s.base) if s.base else self.lookup(s.name)
        self.define(s.name, self.fl.apply(s.op.name, [base, self.lookup(s.value)]))

    def s_Init(self, s) -> None:
        kind = kind_of(s.dtype)
        val = F(float(s.identity)) if kind == "f" else I(int(s.identity))
        self._seed(s.name, val)

    def _seed(self, name: str, val: Val) -> None:
        """Start a mutable accumulator: always a register of its own."""
        reg = self.em.reg({"f": "f", "i": "r", "p": "p"}[val.kind])
        self.owner[reg] = name
        out = Val(val.kind, reg)
        self._mov(out, val)
        self.env[name] = out

    def s_Write(self, s) -> None:
        buf = self.buffer(s.output)
        flat = self.swizzle(getattr(s, "swizzle", "NONE"), self.flat_index(s.output, s.index))
        vals = [self.ints.as_reg(self.ints.to_f32(self.lookup(v))) for v in s.values]
        if s.atomic:
            if buf.space != "global":
                raise CompileError("atomic write to shared memory is not supported")
            for i, v in enumerate(vals):
                self.em.ins("red.global.add.f32", self.address(buf, flat, i * buf.esize), v,
                            comment=f"{buf.name}[...] += {s.values[i]}")
            return
        groups = [vals] if len(vals) in (1, 2, 4) else [[v] for v in vals]
        pos = 0
        space = "shared" if buf.space == "shared" else "global"
        for grp in groups:
            src = grp[0] if len(grp) == 1 else "{" + ", ".join(grp) + "}"
            self.em.ins(f"st.{space}{self._vec(len(grp))}.f32", self.address(buf, flat, pos * buf.esize), src,
                        comment=f"{buf.name}[...] = {', '.join(s.values[pos:pos + len(grp)])}")
            pos += len(grp)

    # ---- control flow ----------------------------------------------------------
    def _seed_loop_accums(self, loop) -> None:
        # Emmy's rule: a reduce loop seeds every Accum of its IMMEDIATE body to the
        # op's identity right before the loop (unless the carrier was seeded outside).
        if not getattr(loop, "seed", True):
            return
        seen = set()
        for st in loop.body:
            if _tname(st) == "Accum" and st.name not in seen:
                seen.add(st.name)
                ident = st.op.identity
                if ident is None:
                    raise CompileError(f"accumulator {st.name!r}: op {st.op.name!r} has no identity")
                kind = kind_of(st.dtype)
                self._seed(st.name, F(float(ident)) if kind == "f" else I(int(ident)))

    def _body_size(self, stmts) -> int:
        n = 0
        for st in stmts:
            n += 1
            for f in ("body", "else_body"):
                sub = getattr(st, f, None)
                if sub:
                    n += self._body_size(sub)
        return n

    def s_Loop(self, s) -> None:
        self._seed_loop_accums(s)
        extent = static_extent(s.axis)
        self._counted_loop(s.axis.name, I(0), I(extent), I(1), s.body, s.unroll)

    def s_StridedLoop(self, s) -> None:
        self._seed_loop_accums(s)
        name = s.axis.name
        start = self.ints.to_i32(self.expr(s.start))
        step = self.ints.to_i32(self.expr(s.step))
        if s.end is not None:
            end = self.ints.to_i32(self.expr(s.end))
            if not end.is_const:
                self.define(name + "_end", end)
                end = self.lookup(name + "_end")
            else:
                self.bind_const(name + "_end", end)
        else:
            end = I(static_extent(s.axis))
        self._counted_loop(name, start, end, step, s.body, s.unroll)

    def _counted_loop(self, var: str, start: Val, end: Val, step: Val, body, unroll: bool) -> None:
        em, ints = self.em, self.ints
        if start.is_const and end.is_const and step.is_const and int(step.v) > 0:
            trips = max(0, -(-(int(end.v) - int(start.v)) // int(step.v)))
            if trips == 0:
                return
            if trips == 1 or (unroll and trips * max(1, self._body_size(body)) <= UNROLL_BUDGET):
                em.comment(f"for {var} in {start.v}..{end.v}:{step.v}  (unrolled x{trips})")
                for k in range(trips):
                    self.bind_const(var, I(int(start.v) + k * int(step.v)))
                    self.body(body)
                return
        # a real loop: the induction variable is one register, the test at the top
        step_op, end_op = ints.operand(step), ints.operand(end)
        reg = em.reg("r")
        em.ins("mov.u32", reg, ints.operand(start), comment=f"for {var} (step {step_op}, end {end_op})")
        if ints.is_nonneg(start) and ints.is_nonneg(step):
            ints.nonneg.add(reg)
        self.env[var] = I(reg)
        self.owner[reg] = var
        head, done = em.label(f"{var}_head"), em.label(f"{var}_done")
        em.place(head)
        p = em.reg("p")
        em.ins("setp.ge.s32", p, reg, end_op)
        em.ins("bra", done, pred=p)
        ints.push()          # values computed in the body do not outlive one iteration
        self.body(body)
        ints.pop()
        em.ins("add.s32", reg, reg, step_op)
        em.ins("bra.uni", head)
        em.place(done)

    def s_Cond(self, s) -> None:
        c = self.as_pred(self.expr(s.cond))
        if c.is_const:                     # statically decided (e.g. inside an unrolled loop)
            self.body(s.body if c.v else s.else_body)
            return
        em = self.em
        els, end = em.label("else"), em.label("endif")
        em.ins("bra", els if s.else_body else end, pred="!" + c.v)
        self.ints.push()     # a value computed in one arm is not available after the join
        self.body(s.body)
        self.ints.pop()
        if s.else_body:
            em.ins("bra.uni", end)
            em.place(els)
            self.ints.push()
            self.body(s.else_body)
            self.ints.pop()
        em.place(end)

    # ---- shared memory and barriers ----------------------------------------------
    def s_Smem(self, s) -> None:
        pass  # declared at module scope by the kernel assembler (see kernel.py)

    def s_Sync(self, s) -> None:
        if s.warp:
            self.em.ins("bar.warp.sync", "0xffffffff")
        elif s.barrier_id == 0 and s.count is None:
            self.em.ins("bar.sync", "0")
        else:
            self.em.ins("bar.sync", str(s.barrier_id), str(s.count))

    def s_CpAsyncCopy(self, s) -> None:
        smem = self.buffer(s.smem)
        flat = self.flatten(smem, [self.ints.to_i32(self.expr(e)) for e in s.smem_index])
        if s.lane_index is not None:
            cols = smem.shape[-1]
            lane = self.ints.add(self.ints.mul(self.ints.to_i32(self.expr(s.lane_index[0])), I(cols)),
                                 self.ints.to_i32(self.expr(s.lane_index[1])))
            flat = self.ints.add(flat, lane)
        flat = self.swizzle(s.swizzle, flat)
        src = self.buffer(s.src)
        sflat = self.flat_index(s.src, s.src_index)
        dst_addr = self.address(smem, flat)
        src_addr = self.address(src, sflat)
        cache = "cg" if s.nbytes == 16 else "ca"
        self.em.ins(f"cp.async.{cache}.shared.global", dst_addr, src_addr, str(s.nbytes),
                    comment=f"{s.smem}[...] <- {s.src}[...] ({s.nbytes} B)")

    def s_CpAsyncCommit(self, s) -> None:
        self.em.ins("cp.async.commit_group")

    def s_CpAsyncWait(self, s) -> None:
        self.em.ins("cp.async.wait_group", str(s.group))

    # ---- cross-thread combines -------------------------------------------------
    def _combine(self, assigns, dtype) -> None:
        for a in assigns:
            args = [self.lookup(x) for x in a.args]
            self.define(a.name, self.fl.apply(a.op.name, args))

    def s_WarpShuffle(self, s) -> None:
        """XOR butterfly over ``length`` lanes: every lane ends with the full fold."""
        em = self.em
        mask = em.reg("r")
        em.ins("activemask.b32", mask, comment=f"WarpShuffle({', '.join(s.state)}, length={s.length})")
        off = int(s.length) // 2
        while off > 0:
            for st, sb in zip(s.state, s.state_b):
                v = self.lookup(st)
                if v.kind != "f":
                    raise CompileError("WarpShuffle over a non-f32 state")
                bits, got, out = em.reg("r"), em.reg("r"), em.reg("f")
                em.ins("mov.b32", bits, self.ints.as_reg(v))
                em.ins("shfl.sync.bfly.b32", got, bits, str(off), "0x1f", mask)
                em.ins("mov.b32", out, got)
                self.define(sb, F(out))
            self._combine(s.combine_states, s.dtype)
            off //= 2

    def s_TreeHalve(self, s) -> None:
        """Halving tree over a shared slab: slot t folds in slot t+s while t < s,
        one barrier per level, then every thread reads the root."""
        em, ints = self.em, self.ints
        tid = self.lookup(s.tid_var)
        if s.barrier_id == 0:
            bar = ("bar.sync", "0")
        else:
            bar = ("bar.sync", str(s.barrier_id), str(s.barrier_count))
        if s.inner is not None:
            iv, scale = s.inner
            ivv = self.lookup(iv)
            slot_of = lambda t: ints.add(ints.mul(t, I(int(scale))), ivv)  # noqa: E731
            root = ivv
        else:
            slot_of = lambda t: t  # noqa: E731
            root = I(0)
        em.comment(f"TreeHalve({', '.join(s.state)}, length={s.length}, tid={s.tid_var})")
        half = int(s.length) // 2
        while half > 0:
            skip = em.label("tree_skip")
            p = ints.cmp("ge", tid, I(half))
            if not p.is_const:
                em.ins("bra", skip, pred=p.v)
            ints.push()
            if not (p.is_const and p.v):
                mine, other = slot_of(tid), slot_of(ints.add(tid, I(half)))
                for buf, st, sb in zip(s.bufs, s.state, s.state_b):
                    b = self.buffer(buf)
                    r1, r2 = em.reg("f"), em.reg("f")
                    em.ins("ld.shared.f32", r1, self.address(b, mine))
                    em.ins("ld.shared.f32", r2, self.address(b, other))
                    self.define(st, F(r1))
                    self.define(sb, F(r2))
                self._combine(s.combine_states, s.dtype)
                for buf, st in zip(s.bufs, s.state):
                    em.ins("st.shared.f32", self.address(self.buffer(buf), mine), self.ints.as_reg(self.lookup(st)))
            ints.pop()
            em.place(skip)
            em.ins(*bar)
            half //= 2
        for buf, st in zip(s.bufs, s.state):
            r = em.reg("f")
            em.ins("ld.shared.f32", r, self.address(self.buffer(buf), root), comment=f"broadcast {st}")
            self.define(st, F(r))

    # ---- statements outside the required set ------------------------------------
    def s_ZeroPrologue(self, s) -> None:
        raise CompileError("ZeroPrologue is not used by any problem and is not supported")


__all__ = ["CodeGen", "Buffer", "CompileError", "f32_hex"]
