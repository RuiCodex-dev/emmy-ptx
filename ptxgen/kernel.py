"""Kernel assembly: IR file -> one complete PTX module.

Stages (the README walks through each one):

1. **Load**     ``Graph.from_dict`` gives typed statement objects; the graph
                level gives every buffer's shape and dtype; ``meta.json`` next
                to the IR gives the parameter order and the launch block size.
2. **Layout**   collect every ``Smem`` declaration and place it: static
                ``.shared`` arrays up to 48 KiB, one dynamic ``.extern``
                pool above that (the launch passes the pool size).
3. **Prologue** parameters, ``cvta.to.global``, thread / block ids, and the
                ``lane`` / ``warp`` names the reduction statements use.
4. **Walk**     ``CodeGen`` lowers the statement tree (lower.py).
5. **Emit**     header, shared declarations, signature, register
                declarations, then the instruction stream.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .emitter import Emitter
from .lower import Buffer, CodeGen, CompileError, _tname
from .values import I

STATIC_SMEM_LIMIT = 48 * 1024
PTX_VERSION = "8.0"     # sm_89 needs >= 7.8; 8.0 assembles for sm_89 and runs on sm_120
PTX_TARGET = "sm_89"


def ident(name: str) -> str:
    """A PTX-safe identifier derived from an IR name."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return s if re.match(r"[A-Za-z_]", s) else "_" + s


def load_meta(ir_path: Path, entry: str) -> dict | None:
    meta_path = ir_path.parent / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    for k in meta.get("kernels", []):
        if k.get("entry") == entry:
            return k
    return None


def walk(stmts):
    for s in stmts:
        yield s
        for f in ("body", "else_body"):
            sub = getattr(s, f, None)
            if sub:
                yield from walk(sub)


def compile_file(ir_path: str | Path) -> str:
    from emmy.compiler.graph import Graph  # the only use of emmy: deserialising the IR

    ir_path = Path(ir_path)
    raw = json.loads(ir_path.read_text())
    graph = Graph.from_dict(raw)

    kernels = [(nid, n) for nid, n in graph.nodes.items() if _tname(n.op) == "KernelOp"]
    if len(kernels) != 1:
        raise CompileError(f"expected exactly one KernelOp in {ir_path.name}, found {len(kernels)}")
    node_id, node = kernels[0]
    op = node.op
    entry = op.name or node_id
    if op.serial:
        raise CompileError("serial axes (runtime coordinate parameters) are not supported")

    # ---- buffers: every tensor the graph names, with its shape and dtype ----
    tensors: dict[str, tuple[tuple[int, ...], str]] = {}
    for n in raw["nodes"].values():
        for t in n.get("outputs", []):
            tensors[t["name"]] = (tuple(t.get("shape") or [1]), t.get("dtype", "f32"))
    out_names = [t["name"] for t in raw["nodes"][node_id].get("outputs", [])]

    meta = load_meta(ir_path, entry)
    if meta is not None:
        arg_order = list(meta["arg_order"])
        launch_block = int(meta["block"][0])
        dyn_smem = int(meta.get("smem_bytes", 0))
        scalar_args = {a["name"] for a in meta.get("args", []) if a.get("scalar")}
    else:   # no launch descriptor: the IR's own order (inputs, then outputs)
        arg_order = list(raw["nodes"][node_id].get("inputs", [])) + out_names
        launch_block = None
        dyn_smem = 0
        scalar_args = set()

    buffers: dict[str, Buffer] = {}
    for name in arg_order:
        if name in scalar_args:
            continue
        if name not in tensors:
            raise CompileError(f"kernel argument {name!r} is not a tensor of the graph")
        shape, dtype = tensors[name]
        b = Buffer(name, shape, dtype, "global")
        b.readonly = name not in out_names     # inputs never alias an output: ld.global.nc
        buffers[name] = b

    # ---- shared memory layout ------------------------------------------------
    smem_decls = []
    seen = set()
    for s in walk(op.body):
        if _tname(s) == "Smem" and s.name not in seen:
            seen.add(s.name)
            smem_decls.append(s)
    smem_lines: list[str] = []
    total = 0
    placed = []
    for s in smem_decls:
        b = Buffer(s.name, s.extents, "f32" if s.dtype in ("float", "f32") else s.dtype, "shared")
        align = max(int(s.align or 0), b.esize)
        total = -(-total // align) * align
        placed.append((b, total, align))
        total += b.esize * max(1, _prod(b.shape))
    use_pool = dyn_smem > 0 or total > STATIC_SMEM_LIMIT
    if use_pool:
        if dyn_smem and total > dyn_smem:
            raise CompileError(f"shared layout needs {total} B but the launch provides {dyn_smem} B")
        smem_lines.append(".extern .shared .align 16 .b8 smem_pool[];"
                          f"  // {total} bytes, sized at launch")
        for b, off, _ in placed:
            b.base, b.base_off = "smem_pool", off
            buffers[b.name] = b
    else:
        for b, _, align in placed:
            sym = "smem_" + ident(b.name)
            nbytes = b.esize * max(1, _prod(b.shape))
            smem_lines.append(f".shared .align {align} .b8 {sym}[{nbytes}];"
                              f"  // {b.name}{list(b.shape)}")
            b.base = sym
            buffers[b.name] = b

    # ---- code generation -----------------------------------------------------
    cg = CodeGen(buffers, launch_block)
    em: Emitter = cg.em
    params = []
    em.comment("parameters")
    for name in arg_order:
        pname = "p_" + ident(name)
        if name in scalar_args:
            params.append(f"\t.param .s32 {pname}")
            r = em.reg("r")
            em.ins("ld.param.s32", r, f"[{pname}]")
            cg.env[name] = I(r)
            continue
        params.append(f"\t.param .u64 {pname}")
        raw_ptr, gptr = em.reg("rd"), em.reg("rd")
        em.ins("ld.param.u64", raw_ptr, f"[{pname}]")
        em.ins("cvta.to.global.u64", gptr, raw_ptr, comment=name)
        buffers[name].base = gptr

    em.comment("thread identity")
    tid, cta, lane, warp = (em.reg("r") for _ in range(4))
    em.ins("mov.u32", tid, "%tid.x")
    em.ins("mov.u32", cta, "%ctaid.x")
    em.ins("and.b32", lane, tid, "31", comment="lane = threadIdx.x & 31")
    em.ins("shr.u32", warp, tid, "5", comment="warp = threadIdx.x >> 5")
    for r in (tid, cta, lane, warp):
        cg.ints.nonneg.add(r)
    cg.env.update({"%tid": I(tid), "%ctaid": I(cta), "lane": I(lane), "warp": I(warp), "tid": I(tid)})

    cg.body(op.body)
    em.ins("ret")

    # ---- assemble ------------------------------------------------------------
    out = [
        f"// {entry}: generated by emmy-ptx from {ascii_name(ir_path.name)}",
        f".version {PTX_VERSION}",
        f".target {PTX_TARGET}",
        ".address_size 64",
        "",
        *smem_lines,
        "" if smem_lines else None,
        f".visible .entry {entry}(",
        ",\n".join(params),
        ")",
    ]
    if launch_block:
        out.append(f".maxntid {launch_block}, 1, 1")
    out.append("{")
    out.extend(em.reg_decls())
    out.append("")
    out.extend(em.lines)
    out.append("}")
    text = "\n".join(line for line in out if line is not None) + "\n"
    text.encode("ascii")   # PTX must be pure ASCII; fail here rather than in ptxas
    return text


def _prod(shape) -> int:
    p = 1
    for d in shape:
        p *= int(d)
    return p


def ascii_name(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")
