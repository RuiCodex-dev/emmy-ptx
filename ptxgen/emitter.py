"""PTX text emission: virtual registers, labels, instructions.

The compiler never does register allocation.  Every value lives in a fresh
virtual register of the right class and ``ptxas`` does the real allocation
downstream, which is what the task asks for.  This module only hands out
register names, counts them per class so the ``.reg`` declarations at the top
of the kernel are exact, and collects the instruction lines.
"""

from __future__ import annotations

import struct

# register class -> (PTX declaration type, name prefix)
REG_CLASSES = {
    "p": (".pred", "%p"),   # predicates (comparisons, guards)
    "r": (".b32", "%r"),    # 32-bit integers (indices, lane ids, bit patterns)
    "f": (".f32", "%f"),    # 32-bit floats (every value the problems compute)
    "rd": (".b64", "%rd"),  # 64-bit integers (global addresses)
}


def f32_hex(value: float) -> str:
    """A float immediate in PTX's exact hex form, e.g. 1.0 -> 0f3F800000.

    Decimal immediates are legal PTX too, but the hex form is exact: the bits
    in the PTX are the bits the kernel computes with, with no second rounding
    by the assembler's float parser.
    """
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(value)))[0]


class Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.counts = {k: 0 for k in REG_CLASSES}
        self.labels = 0
        self.indent = 1

    # ---- registers ---------------------------------------------------------
    def reg(self, cls: str) -> str:
        self.counts[cls] += 1
        return f"{REG_CLASSES[cls][1]}{self.counts[cls]}"

    def reg_decls(self) -> list[str]:
        out = []
        for cls, (ty, prefix) in REG_CLASSES.items():
            n = self.counts[cls]
            if n:
                # registers are numbered from 1, so <n+1> declares %x0 .. %xn
                out.append(f"\t.reg {ty:<6} {prefix}<{n + 1}>;")
        return out

    # ---- labels ------------------------------------------------------------
    def label(self, hint: str) -> str:
        self.labels += 1
        return f"$L_{hint}_{self.labels}"

    def place(self, label: str) -> None:
        self.lines.append(f"{label}:")

    # ---- instructions ------------------------------------------------------
    def ins(self, opcode: str, *operands: str, pred: str | None = None, comment: str | None = None) -> None:
        guard = f"@{pred} " if pred else ""
        text = "\t" * self.indent + f"{guard}{opcode}"
        if operands:
            text += " " + ", ".join(str(o) for o in operands)
        text += ";"
        if comment:
            text += f"  // {ascii_only(comment)}"
        self.lines.append(text)

    def comment(self, text: str) -> None:
        self.lines.append("\t" * self.indent + "// " + ascii_only(text))

    def raw(self, text: str) -> None:
        self.lines.append(text)


def ascii_only(text: str) -> str:
    """PTX rejects any non-ASCII byte, comments included."""
    return text.encode("ascii", "replace").decode("ascii").replace("\n", " ")
