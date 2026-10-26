"""Unit tests for the integer layer: C semantics, affine folding, scoped CSE."""

import pytest

from ptxgen.emitter import Emitter, f32_hex
from ptxgen.values import I, IntOps, Lin, c_div, c_rem


def fresh():
    em = Emitter()
    return em, IntOps(em)


@pytest.mark.parametrize("a,b", [(7, 2), (-7, 2), (7, -2), (-7, -2), (0, 3), (6, 3)])
def test_c_division_truncates_toward_zero(a, b):
    assert c_div(a, b) == int(a / b)
    assert c_rem(a, b) == a - b * int(a / b)


def test_f32_hex_is_exact():
    assert f32_hex(1.0) == "0f3F800000"
    assert f32_hex(0.0) == "0f00000000"
    assert f32_hex(-2.0) == "0fC0000000"
    assert f32_hex(-1e30) == "0fF149F2CA"


def test_constants_fold_without_instructions():
    em, ints = fresh()
    v = ints.add(ints.mul(I(3), I(4)), ints.div(I(9), I(2)))
    assert v == I(16)
    assert em.lines == []


def test_affine_cancellation():
    # ((b*64) + (u*2) + 1) - (b*64)  ->  u*2 + 1, no instruction yet
    em, ints = fresh()
    b, u = I("%r1"), I("%r2")
    v = ints.sub(ints.add(ints.add(ints.mul(b, I(64)), ints.mul(u, I(2))), I(1)), ints.mul(b, I(64)))
    assert ints.lin(v) == Lin((("%r2", 2),), 1)
    assert em.lines == []


def test_exact_division_of_affine_form():
    _, ints = fresh()
    v = ints.div(ints.add(ints.mul(I("%r1"), I(64)), I(128)), I(64))
    assert ints.lin(v) == Lin((("%r1", 1),), 2)


def test_remainder_of_multiple_is_constant_only_when_nonnegative():
    em, ints = fresh()
    ints.nonneg.add("%r1")
    v = ints.rem(ints.add(ints.mul(I("%r1"), I(64)), I(70)), I(64))
    assert v == I(6)
    # without the proof it must emit a real remainder (C sign semantics)
    em2, ints2 = fresh()
    ints2.rem(ints2.add(ints2.mul(I("%r1"), I(64)), I(70)), I(64))
    assert any("rem.s32" in line for line in em2.lines)


def test_power_of_two_division_needs_nonnegativity():
    em, ints = fresh()
    ints.nonneg.add("%r1")
    ints.div(I("%r1"), I(8))
    assert any("shr.u32" in line for line in em.lines)
    em2, ints2 = fresh()
    ints2.div(I("%r1"), I(8))
    assert any("div.s32" in line for line in em2.lines)


def test_materialisation_is_cached():
    em, ints = fresh()
    lin = I(Lin((("%r1", 4), ("%r2", 1)), 0))
    r1 = ints.as_reg(lin)
    n = len(em.lines)
    r2 = ints.as_reg(lin)
    assert r1 == r2 and len(em.lines) == n


def test_cse_scope_drops_inner_entries():
    em, ints = fresh()
    lin = I(Lin((("%r1", 4),), 0))
    ints.push()
    inner = ints.as_reg(lin)
    ints.pop()
    outer = ints.as_reg(lin)
    assert inner != outer          # recomputed: the inner definition does not dominate


def test_cse_outer_entry_visible_inside():
    _, ints = fresh()
    lin = I(Lin((("%r1", 4),), 0))
    outer = ints.as_reg(lin)
    ints.push()
    assert ints.as_reg(lin) == outer
    ints.pop()


def test_comparison_of_constants_folds_to_constant_predicate():
    em, ints = fresh()
    assert ints.cmp("lt", I(1), I(2)).v is True
    assert em.lines == []
