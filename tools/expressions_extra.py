"""Extra practice problems in the five categories, written to resemble the
hidden set ("same kinds, different shapes and ops").  Generated with the task
repository's own tools/generate.py; used only to test this compiler.
"""

import math

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _erf(x):
    from scipy.special import erf
    return erf(x)


d64 = lambda a: a.astype(np.float64)  # noqa: E731
f32 = lambda a: np.asarray(a).astype(np.float32)  # noqa: E731

OPEN = [
    # ---- pointwise --------------------------------------------------------
    dict(id="xp01_gelu", code="torch.nn.functional.gelu(torch.randn(3000))",
         ref=lambda x: [f32(0.5 * d64(x) * (1 + _erf(d64(x) / math.sqrt(2))))], rtol=1e-4, atol=1e-5),
    dict(id="xp02_sigmoid_mul", code="x = torch.randn(1000); y = torch.randn(1000); torch.sigmoid(x) * y",
         ref=lambda x, y: [f32(1 / (1 + np.exp(-d64(x))) * d64(y))], rtol=1e-4, atol=1e-5),
    dict(id="xp03_tanh_exp", code="x = torch.randn(4096); torch.tanh(x) + torch.exp(x * 0.5)",
         ref=lambda x: [f32(np.tanh(d64(x)) + np.exp(d64(x) * 0.5))], rtol=1e-4, atol=1e-5),
    dict(id="xp04_bias_2d", code="x = torch.randn(64, 96); b = torch.randn(96); (x + b) * 3.0 - 1.5",
         ref=lambda x, b: [f32((x + b) * np.float32(3.0) - np.float32(1.5))], rtol=1e-6, atol=1e-6),
    dict(id="xp05_silu", code="torch.nn.functional.silu(torch.randn(8192))",
         ref=lambda x: [f32(d64(x) / (1 + np.exp(-d64(x))))], rtol=1e-4, atol=1e-5),
    dict(id="xp06_div_abs", code="x = torch.randn(2048); y = torch.randn(2048); x / (torch.abs(y) + 1.0)",
         ref=lambda x, y: [f32(d64(x) / (np.abs(d64(y)) + 1.0))], rtol=1e-5, atol=1e-6),
    dict(id="xp07_where", code="x = torch.randn(777); torch.where(x > 0, x, x * 0.1)",
         ref=lambda x: [f32(np.where(x > 0, x, x * np.float32(0.1)))], rtol=1e-6, atol=1e-6),
    dict(id="xp08_clamp_sqrt", code="x = torch.randn(1536); torch.sqrt(torch.clamp(x, min=0.0) + 0.25)",
         ref=lambda x: [f32(np.sqrt(np.maximum(d64(x), 0) + 0.25))], rtol=1e-5, atol=1e-6),
    # ---- reductions --------------------------------------------------------
    dict(id="xr01_max", code="torch.randn(4096).amax()",
         ref=lambda x: [f32(x.max())], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="xr02_row_mean", code="torch.randn(32, 384).mean(-1)",
         ref=lambda x: [f32(d64(x).mean(-1))], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="xr03_sumsq", code="x = torch.randn(10000); (x * x).sum()",
         ref=lambda x: [f32((d64(x) ** 2).sum())], rtol=1e-4, atol=1e-3, knobs=REDUCE_KNOBS),
    dict(id="xr04_row_max", code="torch.randn(128, 200).amax(-1)",
         ref=lambda x: [f32(x.max(-1))], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="xr05_row_min", code="torch.randn(16, 1000).amin(-1)",
         ref=lambda x: [f32(x.min(-1))], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="xr06_abs_sum", code="torch.abs(torch.randn(48, 640)).sum(-1)",
         ref=lambda x: [f32(np.abs(d64(x)).sum(-1))], rtol=1e-4, atol=1e-3, knobs=REDUCE_KNOBS),
    # ---- matmul ------------------------------------------------------------
    dict(id="xm01_matmul_256x128", code="torch.randn(256, 128) @ torch.randn(128, 256)",
         ref=lambda a, b: [f32(d64(a) @ d64(b))], rtol=1e-3, atol=1e-3),
    dict(id="xm02_matmul_small", code="torch.randn(32, 64) @ torch.randn(64, 48)",
         ref=lambda a, b: [f32(d64(a) @ d64(b))], rtol=1e-3, atol=1e-3),
    dict(id="xm03_matmul_tall", code="torch.randn(1024, 64) @ torch.randn(64, 128)",
         ref=lambda a, b: [f32(d64(a) @ d64(b))], rtol=1e-3, atol=1e-3),
    dict(id="xm04_matmul_deep", code="torch.randn(64, 1024) @ torch.randn(1024, 64)",
         ref=lambda a, b: [f32(d64(a) @ d64(b))], rtol=1e-3, atol=1e-3),
    # ---- fused -------------------------------------------------------------
    dict(id="xf01_layernorm", code="x = torch.randn(64, 512); torch.nn.functional.layer_norm(x, (512,))",
         ref=lambda x: [f32((d64(x) - d64(x).mean(-1, keepdims=True))
                            / np.sqrt(d64(x).var(-1, keepdims=True) + 1e-5))], rtol=1e-3, atol=1e-4),
    dict(id="xf02_softmax_wide", code="torch.softmax(torch.randn(32, 1024), dim=-1)",
         ref=lambda x: [f32(_softmax(d64(x)))], rtol=1e-3, atol=1e-5),
    dict(id="xf03_logsumexp", code="torch.logsumexp(torch.randn(64, 256), dim=-1)",
         ref=lambda x: [f32(np.log(np.exp(d64(x)).sum(-1)))], rtol=1e-4, atol=1e-4),
    dict(id="xf04_center", code="x = torch.randn(128, 256); x - x.mean(-1, keepdim=True)",
         ref=lambda x: [f32(d64(x) - d64(x).mean(-1, keepdims=True))], rtol=1e-3, atol=1e-4),
    dict(id="xf05_rmsnorm_384", code="x = torch.randn(32, 384); w = torch.randn(384); "
         "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w",
         ref=lambda x, w: [f32(d64(x) / np.sqrt((d64(x) ** 2).mean(-1, keepdims=True) + 1e-6) * d64(w))],
         rtol=1e-3, atol=1e-4),
    dict(id="xf06_softmax_scaled", code="x = torch.randn(64, 128); torch.softmax(x * 0.5, dim=-1)",
         ref=lambda x: [f32(_softmax(d64(x) * 0.5))], rtol=1e-3, atol=1e-5),
    # ---- attention-like -----------------------------------------------------
    dict(id="xa01_attention_32", code="q = torch.randn(32, 16); kt = torch.randn(16, 32); v = torch.randn(32, 16); "
         "torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax(d64(q) @ d64(kt) / 4.0) @ d64(v))], rtol=1e-3, atol=1e-4),
    dict(id="xa02_attention_256", code="q = torch.randn(256, 64); kt = torch.randn(64, 256); v = torch.randn(256, 64); "
         "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax(d64(q) @ d64(kt) / 8.0) @ d64(v))], rtol=1e-3, atol=1e-4),
    dict(id="xa03_attention_rect", code="q = torch.randn(64, 32); kt = torch.randn(32, 128); v = torch.randn(128, 64); "
         "torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax(d64(q) @ d64(kt) / 5.656854) @ d64(v))], rtol=1e-3, atol=1e-4),
]
