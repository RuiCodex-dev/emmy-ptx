"""Harder practice problems: shapes that do not divide the tiles, large matmuls
(rasterised CTA order), reductions without the pinned schedule (cross-block
atomics), and graphs that emmy splits into several kernels."""

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


d64 = lambda a: a.astype(np.float64)  # noqa: E731
f32 = lambda a: np.asarray(a).astype(np.float32)  # noqa: E731

OPEN = [
    dict(id="hm01_matmul_1024", code="torch.randn(1024, 1024) @ torch.randn(1024, 1024)",
         ref=lambda a, b: [f32(d64(a) @ d64(b))], rtol=1e-3, atol=1e-2),
    dict(id="hm02_matmul_odd", code="torch.randn(100, 70) @ torch.randn(70, 50)",
         ref=lambda a, b: [f32(d64(a) @ d64(b))], rtol=1e-3, atol=1e-3),
    dict(id="hm03_matmul_2048x256", code="torch.randn(2048, 256) @ torch.randn(256, 512)",
         ref=lambda a, b: [f32(d64(a) @ d64(b))], rtol=1e-3, atol=1e-3),
    dict(id="hr01_sum_unpinned", code="torch.randn(1 << 20).sum()",
         ref=lambda x: [f32(d64(x).sum())], rtol=1e-3, atol=1e-2),
    dict(id="hr02_row_sum_unpinned", code="torch.randn(256, 4096).sum(-1)",
         ref=lambda x: [f32(d64(x).sum(-1))], rtol=1e-3, atol=1e-3),
    dict(id="hr03_odd_rows", code="torch.randn(37, 333).sum(-1)",
         ref=lambda x: [f32(d64(x).sum(-1))], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="hp01_odd", code="x = torch.randn(1001); y = torch.randn(1001); x * y + x",
         ref=lambda x, y: [f32(x * y + x)], rtol=1e-6, atol=1e-6),
    dict(id="hp02_3d_bcast", code="x = torch.randn(8, 16, 32); b = torch.randn(32); torch.relu(x - b)",
         ref=lambda x, b: [f32(np.maximum(x - b, 0))], rtol=0, atol=0),
    dict(id="hf01_global_norm", code="x = torch.randn(4096); x / x.abs().sum()",
         ref=lambda x: [f32(d64(x) / np.abs(d64(x)).sum())], rtol=1e-3, atol=1e-6),
    dict(id="hf02_softmax_odd", code="torch.softmax(torch.randn(48, 300), dim=-1)",
         ref=lambda x: [f32(_softmax(d64(x)))], rtol=1e-3, atol=1e-5),
    dict(id="hf03_layernorm_affine", code="x = torch.randn(32, 768); w = torch.randn(768); b = torch.randn(768); "
         "torch.nn.functional.layer_norm(x, (768,), w, b)",
         ref=lambda x, w, b: [f32((d64(x) - d64(x).mean(-1, keepdims=True))
                                  / np.sqrt(d64(x).var(-1, keepdims=True) + 1e-5) * d64(w) + d64(b))],
         rtol=1e-3, atol=1e-4),
    dict(id="hf04_softmax_3d", code="torch.softmax(torch.randn(4, 32, 64), dim=-1)",
         ref=lambda x: [f32(_softmax(d64(x)))], rtol=1e-3, atol=1e-5),
    dict(id="ha01_attention_odd", code="q = torch.randn(48, 40); kt = torch.randn(40, 80); v = torch.randn(80, 24); "
         "torch.softmax(q @ kt / 6.324555, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax(d64(q) @ d64(kt) / 6.324555) @ d64(v))], rtol=1e-3, atol=1e-4),
    dict(id="ha02_attention_512", code="q = torch.randn(512, 64); kt = torch.randn(64, 512); v = torch.randn(512, 64); "
         "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax(d64(q) @ d64(kt) / 8.0) @ d64(v))], rtol=1e-3, atol=1e-4),
]
