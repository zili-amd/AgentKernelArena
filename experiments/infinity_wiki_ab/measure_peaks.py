#!/usr/bin/env python3
"""Measure the throughput a kernel actually attains on this GPU, for roofline.py's attained bounds.

Run inside an AKA container on one idle GPU, e.g.
    AKA_VISIBLE_GPU=7 bash src/scripts/docker_benchmark.sh exec python3 wiki_ab/measure_peaks.py
It prints one JSON object:
  read_bytes_per_s   by working-set size, the best Triton streaming read of a buffer that size, so sizes under
                     the 256 MB Infinity Cache show what a kernel reading them again on every replay attains;
  gemm_flops_per_s   the best hipBLASLt GEMM (torch._scaled_mm for FP8, torch.matmul for FP16) over large shapes.
Timing replays a captured graph without flushing caches, as AKA's benchmark does.
"""
from __future__ import annotations

import json

import torch
import triton
import triton.language as tl


def graph_us(fn, reps: int = 50) -> float:
    fn()
    torch.cuda.synchronize()
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
        with torch.cuda.graph(graph):
            fn()
    torch.cuda.synchronize()
    for _ in range(5):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps * 1e3


@triton.jit
def _read(src, out, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(ITERS):
        offs = (pid * ITERS + i) * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(src + offs, mask=offs < n, other=0)
    tl.store(out + pid, tl.sum(acc))


READ_SIZES_MIB = (16, 32, 64, 128, 192, 256, 384, 512, 4096)


def read_bandwidth(mebibytes: int) -> float:
    n = mebibytes * (1 << 20) // 4
    src = torch.ones(n, dtype=torch.int32, device="cuda")
    best = 0.0
    for block, iters in ((1024, 4), (1024, 16), (2048, 8), (4096, 4), (4096, 8), (8192, 4)):
        out = torch.empty(triton.cdiv(n, block * iters), dtype=torch.int32, device="cuda")
        us = graph_us(lambda: _read[(out.numel(),)](src, out, n, BLOCK=block, ITERS=iters), reps=20)
        best = max(best, n * 4 / (us * 1e-6))
    return best


def gemm_peaks() -> dict[str, float]:
    fp8 = torch.float8_e4m3fnuz if "gfx94" in torch.cuda.get_device_properties(0).gcnArchName else torch.float8_e4m3fn
    one = torch.tensor(1.0, device="cuda")
    best = {"fp8": 0.0, "fp16": 0.0}
    for m, n, k in ((4096, 7168, 16384), (8192, 8192, 8192), (8192, 8192, 16384), (16384, 8192, 8192),
                    (8192, 16384, 8192), (16384, 16384, 8192)):
        a = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.5
        b = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.5
        a8, b8 = a.to(fp8), b.to(fp8)
        us = graph_us(lambda: torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one, out_dtype=torch.float16))
        best["fp8"] = max(best["fp8"], 2 * m * n * k / (us * 1e-6))
        us = graph_us(lambda: torch.matmul(a, b.t()))
        best["fp16"] = max(best["fp16"], 2 * m * n * k / (us * 1e-6))
    return best


def main() -> int:
    print(json.dumps({"device": torch.cuda.get_device_properties(0).gcnArchName,
                      "read_bytes_per_s": {mib: read_bandwidth(mib) for mib in READ_SIZES_MIB},
                      "gemm_flops_per_s": gemm_peaks()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
