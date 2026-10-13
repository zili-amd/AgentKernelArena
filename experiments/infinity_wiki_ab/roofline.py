#!/usr/bin/env python3
"""Per-case MI300X roofline bounds for the screening candidates, and the selection rule.

A case's effective bound is max(roofline, LAUNCH_FLOOR_S): an optimal kernel
still pays launch and timing overhead, so a tiny case cannot show a large
speedup however slow its baseline is. Headroom = sum(baseline) / sum(effective
bound) over the task's performance cases; rule 1 requires >= MIN_HEADROOM.
Without --baselines only the bound distribution is printed. Rules 2-4 (wiki
relevance, task_validator, implemented Triton/HIP) are recorded in manifest.yaml.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

BW = 5.3e12  # MI300X HBM3, bytes/s
PEAK = {"fp8": 2614.9e12, "bf16": 1307.4e12, "fp16": 1307.4e12, "fp32": 163.4e12}  # dense matrix FLOP/s
SIZE = {"fp8": 1, "fp16": 2, "bf16": 2, "fp32": 4}
MIN_HEADROOM = 2.0
LAUNCH_FLOOR_S = 5e-6


def bound(flops: float, nbytes: float, peak: float) -> float:
    return max(flops / peak, nbytes / BW)


def refk_fp8_blockwise_mm(p: dict) -> float:
    c = p["configuration"]
    m, n, k = c["m"], c["n"], c["k"]
    sk, sn = math.ceil(k / 128), math.ceil(n / 128)
    return bound(2 * m * n * k, m * k + n * k + 4 * (m * sk + sn * sk) + 2 * m * n, PEAK["fp8"])


def mla_decode(p: dict) -> float:
    # NHEAD=128, LK=576 (512 NoPE + 64 RoPE), LV=512, FP8 latent KV shared by all heads.
    # Sequence lengths cycle per batch item: ctx//2 + 1, ctx, ctx - 1, 1.
    b, ctx = p["batch"], p["ctx"]
    cycle = (ctx // 2 + 1, ctx, ctx - 1, 1)
    tokens = sum(cycle[i % 4] for i in range(b))
    flops = 128 * tokens * (576 + 512) * 2
    nbytes = tokens * 576 + b * 128 * (576 + 512) * 2 + b * 128 * 4
    return bound(flops, nbytes, PEAK["bf16"])


def moe_gemm(p: dict) -> float:
    a = p["arguments"]
    m, n, k, topk, e, elt = a["M_orig"], a["N"], a["K"], a["top_k"], a["E"], SIZE[a["dtype_str"]]
    rows = m * topk
    return bound(2 * rows * n * k, min(e, rows) * n * k * elt + m * k * elt + rows * n * elt, PEAK[a["dtype_str"]])


def flash_attention_fwd(p: dict) -> float:
    a = p["arguments"]
    z, h, s, d, elt = a["Z"], a["H"], a["N_CTX"], a["D_HEAD"], SIZE[a["dtype_str"]]
    return bound(4 * z * h * s * s * d, 4 * z * h * s * d * elt, PEAK[a["dtype_str"]])


def rmsnorm_fwd(p: dict) -> float:
    a = p["arguments"]
    m, n = a["M"], a["N"]
    return (m * n * (SIZE[a["in_dtype_str"]] + SIZE[a["out_dtype_str"]]) + n * SIZE[a["in_dtype_str"]]) / BW


def softmax(p: dict) -> float:
    a = p["arguments"]
    return 2 * a["M"] * a["N"] * SIZE[a["dtype_str"]] / BW


def topk(p: dict) -> float:
    batch, hidden, k = p["configuration"]
    return (batch * hidden * 4 + batch * k * 12) / BW


def rocmbench_gemm(p: dict) -> float:
    a = p["arguments"]
    m, n, k = a["M"], a["N"], a["K"]
    elt_a, elt_b, elt_c = SIZE[a["in_dtype_a_str"]], SIZE[a["in_dtype_b_str"]], SIZE[a["out_dtype_str"]]
    return bound(2 * m * n * k, m * k * elt_a + k * n * elt_b + m * n * elt_c, PEAK[a["in_dtype_a_str"]])


def aiter_gemm_bias(p: dict) -> float:
    # bf16 x[M,K] @ w[K,N] + bias[N] -> bf16 out[M,N]
    m, n, k = p["configuration"]
    return bound(2 * m * n * k, 2 * (m * k + k * n + n + m * n), PEAK["bf16"])


def gpumode_matmul(p: dict) -> float:
    # fp32 a[..., M, K].matmul(b + b) with b[..., K, N]
    a_shape, b_shape = (item["shape"] for item in p["inputs"])
    batch, (m, k), n = math.prod(a_shape[:-2]), a_shape[-2:], b_shape[-1]
    return bound(2 * batch * m * n * k, 4 * batch * (m * k + k * n + m * n), PEAK["fp32"])


def gpumode_softmax(p: dict) -> float:
    return 2 * math.prod(p["inputs"][0]["shape"]) * SIZE["fp32"] / BW


MODELS = {
    "triton2triton/geak_eval/L1/refk_fp8_blockwise_mm": refk_fp8_blockwise_mm,
    "hip2hip/others/mla_decode": mla_decode,
    "triton2triton/rocmbench/hard/moe_gemm": moe_gemm,
    "triton2triton/rocmbench/hard/test_flashattention_fwd": flash_attention_fwd,
    "triton2triton/rocmbench/medium/rmsnorm_fwd": rmsnorm_fwd,
    "triton2triton/rocmbench/medium/softmax": softmax,
    "triton2triton/geak_eval/L2/topk": topk,
    "triton2triton/rocmbench/hard/gemm": rocmbench_gemm,
    "triton2triton/geak_eval/L3/gemm": aiter_gemm_bias,
    "triton2triton/rocmbench/medium/naive_softmax": softmax,
    "hip2hip/gpumode_large/SimpleMatmulModule": gpumode_matmul,
    "hip2hip/gpumode_large/SoftmaxModule": gpumode_softmax,
}


def performance_cases(aka: Path, task: str) -> list[dict]:
    config = yaml.safe_load((aka / "tasks" / task / "config.yaml").read_text())
    data = json.loads((aka / "tasks" / task / config["evaluation"]["workloads"]).read_text())
    cases = data if isinstance(data, list) else data.get("cases", data)
    return [c for c in cases if "performance" in (c.get("checks") or ["performance"])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aka", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "manifest.yaml")
    parser.add_argument("--baselines", type=Path, help="JSON from screen_baselines.py")
    parser.add_argument("--task", action="append", default=[], help="bound these tasks instead of the manifest's")
    args = parser.parse_args()
    candidates = args.task or yaml.safe_load(args.manifest.read_text())["phases"]["screening"]["candidates"]
    measured = json.loads(args.baselines.read_text()) if args.baselines else {}
    rows = []
    for task in candidates:
        if task not in MODELS:
            print(f"{task}: no roofline model", file=sys.stderr)
            continue
        bounds = {c["test_case_id"]: MODELS[task](c["params"]) for c in performance_cases(args.aka, task)}
        effective = {cid: max(b, LAUNCH_FLOOR_S) for cid, b in bounds.items()}
        row = {"task": task, "cases": len(bounds), "median_bound_us": statistics.median(bounds.values()) * 1e6,
               "share_at_floor": sum(b < LAUNCH_FLOOR_S for b in bounds.values()) / len(bounds)}
        times = measured.get(task, {}).get("median_ms_by_case")
        if times:
            common = [cid for cid in effective if cid in times]
            row["headroom"] = sum(times[c] / 1e3 for c in common) / sum(effective[c] for c in common)
            row["selected"] = row["headroom"] >= MIN_HEADROOM
        rows.append(row)
    for row in rows:
        extra = (f"  headroom={row['headroom']:.1f}x selected={row['selected']}" if "headroom" in row else "")
        print(f"{row['task']:55} cases={row['cases']:4} median_roofline={row['median_bound_us']:9.2f}us "
              f"cases_at_launch_floor={row['share_at_floor']:.0%}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
