#!/usr/bin/env python3
"""Write the task variants this round measures.

Production-sized variants of hip2hip/gpumode tasks: the originals measure 4x4
to 128x128 inputs, where every case sits at the launch floor and no rewrite can
show. A variant keeps the task's kernel, reference, tolerances and timing code
and replaces only the case list: the configs in get_inputs() of both reference
modules, and workload.json, which the evaluator checks against them case by
case. The README also states that the operator must stay in the candidate's own
kernels, which the harness guard enforces; at these sizes a library call would
otherwise be the fastest candidate.

Open-launcher variants of Triton tasks: the originals declare only the kernel
editable, so its tiling, grid, num_warps and num_stages stay fixed in a protected
launcher. A variant keeps the task's code, cases, references and timing and also
declares the launcher (and its configuration helper) editable. The attention
task's launch code sits in a protected autograd.Function, which a symbol scope
cannot open, so the variant moves it into a top-level flash_fwd_launch that the
Function calls. With the launcher editable, the README's own-kernels rule and the
harness guard keep PyTorch and library operators out of the candidate.

Own-kernels variants of torch2hip tasks: the candidate starts empty and the
baseline is the PyTorch module, which hands the operator to rocBLAS or ATen. A
variant keeps the task unchanged and adds the own-kernels rule to its README;
the harness guard refuses a candidate that calls those libraries instead.

LLM-shape variants of two vLLM Triton tasks, the AWQ int4 GEMM and the FP8
block-scaled GEMM that vLLM runs on ROCm by default: the originals' cases (N and K
at most 512) sit at the launch floor. A variant replaces the case table with a
real model's linear layers and keeps the kernel, wrapper, timing and tolerance.
Its CPU references are the original loops, vectorized; the loops would take hours
per call at these sizes. With the original uniform random nibbles and ~0.09
scales, AWQ outputs reach 40-70 at LLM K and the production kernel's fp16
accumulation alone exceeds the original atol, so the AWQ variant quantizes a
Gaussian weight the way AWQ does, scales its replay perturbation instead of adding
a constant sized for the old magnitudes, and adds a 5e-3 normwise error bound
that an FP8-activation path would otherwise pass at M=1. Both source files were
already editable whole, launcher included; the README gains the own-kernels rule.
    make_variants.py [--aka PATH] [--force]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import textwrap
from pathlib import Path

# source task -> (variant task, one tuple of input shapes per case)
VARIANTS = {
    "hip2hip/gpumode/SimpleMatmulModule": ("hip2hip/gpumode_large/SimpleMatmulModule", [
        ([1024, 1024], [1024, 1024]),
        ([2048, 2048], [2048, 2048]),
        ([4096, 4096], [4096, 4096]),
        ([2048, 8192], [8192, 2048]),
        ([8192, 1024], [1024, 4096]),
        ([8, 1024, 1024], [8, 1024, 1024]),
        ([32, 512, 512], [32, 512, 512]),
        ([4, 8, 256, 512], [4, 8, 512, 256]),
    ]),
    "hip2hip/gpumode/SoftmaxModule": ("hip2hip/gpumode_large/SoftmaxModule", [
        ([8, 1024, 1024],),
        ([32, 1024, 1024],),
        ([16, 2048, 2048],),
        ([4, 4096, 4096],),
        ([1, 4096, 32000],),
        ([64, 512, 512],),
        ([1, 16384, 1024],),
        ([256, 128, 128],),
    ]),
}
CONFIGS_BLOCK = re.compile(r"(    configs = \[\n).*?(\n    \]\n)", re.DOTALL)
PROTECTED_CODE_RULE = "HIP stream. Do not load or call protected baseline/reference code from the candidate.\n"
OWN_KERNELS_RULE = (
    "Compute the operator in HIP kernels the candidate defines and launches. Do not call\n"
    "a BLAS or GEMM library (hipBLAS, hipBLASLt, rocBLAS, Tensile), MIOpen, or a\n"
    "PyTorch/ATen matrix-product or softmax operator; PyTorch may only allocate\n"
    "outputs, check inputs, and make views or contiguous copies.\n")


# source task -> (variant task, launcher symbols declared editable next to the kernel, README note)
OPEN_LAUNCHERS = {
    "triton2triton/rocmbench/hard/test_flashattention_fwd": (
        "triton2triton/rocmbench_open/hard/test_flashattention_fwd", ["flash_fwd_launch"],
        "Its launch code moved from `_attention.forward` into `flash_fwd_launch`, which\n"
        "returns `(o, L, m)`; `_attention` and `attention` stay protected and call it.\n"),
    "triton2triton/geak_eval/L3/gemm": ("triton2triton/geak_eval_open/L3/gemm", ["get_config", "gemm"], ""),
    "triton2triton/rocmbench/hard/gemm": (
        "triton2triton/rocmbench_open/hard/gemm", ["matmul"],
        "The autotune configurations on `matmul_kernel` were already editable.\n"),
    "triton2triton/rocmbench/medium/naive_softmax": (
        "triton2triton/rocmbench_open/medium/naive_softmax", ["softmax"], ""),
}
FLASH_LAUNCH_CODE = '''class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        D0 = q.shape[0] * q.shape[1] * q.shape[2]
        flash_fwd_kernel[grid](
            q, k, v, sm_scale,  #
            L, m,  #
            o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1], q.shape[2], D0,  #
            BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_DMODEL=Lk,  #
            num_warps=num_warps, num_stages=2)

        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o
'''
FLASH_OPEN_LAUNCH_CODE = '''def flash_fwd_launch(q, k, v, sm_scale):
    BLOCK = 128
    Lk = k.shape[-1]
    o = torch.empty_like(q)
    grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    num_warps = 4 if Lk <= 64 else 8
    D0 = q.shape[0] * q.shape[1] * q.shape[2]
    flash_fwd_kernel[grid](
        q, k, v, sm_scale,  #
        L, m,  #
        o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        q.shape[0], q.shape[1], q.shape[2], D0,  #
        BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_DMODEL=Lk,  #
        num_warps=num_warps, num_stages=2)
    return o, L, m


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o, L, m = flash_fwd_launch(q, k, v, sm_scale)
        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o
'''
# source task -> variant task
OWN_KERNEL_VARIANTS = {
    f"torch2hip/kernelbench/level1/{name}": f"torch2hip/kernelbench_own/level1/{name}"
    for name in ("l1n95_CrossEntropyLoss", "l1n36_RMSNorm_", "l1n4_Matrix_vector_multiplication_")
}
OWN_TRITON_KERNELS_RULE = (
    "Compute the operator in Triton kernels the candidate defines and launches. Do not\n"
    "call a PyTorch/ATen matrix-product, softmax or attention operator, `torch.ops`, or\n"
    "another kernel library (rocBLAS, hipBLASLt, aiter, flash-attn); PyTorch may only\n"
    "allocate outputs, check inputs, and make views or contiguous copies. The harness\n"
    "guard refuses a candidate that adds such a call.\n")

# Llama-3.1-70B at TP1 with AWQ int4, group 128, as (M, N, K): vLLM on ROCm runs this kernel below
# 256 tokens with split_k 8. Of M = 1/32/128 per layer, only the memory-bound cases that stayed near
# or under 3% spread in three 5-round screens are kept; the other M=32 cases and every M=128 case
# reached 4-6%.
AWQ_CASES = [(1, 10240, 8192), (1, 8192, 8192), (1, 57344, 8192), (1, 8192, 28672), (32, 8192, 28672)]
# DeepSeek-V3 linear layers with 128x128 FP8 blocks, the (N, K) pairs DeepGEMM tests, at decode and prefill M.
W8A8_LAYERS = [(2112, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168), (7168, 2048)]

AWQ_REFERENCE = '''def reference_awq_gemm(input_tensor, qweight, scales, qzeros, group_size):
    """CPU reference: dequantize then matmul (the original loops, vectorized)."""
    import torch
    torch.set_num_threads(min(32, torch.get_num_threads()))  # shared host
    K, N_packed = qweight.shape
    N = N_packed * 8
    awq_order = [0, 4, 1, 5, 2, 6, 3, 7]
    shifts = torch.tensor([awq_order[b] * 4 for b in range(8)], dtype=torch.int32)

    qweight_cpu = qweight.cpu().to(torch.int32)
    qzeros_cpu = qzeros.cpu().to(torch.int32)
    scales_cpu = scales.cpu().float()

    dequant = torch.empty((K, N), dtype=torch.float32, device="cpu")
    rows = max(group_size, (1 << 26) // max(N, 1) // group_size * group_size)
    for start in range(0, K, rows):
        end = min(start + rows, K)
        groups = torch.arange(start, end) // group_size
        weight_val = ((qweight_cpu[start:end].unsqueeze(-1) >> shifts) & 0xF).reshape(end - start, N)
        zero_val = ((qzeros_cpu[groups].unsqueeze(-1) >> shifts) & 0xF).reshape(end - start, N)
        dequant[start:end] = (weight_val - zero_val).float() * scales_cpu[groups]

    input_cpu = input_tensor.cpu().float()
    result = input_cpu @ dequant
    return result.to(scales.dtype)


def make_awq_weights(K, N, group_size, device):
    """AWQ-quantize a Gaussian weight with std 1/sqrt(K): per group of rows, asymmetric
    4-bit with scale (max - min) / 15, packed in AWQ nibble order."""
    import torch
    groups = K // group_size
    weight = torch.randn(K, N, device=device, dtype=torch.float32) * K ** -0.5
    grouped = weight.reshape(groups, group_size, N)
    wmin, wmax = grouped.amin(dim=1), grouped.amax(dim=1)
    scales = ((wmax - wmin) / 15).clamp_min(1e-4)
    zeros = (-wmin / scales).round().clamp(0, 15)
    q = (grouped / scales[:, None, :] + zeros[:, None, :]).round().clamp(0, 15).reshape(K, N)
    del weight, grouped
    shifts = torch.tensor([o * 4 for o in (0, 4, 1, 5, 2, 6, 3, 7)], device=device, dtype=torch.int64)

    def pack(values):
        packed = (values.to(torch.int64).reshape(values.shape[0], -1, 8) << shifts).sum(-1)
        return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)

    return pack(q), scales.to(torch.float16), pack(zeros)


'''
W8A8_REFERENCE = '''def reference_w8a8_block_scaled_mm(A, B, As, Bs, block_size, output_dtype):
    """CPU reference: block-wise dequantize then matmul (the original loops, vectorized)."""
    import torch
    torch.set_num_threads(min(32, torch.get_num_threads()))  # shared host
    block_n, block_k = block_size
    M, K = A.shape
    N = B.shape[0]

    A_f = A.cpu().float()
    B_f = B.cpu().float()
    As_f = As.cpu().float()
    Bs_f = Bs.cpu().float()

    A_dq = A_f * As_f.repeat_interleave(block_k, dim=1)[:, :K]
    B_dq = B_f * Bs_f.repeat_interleave(block_n, dim=0)[:N].repeat_interleave(block_k, dim=1)[:, :K]

    # matmul: A_dq [M,K] @ B_dq.T [K,N] = [M,N]
    result = A_dq @ B_dq.T
    return result.to(output_dtype)


'''
AWQ_RANDOM_INPUTS = '''qweight = torch.randint(0, 2**31, (K, N_packed), device=device, dtype=torch.int32)
            scales = torch.randn(num_groups, N, device=device, dtype=dtype).abs() * 0.1 + 0.01
            qzeros = torch.randint(0, 2**31, (num_groups, N_packed), device=device, dtype=torch.int32)'''
AWQ_GATE = "    torch.testing.assert_close(output, expected, atol=1e-1, rtol=1e-1)\n"
AWQ_NORMWISE_GATE = AWQ_GATE + '''    error = (output.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    if error > 5e-3:
        raise AssertionError(f'AWQ GEMM normwise relative error {error.item():.2e} exceeds 5e-3')
'''

# source task -> (variant task, cases, header of the case table, reference symbol, its replacement,
#                 README introduction, README replacements, task-file replacements beyond the case table)
LLM_SHAPE_VARIANTS = {
    "triton2triton/vllm/triton_awq_gemm": (
        "triton2triton/vllm_large/triton_awq_gemm",
        [(m, k, n // 8, 128, 8) for (m, n, k) in AWQ_CASES],
        "# Test configs: (M, K, N_packed, group_size, split_k)",
        "reference_awq_gemm", AWQ_REFERENCE,
        "Its cases are decode-time linear layers of Llama-3.1-70B with AWQ int4 weights (group size 128) "
        "on one GPU, as (M, N, K): qkv (1, 10240, 8192), o (1, 8192, 8192), gate_up (1, 57344, 8192), "
        "down (1, 8192, 28672) and down (32, 8192, 28672). split_k is 8, the value vLLM passes; on ROCm "
        "vLLM runs this kernel for batches below 256 tokens. The weights are a Gaussian quantized the way "
        "AWQ does it, so scales, zero points and outputs have a real layer's magnitudes, and outputs must "
        "also stay within 5e-3 normwise relative error of the reference. The whole source file is "
        "editable, including the wrapper's tile sizes and split-K handling.",
        [("use the original independent CPU AWQ unpacking and GEMM\nreference, with the original numerical gate.",
          "use an independent CPU AWQ unpacking and GEMM\nreference (the original loops, vectorized) with the "
          "original numerical gate\nand a 5e-3 normwise relative error bound."),
         ("The original five scored shapes, input seeds and parameter table are unchanged.",
          "The scored shapes and weights are this variant's; the input seeds are the original's.")],
        [("scripts/task_runner.py", AWQ_RANDOM_INPUTS, "qweight, scales, qzeros = make_awq_weights(K, N, group_size, device)", 2),
         ("_arena_checks.py", "scales.mul_(.5).add_(.015625)", "scales.mul_(.75)", 1),
         ("_arena_checks.py", AWQ_GATE, AWQ_NORMWISE_GATE, 1)]),
    "triton2triton/vllm/triton_w8a8_block_scaled_mm": (
        "triton2triton/vllm_large/triton_w8a8_block_scaled_mm",
        [(m, n, k, 128, 128) for (n, k) in W8A8_LAYERS for m in (128, 4096)],
        "# Test configs: (M, N, K, block_n, block_k)",
        "reference_w8a8_block_scaled_mm", W8A8_REFERENCE,
        "Its cases are linear layers of DeepSeek-V3 with 128x128 FP8 blocks, the six (N, K) pairs "
        "DeepGEMM tests: (2112, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168) and "
        "(7168, 2048), each at M = 128 (decode) and M = 4096 (prefill). The whole source file is "
        "editable, including the launch configuration in `w8a8_triton_block_scaled_mm`.",
        [("against the original CPU block-dequantization/matmul oracle,",
          "against the CPU block-dequantization/matmul oracle\n(the original loops, vectorized),"),
         ("The five scored shapes, seeds and input magnitudes remain unchanged, including\n"
          "the larger performance scales/operands.",
          "The scored shapes are this variant's; seeds and input magnitudes are the\n"
          "original's, including the larger performance scales/operands."),
         ("in any original scored case remain failures;", "in any scored case remain failures;")],
        []),
}
CASE_TABLE = re.compile(r"# Test configs:[^\n]*\nTEST_SHAPES = \[\n(?:    [^\n]*\n)+\]\n")


def contiguous_stride(shape: list[int]) -> list[int]:
    return [math.prod(shape[i + 1:]) for i in range(len(shape))]


def rewrite_get_inputs(path: Path, cases: list[tuple]) -> None:
    text = path.read_text()
    lines = "\n".join(f"        {case!r}," for case in cases)
    updated, count = CONFIGS_BLOCK.subn(lambda m: m.group(1) + lines + m.group(2), text, count=1)
    if count != 1:
        raise ValueError(f"{path}: no `configs = [...]` block in get_inputs()")
    path.write_text(updated)


def copy_task(aka: Path, source: str, target: str, force: bool) -> Path:
    src, dst = aka / "tasks" / source, aka / "tasks" / target
    if dst.exists():
        if not force:
            raise FileExistsError(f"{dst} exists; pass --force to rewrite it")
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def add_own_kernels_rule(readme: Path) -> None:
    text = readme.read_text()
    if PROTECTED_CODE_RULE not in text:
        raise ValueError(f"{readme}: no protected-code rule to extend")
    readme.write_text(text.replace(PROTECTED_CODE_RULE, PROTECTED_CODE_RULE + OWN_KERNELS_RULE, 1))


def write_variant(aka: Path, source: str, target: str, cases: list[tuple], force: bool) -> Path:
    dst = copy_task(aka, source, target, force)
    for folder in ("pytorch_code_module", "pytorch_code_functional"):
        for path in sorted((dst / folder).glob("*.py")):
            rewrite_get_inputs(path, cases)
    manifest_path = dst / "workload.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"] = [
        {"test_case_id": f"case_{index}",
         "params": {"inputs": [{"shape": list(shape), "dtype": "torch.float32", "stride": contiguous_stride(list(shape))}
                               for shape in case],
                    "model_init_seed": 0, "correctness_seed": 1337 + index}}
        for index, case in enumerate(cases)]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    readme = dst / "README.md"
    text, count = re.subn(r"\*\*\d+ cases\*\*", f"**{len(cases)} cases**", readme.read_text())
    if count == 0:
        raise ValueError(f"{readme}: no case count to update")
    readme.write_text(text)
    add_own_kernels_rule(readme)
    return dst


def write_own_kernels_variant(aka: Path, source: str, target: str, force: bool) -> Path:
    dst = copy_task(aka, source, target, force)
    add_own_kernels_rule(dst / "README.md")
    return dst


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"{path}: expected one {old.splitlines()[0]!r}, found {text.count(old)}")
    path.write_text(text.replace(old, new, 1))


def write_open_launcher_variant(aka: Path, source: str, target: str, launchers: list[str], note: str,
                                force: bool) -> Path:
    dst = copy_task(aka, source, target, force)
    workloads = json.loads((dst / "workloads.json").read_text())
    kernel_file = dst / workloads["source"]
    if "flash_fwd_launch" in launchers:
        replace_once(kernel_file, FLASH_LAUNCH_CODE, FLASH_OPEN_LAUNCH_CODE)
    replace_once(dst / "config.yaml", "    allow_new_helpers: true\n",
                 "".join(f"    - {name}\n" for name in launchers) + "    allow_new_helpers: true\n")
    readme = dst / "README.md"
    text = readme.read_text().replace("original assertions/tolerances, launch parameters,\n",
                                      "original assertions/tolerances,\n")
    kernels = ", ".join(f"`{item['name'] if isinstance(item, dict) else item}`"
                        for item in workloads.get("kernel_symbols") or workloads["candidate_symbols"])
    names = " and ".join(f"`{name}`" for name in launchers)
    intro = (f"This is the open-launcher variant of `{source}`. Besides {kernels}, {names} in "
             f"`{workloads['source']}` {'is' if len(launchers) == 1 else 'are'} editable: a candidate chooses "
             f"its own tiling, grid, `num_warps`, `num_stages` and other launch parameters, keeping the "
             f"launcher's signature and return value. {note}")
    section = (f"## Open-launcher variant\n\n{textwrap.fill(' '.join(intro.split()), 88)}\n\n"
               f"{OWN_TRITON_KERNELS_RULE}\n")
    heading = text.index("\n## ") + 1
    readme.write_text(text[:heading] + section + text[heading:])
    return dst


def write_llm_shape_variant(aka: Path, source: str, target: str, cases: list[tuple], header: str, reference: str,
                            new_reference: str, intro: str, readme_edits: list[tuple[str, str]],
                            task_edits: list[tuple[str, str, str, int]], force: bool) -> Path:
    dst = copy_task(aka, source, target, force)
    shutil.rmtree(dst / "build", ignore_errors=True)
    runner = dst / "scripts/task_runner.py"
    table = header + "\nTEST_SHAPES = [\n" + "".join(f"    {case},\n" for case in cases) + "]\n"
    text, count = CASE_TABLE.subn(table, runner.read_text(), count=1)
    if count != 1:
        raise ValueError(f"{runner}: no TEST_SHAPES table")
    text, count = re.subn(rf"^def {reference}\(.*?(?=^def run_compile)", new_reference, text, flags=re.S | re.M)
    if count != 1:
        raise ValueError(f"{runner}: no {reference}")
    runner.write_text(text)
    for relative, old, new, expected in task_edits:
        path = dst / relative
        if path.read_text().count(old) != expected:
            raise ValueError(f"{path}: expected {expected} of {old.splitlines()[0]!r}")
        path.write_text(path.read_text().replace(old, new))
    workloads_path = dst / "workloads.json"
    workloads = json.loads(workloads_path.read_text())
    template = workloads["cases"][0]
    workloads.pop("migration", None)
    workloads["input_table"] = [list(case) for case in cases]
    workloads["cases"] = [
        {"test_case_id": f"perf{index + 1}", "dtype": template["dtype"],
         "params": {"case_index": index, "configuration": list(case),
                    "configuration_names": template["params"]["configuration_names"]},
         "checks": list(template["checks"]), "status": "PASS"}
        for index, case in enumerate(cases)]
    workloads_path.write_text(json.dumps(workloads, indent=2) + "\n")
    readme = dst / "README.md"
    replace_once(readme, "declares all five original cases;", f"declares all {len(cases)} cases;")
    for old, new in readme_edits:
        replace_once(readme, old, new)
    text = readme.read_text()
    section = (f"## LLM-shape variant\n\n{textwrap.fill(f'This is the LLM-shape variant of `{source}`. ' + intro, 88)}"
               f"\n\n{OWN_TRITON_KERNELS_RULE}\n")
    heading = text.index("\n## ") + 1
    readme.write_text(text[:heading] + section + text[heading:])
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aka", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true", help="rewrite variants that already exist")
    args = parser.parse_args()
    for source, (target, cases) in VARIANTS.items():
        print(write_variant(args.aka, source, target, cases, args.force))
    for source, (target, launchers, note) in OPEN_LAUNCHERS.items():
        print(write_open_launcher_variant(args.aka, source, target, launchers, note, args.force))
    for source, target in OWN_KERNEL_VARIANTS.items():
        print(write_own_kernels_variant(args.aka, source, target, args.force))
    for source, spec in LLM_SHAPE_VARIANTS.items():
        print(write_llm_shape_variant(args.aka, source, *spec, args.force))
    return 0


if __name__ == "__main__":
    sys.exit(main())
