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
    return 0


if __name__ == "__main__":
    sys.exit(main())
