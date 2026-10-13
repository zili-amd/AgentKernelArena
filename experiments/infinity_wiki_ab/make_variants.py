#!/usr/bin/env python3
"""Write production-sized variants of hip2hip/gpumode tasks.

The originals measure 4x4 to 128x128 inputs, where every case sits at the launch
floor and no rewrite can show. A variant keeps the task's kernel, reference,
tolerances and timing code and replaces only the case list: the configs in
get_inputs() of both reference modules, and workload.json, which the evaluator
checks against them case by case. The README also states that the operator must
stay in the candidate's own kernels, which the harness guard enforces; at these
sizes a library call would otherwise be the fastest candidate.
    make_variants.py [--aka PATH] [--force]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
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


def contiguous_stride(shape: list[int]) -> list[int]:
    return [math.prod(shape[i + 1:]) for i in range(len(shape))]


def rewrite_get_inputs(path: Path, cases: list[tuple]) -> None:
    text = path.read_text()
    lines = "\n".join(f"        {case!r}," for case in cases)
    updated, count = CONFIGS_BLOCK.subn(lambda m: m.group(1) + lines + m.group(2), text, count=1)
    if count != 1:
        raise ValueError(f"{path}: no `configs = [...]` block in get_inputs()")
    path.write_text(updated)


def write_variant(aka: Path, source: str, target: str, cases: list[tuple], force: bool) -> Path:
    src, dst = aka / "tasks" / source, aka / "tasks" / target
    if dst.exists():
        if not force:
            raise FileExistsError(f"{dst} exists; pass --force to rewrite it")
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
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
    if PROTECTED_CODE_RULE not in text:
        raise ValueError(f"{readme}: no protected-code rule to extend")
    readme.write_text(text.replace(PROTECTED_CODE_RULE, PROTECTED_CODE_RULE + OWN_KERNELS_RULE, 1))
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aka", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true", help="rewrite variants that already exist")
    args = parser.parse_args()
    for source, (target, cases) in VARIANTS.items():
        print(write_variant(args.aka, source, target, cases, args.force))
    return 0


if __name__ == "__main__":
    sys.exit(main())
