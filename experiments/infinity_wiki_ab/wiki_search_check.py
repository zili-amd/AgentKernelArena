#!/usr/bin/env python3
"""Compare what the knowledge base returns for each task's likely questions at two wiki revisions.

Each query runs the way the forge wiki tool runs it (query.py with the agent's
text after "--"). The report lists the top hits per revision and marks pages
that exist only in the newer one.
    wiki_search_check.py --wiki PATH --before REV [--after REV|worktree] [--limit 5]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

QUERIES = {
    "triton2triton/rocmbench/hard/gemm (variant)": [
        "Triton fp16 GEMM MI300X tuning BLOCK_M BLOCK_N num_stages matrix_instr_nonkdim",
        "persistent GEMM stream-K split-K wave quantization 304 CUs",
        "GEMM output tiles not a multiple of CU count tail wave",
        "XCD-aware program id remap GEMM L2 locality",
    ],
    "triton2triton/rocmbench/medium/naive_softmax (variant)": [
        "softmax Triton row reduction bandwidth bound MI300X",
        "online softmax single pass max and sum normalizer",
        "tl.max tl.sum reduction lowering wave64",
        "softmax long rows vocab 32000 block size num_warps",
    ],
    "hip2hip/gpumode_large/SimpleMatmulModule": [
        "HIP fp32 GEMM MFMA gfx942 v_mfma_f32_32x32x2f32",
        "xf32 precision fp32 matmul MI300X",
        "HIP GEMM LDS tiling register blocking wave tile",
        "FP32 vector peak vs matrix peak MI300X",
    ],
    "hip2hip/gpumode_large/SoftmaxModule": [
        "HIP warp shuffle reduction wave64 __shfl_xor",
        "block reduce shared memory warpSize 64",
        "softmax HIP row reduction vectorized float4 loads",
        "DPP cross-lane reduction gfx942",
    ],
}


def checkout(wiki: Path, rev: str, into: Path) -> Path:
    if rev == "worktree":
        return wiki
    into.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(wiki), "archive", rev], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(into)], input=archive, check=True)
    return into


def hits(root: Path, query: str, limit: int) -> list[str]:
    out = subprocess.run([sys.executable, str(root / "scripts/query.py"), "--paths-only", "--limit", str(limit),
                          "--", query], capture_output=True, text=True, cwd=root).stdout
    return [Path(line.strip()).stem for line in out.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wiki", type=Path, required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", default="worktree")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        before = checkout(args.wiki, args.before, Path(scratch) / "before")
        after = checkout(args.wiki, args.after, Path(scratch) / "after")
        known = {p.stem for p in before.rglob("*.md")}
        print(f"# Wiki search check: {args.before} -> {args.after}\n")
        for task, queries in QUERIES.items():
            print(f"## {task}\n\n| Query | Before (top {args.limit}) | After (top {args.limit}; * = new page) |\n|---|---|---|")
            for query in queries:
                old = hits(before, query, args.limit)
                new = [f"{h}*" if h not in known else h for h in hits(after, query, args.limit)]
                print(f"| {query} | {', '.join(old) or '—'} | {', '.join(new) or '—'} |")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
