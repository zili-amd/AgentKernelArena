#!/usr/bin/env python3
"""Compare what the knowledge base returns for each task's likely questions at two wiki revisions.

Each query runs the way the forge wiki tool runs it (query.py with the agent's
text after "--"). The report lists the top hits per revision and marks pages
that exist only in the newer one. With --calls it also replays the search and
grep calls a run logged (knowledge_wiki_calls.jsonl) with their recorded
arguments, and summarizes how often the top hits changed and which new pages
displaced which.
    wiki_search_check.py --wiki PATH --before REV [--after REV|worktree] [--limit 5]
                         [--calls LABEL=knowledge_wiki_calls.jsonl ...]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
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


# The forge wiki tool's argv and result parsing, so a replay returns what the agent saw.
SEARCH_HIT = re.compile(r"^\s+\[[^\]]+\]\s+(\S+):|^- \*\*id\*\*: `([^`]+)`", re.MULTILINE)
GREP_FILE = re.compile(r"^=== (\S+) \(|^## (\S+)  \(\d+ match", re.MULTILINE)


def replay(root: Path, call: dict) -> list[str]:
    args = call["args"]
    if call["command"] == "search":
        argv = ["query.py"]
        for option in ("architecture", "tag", "type"):
            if args.get(option):
                argv += [f"--{option}", args[option]]
        if args.get("limit") is not None:
            argv += ["--limit", str(args["limit"])]
        argv += ["--", args["query"]]
        pattern = SEARCH_HIT
    else:
        argv = ["grep_wiki.py", *(["--only", args["only"]] if args.get("only") else []), "--", args["pattern"]]
        pattern = GREP_FILE
    out = subprocess.run([sys.executable, str(root / "scripts" / argv[0]), *argv[1:]],
                         capture_output=True, text=True, cwd=root, timeout=60).stdout
    return [Path("".join(groups)).stem for groups in pattern.findall(out)]


def replay_calls(before: Path, after: Path, known: set[str], logs: list[str], top: int) -> None:
    changed = total = grep_changed = grep_total = 0
    surfaced, displaced, grep_new = collections.Counter(), collections.Counter(), collections.Counter()
    sections = []
    # Search returns page IDs (pr-3881); file stems keep the source's case (PR-3881).
    known = {name.lower() for name in known}
    for spec in logs:
        label, _, path = spec.rpartition("=")
        calls = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        rows = []
        for call in calls:
            if call["command"] not in ("search", "grep"):
                continue
            old, new = replay(before, call), replay(after, call)
            logged = [Path(item).stem for item in call["results"]]
            text = call["args"].get("query") or call["args"].get("pattern")
            if call["command"] == "grep":
                # grep_wiki.py lists files in directory-walk order, which differs between checkouts.
                if set(old) != set(logged):
                    print(f"warning: {label} {call['ts']} grep no longer reproduces at the before revision",
                          file=sys.stderr)
                added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
                grep_total += 1
                grep_changed += bool(added or removed)
                grep_new.update(h for h in added if h.lower() not in known)
                marked = [f"{h}*" if h.lower() not in known else h for h in added]
                rows.append(f"| {call['role']} | grep | `{text}` | {len(old)} files | {len(new)} files; added "
                            f"{', '.join(marked) or '—'}; removed {', '.join(removed) or '—'} |")
                continue
            if old != logged:
                print(f"warning: {label} {call['ts']} search no longer reproduces at the before revision",
                      file=sys.stderr)
            total += 1
            changed += old[:top] != new[:top]
            surfaced.update(h for h in new[:top] if h.lower() not in known)
            displaced.update(h for h in old[:top] if h not in new[:top])
            marked = [f"{h}*" if h.lower() not in known else h for h in new[:top]]
            rows.append(f"| {call['role']} | search | `{text}` | {', '.join(old[:top]) or '—'} | "
                        f"{', '.join(marked) or '—'} |")
        sections.append(f"### {label}\n\n| Role | Tool | Query or pattern | Before (top {top}) | "
                        f"After (top {top}; * = new page) |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    print(f"## Replay of logged calls\n\n{changed} of {total} search calls change their top {top}; "
          f"{grep_changed} of {grep_total} grep calls change their file set.\n")
    print(f"New pages entering a search top {top}: "
          + (", ".join(f"{k} ({v})" for k, v in surfaced.most_common()) or "none"))
    print(f"\nPages leaving a search top {top}: "
          + (", ".join(f"{k} ({v})" for k, v in displaced.most_common()) or "none"))
    print("\nNew pages added to grep results: " + (", ".join(f"{k} ({v})" for k, v in grep_new.most_common()) or "none"))
    print("\n" + "\n".join(sections))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wiki", type=Path, required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", default="worktree")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--calls", nargs="*", default=[], metavar="LABEL=PATH")
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
        if args.calls:
            replay_calls(before, after, known, args.calls, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
