#!/usr/bin/env python3
"""Static prefilter of implemented Triton and HIP tasks for the rewrite-and-tune screen.

A task passes when its editable code launches at most MAX_LAUNCHED distinct kernels
(dispatch variants count separately here), every launch of an editable kernel
sits in editable code (so tile sizes, warps and the grid are the candidate's to
choose), the editable code delegates no compute to a library, and nothing marks
it gfx950-only. The measured screen then requires exactly one kernel per call.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import yaml

CATEGORIES = ("triton2triton", "hip2hip")
MAX_LAUNCHED = 3
SUBSCRIPT_START = re.compile(r"\b(\w+)\[")
LIBRARY_PY = re.compile(r"\btorch\.(mm|matmul|bmm|baddbmm|addmm|einsum|conv\d?d|_scaled_mm|cumsum|sort|topk|softmax|"
                        r"layer_norm|rms_norm)\b|\bF\.(linear|conv\d?d|softmax|layer_norm|rms_norm|"
                        r"scaled_dot_product_attention)\b|\baiter\.|\bflash_attn\b")
KERNEL_HIP = re.compile(r"__global__\s+(?:void\s+)?(?:__launch_bounds__\([^)]*\)\s+)?(?:static\s+)?void\s+(\w+)")
LAUNCH_HIP = re.compile(r"\b(\w+)\s*(?:<[^;{}]*?>)?\s*<<<|hipLaunchKernelGGL\(\s*(?:HIP_KERNEL_NAME\()?\(?\s*(\w+)")
LIBRARY_HIP = re.compile(r"hipblas|rocblas|hipblaslt|miopen|rocprim|hipcub|thrust::|at::(?:mm|matmul|bmm|conv)|torch::(?:mm|matmul)")
ARCH_ONLY = re.compile(r"gfx950|gfx12|MI355|mxfp4|\bfp4\b|e2m1|CDNA4|RDNA", re.IGNORECASE)


def python_launches(text: str) -> set[str]:
    """Names called as `name[grid](...)`; the grid subscript may nest brackets."""
    names = set()
    for match in SUBSCRIPT_START.finditer(text):
        depth, i = 1, match.end()
        while i < len(text) and depth:
            depth += {"[": 1, "]": -1}.get(text[i], 0)
            i += 1
        while i < len(text) and text[i] in " \t":
            i += 1
        if depth == 0 and i < len(text) and text[i] == "(":
            names.add(match.group(1))
    return names


def editable_entries(candidate: dict) -> list[tuple[str, str, set[str]]]:
    out = []
    for entry in candidate.get("editable", []):
        if isinstance(entry, str):
            out.append((entry, "file", set()))
        else:
            out.append((entry["path"], entry.get("scope", "file"), set(entry.get("symbols") or [])))
    return out


def python_sources(task_dir: Path) -> dict[str, str]:
    return {str(p.relative_to(task_dir)): p.read_text(errors="replace") for p in task_dir.rglob("*.py")
            if "__pycache__" not in p.parts and "build" not in p.parts and not p.name.startswith("_arena")}


def split_python(text: str, symbols: set[str]) -> tuple[str, str]:
    """Return (editable, protected) source text of one file for a symbols scope."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "", text
    editable, protected = [], []
    for node in tree.body:
        segment = ast.get_source_segment(text, node) or ""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in symbols:
            decorators = "\n".join("@" + ast.get_source_segment(text, d) for d in node.decorator_list)
            editable.append(decorators + "\n" + segment)
        else:
            protected.append(segment)
    return "\n".join(editable), "\n".join(protected)


def jit_kernels(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any("jit" in ast.unparse(d) for d in node.decorator_list):
            names.add(node.name)
    return names


def triton_facts(task_dir: Path, entries) -> dict:
    sources = python_sources(task_dir)
    editable_text, protected_text = [], []
    for rel, text in sources.items():
        match = next((e for e in entries if e[0] == rel), None)
        if match is None:
            protected_text.append(text)
        elif match[1] == "file":
            editable_text.append(text)
        else:
            e, p = split_python(text, match[2])
            editable_text.append(e)
            protected_text.append(p)
    editable, protected = "\n".join(editable_text), "\n".join(protected_text)
    kernels = set().union(*(jit_kernels(t) for t in sources.values())) if sources else set()
    editable_kernels = jit_kernels(editable) | {k for k in kernels if re.search(rf"def {k}\b", editable)}
    launched_editable = python_launches(editable) & kernels
    launched_protected = python_launches(protected) & editable_kernels
    return {"kernels_launched_in_editable": sorted(launched_editable),
            "editable_kernels_launched_by_protected_code": sorted(launched_protected),
            "library_calls_in_editable": sorted({m.group(0) for m in LIBRARY_PY.finditer(editable)}),
            "editable_text": editable}


def hip_facts(task_dir: Path, entries) -> dict:
    editable = "\n".join((task_dir / rel).read_text(errors="replace") for rel, _, _ in entries if (task_dir / rel).is_file())
    defined = set(KERNEL_HIP.findall(editable))
    launched = {a or b for a, b in LAUNCH_HIP.findall(editable)} & (defined or {a or b for a, b in LAUNCH_HIP.findall(editable)})
    return {"kernels_launched_in_editable": sorted(launched),
            "editable_kernels_launched_by_protected_code": [] if launched else sorted(defined),
            "library_calls_in_editable": sorted(set(LIBRARY_HIP.findall(editable))),
            "editable_text": editable}


def performance_cases(task_dir: Path, config: dict) -> int:
    path = task_dir / (config.get("evaluation") or {}).get("workloads", "")
    if not path.is_file():
        return 0
    data = json.loads(path.read_text())
    cases = data if isinstance(data, list) else data.get("cases", [])
    return sum("performance" in (c.get("checks") or ["performance"]) for c in cases if isinstance(c, dict))


def examine(config_path: Path, tasks_root: Path) -> dict:
    task_dir = config_path.parent
    config = yaml.safe_load(config_path.read_text())
    candidate = config.get("candidate") or {}
    entries = editable_entries(candidate)
    language = candidate.get("language")
    facts = (triton_facts if language == "triton" else hip_facts)(task_dir, entries)
    readme = (task_dir / "README.md").read_text(errors="replace") if (task_dir / "README.md").is_file() else ""
    arch = sorted({m.group(0).lower() for m in ARCH_ONLY.finditer(readme + facts.pop("editable_text"))})
    row = {"task": str(task_dir.relative_to(tasks_root)), "language": language,
           "initial_state": candidate.get("initial_state"), "baseline": (config.get("baseline") or {}).get("kind"),
           "scopes": sorted({e[1] for e in entries}), "performance_cases": performance_cases(task_dir, config),
           "arch_markers": arch, **facts}
    reasons = []
    if row["initial_state"] != "implemented":
        reasons.append("not implemented")
    if not 1 <= len(row["kernels_launched_in_editable"]) <= MAX_LAUNCHED:
        reasons.append(f"{len(row['kernels_launched_in_editable'])} kernels launched in editable code")
    if row["editable_kernels_launched_by_protected_code"]:
        reasons.append("launched by protected code")
    if row["library_calls_in_editable"]:
        reasons.append("library compute")
    if arch:
        reasons.append("arch-specific")
    if row["performance_cases"] < 3:
        reasons.append("fewer than 3 performance cases")
    row["rejected_because"] = reasons
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aka", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, help="write every row as JSON here")
    args = parser.parse_args()
    tasks_root = args.aka / "tasks"
    rows = []
    for category in CATEGORIES:
        for config_path in sorted((tasks_root / category).rglob("config.yaml")):
            try:
                rows.append(examine(config_path, tasks_root))
            except Exception as exc:  # one malformed task must not stop the prefilter
                rows.append({"task": str(config_path.parent.relative_to(tasks_root)), "rejected_because": [f"error: {exc}"]})
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
    passing = [r for r in rows if not r["rejected_because"]]
    for row in passing:
        print(f"{row['task']:70} {row['language']:6} cases={row['performance_cases']:3} "
              f"kernels={','.join(row['kernels_launched_in_editable'])}")
    tally: dict[str, int] = {}
    for row in rows:
        for reason in row["rejected_because"]:
            key = reason.split(" kernels")[0] + " kernels" if "kernels launched" in reason else reason
            tally[key] = tally.get(key, 0) + 1
    print(f"\n{len(passing)} of {len(rows)} pass; rejections: {tally}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
