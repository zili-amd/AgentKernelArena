#!/usr/bin/env python3
"""Freeze each mxfp4_seed task's first correct FlyDSL port as the initial candidate of its mxfp4_moe_ab variant.

    spur_freeze_seeds.py ROOT

For every task of the mxfp4_seed phase, the lowest repeat whose engine committed "forge-rewrite: initial correct
flydsl port" supplies the seed: kernel.py as that commit holds it, before the rewrite's search changed anything.
Each arm checkout then gets tasks/Aiter-task-seeded/<task>, a copy of tasks/Aiter-task/<task> declared implemented
with that kernel, and evaluate.py's initial-state check following the declared state (as in tasks/Aiter-task-mi300).
ROOT/seeds/<task>/ keeps the kernel and its provenance. Prints the seeded task paths, one per line, on stdout;
tasks without a port are reported on stderr and left out.
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PORT_SUBJECT = "forge-rewrite: initial correct flydsl port"
SEED_PHASE, SEEDED_SUITE = "mxfp4_seed", "Aiter-task-seeded"
MI300_REFERENCE = "tasks/Aiter-task-mi300/gemm_a16w16_nt_n128_k6144/scripts/evaluate.py"
NOTE = """# Seeded variant of `tasks/Aiter-task/{name}`

The workload, inputs, reference, comparator, timing and baseline entry are unchanged. The candidate is declared
`implemented`: `kernel.py` is the first correct FlyDSL port KernelForge's FlyDSL rewrite committed for the original
task in the `mxfp4_seed` phase (run `{tag}`, engine commit `{commit}`), taken before the rewrite's search changed it.
Both arms of `mxfp4_moe_ab` start from it, so both run forge-loop instead of the rewrite. `scripts/evaluate.py`'s
initial-state check follows the state `config.yaml` declares. Everything below is the original task's README.

"""


def git(engine: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(engine), *args], check=True, capture_output=True, text=True).stdout


def function_source(path: Path, name: str) -> str:
    text = path.read_text()
    start = text.index(f"def {name}(")
    return text[start:text.index("\n\n\ndef ", start)]


def find_port(aka: Path, task: str, repeat: int, entry: str) -> dict | None:
    tag = f"{SEED_PHASE}-{task.split('/', 1)[1].replace('/', '.')}-r{repeat}-kf"
    for run in sorted(aka.glob(f"workspace_*/run_*_{tag}")):
        for engine in sorted([*run.glob("*-forge-*/engine"), *run.glob("*/*-forge-*/engine")]):
            try:
                log = git(engine, "log", "--all", "--reverse", "--format=%H%x09%ct%x09%s")
            except subprocess.CalledProcessError:
                continue
            for line in log.splitlines():
                commit, when, subject = line.split("\t", 2)
                if subject != PORT_SUBJECT:
                    continue
                paths = [p for p in git(engine, "show", "--name-only", "--format=", commit).splitlines()
                         if Path(p).name == entry]
                if len(paths) != 1:
                    print(f"{tag}: port {commit[:12]} holds {len(paths)} {entry} files; skipped", file=sys.stderr)
                    continue
                return {"tag": tag, "engine": str(engine), "commit": commit, "committed_unix": int(when),
                        "path": paths[0], "source": git(engine, "show", f"{commit}:{paths[0]}")}
    return None


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    manifest = yaml.safe_load((HERE / "manifest.yaml").read_text())
    phase = manifest["phases"][SEED_PHASE]
    arms = (root / "dist" / "arms.txt").read_text().split()
    aka_kf = root / "aka-kf"
    seeded = []
    for task in phase["tasks"]:
        name = task.split("/", 1)[1]
        original = aka_kf / "tasks" / task
        config = yaml.safe_load((original / "config.yaml").read_text())
        entry, symbol = config["candidate"]["entrypoints"][0]["file"], config["candidate"]["entrypoints"][0]["symbol"]
        port = next(filter(None, (find_port(aka_kf, task, r, entry) for r in range(phase["repeats"]))), None)
        if port is None:
            print(f"{task}: no repeat committed a correct port; no seeded variant", file=sys.stderr)
            continue
        tree = ast.parse(port["source"])
        if not any(isinstance(node, ast.FunctionDef) and node.name == symbol for node in tree.body):
            print(f"{task}: the port from {port['tag']} defines no {symbol}; no seeded variant", file=sys.stderr)
            continue
        keep = root / "seeds" / name
        keep.mkdir(parents=True, exist_ok=True)
        (keep / entry).write_text(port["source"])
        (keep / "provenance.json").write_text(json.dumps({k: v for k, v in port.items() if k != "source"}, indent=2))
        for arm in arms:
            aka = root / f"aka-{arm}"
            reference = function_source(aka / MI300_REFERENCE, "check_initial_state")
            source_task = aka / "tasks" / task
            target = aka / "tasks" / SEEDED_SUITE / name
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(source_task, target, ignore=shutil.ignore_patterns("__pycache__"))
            (target / entry).write_text(port["source"])
            cfg = (target / "config.yaml").read_text()
            assert cfg.count("initial_state: unimplemented") == 1, f"{target}: unexpected config.yaml"
            cfg = cfg.replace("initial_state: unimplemented", "initial_state: implemented")
            cfg = cfg.replace(f"description: Implement {name} in FlyDSL over every declared case.",
                              f"description: Optimize the seeded FlyDSL {name} kernel on gfx950 over every declared case.")
            (target / "config.yaml").write_text(cfg)
            evaluate = target / "scripts" / "evaluate.py"
            text = evaluate.read_text()
            original_check = function_source(evaluate, "check_initial_state")
            assert text.count(original_check) == 1
            evaluate.write_text(text.replace(original_check, reference))
            readme = target / "README.md"
            readme.write_text(NOTE.format(name=name, tag=port["tag"], commit=port["commit"][:12]) + readme.read_text())
        print(f"{task}: seeded from {port['tag']} commit {port['commit'][:12]}", file=sys.stderr)
        seeded.append(f"{SEEDED_SUITE}/{name}")
    print("\n".join(seeded))
    return 0


if __name__ == "__main__":
    sys.exit(main())
