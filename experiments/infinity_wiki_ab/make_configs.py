#!/usr/bin/env python3
"""Write one AgentKernelArena run config per (phase, task, arm) from manifest.yaml.

With --aka, each config is also validated by AKA's own loaders: the forge or
GEAK option parser, and knowledge_wiki.validate_run (with --wiki-root standing in
for the AKA_KNOWLEDGE_WIKI_ROOT the runner forwards).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def task_slug(task: str) -> str:
    return task.replace("/", "__")


def run_config(manifest: dict, phase: dict, task: str, arm: str) -> dict:
    spec = manifest["arms"][arm]
    agent = {"template": spec["template"], "model": manifest["pins"]["model"],
             "timeout_seconds": phase["timeout_seconds"]}
    if spec["template"] == "forge":
        agent["experience_kb"] = "isolated"
        targets = phase.get("speedup_targets")
        if targets is not None:
            if task not in targets:
                raise ValueError(f"the phase sets speedup targets but none for {task}")
            agent["target_mean_case_speedup"] = targets[task]
    if spec.get("wiki_guided"):
        # The playbook is built from the knowledge base, so a guided arm without one would plan unguided.
        if spec["template"] != "forge" or not spec["knowledge_wiki"]:
            raise ValueError(f"arm {arm}: wiki_guided needs the forge template and a knowledge_wiki pin")
        agent["wiki_guided"] = True
    config = {"agent": agent, "tasks": [task], "target_gpu_model": "MI300",
              "log_directory": "logs", "workspace_directory_prefix": "workspace"}
    if spec["knowledge_wiki"]:
        config["knowledge_wiki"] = {"enabled": True}
    return config


def validate(config: dict, aka: Path, wiki_root: Path | None) -> None:
    if str(aka) not in sys.path:
        sys.path.insert(0, str(aka))
    import os
    from src import knowledge_wiki
    template = config["agent"]["template"]
    if template == "forge":
        from agents.forge.adapter import _config
        _config(config)
    else:
        from agents.geak.launch_agent import load_options
        load_options(config)
    previous = os.environ.pop(knowledge_wiki.ROOT_ENV, None)
    if wiki_root:
        os.environ[knowledge_wiki.ROOT_ENV] = str(wiki_root)
    try:
        wiki = knowledge_wiki.validate_run(config, template)
    finally:
        os.environ.pop(knowledge_wiki.ROOT_ENV, None)
        if previous is not None:
            os.environ[knowledge_wiki.ROOT_ENV] = previous
    if (wiki is not None) != ("knowledge_wiki" in config):
        raise ValueError("knowledge_wiki resolution does not match the arm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.yaml")
    parser.add_argument("--out", type=Path, default=HERE / "configs")
    parser.add_argument("--aka", type=Path, help="AgentKernelArena checkout used to validate every config")
    parser.add_argument("--wiki-root", type=Path, help="knowledge base checkout for validating wiki arms")
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    written = 0
    for phase_name, phase in manifest["phases"].items():
        if "tasks" not in phase:
            continue
        for task in phase["tasks"] + phase.get("optional_tasks", []):
            if args.aka and not (args.aka / "tasks" / task / "config.yaml").is_file():
                raise SystemExit(f"task not found in {args.aka}: {task}")
            for arm in manifest["arms"]:
                config = run_config(manifest, phase, task, arm)
                if args.aka:
                    validate(config, args.aka, args.wiki_root)
                path = args.out / phase_name / task_slug(task) / f"{arm}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                written += 1
    print(f"wrote {written} configs under {args.out}" + (" (validated)" if args.aka else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
