#!/usr/bin/env python3
"""Measure screening candidates' baseline case times with AKA's own task harness (no agent).

Run inside an AKA container from the checkout root, e.g. piped into
`AKA_VISIBLE_GPU=0 make docker-shell`. Each candidate is materialized exactly as
a run would be, validated with the initial-task checks, then its frozen baseline
performance action runs --rounds times. Output feeds roofline.py --baselines.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import yaml


def screen(task: str, aka: Path, run_directory: Path, rounds: int, logger: logging.Logger) -> dict:
    from src.preprocessing import setup_workspace
    from src.task_materialization import verify_original_materialization
    from src.task_run import _state_directory
    from src.task_runtime import bind_session_runtime
    from src.task_session import TaskSession
    from src.task_spec import load_task_spec

    # AKA's `task_config_dir` arguments are the task's config.yaml path.
    config_path = aka / "tasks" / task / "config.yaml"
    spec = load_task_spec(config_path, task_id=task)
    workspace = setup_workspace(str(config_path), run_directory, time.strftime("%Y%m%d_%H%M%S"), logger, task_name=task)
    verify_original_materialization(workspace)
    session = TaskSession.create(spec, workspace, _state_directory(workspace), logger)
    runtime = bind_session_runtime(session)
    initial = session.validate_initial()
    record = {"workspace": str(workspace), "runtime": runtime, "initial_accepted": initial.accepted,
              "initial_errors": list(initial.errors)}
    if not initial.accepted:
        return record
    by_case: dict[str, list[float]] = {}
    for _ in range(rounds):
        result = session._execute("baseline", "performance", "task_validation").result
        for row in result.cases:
            by_case.setdefault(row["test_case_id"], []).append(float(row["execution_time_ms"]))
    record["rounds"] = rounds
    record["median_ms_by_case"] = {cid: statistics.median(times) for cid, times in by_case.items()}
    record["max_relative_spread"] = max((max(t) - min(t)) / statistics.median(t) for t in by_case.values())
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True, help="scratch run directory for workspaces")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--aka", type=Path, default=Path.cwd())
    parser.add_argument("--only", action="append", default=[], help="screen only these tasks (repeatable)")
    args = parser.parse_args()
    sys.path.insert(0, str(args.aka))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("screen")
    candidates = yaml.safe_load(args.manifest.read_text())["phases"]["screening"]["candidates"]
    results = json.loads(args.out.read_text()) if args.out.is_file() else {}
    for task in args.only or candidates:
        args.run_directory.mkdir(parents=True, exist_ok=True)
        try:
            results[task] = screen(task, args.aka, args.run_directory, args.rounds, logger)
        except Exception as exc:  # one candidate's failure must not stop the screen
            results[task] = {"error": f"{type(exc).__name__}: {exc}"}
        args.out.write_text(json.dumps(results, indent=2, sort_keys=True))
        status = results[task].get("error") or ("accepted" if results[task].get("initial_accepted") else
                                                 "; ".join(results[task].get("initial_errors", [])))
        logger.info("%s: %s", task, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
