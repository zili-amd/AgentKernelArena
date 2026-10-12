#!/usr/bin/env python3
"""Collect one phase's runs from the ledger and compare the paired arms.

    collect_results.py ROOT [--phase main] [--control kf] [--treatment kf_wiki] [--json OUT]

Per run: AKA's speedup_ratio (failed, incorrect or unscoreable counts as 1.0, as
pre-registered), pass, cost and knowledge base calls by role. Per pair (task,
repeat): log(treatment / control). Then the exact two-sided Wilcoxon signed-rank
p-value and a bootstrap 95% CI of the median paired ratio. These are in-run
timings; headline numbers come from the quiet re-measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
import json
import math
from pathlib import Path
import random
import statistics

import yaml


def runs(root: Path, phase: str) -> list[dict]:
    started: dict[str, dict] = {}
    finished = []
    for line in (root / "invocations.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("phase") != phase:
            continue
        if event["event"] == "start":
            started[event["run_suffix"]] = event
        elif event["event"] == "end":
            finished.append({**started.get(event["run_suffix"], {}), **event, "started": started.get(
                event["run_suffix"], {}).get("time")})
    # A rerun after an exclusion reuses the suffix; its last end wins.
    return list({run["run_suffix"]: run for run in finished}.values())


def run_directory(root: Path, run: dict) -> Path | None:
    found = sorted(root.glob(f"aka-{run['arm']}/workspace_*/run_*_{run['run_suffix']}"))
    return found[-1] if found else None


def measure(root: Path, run: dict) -> dict:
    record = {key: run.get(key) for key in ("task", "repeat", "arm", "gpu", "run_suffix", "exit_code",
                                            "started", "time")}
    directory = run_directory(root, run)
    record.update(run_dir=str(directory) if directory else None, passed=False, speedup=1.0,
                  raw_speedup=None, cost_usd=None, wiki_calls=0, wiki_calls_by_role={})
    if directory is None:
        return record
    results = [p for p in directory.glob("*/task_result.yaml") if "-forge-" not in p.parent.name]
    if results:
        result = yaml.safe_load(results[0].read_text()) or {}
        raw = result.get("speedup_ratio")
        record["raw_speedup"] = raw
        record["passed"] = bool(result.get("pass_compilation") and result.get("pass_correctness")
                                and isinstance(raw, (int, float)) and raw > 0)
        record["speedup"] = float(raw) if record["passed"] else 1.0
    for usage in directory.glob("*-forge-*/engine/forge_experiments/llm_usage.json"):
        record["cost_usd"] = (record["cost_usd"] or 0.0) + float(json.loads(usage.read_text()).get("total_cost_usd") or 0)
    roles: Counter = Counter()
    for log in directory.glob("*-forge-*/knowledge_wiki_calls.jsonl"):
        for line in log.read_text().splitlines():
            if line.strip():
                roles[json.loads(line).get("role") or "shell"] += 1
    record["wiki_calls"] = sum(roles.values())
    record["wiki_calls_by_role"] = dict(roles)
    return record


def wilcoxon_exact(differences: list[float]) -> float | None:
    """Two-sided exact p-value of the signed-rank statistic; zeros dropped, ties mid-ranked."""
    values = [d for d in differences if d != 0]
    n = len(values)
    if n == 0 or n > 20:
        return None
    order = sorted(range(n), key=lambda i: abs(values[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(values[order[j + 1]]) == abs(values[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    observed = sum(r for r, v in zip(ranks, values) if v > 0)
    expected = sum(ranks) / 2
    extreme = sum(1 for signs in product((0, 1), repeat=n)
                  if abs(sum(r for r, s in zip(ranks, signs) if s) - expected) >= abs(observed - expected) - 1e-12)
    return extreme / 2 ** n


def bootstrap_median_ci(values: list[float], rounds: int = 10_000, seed: int = 0) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(rounds))
    return medians[int(0.025 * rounds)], medians[int(0.975 * rounds) - 1]


def summarize(records: list[dict], control: str, treatment: str) -> dict:
    arms = {}
    for arm in sorted({r["arm"] for r in records}):
        mine = [r for r in records if r["arm"] == arm]
        speedups = [r["speedup"] for r in mine]
        arms[arm] = {"runs": len(mine), "pass_rate": sum(r["passed"] for r in mine) / len(mine),
                     "arithmetic_mean": statistics.fmean(speedups),
                     "geometric_mean": math.exp(statistics.fmean(math.log(s) for s in speedups)),
                     "cost_usd": sum(r["cost_usd"] or 0 for r in mine),
                     "wiki_calls": sum(r["wiki_calls"] for r in mine)}
    by_pair = {(r["task"], r["repeat"], r["arm"]): r for r in records}
    pairs = []
    for (task, repeat, arm), record in sorted(by_pair.items()):
        other = by_pair.get((task, repeat, control))
        if arm == treatment and other is not None:
            pairs.append({"task": task, "repeat": repeat,
                          "log_ratio": math.log(record["speedup"] / other["speedup"]),
                          "ratio": record["speedup"] / other["speedup"]})
    logs = [p["log_ratio"] for p in pairs]
    ci = bootstrap_median_ci([p["ratio"] for p in pairs])
    return {"arms": arms, "pairs": pairs,
            "median_ratio": statistics.median(p["ratio"] for p in pairs) if pairs else None,
            "median_ratio_ci95": ci, "wilcoxon_p": wilcoxon_exact(logs)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--phase", default="main")
    parser.add_argument("--control", default="kf")
    parser.add_argument("--treatment", default="kf_wiki")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    records = sorted((measure(args.root, run) for run in runs(args.root, args.phase)),
                     key=lambda r: (r["task"], r["repeat"], r["arm"]))
    for r in records:
        print(f"{r['task']:<55} r{r['repeat']} {r['arm']:<8} exit={r['exit_code']} pass={int(r['passed'])} "
              f"speedup={r['speedup']:.4f} cost=${r['cost_usd'] or 0:.2f} wiki_calls={r['wiki_calls']} "
              f"{r['wiki_calls_by_role'] or ''}")
    summary = summarize(records, args.control, args.treatment) if records else {}
    for arm, stats in summary.get("arms", {}).items():
        print(f"{arm}: runs={stats['runs']} pass_rate={stats['pass_rate']:.2f} mean={stats['arithmetic_mean']:.4f} "
              f"geomean={stats['geometric_mean']:.4f} cost=${stats['cost_usd']:.2f} wiki_calls={stats['wiki_calls']}")
    if summary.get("pairs"):
        print(f"{len(summary['pairs'])} pairs: median {args.treatment}/{args.control} = {summary['median_ratio']:.4f}, "
              f"bootstrap 95% CI {summary['median_ratio_ci95']}, Wilcoxon p = {summary['wilcoxon_p']}")
    if args.json:
        args.json.write_text(json.dumps({"runs": records, **summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
