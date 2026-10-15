#!/usr/bin/env python3
"""Collect one phase's runs from the ledger and compare the paired arms.

    collect_results.py ROOT [--phase main] [--control kf] [--treatment kf_wiki] [--json OUT]

Per run: AKA's speedup_ratio (failed, incorrect or unscoreable counts as 1.0, as
pre-registered), pass, cost and knowledge base calls by role, and from the run's
forge events how fast its incumbent climbed: the hours to the phase's speedup
target and the time-averaged best speedup over the phase budget. Per pair (task,
repeat): log(treatment / control) of both. Then the exact two-sided Wilcoxon
signed-rank p-value and a bootstrap 95% CI of the median paired ratio. These are
in-run timings; headline numbers come from the quiet re-measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import random
import statistics

import yaml

HERE = Path(__file__).resolve().parent


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


def _utc(text: str | None) -> datetime | None:
    """The ledger's ``...Z`` time or a forge event's ``ts``, both UTC."""
    for layout in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(text or ""), layout).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def trajectory(directory: Path, *, started: str | None, target: float | None, horizon_sec: float | None) -> dict:
    """How fast a run's incumbent climbed, read off its forge events (forge's own in-run measurements).

    Time runs from the ledger's start of the run, so setup and initialization count. hours_to_target is when the
    incumbent first reached ``target``; time_avg_speedup is the incumbent's best speedup averaged over
    ``horizon_sec`` (the phase budget), holding its last value after the run ends, so a run stopped early at its
    target is not penalized for the time it did not use. guided_hours and guided_exit describe a --wiki-guided run.
    """
    stamped = []
    for path in directory.glob("*-forge-*/engine/forge_experiments/events.jsonl"):
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line) if line.strip() else None
            except json.JSONDecodeError:
                continue
            when = _utc(event.get("ts")) if isinstance(event, dict) else None
            if when is not None:
                stamped.append((when, event))
    if not stamped:
        return {}
    stamped.sort(key=lambda pair: pair[0])
    origin = _utc(started) or stamped[0][0]
    elapsed = max(0.0, (stamped[-1][0] - origin).total_seconds())
    horizon = float(horizon_sec) if horizon_sec else max(elapsed, 1.0)
    # The incumbent before the first measured iteration is the baseline.
    best, area, clock, hours_to_target = 1.0, 0.0, 0.0, None
    for when, event in stamped:
        value = event.get("best_after_mean_case_speedup")
        if event.get("type") != "iteration_result" or type(value) not in (int, float):
            continue
        seconds = min(max(0.0, (when - origin).total_seconds()), horizon)
        area += best * (seconds - clock)
        clock, best = seconds, max(best, float(value))
        if hours_to_target is None and target and best >= target:
            hours_to_target = seconds / 3600
    area += best * (horizon - clock)
    result = {"hours_to_target": hours_to_target, "time_avg_speedup": area / horizon, "hours": elapsed / 3600}
    if any(event.get("guidance_phase") or str(event.get("type", "")).startswith("wiki_") for _, event in stamped):
        free = next(((when, event) for when, event in stamped
                     if event.get("type") == "wiki_guidance_phase" and event.get("phase") == "FREE"), None)
        result["guided_hours"] = ((free[0] - origin).total_seconds() if free else elapsed) / 3600
        result["guided_exit"] = ",".join(free[1].get("reason_codes") or []) if free else "still GUIDED"
    return result


def measure(root: Path, run: dict, *, target: float | None = None, horizon_sec: float | None = None) -> dict:
    record = {key: run.get(key) for key in ("task", "repeat", "arm", "gpu", "run_suffix", "exit_code",
                                            "started", "time")}
    directory = run_directory(root, run)
    record.update(run_dir=str(directory) if directory else None, passed=False, speedup=1.0,
                  raw_speedup=None, cost_usd=None, wiki_calls=0, wiki_calls_by_role={},
                  hours_to_target=None, time_avg_speedup=None, guided_hours=None, guided_exit="")
    if directory is None:
        return record
    record.update(trajectory(directory, started=run.get("started"), target=target, horizon_sec=horizon_sec))
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
        reached = [r["hours_to_target"] for r in mine if r.get("hours_to_target") is not None]
        averaged = [r["time_avg_speedup"] for r in mine if r.get("time_avg_speedup") is not None]
        arms[arm] = {"runs": len(mine), "pass_rate": sum(r["passed"] for r in mine) / len(mine),
                     "arithmetic_mean": statistics.fmean(speedups),
                     "geometric_mean": math.exp(statistics.fmean(math.log(s) for s in speedups)),
                     "cost_usd": sum(r["cost_usd"] or 0 for r in mine),
                     "wiki_calls": sum(r["wiki_calls"] for r in mine),
                     "reached_target": len(reached),
                     "median_hours_to_target": statistics.median(reached) if reached else None,
                     "mean_time_avg_speedup": statistics.fmean(averaged) if averaged else None}
    by_pair = {(r["task"], r["repeat"], r["arm"]): r for r in records}
    pairs = []
    for (task, repeat, arm), record in sorted(by_pair.items()):
        other = by_pair.get((task, repeat, control))
        if arm == treatment and other is not None:
            pair = {"task": task, "repeat": repeat,
                    "log_ratio": math.log(record["speedup"] / other["speedup"]),
                    "ratio": record["speedup"] / other["speedup"],
                    "hours_to_target": (other.get("hours_to_target"), record.get("hours_to_target")),
                    "time_avg_ratio": None}
            if record.get("time_avg_speedup") and other.get("time_avg_speedup"):
                pair["time_avg_ratio"] = record["time_avg_speedup"] / other["time_avg_speedup"]
            pairs.append(pair)
    logs = [p["log_ratio"] for p in pairs]
    ci = bootstrap_median_ci([p["ratio"] for p in pairs])
    averaged = [p["time_avg_ratio"] for p in pairs if p["time_avg_ratio"] is not None]
    return {"arms": arms, "pairs": pairs,
            "median_ratio": statistics.median(p["ratio"] for p in pairs) if pairs else None,
            "median_ratio_ci95": ci, "wilcoxon_p": wilcoxon_exact(logs),
            "median_time_avg_ratio": statistics.median(averaged) if averaged else None,
            "time_avg_ratio_ci95": bootstrap_median_ci(averaged),
            "time_avg_wilcoxon_p": wilcoxon_exact([math.log(value) for value in averaged])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--phase", default="main")
    parser.add_argument("--control", default="kf")
    parser.add_argument("--treatment", default="kf_wiki")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    spec = yaml.safe_load((HERE / "manifest.yaml").read_text())["phases"].get(args.phase, {})
    targets = spec.get("speedup_targets") or {}
    records = sorted((measure(args.root, run, target=targets.get(run["task"]), horizon_sec=spec.get("timeout_seconds"))
                      for run in runs(args.root, args.phase)),
                     key=lambda r: (r["task"], r["repeat"], r["arm"]))
    for r in records:
        reached = f"{r['hours_to_target']:.2f}h" if r["hours_to_target"] is not None else "-"
        averaged = f"{r['time_avg_speedup']:.4f}" if r["time_avg_speedup"] is not None else "-"
        print(f"{r['task']:<55} r{r['repeat']} {r['arm']:<8} exit={r['exit_code']} pass={int(r['passed'])} "
              f"speedup={r['speedup']:.4f} to_target={reached} time_avg={averaged} cost=${r['cost_usd'] or 0:.2f} "
              f"wiki_calls={r['wiki_calls']} {r['wiki_calls_by_role'] or ''}")
    summary = summarize(records, args.control, args.treatment) if records else {}
    for arm, stats in summary.get("arms", {}).items():
        print(f"{arm}: runs={stats['runs']} pass_rate={stats['pass_rate']:.2f} mean={stats['arithmetic_mean']:.4f} "
              f"geomean={stats['geometric_mean']:.4f} cost=${stats['cost_usd']:.2f} wiki_calls={stats['wiki_calls']}")
    if summary.get("pairs"):
        print(f"{len(summary['pairs'])} pairs: median {args.treatment}/{args.control} = {summary['median_ratio']:.4f}, "
              f"bootstrap 95% CI {summary['median_ratio_ci95']}, Wilcoxon p = {summary['wilcoxon_p']}")
        if summary["median_time_avg_ratio"] is not None:
            print(f"time-averaged best speedup: median {args.treatment}/{args.control} = "
                  f"{summary['median_time_avg_ratio']:.4f}, bootstrap 95% CI {summary['time_avg_ratio_ci95']}, "
                  f"Wilcoxon p = {summary['time_avg_wilcoxon_p']}")
    if args.json:
        args.json.write_text(json.dumps({"runs": records, **summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
