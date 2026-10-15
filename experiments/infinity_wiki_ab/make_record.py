#!/usr/bin/env python3
"""Write one phase's test record from the ledger, the run artifacts and TEST_RECORD_TEMPLATE.md.

    make_record.py ROOT [--phase main] [--out ROOT/results/<phase>-record.md]

launch_wave.sh runs it after every pair, so the record follows the runs. The
agent's section, between the agent-notes markers, is carried over from the
record already at --out.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
import itertools
import json
import os
from pathlib import Path
import re
import statistics
from string import Template
import sys
import tempfile
import time
from urllib.parse import urlparse

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_run  # noqa: E402
import collect_results  # noqa: E402
import roofline  # noqa: E402

NOTES = re.compile(r"<!-- agent-notes:start -->.*?<!-- agent-notes:end -->", re.DOTALL)
DEFAULT_NOTES = """<!-- agent-notes:start -->
Fill in after reviewing the runs; this section is kept when the record is regenerated.

- Per task: the optimization directions each arm kept, and where it consulted the
  knowledge base (roles and pages in the call log).
- Anomalies, exclusions and reruns, with the audit verdicts behind them.
- What this phase shows, within what the pre-registration allows.
<!-- agent-notes:end -->"""


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def task_result_dir(run_dir: Path) -> Path | None:
    found = [p.parent for p in run_dir.glob("*/task_result.yaml") if "-forge-" not in p.parent.name]
    return found[0] if found else None


def perf_cases(run_dir: Path | None, name: str) -> dict[str, dict]:
    directory = task_result_dir(run_dir) if run_dir else None
    path = directory / name if directory else None
    if not path or not path.is_file():
        return {}
    rows = (yaml.safe_load(path.read_text()) or {}).get("test_cases") or []
    return {row["test_case_id"]: row for row in rows if isinstance(row.get("execution_time_ms"), (int, float))}


def bounds(task: str, cases: dict[str, dict]) -> dict[str, float]:
    model = roofline.MODELS.get(task)
    return {cid: model(row.get("params") or {}) for cid, row in cases.items()} if model else {}


def audit_verdict(root: Path, run: dict, arms: list[str], model: str, host: str) -> str:
    run_dir = collect_results.run_directory(root, run)
    if run_dir is None:
        return "no run directory"
    home = root / f"aka-{run['arm']}" / "wiki_ab_results" / f"home-{run['run_suffix']}"
    report = audit_run.audit(Namespace(
        run_dir=[str(run_dir)] + ([str(home)] if home.is_dir() else []), arm=run["arm"], expect_model=model,
        other_root=[str(root / f"aka-{arm}") for arm in arms if arm != run["arm"]],
        allow_host=[host] if host else [], wiki_log=None))
    fails, reviews, warns = (report.count(level) for level in ("FAIL", "REVIEW", "WARN"))
    verdict = "INVALID" if fails else "UNVERIFIED" if warns else "OK"
    return f"{verdict} ({fails} fail, {reviews} review)"


class Record:
    def __init__(self, root: Path, phase: str) -> None:
        self.root, self.phase = root, phase
        self.manifest = yaml.safe_load((HERE / "manifest.yaml").read_text())
        self.spec = self.manifest["phases"][phase]
        arms_file = root / "dist" / "arms.txt"
        self.arms = arms_file.read_text().split() if arms_file.is_file() else list(self.manifest["arms"])
        # Only this experiment's arms: the ledger also holds runs of arms an earlier pack ran.
        self.runs = [run for run in (collect_results.runs(root, phase) if (root / "invocations.jsonl").is_file() else [])
                     if run["arm"] in self.arms]
        self.records = sorted((collect_results.measure(root, run) for run in self.runs),
                              key=lambda r: (r["task"], r["repeat"], r["arm"]))
        self.tasks = list(self.spec.get("tasks") or sorted({r["task"] for r in self.records}))
        # Each task's starting point, from the first run that measured it.
        self.baselines: dict[str, dict[str, dict]] = {}
        for record in self.records:
            if record["task"] not in self.baselines and record["run_dir"]:
                cases = perf_cases(Path(record["run_dir"]), "baseline_perf.yaml")
                if cases:
                    self.baselines[record["task"]] = cases

    def wiki_pin(self, arm: str) -> str:
        value = self.manifest["arms"][arm]["knowledge_wiki"]
        return value if isinstance(value, str) else "wiki" if value else ""

    def what(self) -> dict[str, str]:
        hardware = self.manifest.get("hardware", {})
        rows = []
        for arm in self.arms:
            pin = self.wiki_pin(arm)
            spec = self.manifest["pins"].get(pin, {}) if pin else {}
            rows.append([f"`{arm}`", self.manifest["arms"][arm]["template"],
                         f"`{pin}`: {spec.get('branch', '?')} @ `{str(spec.get('commit', ''))[:12]}`" if pin else "none",
                         f"`knowledge_wiki: {{enabled: true}}`, `src/infinity-{pin.replace('_', '-')}`" if pin else "none"])
        pins_path = self.root / "dist" / "pins.txt"
        pins = [line.split() for line in pins_path.read_text().splitlines()] if pins_path.is_file() else []
        pin_rows = [[name, f"`{commit}`"] for name, commit in pins]
        pin_rows += [["model", f"`{self.manifest['pins']['model']}`"],
                     ["runtime image", f"`{hardware.get('docker_image')}`"]]
        runtime = self.runtime_identity()
        if runtime:
            pin_rows.append(["runtime", ", ".join(f"{k} {v}" for k, v in runtime.items() if v)])
        return {
            "question": self.manifest.get("question", ""),
            "arms": table(["Arm", "System", "Knowledge base", "Config difference"], rows),
            "held_equal": (f"the {len(self.tasks)} tasks below, each task's frozen initial implementation and an "
                           f"empty per-run Experience KB, {self.spec['timeout_seconds'] / 3600:g} h of wall clock "
                           f"per run, {self.spec.get('repeats', 1)} repeat(s), the model and the pinned code "
                           "below, and paired launches on adjacent GPUs that swap between repeats."),
            "pins": table(["Pinned", "Version"], pin_rows),
        }

    def runtime_identity(self) -> dict:
        for record in self.records:
            directory = task_result_dir(Path(record["run_dir"])) if record["run_dir"] else None
            if directory:
                result = yaml.safe_load((directory / "task_result.yaml").read_text()) or {}
                identity = result.get("runtime_identity") or {}
                return {key: identity.get(key) for key in ("gpu_name", "gpu_arch", "torch", "triton", "hip")}
        return {}

    def starting_point(self) -> str:
        parts = []
        for task in self.tasks:
            config_path = self.root / f"aka-{self.arms[0]}" / "tasks" / task / "config.yaml"
            candidate = (yaml.safe_load(config_path.read_text()) or {}).get("candidate", {}) if config_path.is_file() else {}
            editable = ", ".join(f"`{e['path'] if isinstance(e, dict) else e}`" for e in candidate.get("editable", []))
            cases = self.baselines.get(task, {})
            sums = [sum(row["execution_time_ms"] for row in perf_cases(Path(r["run_dir"]), "baseline_perf.yaml").values())
                    for r in self.records if r["task"] == task and r["run_dir"]]
            sums = [s for s in sums if s]
            measured = (f"{len(cases)} performance cases, Σ baseline {sums[0]:.4f} ms "
                        f"(range over {len(sums)} runs: {min(sums):.4f} to {max(sums):.4f} ms)") if sums else "not measured yet"
            parts.append(f"- **{task}**: {candidate.get('language', '?')} candidate, initial state "
                         f"`{candidate.get('initial_state', '?')}`, editable {editable or '?'}. Baseline: the task's "
                         f"frozen initial implementation; {measured}.")
        return "\n".join(parts)

    def roofline_and_ceiling(self) -> tuple[str, str]:
        peaks = ", ".join(f"{dtype.upper()} {flops / 1e12:.1f}" for dtype, flops in roofline.PEAK.items())
        parts = [f"Each case's bound is max(FLOPs / peak, bytes / HBM bandwidth) on MI300X ({peaks} TFLOP/s dense, "
                 f"HBM {roofline.BW / 1e12:.1f} TB/s); its effective bound adds the "
                 f"{roofline.LAUNCH_FLOOR_S * 1e6:.0f} µs launch floor an optimal kernel still pays (`roofline.py`)."]
        ceiling_rows = []
        for task in self.tasks:
            cases = self.baselines.get(task, {})
            bound = bounds(task, cases)
            if not cases or not bound:
                parts.append(f"\n**{task}**: " + ("no roofline model" if task not in roofline.MODELS else "no baseline yet"))
                continue
            effective = {cid: max(b, roofline.LAUNCH_FLOOR_S) for cid, b in bound.items()}
            rows = []
            for cid, row in cases.items():
                ms = row["execution_time_ms"]
                params = json.dumps((row.get("params") or {}).get("configuration") or (row.get("params") or {}).get(
                    "arguments") or row.get("params"), separators=(",", ":"))
                rows.append([cid, f"`{params[:70]}`", f"{ms * 1e3:.2f}", f"{bound[cid] * 1e6:.2f}",
                             f"{effective[cid] * 1e6:.2f}", f"{ms / 1e3 / effective[cid]:.1f}x"])
            floor_share = sum(b < roofline.LAUNCH_FLOOR_S for b in bound.values()) / len(bound)
            parts.append(f"\n**{task}**: {len(bound)} cases, median roofline {statistics.median(bound.values()) * 1e6:.2f} µs, "
                         f"{floor_share:.0%} at the launch floor.\n\n<details><summary>Per case</summary>\n\n"
                         + table(["Case", "Shape", "Baseline µs", "Roofline µs", "Effective bound µs", "Baseline / bound"],
                                 rows) + "\n\n</details>")
            total_ms = sum(cases[c]["execution_time_ms"] for c in effective)
            ceiling_rows.append([task, len(effective), f"{total_ms:.4f}", f"{sum(effective.values()) * 1e3:.4f}",
                                 f"{total_ms / 1e3 / sum(effective.values()):.1f}x",
                                 f"{statistics.fmean(cases[c]['execution_time_ms'] / 1e3 / e for c, e in effective.items()):.1f}x"])
        ceiling = ("The fastest any candidate could be if every case ran at its effective bound. The time-weighted "
                   "ceiling is Σ baseline / Σ bound; the per-case mean ceiling bounds AKA's score, the mean of "
                   "per-case speedups.\n\n" + table(["Task", "Cases", "Σ baseline ms", "Σ bound ms",
                                                     "Time-weighted ceiling", "Per-case mean ceiling (AKA score)"],
                                                    ceiling_rows)) if ceiling_rows else "No baseline measured yet."
        return "\n".join(parts), ceiling

    def results(self) -> str:
        if not self.records:
            return "No finished runs yet."
        host = urlparse(os.environ.get("ANTHROPIC_BASE_URL", "")).hostname or ""
        model = self.manifest["pins"]["model"]
        by_suffix = {run["run_suffix"]: run for run in self.runs}
        rows = []
        for r in self.records:
            cases = self.baselines.get(r["task"], {})
            bound = {cid: max(b, roofline.LAUNCH_FLOOR_S) for cid, b in bounds(r["task"], cases).items()}
            optimized = perf_cases(Path(r["run_dir"]), "optimized_perf.yaml") if r["run_dir"] else {}
            ceiling = statistics.fmean(cases[c]["execution_time_ms"] / 1e3 / b for c, b in bound.items()) if bound else None
            common = [c for c in bound if c in optimized]
            roof = (f"{sum(bound[c] for c in common) / sum(optimized[c]['execution_time_ms'] / 1e3 for c in common):.1%}"
                    if common and r["passed"] else "-")
            rows.append([r["task"].split("/", 1)[-1], r["repeat"], r["arm"], r["gpu"], r["exit_code"],
                         "yes" if r["passed"] else "no", f"{r['speedup']:.4f}",
                         f"{r['speedup'] / ceiling:.1%}" if ceiling else "-", roof,
                         f"${r['cost_usd']:.2f}" if r["cost_usd"] is not None else "-",
                         f"{r['wiki_calls']} {r['wiki_calls_by_role'] or ''}".strip(),
                         audit_verdict(self.root, by_suffix[r["run_suffix"]], self.arms, model, host)])
        parts = ["Speedup is AKA's mean of per-case speedups; a failed, incorrect or unscoreable run counts as "
                 "1.0. *Share of ceiling* is speedup / the per-case mean ceiling; *roofline* is Σ bound / Σ final "
                 "time.\n", table(["Task", "Repeat", "Arm", "GPU", "Exit", "Pass", "Speedup", "Share of ceiling",
                                    "Roofline", "Cost", "KB calls", "Audit"], rows)]
        summary = collect_results.summarize(self.records, *(self.arms + ["", ""])[:2])
        parts.append("\n" + table(["Arm", "Runs", "Pass rate", "Mean speedup", "Geomean", "Cost", "KB calls"], [
            [arm, s["runs"], f"{s['pass_rate']:.0%}", f"{s['arithmetic_mean']:.4f}", f"{s['geometric_mean']:.4f}",
             f"${s['cost_usd']:.2f}", s["wiki_calls"]] for arm, s in summary["arms"].items()]))
        # Every arm against every later one: with kf first, each wiki arm against kf, then the wiki arms.
        for control, treatment in itertools.combinations(self.arms, 2):
            summary = collect_results.summarize(self.records, control, treatment)
            if not summary["pairs"]:
                continue
            ci = summary["median_ratio_ci95"]
            parts.append(f"\n**{treatment} over {control}**: {len(summary['pairs'])} pair(s), median ratio "
                         f"{summary['median_ratio']:.4f}" + (f", bootstrap 95% CI {ci[0]:.4f} to {ci[1]:.4f}" if ci else "")
                         + (f", exact Wilcoxon p = {summary['wilcoxon_p']:.4f}" if summary["wilcoxon_p"] is not None else "")
                         + ".\n\n" + table(["Task", "Repeat", "Ratio"], [[p["task"], p["repeat"], f"{p['ratio']:.4f}"]
                                                                          for p in summary["pairs"]]))
        return "\n".join(parts)

    def render(self, previous: str) -> str:
        roof, ceiling = self.roofline_and_ceiling()
        notes = NOTES.search(previous)
        values = {"experiment": self.manifest.get("experiment", ""), "phase": self.phase, "root": str(self.root),
                  "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), **self.what(),
                  "starting_point": self.starting_point(), "roofline": roof, "upper_bound": ceiling,
                  "results": self.results(), "agent_notes": notes.group(0) if notes else DEFAULT_NOTES}
        return Template((HERE / "TEST_RECORD_TEMPLATE.md").read_text()).safe_substitute(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--phase", default="main")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.root / "results" / f"{args.phase}-record.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = Record(args.root, args.phase).render(out.read_text() if out.is_file() else "")
    # Pairs finish concurrently; each writes a complete record and the last one wins.
    with tempfile.NamedTemporaryFile("w", dir=out.parent, delete=False, suffix=".tmp") as handle:
        handle.write(text)
    os.replace(handle.name, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
