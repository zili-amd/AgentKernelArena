#!/usr/bin/env python3
"""Read-only post-run audit for the kernel wiki A/B protocol (rules 1-4).

Exit status: 0 = OK, 1 = at least one FAIL (run is invalid), 2 = usage error,
3 = UNVERIFIED (no FAIL, but a WARN means some rule could not be checked).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ARMS = ("kf", "kf_wiki", "kf_wiki_branch", "geak", "geak_wiki")
FORGE_ARMS = {"kf", "kf_wiki", "kf_wiki_branch"}
GEAK_ARMS = {"geak", "geak_wiki"}
WIKI_ARMS = {"kf_wiki", "kf_wiki_branch", "geak_wiki"}

TEXT_SUFFIXES = {".log", ".jsonl", ".txt", ".out", ".err", ".json"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", "site-packages", ".aka-pyuserbase"}
MAX_BYTES = 64 * 1024 * 1024
MAX_PRINTED = 20

WIKI_RE = re.compile(r"ROCmKernelWiki|rocm[-_]kernel[-_]wiki|InstinctWiki|infinity[-_]wiki", re.IGNORECASE)
WARM_RE = re.compile(r"\[kb\] warm-start applied|injecting top solutions as reference")
DRIVER_REPLACED_RE = re.compile(r"\[prepare\] prepared measurement driver")
DRIVER_KEPT_RE = re.compile(r"\[prepare\] task already conforms")
NET_CMD_RE = re.compile(
    r"\b(curl|wget|git\s+clone|git\s+fetch|git\s+pull|pip3?\s+download|huggingface-cli\s+download)\b"
)
URL_RE = re.compile(r"https?://([A-Za-z0-9.-]+)")
MODEL_KEYS = ("requested_model", "init_model", "assistant_models", "workflow_models")
DEFAULT_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}
WIKI_LOG_NAME = "knowledge_wiki_calls.jsonl"


class Report:
    def __init__(self) -> None:
        self.items: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def add(self, level: str, check: str, detail: str) -> None:
        self.items[level].append((check, detail))

    def count(self, level: str) -> int:
        return len(self.items[level])

    def print(self) -> None:
        for level in ("FAIL", "REVIEW", "WARN", "INFO"):
            entries = self.items[level]
            for check, detail in entries[:MAX_PRINTED]:
                print(f"{level:6} [{check}] {detail}")
            if len(entries) > MAX_PRINTED:
                print(f"{level:6} ... {len(entries) - MAX_PRINTED} more {level} item(s)")


def iter_text_files(root: Path):
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES and path.stat().st_size <= MAX_BYTES:
            yield path


def walk_json(node, found: dict[str, list]) -> None:
    if isinstance(node, dict):
        if "instructions_sha256" in node and "commit" in node:
            found["wiki_identity"].append(node)
        for key, value in node.items():
            if key == "kb_experience" and isinstance(value, dict):
                found["kb_experience"].append(value)
            if key == "experience_kb" and isinstance(value, dict) and "mode" in value:
                found["experience_kb"].append(value)
            if key in MODEL_KEYS:
                found["models"].append((key, value))
            walk_json(value, found)
    elif isinstance(node, list):
        for item in node:
            walk_json(item, found)


def flatten_models(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        return [str(key) for key in value]
    if isinstance(value, list):
        return [model for item in value for model in flatten_models(item)]
    return []


def kb_hit(record: dict) -> bool:
    read = record.get("read") if isinstance(record.get("read"), dict) else {}
    return bool(read.get("applied") or read.get("candidate") or read.get("read_reason") == "hit")


def snippet(line: str, match: re.Match) -> str:
    start = max(0, match.start() - 60)
    return line[start:match.end() + 80].strip()


def audit(args: argparse.Namespace) -> Report:
    report = Report()
    allowed_hosts = DEFAULT_ALLOWED_HOSTS | {host.lower() for host in args.allow_host}
    other_roots = [str(Path(root).resolve()) for root in args.other_root]
    kb_records: list[tuple[Path, dict]] = []
    isolation_records: list[tuple[Path, dict]] = []
    wiki_identities: list[tuple[Path, dict]] = []
    wiki_logs: list[Path] = []
    model_records: list[tuple[Path, str, str]] = []
    url_hosts: dict[str, list[str]] = defaultdict(list)
    wiki_mentions = 0
    drivers_kept = drivers_replaced = 0
    scanned = 0

    for run_dir in args.run_dir:
        root = Path(run_dir)
        if not root.is_dir():
            report.add("FAIL", "input", f"run directory not found: {root}")
            continue
        for path in iter_text_files(root):
            scanned += 1
            if path.name == WIKI_LOG_NAME:
                wiki_logs.append(path)
            try:
                text = path.read_text(errors="replace")
            except OSError as error:
                report.add("WARN", "input", f"unreadable {path}: {error}")
                continue
            if path.suffix == ".json":
                try:
                    found: dict[str, list] = defaultdict(list)
                    walk_json(json.loads(text), found)
                    kb_records += [(path, record) for record in found["kb_experience"]]
                    isolation_records += [(path, record) for record in found["experience_kb"]]
                    wiki_identities += [(path, record) for record in found["wiki_identity"]]
                    model_records += [
                        (path, key, model) for key, value in found["models"] for model in flatten_models(value)
                    ]
                except ValueError:
                    pass
            for lineno, line in enumerate(text.splitlines(), 1):
                where = f"{path}:{lineno}"
                if args.arm in FORGE_ARMS:
                    for match in WARM_RE.finditer(line):
                        report.add("FAIL", "rule1-kb", f"{where}: {snippet(line, match)}")
                    if path.name == "engine.log":
                        if DRIVER_REPLACED_RE.search(line):
                            drivers_replaced += 1
                            report.add("FAIL", "measure",
                                       f"Forge replaced the Arena measurement driver {where}")
                        drivers_kept += bool(DRIVER_KEPT_RE.search(line))
                for match in WIKI_RE.finditer(line):
                    wiki_mentions += 1
                    if args.arm not in WIKI_ARMS:
                        report.add("FAIL", "rule3-wiki", f"wiki reference in no-wiki arm {where}: {snippet(line, match)}")
                for match in NET_CMD_RE.finditer(line):
                    report.add("REVIEW", "rule4-net", f"{where}: {snippet(line, match)}")
                for match in URL_RE.finditer(line):
                    host = match.group(1).lower().rstrip(".")
                    if host not in allowed_hosts:
                        url_hosts[host].append(where)
                for other in other_roots:
                    if other in line:
                        report.add("FAIL", "rule4-fs", f"path into another arm's checkout {where}: {other}")

    if scanned == 0:
        report.add("WARN", "input", "no log/transcript/JSON files found; nothing was audited")

    if args.arm in FORGE_ARMS:
        if drivers_kept:
            report.add("INFO", "measure", f"{drivers_kept} engine log(s) kept the Arena measurement driver")
        elif not drivers_replaced:
            report.add("WARN", "measure", "no engine.log preflight verdict found; driver ownership is unverified")
        if not kb_records:
            report.add("WARN", "rule1-kb", "no JSON with kb_experience found; rule 1 is unverified")
        for path, record in kb_records:
            if kb_hit(record):
                read = record.get("read") or {}
                report.add(
                    "FAIL", "rule1-kb",
                    f"{path}: KB read hit (read_reason={read.get('read_reason')!r}, "
                    f"candidate={read.get('candidate')}, applied={read.get('applied')}, "
                    f"solution={read.get('solution_slug')!r})",
                )
        if kb_records:
            clean = sum(1 for _, record in kb_records if not kb_hit(record))
            report.add("INFO", "rule1-kb", f"{clean}/{len(kb_records)} kb_experience record(s) show no KB read")
        if not isolation_records:
            report.add("WARN", "rule1-kb", "no adapter experience_kb status found; is AKA on the wiki-ab-infra branch?")
        for path, record in isolation_records:
            if record.get("mode") != "isolated":
                report.add("FAIL", "rule1-kb", f"{path}: experience_kb mode {record.get('mode')!r}, expected 'isolated'")

    if args.arm in GEAK_ARMS and not model_records:
        report.add("WARN", "rule2-model", "no GEAK runtime model fields found; rule 2 is unverified")
    models = sorted({model for _, _, model in model_records})
    if models:
        report.add("INFO", "rule2-model", f"models seen: {', '.join(models)}")
    if args.expect_model:
        for path, key, model in model_records:
            if not model.startswith(args.expect_model):
                report.add("FAIL", "rule2-model", f"{path}: {key}={model!r}, expected {args.expect_model!r}")

    for host, locations in sorted(url_hosts.items(), key=lambda item: -len(item[1])):
        report.add("REVIEW", "rule4-url", f"{host}: {len(locations)} hit(s), first at {locations[0]}")

    versions = {(record.get("commit"), record.get("dirty"), record.get("instructions_sha256"))
                for _, record in wiki_identities}
    if args.arm in WIKI_ARMS:
        report.add("INFO", "rule3-wiki", f"{wiki_mentions} wiki reference(s) in run artifacts")
        if not wiki_identities:
            report.add("FAIL", "rule3-wiki", "no knowledge_wiki identity in adapter status; the wiki was not configured")
        if len(versions) > 1:
            report.add("FAIL", "rule3-wiki", f"inconsistent wiki versions across runs: {sorted(map(str, versions))}")
        elif versions:
            commit, dirty, sha = next(iter(versions))
            report.add("WARN" if dirty else "INFO", "rule3-wiki",
                       f"wiki commit {commit} dirty={dirty} instructions_sha256={str(sha)[:12]}")
    elif wiki_identities:
        report.add("FAIL", "rule3-wiki", f"no-wiki arm configured the wiki: {wiki_identities[0][0]}")
    if args.wiki_log is not None:
        wiki_logs.append(Path(args.wiki_log))
    calls = sum(sum(1 for line in log.read_text().splitlines() if line.strip())
                for log in wiki_logs if log.is_file())
    if args.arm in WIKI_ARMS:
        report.add("INFO" if calls else "REVIEW", "rule3-wiki",
                   f"{calls} logged wiki call(s) across {len(wiki_logs)} log(s)")
    elif calls:
        report.add("FAIL", "rule3-wiki", f"no-wiki arm has {calls} logged wiki call(s): {wiki_logs[0]}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="+", help="AKA run directory (or forge/GEAK artifact directory) to audit")
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--expect-model", default=None, help="model id every GEAK runtime field must start with")
    parser.add_argument("--other-root", action="append", default=[], help="another arm's AKA checkout (repeatable)")
    parser.add_argument("--allow-host", action="append", default=[], help="URL host not to flag, e.g. the LLM gateway")
    parser.add_argument("--wiki-log", default=None,
                        help=f"extra wiki call log; {WIKI_LOG_NAME} files under the run are found automatically")
    args = parser.parse_args()

    report = audit(args)
    report.print()
    fails, reviews, warns = report.count("FAIL"), report.count("REVIEW"), report.count("WARN")
    verdict, status = ("INVALID", 1) if fails else ("UNVERIFIED", 3) if warns else ("OK", 0)
    print(f"RESULT: {verdict} (arm={args.arm}; FAIL={fails} REVIEW={reviews} WARN={warns})")
    return status


if __name__ == "__main__":
    sys.exit(main())
