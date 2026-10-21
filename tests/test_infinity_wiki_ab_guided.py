"""The guided round of the InstinctWiki A/B: its arm config and its convergence readings."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "infinity_wiki_ab"
sys.path.insert(0, str(EXPERIMENT))

import collect_results  # noqa: E402
import make_configs  # noqa: E402

MANIFEST = yaml.safe_load((EXPERIMENT / "manifest.yaml").read_text())
TASK = "torch2hip/kernelbench_own/level1/l1n95_CrossEntropyLoss"


def test_the_guided_arm_runs_its_own_engine_with_the_knowledge_base():
    arm = MANIFEST["arms"]["kf_wiki_guided"]
    assert arm["wiki_guided"] is True
    assert arm["knowledge_wiki"] == MANIFEST["arms"]["kf_wiki"]["knowledge_wiki"]
    assert "wiki-guided" in MANIFEST["pins"][arm["hyperloom"]]["branch"]


def test_only_the_guided_arm_asks_forge_to_plan_from_the_playbook():
    phase = MANIFEST["phases"]["main"]
    guided = make_configs.run_config(MANIFEST, phase, TASK, "kf_wiki_guided")
    assert guided["agent"]["wiki_guided"] is True
    assert guided["knowledge_wiki"] == {"enabled": True}
    assert guided["agent"]["target_mean_case_speedup"] == phase["speedup_targets"][TASK]
    for arm in ("kf", "kf_wiki"):
        assert "wiki_guided" not in make_configs.run_config(MANIFEST, phase, TASK, arm)["agent"]


def test_a_by_slot_phase_starts_each_task_repeat_on_its_own_gpu_repeat_by_repeat(tmp_path):
    phase = "guided3h_triton_hip"
    spec = MANIFEST["phases"][phase]
    # The phase's own schedule overrides the round's, which the other phases keep.
    assert spec["schedule"]["waves"] == "by-slot" and MANIFEST["schedule"]["waves"] == "by-task"
    prefixes = ("triton2triton/vllm_large/triton_awq_gemm", "triton2triton/rocmbench_open/hard/",
                "hip2hip/gpumode_large/SoftmaxModule")
    kept = [task for task in spec["tasks"] if task.startswith(prefixes)]
    root = tmp_path / "root"
    (root / "dist").mkdir(parents=True)
    (root / "dist" / "arms.txt").write_text("kf_wiki_guided\n")
    arm_dir = root / "aka-kf_wiki_guided" / "wiki_ab"
    for task in spec["tasks"]:
        config = arm_dir / "configs" / phase / task.replace("/", "__") / "kf_wiki_guided.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("{}\n")
    (arm_dir / "ca-certificates.crt").write_text("certificate\n")
    claude = root / "agent-cli" / "bin" / "claude"
    claude.parent.mkdir(parents=True)
    claude.write_text("#!/bin/sh\n")
    claude.chmod(0o755)
    (root / "claude-home" / ".claude").mkdir(parents=True)
    (root / "claude-home" / ".claude.json").write_text("{}\n")

    out = subprocess.run(
        ["bash", str(EXPERIMENT / "run_main.sh"), str(root), "--dry-run"],
        env={**os.environ, "PHASE": phase, "TASKS_PREFIX": " ".join(prefixes)},
        capture_output=True, text=True, check=True,
    ).stdout

    order = [(repeat, task) for repeat in range(3) for task in kept]
    assert re.findall(r"slot: r(\d) (\S+) on GPU base (\d)", out) == [
        (str(repeat), task, str(index % 8)) for index, (repeat, task) in enumerate(order)
    ]
    launched = re.findall(r"^GPU (\d) .*--run-suffix (\S+)\"?$", out, re.MULTILINE)
    assert sorted(launched) == sorted(
        (str(index % 8), f"{phase}-{task.split('/', 1)[1].replace('/', '.')}-r{repeat}-kf_wiki_guided")
        for index, (repeat, task) in enumerate(order)
    )


def test_the_audit_knows_every_manifest_arm_and_which_may_use_the_wiki():
    import audit_run

    for name, arm in MANIFEST["arms"].items():
        assert name in audit_run.ARMS
        assert (name in audit_run.WIKI_ARMS) == bool(arm.get("knowledge_wiki"))
        if arm.get("template") == "forge":
            assert name in audit_run.FORGE_ARMS


def test_the_guided_arms_wiki_calls_are_not_held_against_it(tmp_path):
    import argparse

    import audit_run

    (tmp_path / "knowledge_wiki_calls.jsonl").write_text('{"role": "playbook", "command": "page"}\n')
    (tmp_path / "engine.log").write_text("Knowledge base: /root/src/infinity-wiki\n")

    def fails(arm):
        report = audit_run.audit(argparse.Namespace(run_dir=[str(tmp_path)], arm=arm, expect_model=None,
                                                    other_root=[], allow_host=[], wiki_log=None))
        return [check for check, _ in report.items["FAIL"] if check == "rule3-wiki" and "no-wiki arm" in _]

    assert fails("kf_wiki_guided") == []
    assert fails("kf")


def test_a_guided_arm_without_a_knowledge_base_is_refused():
    manifest = {**MANIFEST, "arms": {**MANIFEST["arms"], "bad": {"template": "forge", "hyperloom": "hyperloom",
                                                                 "knowledge_wiki": False, "wiki_guided": True}}}
    with pytest.raises(ValueError, match="wiki_guided needs"):
        make_configs.run_config(manifest, MANIFEST["phases"]["main"], TASK, "bad")


def _run_dir(tmp_path, events):
    forge = tmp_path / "aka-kf_wiki_guided" / "workspace_MI300_forge" / "run_1_main-x-r0-kf_wiki_guided" / "t-forge-1"
    path = forge / "engine" / "forge_experiments" / "events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    return forge.parent


def test_the_incumbent_trajectory_gives_time_to_target_and_a_time_averaged_speedup(tmp_path):
    # Started at 00:00; best 2.0 from 01:00, 4.0 from 03:00; the run ends at 03:00 and holds 4.0 to the horizon.
    directory = _run_dir(tmp_path, [
        {"ts": "2026-10-15 00:10:00", "type": "session_started"},
        {"ts": "2026-10-15 01:00:00", "type": "iteration_result", "best_after_mean_case_speedup": 2.0},
        {"ts": "2026-10-15 02:00:00", "type": "iteration_result", "best_after_mean_case_speedup": 2.0},
        {"ts": "2026-10-15 03:00:00", "type": "iteration_result", "best_after_mean_case_speedup": 4.0},
    ])

    result = collect_results.trajectory(directory, started="2026-10-15T00:00:00Z", target=3.0, horizon_sec=4 * 3600)

    assert result["hours_to_target"] == pytest.approx(3.0)
    # (1.0 * 1 h + 2.0 * 2 h + 4.0 * 1 h) / 4 h
    assert result["time_avg_speedup"] == pytest.approx(9.0 / 4)
    assert result["hours"] == pytest.approx(3.0)
    assert "guided_hours" not in result


def test_a_run_short_of_its_target_has_no_time_to_target(tmp_path):
    directory = _run_dir(tmp_path, [
        {"ts": "2026-10-15 00:00:00", "type": "session_started"},
        {"ts": "2026-10-15 02:00:00", "type": "iteration_result", "best_after_mean_case_speedup": 2.0},
    ])
    result = collect_results.trajectory(directory, started=None, target=3.0, horizon_sec=4 * 3600)
    assert result["hours_to_target"] is None
    assert result["time_avg_speedup"] == pytest.approx((1.0 * 2 + 2.0 * 2) / 4)


def test_a_guided_run_reports_how_long_it_planned_from_the_playbook(tmp_path):
    directory = _run_dir(tmp_path, [
        {"ts": "2026-10-15 00:00:00", "type": "session_started"},
        {"ts": "2026-10-15 00:30:00", "type": "wiki_playbook_built", "items": ["P1"]},
        {"ts": "2026-10-15 01:00:00", "type": "iteration_result", "best_after_mean_case_speedup": 2.0,
         "guidance_phase": "GUIDED"},
        {"ts": "2026-10-15 02:30:00", "type": "wiki_guidance_phase", "phase": "FREE",
         "reason_codes": ["PLAYBOOK_EXHAUSTED"]},
        {"ts": "2026-10-15 03:00:00", "type": "iteration_result", "best_after_mean_case_speedup": 2.5,
         "guidance_phase": "FREE"},
    ])
    result = collect_results.trajectory(directory, started="2026-10-15T00:00:00Z", target=None, horizon_sec=None)
    assert result["guided_hours"] == pytest.approx(2.5)
    assert result["guided_exit"] == "PLAYBOOK_EXHAUSTED"


def test_pairs_compare_the_time_averaged_speedup_too():
    def record(arm, speedup, avg, hours):
        return {"task": TASK, "repeat": 0, "arm": arm, "speedup": speedup, "passed": True, "cost_usd": 1.0,
                "wiki_calls": 0, "hours_to_target": hours, "time_avg_speedup": avg}

    summary = collect_results.summarize([record("kf", 4.0, 3.0, 5.0), record("kf_wiki_guided", 4.0, 3.3, 2.0)],
                                        "kf", "kf_wiki_guided")

    [pair] = summary["pairs"]
    assert pair["ratio"] == pytest.approx(1.0)
    assert pair["time_avg_ratio"] == pytest.approx(1.1)
    assert pair["hours_to_target"] == (5.0, 2.0)
    assert summary["median_time_avg_ratio"] == pytest.approx(1.1)
    assert summary["arms"]["kf_wiki_guided"]["median_hours_to_target"] == pytest.approx(2.0)
    assert summary["arms"]["kf"]["reached_target"] == 1
