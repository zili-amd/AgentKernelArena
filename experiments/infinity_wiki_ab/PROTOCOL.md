<!-- Copied from the kernel-wiki-ab-protocol skill. skills/infinity-wiki-ab-test and skills/infinity-wiki-ab-setup say how to set up and run. -->

# Kernel wiki A/B protocol

A run that breaks a MUST rule is invalid. Exclude it from analysis, record the
reason in the manifest, and rerun it. Never repair a contaminated run in
post-processing.

## The experiment

| Arm | System (AKA `agent.template`) | Wiki |
|---|---|---|
| `kf` | `forge` (Hyperloom KernelForge) | none |
| `kf_wiki` | `forge` | `knowledge_wiki: {enabled: true}` |
| `geak` | `geak` | none |
| `geak_wiki` | `geak` | `knowledge_wiki: {enabled: true}` |

Primary comparisons are within a system: `kf_wiki` vs `kf`, `geak_wiki` vs
`geak`. `kf` vs `geak` is secondary (see rule 9).

Hardware is MI300X (gfx942). For SSH, Docker exec and gateway credentials on the
box, use the `mi300x-single-node-testing` skill.

The experiment needs the AgentKernelArena branch `infinity-wiki-AB-test` (built
on `wiki-ab-infra`). It isolates the forge Experience KB, adds `knowledge_wiki`,
and provisions forge in Docker. AKA `main` does none of these.

Pin these and record them in the manifest. They must be identical across arms:
- AgentKernelArena commit (branch `infinity-wiki-AB-test`);
- Hyperloom commit (`HYPERLOOM_ROOT`, installed by `make docker-setup-forge`);
- GEAK commit (`GEAK_HOME`; it must be exactly the `UPSTREAM_REVISION` in `agents/geak/compatibility.py`, or GEAK preflight fails);
- wiki commit: InstinctWiki (`AMD-BRAIN-Internal/infinity-wiki`, checked out on the node at `ROOT/src/infinity-wiki`). It changes daily; pin one commit for the whole experiment. Some of its kernel pages are built from AKA tasks and their baseline numbers, so before adding a task, grep the wiki for the task path and exclude or record any task it covers;
- Docker image digest, ROCm version, model id, and LLM gateway.

## Rules

### 1. Isolate the forge Experience KB (MUST, critical)

forge-loop defaults `--experience-kb` and `--kb-warmstart` to on. With a shared
store, a later run warm-starts from an earlier run's best patch for the same
operator and hardware, across arms and across repeats. GEAK's KB is forced off
inside AKA, so a forge KB left on would also make the two systems unequal.

On `wiki-ab-infra`, the forge adapter handles this by default
(`experience_kb: isolated`):
- each invocation gets a fresh, empty `experience_kb/` in its artifact directory;
- `KNOWLEDGE_STORE_MODE=local` is set;
- `KB_STORE_URL` and `KB_STORE_TOKEN` are removed;
- Claude auto memory is off.

Never set `agent.experience_kb: engine` in any arm. On AKA `main` this
protection does not exist; do not run the experiment there.

Verify after the run:
- each `arena_forge_status.json` has `experience_kb.mode == "isolated"`;
- each forge result's `kb_experience.read` has `read_reason != "hit"`, `candidate == false` and `applied == false`;
- forge output has no `[kb] warm-start applied` line and no `injecting top solutions` line.

`audit_run.py` checks all of this.

### 2. Same model and budget in every arm (MUST)

The defaults differ:
- forge: `model: claude-opus-5`, `timeout_seconds: 87300`.
- GEAK: `model: null` (falls back to GEAK's own default), `timeout_seconds: 43200`, `budget: 6`.

Set both keys explicitly in every run config:

```yaml
agent:
  template: forge          # or geak
  model: claude-opus-5     # same id in all four arms
  timeout_seconds: 43200   # same wall clock in all four arms
target_gpu_model: MI300
```

Leave the other knobs at their defaults, but keep them identical between the
two arms of the same system.

Unset `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL` and
`CLAUDE_CODE_SUBAGENT_MODEL`, or set them identically for every arm. The runner
forwards them into GEAK containers, where they can change the model silently.

Verify: GEAK's runtime fields `requested_model`, `init_model`,
`assistant_models` and `workflow_models` must all equal the configured id.
Record the wall clock used and the tokens or cost for each run. GEAK can stop
early once its direction budget is spent. Report that; don't pad it.

### 3. Identical wiki access, and none in no-wiki arms (MUST)

Access goes through AKA's `knowledge_wiki` feature (`docs/how-to/knowledge-wiki.md`
on `wiki-ab-infra`), not through MCP. Forge has no generic external MCP hook,
and both systems' agents have Bash.

- **Wiki arms:** add `knowledge_wiki: {enabled: true}` to the run config. Start with `AKA_KNOWLEDGE_WIKI_ROOT=<wiki checkout>` on the host (`launch_wave.sh` sets it). Any checkout with `scripts/query.py`, `get_page.py` and `grep_wiki.py` works; the instructions (version 2) name only the options KernelWiki-style wikis share.
- **Forge delivery:** forge's planning roles (dispatch, three specialists, synthesis, plan critic, supervisor) have no shell, so shell instructions reach only the implementer and the Analysis Agent. With AKA `263c60a4`+ and Hyperloom `forge-knowledge-wiki-tool` (`da36f813e`), the adapter sets `FORGE_KNOWLEDGE_WIKI_ROOT`/`_LOG` and the engine serves `mcp__knowledge_wiki__search|page|grep|guide` to every role with tools; each log line names the `role`. Both forge arms run that same Hyperloom commit; the `kf` arm has the variables unset. AKA refuses a wiki arm on an engine whose probe lacks `knowledge_wiki_tool`. `guide` returns the wiki's own `SKILL.md`, so the main phase needs a correct one.
- **Mounting:** the runner mounts the checkout read-only only for such runs. It refuses a checkout inside the AKA repository.
- **Same instructions:** both adapters append the same instruction block (`INSTRUCTIONS_VERSION` and SHA-256 recorded) to every agent's task contract. Agents run `src/knowledge_wiki.py search|page|grep`.
- **Call logging:** each call is logged to `knowledge_wiki_calls.jsonl` in the run's artifact directory, with the arguments, exit code, duration and returned page IDs.
- **No-wiki arms:** the key is absent, so there is no mount, no variable and no instructions. Don't keep any wiki clone under the AKA repository.
- **Other agents:** a run that enables the key for any agent other than forge or GEAK stops before its first task.

Verify:
- Every wiki-arm run records the same wiki commit, `dirty == false` and the same instruction SHA-256. It also logs at least one call. A run with zero calls means the treatment was not delivered. Keep it in the intention-to-treat analysis, but report the count.
- No-wiki arms have no wiki identity and no call log.

### 4. No leakage through Bash, the network or the filesystem (MUST audit)

Neither runner exposes WebSearch or WebFetch:
- forge: Read, Grep, Glob, Edit, Write, Bash.
- GEAK in AKA: Workflow, Bash, Read, Write. GEAK's DRA web research is forced off.

Leakage is still possible: the default container uses `--network=host`, and
the AKA repository root, with every arm's run directory, is mounted. Bash can
therefore clone the public wiki, fetch GPU MODE winning solutions, or read
another arm's workspace.

- Use a separate AKA checkout per arm (`git worktree add ../aka-<arm> wiki-ab-infra` or a separate clone). Each container then mounts only its own arm's runs.
- If the box allows it, add an egress allowlist that permits only the LLM gateway. Otherwise rely on the audit.
- Run `audit_run.py` with `--other-root` pointing at the other arms' checkouts. A confirmed wiki or solution fetch in a no-wiki arm, or any read of another arm's checkout, invalidates the run.
- GEAK's adapter keeps no model transcripts in its artifacts, so the audit cannot see GEAK's Bash commands there. In the pilot, find the Claude Code session logs (under the container's `~/.claude/projects/`) and pass that directory to the audit too.
- By default the runner mounts the host `~/.claude` read-write, so sessions from all arms accumulate there. Prefer `AGENT_HOME_ISOLATION=1`, which mounts it read-only into a per-worker home. Confirm in the pilot that authentication still works.

Access controls do not cover what the wiki itself says about a task. Before the
first main launch, classify every task against the pinned wiki and record it as
`audits/overlap-<wiki>-<commit>.yaml` in the experiment repository:
- **self-reference**: a page or artifact describes the task's own baseline code. The wiki can only restate it, so the task cannot show an effect. Drop it.
- **solution-available**: the wiki holds a portable optimized implementation of the same operator, dtype and target. Report it separately; it measures finding and porting an answer.
- **transfer**: only related techniques from other kernels. Only these tasks support claims about transferable knowledge.

Grep the task's kernel function names across the whole wiki checkout, including
PR diffs, and read every PR for the operator on the target architecture. A
match on task paths alone is not enough. Never evaluate on a wiki's own
examples or on tasks its pages were written from.

### 5. Pair the arms in time and on hardware (MUST)

- Launch the four arms of the same (task, repeat) together, on four GPUs of one node. Changing conditions such as gateway latency, rate limits and node load then hit every arm equally.
- Never run all of one arm before the next.
- Rotate GPUs across repeats as a Latin square: arm `i` in repeat `r` gets GPU `base + (i + r) % 4`.
- Use at least 3 repeats per (arm, task), and fix the count before the first launch.

Each arm is its own single-GPU invocation, with `AKA_VISIBLE_GPU=<id>` set on
the host for `make docker-run`. AKA verifies its parallel Docker path only for
cursor, claude_code, codex and task_validator. For forge and GEAK, confirm in
the pilot that each container sees exactly its assigned GPU (`rocm-smi` inside
the container, plus `ROCR_VISIBLE_DEVICES`).

Forge runs need `HYPERLOOM_ROOT=<pinned checkout>` on the host. Run
`make docker-setup-forge` once per Hyperloom revision. It installs the committed
`HEAD` only, so commit or discard local changes first. GEAK runs need `GEAK_HOME`
at the pinned revision. Only the configured agent's CLI is provisioned:
- Claude Code for GEAK, and for forge unless it sets `agent.agent_backend: codex`;
- gateway credentials are forwarded by name for both.

Known issue: against Hyperloom `main` `f2b32cdc9`, AKA's
`test_installed_upstream_initialization` fails, on AKA `main` as well. That covers
forge's HIP/Triton initialization workflow, which writes an implementation
before optimizing. Prefer tasks with an existing implementation (forge
`optimize` workflow), or fix that path before using such tasks.

### 6. Take headline numbers from a quiet re-measurement (MUST)

AKA measures the baseline once before the agent and the final kernel once
after, each with one `performance_command` invocation, while other arms load
the node. That is fine for pass/fail, but too noisy for effects under 10%.

Wait until all runs have finished and the node is idle. Check
`rocm-smi --showpids` and `who`, because the box can be shared. Then, for each
task:
- Re-measure the baseline and every arm's final kernel on the same GPU, using the task's own `performance_command`.
- Interleave the order: ABCD, then DCBA, and so on.
- Run at least 5 rounds.
- Report the median and a bootstrap 95% confidence interval.

If `scripts/remeasure.py` does not exist yet, label every speedup as
provisional (in-run) in reports.

### 7. Held-out shapes (MUST for headline claims)

1. Generate held-out shapes once per task, before the runs, with `python3 -m src.held_out.generate_heldout`, and review them.
2. During runs, keep `held_out_tests/` outside every mounted path. It defaults to the AKA repository root, which agents can read.
3. After the runs, run this for every run directory:

   ```bash
   python3 -m src.held_out.run_heldout_eval --run-dir <run_dir> \
       --heldout-dir <held_out_dir> --tasks-dir tasks/
   ```

   It requires the same captured GPU and runtime identity as the original run.
4. Report conditional correctness, P(opt correct | orig correct), per arm.

### 8. Tasks must be valid on gfx942 (MUST)

Use `target_gpu_model: MI300`. Use only tasks whose images and shapes target
MI300X:
- `tasks/image_kernel/mi300x_sglang_hip_mha_batch_prefill`
- `tasks/image_kernel/mi300x_sglang_hip_pa_decode`
- `tasks/image_kernel/mi300x_sglang_hip_pa_ragged`
- `tasks/image_kernel/mi300x_sglang_triton_fp8_gemm`
- `tasks/image_kernel/mi300x_sglang_triton_gemm`
- tasks we author for gfx942

Exclude every `mi355x_*` task, even one whose `required_arch` is `any`.

gfx942 has no FP4 hardware, so MXFP4 tasks there are emulated. They are still
valid tasks, but the baseline must be the gfx942 production path, not a gfx950
kernel.

### 9. Scoring and analysis (MUST)

- **Source of numbers:** correctness and timing come only from AKA's central evaluation and the rule 6 re-measurement, never from forge or GEAK self-reports. forge's own SNR parse can mask NaN in non-first cases.
- **Pre-registration:** before the first launch, write into the manifest the tasks, repeat count, primary metric and failure handling.
- **Default metric:** per-task speedup over the baseline, with a failed or incorrect final counted as 1.0×, and pass rate reported separately. AKA writes `average_speedup: 0.0` for failures and mismatched cases. Convert those values to 1.0× and count them as failures. Never average the raw 0.0 values.
- **Aggregates:** report three:
  - per-case arithmetic mean, which is AKA's default and forge's own objective;
  - geometric mean;
  - time-weighted mean, Σ baseline / Σ optimized.

  AKA's default matches forge's optimization target, so it favors forge in `kf` vs `geak`. Within-system comparisons are unaffected.
- **Cost:** report per run the wall clock, tokens and wiki call count.

### 10. Forge measures with the Arena driver (MUST)

Forge's task preparation runs a preflight on the AKA-generated
`arena_forge_driver.py`. If any check fails, a preparation agent rewrites the
driver. The rewritten driver gates candidates with its own SNR check instead of
the task's correctness action. In the first pilot it also printed raw case IDs
(`case/0`) while `baseline.json` holds URL-encoded ones (`case%2F0`). No
candidate got a per-case speedup, so a correct 5.4x candidate was reverted and
the run delivered 1.00x. An LLM-written driver also differs between arms.

AKA `cda55898` and later answers `--profile-run` with the task's own
performance action, so the preflight passes and the driver is kept.

Verify: the forge output has `[prepare] task already conforms to the driver
contract; skipping` and no `[prepare] prepared measurement driver`.
`audit_run.py` FAILs a run whose `engine.log` shows a replaced driver.

### Node notes (learned on a shared MI300X node)

- A gateway may need `ANTHROPIC_CUSTOM_HEADERS` (for example a subscription-key header) besides `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`. AKA `bf10e160` and later forwards all three. Pass them through ssh stdin, never argv.
- A gateway certificate signed by a private CA is not trusted by the runtime image. `setup_node.sh` copies the host CA bundle into each checkout, and `launch_wave.sh` points `NODE_EXTRA_CA_CERTS`, `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` at it.
- A shared node's `/root/.claude` can hold another user's MCP servers and skills. `launch_wave.sh` sets `HOME=ROOT/claude-home`, which stays empty.
- GEAK refuses to run Claude as root, and the runtime container runs as uid 0. GEAK arms need a non-root runtime user first.

## Known non-issues

These were verified at the pinned commits in [reference.md](reference.md).
Re-check them only if those commits change.

- AKA's cheatsheet prompt is not injected into forge or GEAK. Only cursor, claude_code, codex and deepseek_harness receive it.
- GEAK inside AKA forces `warm_start`, `update_experience`, `use_learned_kb`, `dra_enabled` and `use_expert_skills` off.
- forge's PR knowledge base is opt-in (`--pr-kb`), and forge does not read `HYPERLOOM_KB_URL`.
- Built-in static knowledge stays on in both arms of a system: forge's knowledge maps and GEAK's `perf_knowledge`. It is part of the "original" system.

## Workflow

Run a pilot first: 1 task × 4 arms × 1 repeat, through the whole checklist.
Fix everything the audit flags, then scale up.

```
Pre-flight
- [ ] Manifest: pins (identical across arms) + pre-registration written
- [ ] One AKA checkout per arm, all on wiki-ab-infra; held-out data and wiki clone outside all mounts
- [ ] Run configs: same agent.model and agent.timeout_seconds; gfx942-valid tasks only; no experience_kb override
- [ ] ANTHROPIC_*MODEL and CLAUDE_CODE_SUBAGENT_MODEL env unset or identical
- [ ] knowledge_wiki.enabled only in wiki arms; wiki checkout clean at the pinned commit
- [ ] Task-wiki overlap audit written (rule 4); self-reference tasks dropped
- [ ] Closed-book probe of the wiki's task-relevant facts (`closed_book/run.py`), so a null result can be read against what the model already knew
- [ ] HYPERLOOM_ROOT clean at the pinned commit; make docker-setup-forge done; GEAK_HOME at UPSTREAM_REVISION
- [ ] Each container sees exactly its assigned GPU
- [ ] Node idle apart from this experiment
Run
- [ ] Four arms of the same (task, repeat) launched together; GPU rotation applied
Post-run
- [ ] python3 audit_run.py --arm <arm> [--expect-model <id>] [--other-root <path> ...] <run_dir> [<session_log_dir>]  (exit 0)
- [ ] Held-out evaluation for every run
- [ ] Quiet re-measurement
- [ ] Report: within-system deltas with CIs first, then cross-system; three aggregates; cost
```

The tooling lives in this directory; the two skills under `skills/` say how to set up and run, and `deploy.sh`, `start.sh` and `status.sh` drive a machine over SSH. `make_record.py` fills `TEST_RECORD_TEMPLATE.md` after every pair:
- `manifest.yaml`: pins, tasks, budgets and the pre-registration.
- `make_configs.py --aka <checkout> --wiki-root <wiki>`: writes and validates one run config per phase, task and arm.
- `pack.sh`: builds one pinned git bundle per repository into `dist/`.
- `setup_node.sh ROOT [--provision|--check]`: unpacks the bundles on the node into one AKA checkout per arm, plus Hyperloom, GEAK and the wiki.
- `launch_wave.sh ROOT PHASE TASK REPEAT GPU_BASE [--dry-run]`: starts the `ARMS` of one (task, repeat) together with the Latin-square GPU rotation and records them in `invocations.jsonl`.
- `run_main.sh ROOT [--dry-run]`: the main phase, one wave per repeat, every main task as a pair on its own GPU pair; busy GPUs are retried, never taken.
- `collect_results.py ROOT [--phase main]`: per-run speedup (failures as 1.0), pass, cost and wiki calls by role; paired log ratios, exact Wilcoxon, bootstrap CI.
- `closed_book/run.py [--self-test] [--out DIR] QUESTIONS...`: asks the experiment model the wiki's task-relevant questions with no wiki or tools, and grades the answers against regexes. The regexes are a first pass, so read `answers.jsonl` before reporting a rate. Needs gateway credentials in the environment.
- `screen.sh ROOT [GPU] [--dry-run]`: baseline-only screening, with no agent. It measures every screening candidate with AKA's own harness (`screen_baselines.py`), then `roofline.py` applies the selection rule: headroom of at least 3x over max(roofline, 5 us). Only tasks that pass become main-phase tasks. Never add a near-optimal production kernel; it cannot show a wiki effect.

Keep one `manifest.yaml` per experiment. It holds the pins, the
pre-registration, every invocation (arm, repeat, GPU, run directory, start and
end times) and every exclusion with its reason.

## Audit script

`audit_run.py` is read-only. It exits with one of these codes:
- `0` (OK): the run passed.
- `1` (INVALID): at least one FAIL. The run is invalid.
- `3` (UNVERIFIED): no FAIL, but some rule could not be checked, for example
  because no forge result JSON or GEAK model fields were found. Treat it as not
  passed.

REVIEW lines need a human or agent decision, recorded in the manifest.

```bash
python3 experiments/infinity_wiki_ab/audit_run.py \
    --arm kf --expect-model claude-opus-5 \
    --other-root /path/aka-kf_wiki --other-root /path/aka-geak --other-root /path/aka-geak_wiki \
    /path/aka-kf/workspace_MI300_forge/run_YYYYMMDD_HHMMSS
```

It finds `knowledge_wiki_calls.jsonl` files on its own, and checks the
adapters' `experience_kb` and `knowledge_wiki` status records.

It has run on the first real forge pilot (refk_fp8_blockwise_mm on MI300X). It
found the `kb_experience` records, the isolation status and the Claude session
transcripts (pass the run's `wiki_ab_results/home-<tag>` directory). Its rule 4
REVIEW lines there were text, not fetches: Claude Code's own Bash tool
description contains `curl -s url | jq`, and prompts cite URLs. To check what
actually ran, read the `tool_use` blocks in the transcripts. It has not yet run
on a GEAK run.

## Additional resources

- Evidence (file:line at the pinned commits) for every rule: [reference.md](reference.md)
