# Evidence for the protocol rules

Pinned commits at the time of verification. Rule 1 and rule 3 also cite `main`;
the branch section below says what changed there.

| Repo | Commit |
|---|---|
| AgentKernelArena `main` | `a4217436` |
| AgentKernelArena `wiki-ab-infra` | 3 commits on `a4217436` |
| Hyperloom `main` | `f2b32cdc9` |
| GEAK pinned by AKA | `c0c0e2ae` |
| ROCmKernelWiki | `9252153f` |

## The `wiki-ab-infra` branch

The branch replaces the manual workarounds that `main` would need:
- `Isolate the forge Experience KB per invocation by default`:
  - `agents/forge/adapter.py` (`isolate_experience_kb`, `experience_kb` config key);
  - `agent_config.yaml`;
  - README;
  - tests in `tests/test_forge_v2.py`.
- `Offer a run-level kernel knowledge base identically to Forge and GEAK`:
  - `src/knowledge_wiki.py` (config, instructions, logged query CLI, identity);
  - hooks in `agents/forge/program.py` and `adapter.py`, and in `agents/geak/compatibility.py` (`arena_contract`) and `launch_agent.py`;
  - the `main.py` check that rejects other agents;
  - the runner mount;
  - `docs/how-to/knowledge-wiki.md`;
  - tests in `tests/test_knowledge_wiki.py` and `tests/test_docker_benchmark.sh`.
- `Provision the forge agent in the Docker runner`:
  - `FORGE_RUNTIME`, the `HYPERLOOM_ROOT` read-only mount, the backend CLI, credential forwarding, `make docker-setup-forge` (installs the committed `HEAD`, records `.aka-hyperloom-revision`) and the `--arena-probe` check;
  - shell tests, including the install logic.

Checks that passed on CPU:
- the forge, GEAK, wiki and `main.py` Python tests: 448 passed, with `GEAK_TEST_CHECKOUT` and a venv with the GEAK SDK and CPU torch;
- `make check-docker-runner`;
- mutation checks of the new shell assertions.

With `AKA_FORGE_PROBE_PYTHON` pointing at an interpreter with `Hyperloom[forge]` installed:
- the 4 cases of `test_installed_upstream_initialization` fail on `main` as well;
- the other real-engine probes pass.

Nothing on the branch has run on a GPU yet.

If any of these commits changes, re-grep the cited lines before relying on a
rule. Paths are relative to the repository root. Hyperloom paths are relative to
`src/kernelforge/`.

## Rule 1: forge Experience KB

- Hyperloom `cli.py:877-891`: `--experience-kb/--no-experience-kb` and `--kb-warmstart/--no-kb-warmstart`, both `default=True`.
- Hyperloom `knowledge/experience_store.py:55-77`: `KNOWLEDGE_STORE_MODE` defaults to `local`. The root is resolved from `KNOWLEDGE_LOCAL_ROOT`, then `USER_DATA_PATH/knowledge`, then `~/.cache/hyperloom/knowledge`. Remote mode needs `KB_STORE_URL` and `KB_STORE_TOKEN`.
- Hyperloom `knowledge/experience_integration.py`:
  - `:1088` sets `read_reason="hit"` when the store returns solutions.
  - `:1253` prints `[kb] warm-start applied`.
  - `:1270-1272` prints `reference-only ... injecting top solutions as reference`.
  - `:1439-1458` defines `kb_read_status`, whose fields are `candidate`, `read_reason`, `applied` and `solution_slug`.
- Hyperloom `cli.py:1628` writes `kb_experience.read` into the result and experiment JSON.
- AKA `agents/forge/adapter.py`:
  - `:60-62` lists the only accepted overrides, and none of them controls the KB.
  - `:230-275` builds the forge-loop command with no KB flag.
  - `:311` passes `os.environ.copy()` to forge.
- AKA `agents/forge/README.md:350-356` says "knowledge warm starts follow the engine defaults".
- AKA `src/scripts/docker_benchmark.sh`:
  - `:1028-1050` is the explicit `-e` list, which has no `KNOWLEDGE_*`.
  - `:990` sets the container `HOME`, which defaults to the host `$HOME` path.
- AKA `agents/geak/compatibility.py:200-202` forces GEAK's warm start, experience update, learned KB, DRA and expert skills off. This is the asymmetry.
- Searching Hyperloom `src/kernelforge` for `hyperloom_kb` or `HYPERLOOM_KB` finds no hits, so forge does not use that service.

## Rule 2: model and budget parity

- AKA `agents/forge/agent_config.yaml` contains `model: claude-opus-5`, `timeout_seconds: 87300`, `session_timeout_seconds: 3600` and `initialization_budget_fraction: 0.4`.
- AKA `agents/geak/agent_config.yaml` contains `model: null`, `effort: high`, `budget: 6`, `min_improve: 0.02`, `deep_cost: 2` and `timeout_seconds: 43200`.
- AKA `agents/geak/launch_agent.py`:
  - `:32-47`: any key of `agent_config.yaml` can be overridden from the run config's `agent:` mapping.
  - `:28-29`: the runtime fields `requested_model`, `init_model`, `assistant_models` and `workflow_models` are recorded.
- AKA `agents/geak/compatibility.py:203` passes `arena_model` through to the GEAK workflow options.
- AKA `src/scripts/docker_benchmark.sh:1117-1130` forwards `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_*_MODEL` and `CLAUDE_CODE_SUBAGENT_MODEL` into GEAK containers. It does so only when `GEAK_RUNTIME=1` and the agent list contains claude_code.

## Rule 3 and rule 4: wiki access and leakage

- Hyperloom `agent_backends/claude.py:397-424`: forge's Claude backend allows Read, Grep, Glob, Edit, Write and Bash, plus `policy.extra_tools`. MCP tools reach the session through `allowed_tools` (`:259`).
- AKA `agents/geak/workflow_runner.py:46` sets `ALLOWED_TOOLS = ["Workflow", "Bash", "Read", "Write"]`. GEAK's own `interface/run_e2e.py` adds WebSearch and WebFetch, but AKA does not use that path.
- GEAK `kernel_workflow/kernel_lane.js:11,166,911` (at `c0c0e2ae`): only the opt-in DRA research stage uses WebSearch and WebFetch. AKA forces DRA off.
- AKA `agents/geak/README.md:102-106`: GEAK's adapter artifacts omit runtime environment dumps, raw task stdout and model transcripts.
- AKA `src/scripts/docker_benchmark.sh`, `mount_agent` claude_code branch: without `AGENT_HOME_ISOLATION=1`, host `~/.claude` and `~/.claude.json` are mounted read-write at the same path. With it, they are mounted read-only under `AGENT_STATE_MOUNT_ROOT`. `parallel-run` sets isolation itself.
- AKA `src/scripts/docker_benchmark.sh`:
  - `:1020-1024`: by default (`AKA_DOCKER_PRIVILEGED=1`) containers use `--ipc=host --network=host --privileged`. With `0`, Docker's default bridge network applies, which still allows egress.
  - `:1205-1214`: in parallel runs the artifact namespace is mounted read-only, but the broad repository mount stays readable, including the other runs' workspaces.
- AKA `README.md` states that agents are "not a security sandbox" and can access mounted repository and authentication state.
- The wiki is public, at `github.com/jhinpan/ROCmKernelWiki` and the fork `AMD-AGI/ROCmKernelWiki`, so `git clone` works from Bash.

## Rule 5: pairing and GPU masking

- AKA `README.md:292-295`: the Docker parallel path is verified only for cursor, claude_code, codex and task_validator. GEAK needs its own GPU-ID configuration. The runner script does not mention forge.
- AKA `src/scripts/docker_benchmark.sh:1092-1100`: `AKA_VISIBLE_GPU` sets `ROCR_VISIBLE_DEVICES` to the host GPU and `HIP_VISIBLE_DEVICES` to the logical GPU.

## Rule 6: measurement

- AKA `src/performance.py:278-333`: `measure_performance` runs each `performance_command` once, parses the report and returns on the first parseable result. It does no repeats and no interleaving.
- AKA `src/performance.py:334-359`: `measure_baseline` runs once before the agent starts and is saved to `baseline_perf.yaml`.
- AKA `src/evaluator.py:135` is the centralized `evaluate_kernel`, which measures the final kernel once after the agent.
- The forge adapter passes AKA's baseline into forge as `--baseline-json` (`agents/forge/adapter.py:258`), so forge's internal anchor matches AKA's.

## Rule 7: held-out shapes

- AKA `src/held_out/README.md`:
  - The workflow is optional and invoked manually: `generate_heldout`, then `run_heldout_eval`.
  - `held_out_tests/` is gitignored but lives under the repository root.
  - V2 evaluation requires the same captured GPU and runtime identity.
  - New performance inputs must be absent from the original manifest, so the tool cannot re-measure the original shapes.

## Rule 8: tasks

- MI300X-native image-kernel tasks: `tasks/image_kernel/mi300x_sglang_{hip_mha_batch_prefill,hip_pa_decode,hip_pa_ragged,triton_fp8_gemm,triton_gemm}`.
- There are 16 `tasks/**/mi355x_*` directories. They come from MI355X images and shapes even when `platform_support.required_arch` is `any`.

## Rule 9: scoring

- AKA `src/testcases.py:549-638`: `calculate_average_speedup`, called from `src/evaluator.py:401`, returns the arithmetic mean of per-case baseline/optimized ratios (`:636`).
  - It returns `0.0` when cases do not match completely, the workload or benchmark method changed, or a case time is invalid. The evaluator also uses `0.0` for compilation or correctness failure (`src/evaluator.py:509,518`).
  - This mean matches forge's `calculate_mean_case_speedup` objective (Hyperloom `loop/scoring.py`, `mcp_server/tools/bench.py:226-266`).
  - GEAK optimizes a weighted aggregate, Σw / Σ(w/s).
- Hyperloom `mcp_server/tools/test.py:58`: the SNR parse takes the first `SNR: <x> dB` match and skips `nan`. Combined with `min(snr_values)` in the example drivers, this can mask NaN in non-first cases. AKA's central correctness check is independent of it.

## Known non-issues

- AKA `src/module_registration.py:137-140`: the standard prompt builder, which carries AKA's cheatsheet, serves only cursor, claude_code, codex and deepseek_harness.
- Hyperloom `cli.py:911`: `--pr-kb/--no-pr-kb` is opt-in.
