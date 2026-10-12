---
name: infinity-wiki-ab-test
description: >-
  Set up and run the KernelForge vs KernelForge + InstinctWiki (infinity-wiki)
  A/B test through AgentKernelArena on any MI300X machine: clone the
  infinity-wiki-AB-test branch, pin and pack the repositories, set up the GPU
  node, run the pilot and the main phase, audit every run, and compute the
  paired statistics. Use when the user wants to set up, run, resume, monitor,
  audit or analyze the infinity-wiki A/B test, KernelForge + wiki, or kf vs
  kf_wiki.
---

# KernelForge × InstinctWiki A/B test

Question: does read-only access to InstinctWiki make the kernels KernelForge
writes on MI300X faster? Read [PROTOCOL.md](PROTOCOL.md) before any launch; a
run that breaks one of its MUST rules is invalid.

| | `kf` | `kf_wiki` |
|---|---|---|
| Run config | from `manifest.yaml` | the same plus `knowledge_wiki: {enabled: true}` |
| Knowledge base | none | every forge role gets `mcp__knowledge_wiki__search/page/grep/guide` |

Held equal: the four main tasks (baselines far from the MI300X roofline,
`screening/`), the starting point (each task's frozen initial implementation and
an empty per-run Experience KB), the wall clock (2 h pilot, 6 h main), the model,
the pinned code (`manifest.yaml`, `dist/pins.txt`), and time and hardware (the two
arms of a pair start together on adjacent GPUs and swap GPUs between repeats).
Score: AKA `speedup_ratio` per run, failures count as 1.0; per pair
log(kf_wiki / kf); exact Wilcoxon and a bootstrap CI over 4 tasks × 3 repeats.

## Requirements

- **Packing machine:** git with read access to `AMD-AGI/AgentKernelArena`,
  `AMD-AGI/Hyperloom` and `AMD-BRAIN-Internal/infinity-wiki`; python3 with PyYAML.
- **GPU node:** MI300X (gfx942), Docker, `rocm-smi`, python3 with PyYAML, about
  80 GB of disk for the runtime image, internet access on first provision (Docker
  image, Node.js, npm). Two free GPUs per pair; a full main wave uses eight.
- **LLM gateway:** `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` (or
  `ANTHROPIC_AUTH_TOKEN`) and, if the gateway needs one, `ANTHROPIC_CUSTOM_HEADERS`.
  Never print them or put them in argv; pipe them through ssh stdin (step 6).
- **Shared nodes:** never stop or touch another user's process or container.
  `launch_wave.sh` refuses a GPU another process uses.

## 1. Get the tooling

```bash
git clone -b infinity-wiki-AB-test https://github.com/AMD-AGI/AgentKernelArena.git
cd AgentKernelArena/experiments/infinity_wiki_ab     # every local command below runs here
```

## 2. Pin the wiki (before every main launch)

The wiki changes daily, so pin the newest commit at launch time:

1. Check that its `SKILL.md` describes the real corpus. The `guide` tool hands it
   to the agents verbatim; at `62f09f04` it still says the corpus is unbuilt, and
   the main phase must not start with that text.
2. Repeat the overlap audit for every main task: grep the task path and its kernel
   function names across the whole checkout, `artifacts/` included. A task the wiki
   covers (its own baseline or a portable solution) is reported separately.
3. Set `pins.wiki.commit` in `manifest.yaml`, commit, and push the branch.

## 3. Pack (where GitHub is reachable)

```bash
ARMS="kf kf_wiki" bash pack.sh       # HYPERLOOM_SRC=/path or WIKI_SRC=/path to use local checkouts
python3 make_configs.py --aka "$(git rev-parse --show-toplevel)" --wiki-root <wiki checkout> --out /tmp/ab-configs
```

`pack.sh` pins AgentKernelArena to this checkout's clean HEAD and writes
`dist/`: one bundle per repository, `pins.txt`, `arms.txt`, and `experiment/`
(these files plus every run config). The second command validates the configs
with the platform's own loaders.

## 4. Set up the GPU node

```bash
ROOT=/root/wiki-ab                                    # any directory with space
rsync -a dist/ <node>:$ROOT/dist/
ssh <node> bash $ROOT/dist/experiment/setup_node.sh $ROOT --provision
ssh <node> bash $ROOT/dist/experiment/setup_node.sh $ROOT --check
```

This creates one AKA checkout per arm (`aka-kf`, `aka-kf_wiki`), `src/hyperloom`
and `src/infinity-wiki`, pulls the runtime image, installs the pinned agent CLI
and an empty Claude home, and installs forge into each checkout's container
target. After a pin change, rerun steps 3 and 4: existing checkouts fast-forward
and forge is reinstalled. Never do that while a run uses those checkouts.

## 5. Screening (only for new tasks or another GPU model)

```bash
ssh <node> bash $ROOT/dist/experiment/screen.sh $ROOT <gpu>
```

It measures baselines only, with no agent. A main task needs headroom of at least
3x over max(roofline, 5 µs). The MI300X results that chose the four tasks are in
`screening/`.

## 6. Pilot: one pair, 2 hours

From the machine holding the credentials (an env file with the variables above):

```bash
{ printf 'set -a\n'; cat <env file>; printf '\nset +a\n'
  echo "cd $ROOT && ARMS='kf kf_wiki' setsid nohup bash dist/experiment/launch_wave.sh $ROOT pilot \
triton2triton/geak_eval/L1/refk_fp8_blockwise_mm <repeat> <gpu base> > logs/pilot-r<repeat>.log 2>&1 < /dev/null &"
} | ssh <node> bash -s
```

Use a repeat number not used before; it names the run directories. Check:

- within 10 minutes, each arm's log (`$ROOT/logs/pilot/<task>/r<repeat>/<arm>.log`)
  shows `[prepare] task already conforms to the driver contract; skipping`, and only
  `kf_wiki`'s shows `Knowledge base: <root>`;
- within an hour, `kf_wiki`'s `knowledge_wiki_calls.jsonl` (in the run's
  `*-forge-*` directory) has lines, including specialist roles; `kf` has none;
- forge's KEEP/REVERT lines report a mean case speedup, never `None`.

## 7. Main phase

Same credential pipe as step 6, with this command:

```bash
cd $ROOT && setsid nohup bash dist/experiment/run_main.sh $ROOT > logs/main.log 2>&1 < /dev/null &
```

Three waves (one per repeat) of four pairs on GPU pairs 0-1 to 6-7, about
19 hours. A pair whose GPUs are busy is retried every 5 minutes for an hour, then
skipped and logged. Monitor `$ROOT/invocations.jsonl` and `$ROOT/logs/main.log`.

## 8. Audit every run (on the node)

```bash
python3 $ROOT/dist/experiment/audit_run.py --arm <arm> --expect-model claude-opus-5 \
    --allow-host <gateway host> --other-root $ROOT/aka-<other arm> \
    $ROOT/aka-<arm>/workspace_MI300_forge/run_*_<run suffix> $ROOT/aka-<arm>/wiki_ab_results/home-<run suffix>
```

Exit 0 is required. A FAIL invalidates the run: record it in `manifest.yaml`
`exclusions` and rerun it on the same GPU slot. Settle every REVIEW line by
reading the transcripts' `tool_use` blocks.

## 9. Results

```bash
python3 $ROOT/dist/experiment/collect_results.py $ROOT --phase main --json $ROOT/results/main.json
```

These are in-run timings. Before headline numbers, run the quiet
re-measurement and the held-out shapes (PROTOCOL.md rules 6 and 7).

## Known issues

- GEAK arms need a non-root runtime user: GEAK refuses to run Claude as root.
- Hyperloom's `test_hookless_resume_republication_clears_a_corrected_refusal`
  fails on `main` too.
- Evidence for every rule, at the pinned commits: [reference.md](reference.md).
