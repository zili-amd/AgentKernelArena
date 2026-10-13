---
name: infinity-wiki-ab-test
description: >-
  Runs the KernelForge vs KernelForge + InstinctWiki (infinity-wiki) A/B test
  end to end for a user: asks what to test, on which MI300X machine and with
  which credential file, sets the machine up over SSH, starts and monitors the
  runs, and returns the filled test record. Use when the user asks to run,
  resume, check on or report the infinity-wiki A/B test, KernelForge + wiki,
  or kf vs kf_wiki.
---

# Run the KernelForge × InstinctWiki A/B test

`W` is the local directory that holds the three repositories (`AgentKernelArena`,
`Hyperloom`, `infinity-wiki`); `E` is `W/AgentKernelArena/experiments/infinity_wiki_ab`.
If this skill directory is a symlink, `W` is the directory above the
AgentKernelArena checkout it resolves into; otherwise ask the user. Every script
below is in `E` and runs on the user's machine, not on the test machine.

## 1. Ask the user

Ask everything in one round (AskQuestion where available), offering these defaults:

| Question | Default |
|---|---|
| What to test | `pilot`: one task, the arms side by side, 2 h. `main`: the manifest's main tasks × arms × repeats at 6 h; as pinned, 4 tasks × 3 arms (`kf`, `kf_wiki` on wiki main, `kf_wiki_branch` on the wiki branch) × 1 repeat, two waves of two tasks on 6 GPUs, about 12 h |
| Arms | `--arms "kf kf_wiki kf_wiki_branch"` for `deploy.sh`; `knowledge_wiki` in `manifest.yaml` names each arm's wiki pin |
| Test machine | SSH target `USER@HOST` and an SSH key file; key-based login without a prompt |
| Working directory on it | `/root/wiki-ab` |
| Credentials | a local `.env` with `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_CUSTOM_HEADERS` if the gateway needs it |
| Where the record goes | `W/infinity-wiki-ab-results/` |

Check the `.env` by variable names only (`grep -oE '^[A-Z_]+=' FILE`). Never print,
copy or quote its values, and never put them in a command line.

A pilot costs about $20 per run, so $40 per pair. Tell the user before starting
`main`, which runs 24 runs of 6 hours.

## 2. Check the definition

Read `E/PROTOCOL.md` once. Then:

- **Custom scope** (other tasks, budget or repeats): edit `phases` in
  `E/manifest.yaml`, keep only tasks that pass screening (headroom ≥ 3x), commit.
- **Before `main`**: pin the newest infinity-wiki commit. Its `SKILL.md` must
  describe the real corpus, because the `guide` tool hands it to the agents
  verbatim. Repeat the overlap audit (grep each main task's path and kernel
  names across the wiki, `artifacts/` included). Then set `pins.wiki.commit` and
  commit. Never change `preregistration` after the first `main` launch.

## 3. Set up the machine

Follow `E/skills/infinity-wiki-ab-setup/SKILL.md`. In short:
`bash E/deploy.sh --host H --root R --key K`. It refuses to run while a test
on that machine is still in progress.

## 4. Start

```bash
bash E/start.sh --host H --root R --key K --env-file F pilot    # or: main
```

For `pilot` it picks the manifest's pilot task, the next unused repeat number and
the first free pair of GPUs (`--task`, `--repeat`, `--gpu-base` override them). It
never starts on a GPU another process uses.

## 5. Monitor

```bash
bash E/status.sh --host H --root R --key K --phase P --env-file F
```

It lists each run's state, its latest forge decision and its knowledge base call
count, and regenerates the test record on the machine. For a pilot, check:

- within 10 minutes, every arm shows `[prepare] task already conforms`, and only
  `kf_wiki` shows `Knowledge base:`;
- within an hour, `kf_wiki` has knowledge base calls, specialist roles among them,
  and `kf` has none;
- KEEP and REVERT lines report a mean case speedup, never `None`.

If a check fails, report it to the user rather than working around it. Never
stop or touch a process or container you cannot tie to a run in this
experiment's ledger (`R/invocations.jsonl`). For `main`, tell the user it has
started and how to resume ("check on the infinity-wiki A/B test"), instead of
holding the session for 19 hours.

## 6. Record and report

When every run of the phase has finished:

1. `bash E/status.sh ... --phase P --env-file F --fetch W/infinity-wiki-ab-results`
   copies `P-record.md` back. Sections 1 to 5 are already filled: what is tested,
   the starting point, the roofline analysis, the theoretical ceiling, and the
   results with an audit verdict per run.
2. Every run must audit `OK`. Settle each REVIEW or UNVERIFIED verdict by reading
   the run's transcripts (`R/aka-<arm>/wiki_ab_results/home-<run suffix>`, the
   `tool_use` blocks). An INVALID run is excluded: record why in
   `manifest.yaml` `exclusions`, rerun it with the same repeat and GPU base, and
   regenerate.
3. Fill section 6, between the `agent-notes` markers:
   - which directions each arm kept (the KEEP lines in `R/logs/P/<task>/r<N>/<arm>.log`);
   - how the wiki was used (`knowledge_wiki_calls.jsonl`: roles, queries, pages);
   - anomalies and exclusions;
   - what the phase shows, within the pre-registration. A pilot shows only that
     the pipeline works.
4. Copy the filled record back to `R/results/` so that later regenerations keep
   section 6.
5. Reply to the user with the record's path, the per-arm table, the paired
   ratio, the cost, and what comes next. In-run timings are not headline
   numbers; PROTOCOL.md rule 6 needs a quiet re-measurement first.
