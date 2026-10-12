---
name: infinity-wiki-ab-setup
description: >-
  Sets up an MI300X machine over SSH for the KernelForge × InstinctWiki A/B
  test from three local repositories in one directory (AgentKernelArena on
  infinity-wiki-AB-test, Hyperloom, infinity-wiki): checks the pinned commits,
  packs them, checks the machine, copies and provisions. Use when the
  infinity-wiki-ab-test skill reaches setup, or when the user asks to set up or
  refresh a machine for this test.
---

# Set up a machine for the KernelForge × InstinctWiki A/B test

`W` and `E` are as in the infinity-wiki-ab-test skill.

## Local layout

```text
W/
├── AgentKernelArena/   branch infinity-wiki-AB-test, no uncommitted changes (its HEAD is pinned)
├── Hyperloom/          must contain the manifest's Hyperloom pin (branch forge-knowledge-wiki-tool)
└── infinity-wiki/      must contain the manifest's wiki pin
```

To get them:

```bash
git clone -b infinity-wiki-AB-test https://github.com/zili-amd/AgentKernelArena.git
git clone -b forge-knowledge-wiki-tool https://github.com/AMD-AGI/Hyperloom.git
git clone https://github.com/AMD-BRAIN-Internal/infinity-wiki.git          # needs read access
ln -s "$PWD"/AgentKernelArena/experiments/infinity_wiki_ab/skills/* ~/.cursor/skills/   # or ~/.claude/skills/
```

An existing Hyperloom or infinity-wiki checkout works as long as it contains the
pinned commits; `deploy.sh` fetches a missing pin from `origin` once. Set
`HYPERLOOM_DIR` or `WIKI_DIR` for other directory names.

## Machine requirements

- MI300X (gfx942). Another GPU model needs its own screening and run configs.
- Docker usable by the SSH user, `rocm-smi` and `rocminfo`, python3 with PyYAML,
  and `/etc/ssl/certs/ca-certificates.crt`.
- Key-based SSH that never prompts (`BatchMode`).
- On first provision: internet access for the runtime image, Node.js and npm,
  and about 110 GB of free disk.

## Deploy

```bash
bash E/deploy.sh --host H --root R --key K --dry-run   # local checks and packing only
bash E/deploy.sh --host H --root R --key K
```

It:
1. checks that the three checkouts hold their pins;
2. packs `E/dist/` (one bundle per repository, the run configs and the tools);
3. checks the machine: GPU architecture, GPUs in use, Docker, PyYAML, the CA
   bundle, free disk, and runs still in progress;
4. copies `dist/` to `R/dist/`;
5. runs `setup_node.sh R --provision`: one AKA checkout per arm, `src/hyperloom`
   and `src/infinity-wiki`, the runtime image, the pinned agent CLI, an empty
   Claude home, and forge installed into each arm's checkout.

The last lines repeat `setup_node.sh --check`: each checkout and the image at its
pin, and the GPUs in use.

| Message | Meaning and fix |
|---|---|
| `lacks the hyperloom pin` | `git -C W/Hyperloom fetch origin forge-knowledge-wiki-tool` |
| `lacks the wiki pin` | `git -C W/infinity-wiki fetch origin` |
| `commit or stash the changes` | the AKA pin is its HEAD; commit first |
| `needs MI300X (gfx942)` | wrong machine |
| `docker is not usable` | the SSH user needs Docker access |
| `runs still in progress` | wait (`status.sh`); never redeploy under a running test |
| a failed image pull or npm install | the machine lacks internet; pre-pull the image AKA selects for gfx942 (`DEFAULT_DOCKER_IMAGE_GFX942` in `src/scripts/docker_benchmark.sh`) |

After any pin change, commit in AgentKernelArena and run `deploy.sh` again.
Existing checkouts fast-forward and forge is reinstalled.
