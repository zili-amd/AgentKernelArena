#!/usr/bin/env bash
# Baseline-only screening on the node: measure every screening candidate with
# AKA's own harness (no agent), then apply the selection rule.
#   screen.sh ROOT [GPU] [--dry-run]      ROUNDS=5 by default
# Results: ROOT/results/baselines.json and ROOT/results/screening.txt.
set -euo pipefail

ROOT="$(realpath -m "${1:?usage: screen.sh ROOT [GPU] [--dry-run]}")"
GPU="${2:-0}"
DRY_RUN=0; [[ "${3:-}" == "--dry-run" ]] && DRY_RUN=1
CHECKOUT="$ROOT/aka-kf"   # any arm's checkout; screening runs no agent
OUT="wiki_ab_results/baselines.json"

measure=(bash src/scripts/docker_benchmark.sh exec python3 wiki_ab/screen_baselines.py
         --manifest wiki_ab/manifest.yaml --out "$OUT" --run-directory /tmp/wiki-ab-screen
         --rounds "${ROUNDS:-5}")
select=(bash src/scripts/docker_benchmark.sh exec python3 wiki_ab/roofline.py
        --aka . --manifest wiki_ab/manifest.yaml --baselines "$OUT")

[[ -f "$CHECKOUT/wiki_ab/screen_baselines.py" ]] || { echo "ERROR: run setup_node.sh first" >&2; exit 1; }
if [[ "$DRY_RUN" == "1" ]]; then
    printf 'cd %s && AKA_VISIBLE_GPU=%s %s\n' "$CHECKOUT" "$GPU" "${measure[*]}"
    printf 'cd %s && AKA_VISIBLE_GPU=%s %s\n' "$CHECKOUT" "$GPU" "${select[*]}"
    exit 0
fi
busy="$(rocm-smi --showpidgpus 2>/dev/null | awk '/is using [1-9]/ {grab=1; next} grab && /^[0-9 ]+$/ {print; grab=0}' | tr ' ' '\n' | sort -u)"
if [[ "${ALLOW_BUSY:-0}" != "1" ]] && grep -qx "$GPU" <<< "$busy"; then
    echo "ERROR: GPU $GPU is in use by another process (rocm-smi --showpidgpus); ALLOW_BUSY=1 overrides" >&2
    exit 1
fi
mkdir -p "$CHECKOUT/wiki_ab_results" "$ROOT/results"
(cd "$CHECKOUT" && AKA_VISIBLE_GPU="$GPU" "${measure[@]}")
cp "$CHECKOUT/$OUT" "$ROOT/results/baselines.json"
(cd "$CHECKOUT" && AKA_VISIBLE_GPU="$GPU" "${select[@]}") | tee "$ROOT/results/screening.txt"
