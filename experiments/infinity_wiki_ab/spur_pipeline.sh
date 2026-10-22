#!/usr/bin/env bash
# The whole mxfp4 MoE A/B inside one exclusive Spur allocation, unattended:
# provision -> mxfp4_seed (kf alone) -> freeze each task's first correct port -> mxfp4_moe_ab (kf vs kf_wiki_guided).
#   spur_pipeline.sh ROOT ENV_FILE [STOP_FILE]
# STOP_FILE (touched at the end, success or not) tells the holder job to release the node.
set -uo pipefail

ROOT="$(realpath -m "${1:?usage: spur_pipeline.sh ROOT ENV_FILE [STOP_FILE]}")"
ENV_FILE="${2:?usage: spur_pipeline.sh ROOT ENV_FILE [STOP_FILE]}"
STOP_FILE="${3:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
release() { [[ -z "$STOP_FILE" ]] || touch "$STOP_FILE"; }
trap release EXIT
stamp() { echo "$(date -u +%FT%TZ) $*"; }

stamp "provision"
bash "$HERE/spur_provision.sh" "$ROOT" || { stamp "provision FAILED"; exit 1; }

stamp "mxfp4_seed"
bash "$HERE/spur_seed.sh" "$ROOT" "$ENV_FILE" || stamp "mxfp4_seed reported failures; freezing what exists"

stamp "freeze seeds"
mapfile -t SEEDED < <(python3 "$HERE/spur_freeze_seeds.py" "$ROOT")
(( ${#SEEDED[@]} )) || { stamp "no task has a seed; mxfp4_moe_ab not started"; exit 1; }
stamp "seeded: ${SEEDED[*]}"

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a
export AKA_GPU_ARCH=gfx950 AKA_SKIP_DEV_MEM=1
stamp "mxfp4_moe_ab"
PHASE=mxfp4_moe_ab TASKS_PREFIX="${SEEDED[*]}" bash "$HERE/run_main.sh" "$ROOT" > "$ROOT/logs/mxfp4_moe_ab.log" 2>&1
stamp "mxfp4_moe_ab ended rc=$?"
