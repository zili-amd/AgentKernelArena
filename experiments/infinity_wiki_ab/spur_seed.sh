#!/usr/bin/env bash
# Run the mxfp4_seed phase on a provisioned Spur allocation: kf alone, every (task, repeat) on its own GPU at once.
#   spur_seed.sh ROOT ENV_FILE
# ENV_FILE holds the LLM gateway credentials; it is sourced, never printed.
set -euo pipefail

ROOT="$(realpath -m "${1:?usage: spur_seed.sh ROOT ENV_FILE}")"
ENV_FILE="${2:?usage: spur_seed.sh ROOT ENV_FILE}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE=mxfp4_seed
read -r -a GPUS <<< "$(tr ',' ' ' <<< "${SPUR_JOB_GPUS:-${SLURM_JOB_GPUS:-0 1 2 3 4 5 6 7}}")"
mapfile -t TASKS < <(python3 -c "import sys, yaml; print('\n'.join(yaml.safe_load(open(sys.argv[1]))['phases'][sys.argv[2]]['tasks']))" "$HERE/manifest.yaml" "$PHASE")
REPEATS="$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['phases'][sys.argv[2]]['repeats'])" "$HERE/manifest.yaml" "$PHASE")"
(( ${#TASKS[@]} * REPEATS <= ${#GPUS[@]} )) || { echo "ERROR: ${#TASKS[@]} tasks x $REPEATS repeats need more than ${#GPUS[@]} GPUs" >&2; exit 1; }

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a
export AKA_GPU_ARCH=gfx950 AKA_SKIP_DEV_MEM=1 ARMS=kf
mkdir -p "$ROOT/logs"

slot=0 pids=()
for (( r = 0; r < REPEATS; r++ )); do
    for task in "${TASKS[@]}"; do
        bash "$HERE/launch_wave.sh" "$ROOT" "$PHASE" "$task" "$r" "${GPUS[$slot]}" &
        pids+=("$!")
        echo "$(date -u +%FT%TZ) $PHASE $task r$r on GPU ${GPUS[$slot]}"
        slot=$((slot + 1))
        sleep 20
    done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
echo "$(date -u +%FT%TZ) $PHASE done; $failed launch(es) failed"
