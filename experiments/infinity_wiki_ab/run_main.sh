#!/usr/bin/env bash
# Run the main phase in waves: each wave launches as many main tasks as there are
# GPU bases, every task with all of dist/arms.txt, and waits for all of them.
#   run_main.sh ROOT [--dry-run]
# Task j of a wave uses GPU base GPU_BASES[j]; launch_wave.sh places arm i on
# base + (i + repeat + task index) % arms. A group whose GPUs are busy is retried
# every RETRY_S seconds up to RETRIES times, then skipped and logged; other users'
# processes are never touched.
set -euo pipefail

ROOT="$(realpath -m "${1:?usage: run_main.sh ROOT [--dry-run]}")"
DRY_RUN="${2:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/manifest.yaml"
RETRY_S="${RETRY_S:-300}"
RETRIES="${RETRIES:-12}"
ARMS="$(tr '\n' ' ' < "$ROOT/dist/arms.txt")"
export ARMS
read -r -a ARM_LIST <<< "$ARMS"

mapfile -t TASKS < <(python3 -c "import sys, yaml; print('\n'.join(yaml.safe_load(open(sys.argv[1]))['phases']['main']['tasks']))" "$MANIFEST")
REPEATS="$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['phases']['main']['repeats'])" "$MANIFEST")"
read -r -a GPU_BASES <<< "$(python3 -c "import sys, yaml; print(*yaml.safe_load(open(sys.argv[1]))['schedule']['gpu_bases'])" "$MANIFEST")"
for (( b = 1; b < ${#GPU_BASES[@]}; b++ )); do
    (( GPU_BASES[b] - GPU_BASES[b - 1] >= ${#ARM_LIST[@]} )) \
        || { echo "ERROR: GPU bases ${GPU_BASES[*]} overlap for ${#ARM_LIST[@]} arms per task" >&2; exit 1; }
done

group() {  # repeat task base task-index
    local attempt
    for (( attempt = 1; attempt <= RETRIES; attempt++ )); do
        if ROTATE="$4" bash "$HERE/launch_wave.sh" "$ROOT" main "$2" "$1" "$3" $DRY_RUN; then
            return 0
        fi
        echo "$(date -u +%FT%TZ) r$1 $2: launch refused (attempt $attempt/$RETRIES); retrying in ${RETRY_S}s"
        sleep "$RETRY_S"
    done
    echo "$(date -u +%FT%TZ) r$1 $2: SKIPPED after $RETRIES refused launches"
    return 1
}

per_wave=${#GPU_BASES[@]}
for (( r = 0; r < REPEATS; r++ )); do
    for (( first = 0; first < ${#TASKS[@]}; first += per_wave )); do
        echo "$(date -u +%FT%TZ) wave r$r tasks $first-$(( first + per_wave - 1 )): arms $ARMS"
        pids=()
        for (( j = first; j < first + per_wave && j < ${#TASKS[@]}; j++ )); do
            group "$r" "${TASKS[$j]}" "${GPU_BASES[$(( j - first ))]}" "$j" &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
        echo "$(date -u +%FT%TZ) wave r$r tasks $first-$(( first + per_wave - 1 )) done; $failed group(s) skipped"
    done
done
[[ -n "$DRY_RUN" ]] || python3 "$HERE/make_record.py" "$ROOT" --phase main
