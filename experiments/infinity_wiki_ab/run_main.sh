#!/usr/bin/env bash
# Run the main phase: one wave per repeat, each wave launching every main task as
# a (kf, kf_wiki) pair on its own GPU pair, then waiting for all of them.
#   run_main.sh ROOT [--dry-run]
# Task j of repeat r gets GPU base GPU_BASES[(j + r) % 4]; launch_wave.sh rotates the
# two arms within the pair. A pair whose GPUs are busy is retried every RETRY_S
# seconds up to RETRIES times, then skipped and logged; other users' processes are
# never touched.
set -euo pipefail

ROOT="$(realpath -m "${1:?usage: run_main.sh ROOT [--dry-run]}")"
DRY_RUN="${2:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/manifest.yaml"
RETRY_S="${RETRY_S:-300}"
RETRIES="${RETRIES:-12}"
export ARMS="kf kf_wiki"

mapfile -t TASKS < <(python3 -c "import sys, yaml; print('\n'.join(yaml.safe_load(open(sys.argv[1]))['phases']['main']['tasks']))" "$MANIFEST")
REPEATS="$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['phases']['main']['repeats'])" "$MANIFEST")"
read -r -a GPU_BASES <<< "$(python3 -c "import sys, yaml; print(*yaml.safe_load(open(sys.argv[1]))['schedule']['gpu_bases'])" "$MANIFEST")"
(( ${#TASKS[@]} <= ${#GPU_BASES[@]} )) || { echo "ERROR: more tasks than GPU pairs in one wave" >&2; exit 1; }

pair() {  # repeat task base
    local attempt
    for (( attempt = 1; attempt <= RETRIES; attempt++ )); do
        if bash "$HERE/launch_wave.sh" "$ROOT" main "$2" "$1" "$3" $DRY_RUN; then
            return 0
        fi
        echo "$(date -u +%FT%TZ) r$1 $2: launch refused (attempt $attempt/$RETRIES); retrying in ${RETRY_S}s"
        sleep "$RETRY_S"
    done
    echo "$(date -u +%FT%TZ) r$1 $2: SKIPPED after $RETRIES refused launches"
    return 1
}

for (( r = 0; r < REPEATS; r++ )); do
    echo "$(date -u +%FT%TZ) wave r$r: ${#TASKS[@]} pairs"
    pids=()
    for j in "${!TASKS[@]}"; do
        pair "$r" "${TASKS[$j]}" "${GPU_BASES[$(( (j + r) % ${#GPU_BASES[@]} ))]}" &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
    echo "$(date -u +%FT%TZ) wave r$r done; $failed pair(s) skipped"
done
