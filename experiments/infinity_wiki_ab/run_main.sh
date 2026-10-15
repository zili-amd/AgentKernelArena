#!/usr/bin/env bash
# Run the main phase in waves, every (task, repeat) with all of dist/arms.txt, waiting for each wave.
#   [TASKS_PREFIX="triton2triton/ ..."] run_main.sh ROOT [--dry-run]
# schedule.waves in the manifest picks the wave shape:
#   by-repeat  each wave is one repeat of as many tasks as there are GPU bases; task j uses GPU_BASES[j].
#   by-task    each wave is one task with all its repeats at once, on the first free GPU bases;
#              the next task starts only after every run of the wave has ended.
# TASKS_PREFIX keeps only the main tasks starting with one of its prefixes (one machine per task family).
# launch_wave.sh places arm i on base + (i + repeat + task index) % arms, the task index counted
# in the manifest's full list. A group whose GPUs are busy is retried every RETRY_S seconds up to
# RETRIES times, then skipped and logged; other users' processes are never touched.
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

mapfile -t ALL_TASKS < <(python3 -c "import sys, yaml; print('\n'.join(yaml.safe_load(open(sys.argv[1]))['phases']['main']['tasks']))" "$MANIFEST")
REPEATS="$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['phases']['main']['repeats'])" "$MANIFEST")"
WAVES="$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['schedule'].get('waves', 'by-repeat'))" "$MANIFEST")"
read -r -a GPU_BASES <<< "$(python3 -c "import sys, yaml; print(*yaml.safe_load(open(sys.argv[1]))['schedule']['gpu_bases'])" "$MANIFEST")"
for (( b = 1; b < ${#GPU_BASES[@]}; b++ )); do
    (( GPU_BASES[b] - GPU_BASES[b - 1] >= ${#ARM_LIST[@]} )) \
        || { echo "ERROR: GPU bases ${GPU_BASES[*]} overlap for ${#ARM_LIST[@]} arms per task" >&2; exit 1; }
done

# TASKS keeps the selected tasks; TASK_INDEX their position in the manifest, which fixes each arm's GPU rotation.
read -r -a PREFIXES <<< "${TASKS_PREFIX:-}"
TASKS=() TASK_INDEX=()
for (( j = 0; j < ${#ALL_TASKS[@]}; j++ )); do
    keep=$(( ${#PREFIXES[@]} == 0 ))
    for prefix in "${PREFIXES[@]}"; do [[ "${ALL_TASKS[$j]}" == "$prefix"* ]] && keep=1; done
    (( keep )) && { TASKS+=("${ALL_TASKS[$j]}"); TASK_INDEX+=("$j"); }
done
(( ${#TASKS[@]} )) || { echo "ERROR: no main task matches TASKS_PREFIX=${TASKS_PREFIX:-}" >&2; exit 1; }
[[ "$WAVES" == by-repeat || "$WAVES" == by-task ]] || { echo "ERROR: schedule.waves must be by-repeat or by-task" >&2; exit 1; }
[[ "$WAVES" == by-repeat ]] || (( REPEATS <= ${#GPU_BASES[@]} )) \
    || { echo "ERROR: $REPEATS repeats need as many GPU bases; the manifest has ${#GPU_BASES[@]}" >&2; exit 1; }

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

free_bases() {  # GPU bases whose GPUs no process uses, in manifest order
    local busy b g ok
    busy=" $(rocm-smi --showpidgpus 2>/dev/null | awk '/is using [1-9]/ {grab=1; next} grab && /^[0-9 ]+$/ {print; grab=0}' | tr '\n' ' ') "
    for b in "${GPU_BASES[@]}"; do
        ok=1
        for (( g = b; g < b + ${#ARM_LIST[@]}; g++ )); do [[ "$busy" == *" $g "* ]] && ok=0; done
        (( ok )) && printf '%s ' "$b"
    done
    return 0
}

echo "$(date -u +%FT%TZ) main: $WAVES waves of ${TASKS[*]}; $REPEATS repeats; arms $ARMS"
if [[ "$WAVES" == by-task ]]; then
    for (( t = 0; t < ${#TASKS[@]}; t++ )); do
        task="${TASKS[$t]}" bases=()
        for (( attempt = 1; attempt <= RETRIES; attempt++ )); do
            if [[ -n "$DRY_RUN" ]]; then bases=("${GPU_BASES[@]}"); else read -r -a bases <<< "$(free_bases)"; fi
            (( ${#bases[@]} >= REPEATS )) && break
            echo "$(date -u +%FT%TZ) $task: only ${#bases[@]} free GPU bases (attempt $attempt/$RETRIES); retrying in ${RETRY_S}s"
            bases=()
            sleep "$RETRY_S"
        done
        (( ${#bases[@]} >= REPEATS )) || { echo "$(date -u +%FT%TZ) $task: SKIPPED, too few free GPUs"; continue; }
        echo "$(date -u +%FT%TZ) wave $task: repeats 0-$(( REPEATS - 1 )) on GPU bases ${bases[*]:0:$REPEATS}"
        pids=()
        for (( r = 0; r < REPEATS; r++ )); do
            group "$r" "$task" "${bases[$r]}" "${TASK_INDEX[$t]}" &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
        echo "$(date -u +%FT%TZ) wave $task done; $failed group(s) skipped"
    done
else
    per_wave=${#GPU_BASES[@]}
    for (( r = 0; r < REPEATS; r++ )); do
        for (( first = 0; first < ${#TASKS[@]}; first += per_wave )); do
            echo "$(date -u +%FT%TZ) wave r$r tasks $first-$(( first + per_wave - 1 )): arms $ARMS"
            pids=()
            for (( j = first; j < first + per_wave && j < ${#TASKS[@]}; j++ )); do
                group "$r" "${TASKS[$j]}" "${GPU_BASES[$(( j - first ))]}" "${TASK_INDEX[$j]}" &
                pids+=("$!")
            done
            failed=0
            for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
            echo "$(date -u +%FT%TZ) wave r$r tasks $first-$(( first + per_wave - 1 )) done; $failed group(s) skipped"
        done
    done
fi
[[ -n "$DRY_RUN" ]] || python3 "$HERE/make_record.py" "$ROOT" --phase main
