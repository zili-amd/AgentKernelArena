#!/usr/bin/env bash
# Run the main phase (or PHASE, another manifest phase with tasks and repeats) in waves, every (task, repeat)
# with all of dist/arms.txt, waiting for each wave.
#   [PHASE=main] [TASKS_PREFIX="triton2triton/ ..."] run_main.sh ROOT [--dry-run]
# schedule.waves in the manifest (or the phase's own schedule, which overrides it) picks the wave shape:
#   by-repeat  each wave is one repeat of as many tasks as there are GPU bases; task j uses GPU_BASES[j].
#   by-task    each wave is one task with all its repeats at once, on the first free GPU bases;
#              the next task starts only after every run of the wave has ended.
#   by-slot    every (task, repeat) starts on the next free GPU base as soon as there is one, repeat by repeat
#              so every task's first repeats start first; a base stays held while its group runs.
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
SLOT_POLL_S="${SLOT_POLL_S:-60}"
SLOT_SETTLE_S="${SLOT_SETTLE_S:-20}"
ARMS="$(tr '\n' ' ' < "$ROOT/dist/arms.txt")"
export ARMS
read -r -a ARM_LIST <<< "$ARMS"

PHASE="${PHASE:-main}"
[[ "$PHASE" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "ERROR: PHASE must name a manifest phase" >&2; exit 2; }
mapfile -t ALL_TASKS < <(python3 -c "import sys, yaml; print('\n'.join(yaml.safe_load(open(sys.argv[1]))['phases'][sys.argv[2]]['tasks']))" "$MANIFEST" "$PHASE")
REPEATS="$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['phases'][sys.argv[2]]['repeats'])" "$MANIFEST" "$PHASE")"
SCHEDULE="import sys, yaml; m = yaml.safe_load(open(sys.argv[1])); s = {**m['schedule'], **(m['phases'][sys.argv[2]].get('schedule') or {})}"
WAVES="$(python3 -c "$SCHEDULE; print(s.get('waves', 'by-repeat'))" "$MANIFEST" "$PHASE")"
read -r -a GPU_BASES <<< "$(python3 -c "$SCHEDULE; print(*s['gpu_bases'])" "$MANIFEST" "$PHASE")"
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
(( ${#TASKS[@]} )) || { echo "ERROR: no $PHASE task matches TASKS_PREFIX=${TASKS_PREFIX:-}" >&2; exit 1; }
[[ "$WAVES" == by-repeat || "$WAVES" == by-task || "$WAVES" == by-slot ]] \
    || { echo "ERROR: schedule.waves must be by-repeat, by-task or by-slot" >&2; exit 1; }
[[ "$WAVES" != by-task ]] || (( REPEATS <= ${#GPU_BASES[@]} )) \
    || { echo "ERROR: $REPEATS repeats need as many GPU bases; the manifest has ${#GPU_BASES[@]}" >&2; exit 1; }

group() {  # repeat task base task-index
    local attempt
    for (( attempt = 1; attempt <= RETRIES; attempt++ )); do
        if ROTATE="$4" bash "$HERE/launch_wave.sh" "$ROOT" "$PHASE" "$2" "$1" "$3" $DRY_RUN; then
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

echo "$(date -u +%FT%TZ) $PHASE: $WAVES waves of ${TASKS[*]}; $REPEATS repeats; arms $ARMS"
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
elif [[ "$WAVES" == by-slot ]]; then
    declare -A held=()
    pids=() launched=0
    for (( r = 0; r < REPEATS; r++ )); do
        for (( t = 0; t < ${#TASKS[@]}; t++ )); do
            base=""
            for (( waited = 0; ; waited++ )); do
                for b in "${!held[@]}"; do kill -0 "${held[$b]}" 2>/dev/null || unset "held[$b]"; done
                if [[ -n "$DRY_RUN" ]]; then base="${GPU_BASES[$(( launched % ${#GPU_BASES[@]} ))]}"; break; fi
                # A base this scheduler launched on stays held until its group ends, even before its GPUs look busy.
                for b in $(free_bases); do [[ -z "${held[$b]:-}" ]] && { base="$b"; break; }; done
                [[ -n "$base" ]] && break
                (( waited % 10 )) || echo "$(date -u +%FT%TZ) r$r ${TASKS[$t]}: waiting for a free GPU base"
                sleep "$SLOT_POLL_S"
            done
            echo "$(date -u +%FT%TZ) slot: r$r ${TASKS[$t]} on GPU base $base"
            group "$r" "${TASKS[$t]}" "$base" "${TASK_INDEX[$t]}" &
            held[$base]=$!
            pids+=("$!")
            launched=$((launched + 1))
            [[ -n "$DRY_RUN" ]] || sleep "$SLOT_SETTLE_S"
        done
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
    echo "$(date -u +%FT%TZ) slots done; $failed group(s) skipped"
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
[[ -n "$DRY_RUN" ]] || python3 "$HERE/make_record.py" "$ROOT" --phase "$PHASE"
