#!/usr/bin/env bash
# Launch the ARMS of one (task, repeat) together, one GPU each.
#   launch_wave.sh ROOT PHASE TASK REPEAT GPU_BASE [--dry-run]
# Arm i runs on GPU GPU_BASE + (i + REPEAT + ROTATE) % N for N launched arms (Latin square across repeats).
# Credentials come from the caller's environment and are forwarded by name only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(realpath -m "${1:?usage: launch_wave.sh ROOT PHASE TASK REPEAT GPU_BASE [--dry-run]}")"
PHASE="${2:?phase}"; TASK="${3:?task}"; REPEAT="${4:?repeat}"; GPU_BASE="${5:?gpu base}"
DRY_RUN=0; [[ "${6:-}" == "--dry-run" ]] && DRY_RUN=1
# Extra GPU rotation, so that tasks sharing a repeat do not all put each arm in the same position.
ROTATE="${ROTATE:-0}"
# ARMS="kf" launches a subset (e.g. reference runs without the wiki).
read -r -a ARMS <<< "${ARMS:-kf kf_wiki}"
SLUG="${TASK//\//__}"
LOG_DIR="$ROOT/logs/$PHASE/$SLUG/r$REPEAT"
LEDGER="$ROOT/invocations.jsonl"
# Model overrides the runner forwards into GEAK containers; unset so every arm uses agent.model.
UNSET=(ANTHROPIC_MODEL ANTHROPIC_DEFAULT_OPUS_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL
       ANTHROPIC_DEFAULT_HAIKU_MODEL CLAUDE_CODE_SUBAGENT_MODEL
       KB_STORE_URL KB_STORE_TOKEN KNOWLEDGE_STORE_MODE KNOWLEDGE_LOCAL_ROOT AKA_KNOWLEDGE_WIKI_ROOT)

for arm in "${ARMS[@]}"; do
    [[ -f "$ROOT/aka-$arm/wiki_ab/configs/$PHASE/$SLUG/$arm.yaml" ]] \
        || { echo "ERROR: no config for $arm/$PHASE/$SLUG; run setup_node.sh first" >&2; exit 1; }
    [[ -s "$ROOT/aka-$arm/wiki_ab/ca-certificates.crt" ]] \
        || { echo "ERROR: no CA bundle in aka-$arm/wiki_ab; rerun setup_node.sh" >&2; exit 1; }
done
[[ -x "$ROOT/agent-cli/bin/claude" && -f "$ROOT/claude-home/.claude.json" && -d "$ROOT/claude-home/.claude" ]] \
    || { echo "ERROR: run setup_node.sh --provision (agent CLI and empty Claude home)" >&2; exit 1; }
[[ -z "$(ls -A "$ROOT/claude-home/.claude")" ]] \
    || { echo "ERROR: $ROOT/claude-home/.claude must stay empty" >&2; exit 1; }
if [[ "$DRY_RUN" == "0" ]]; then
    [[ -n "${ANTHROPIC_AUTH_TOKEN:-}${ANTHROPIC_API_KEY:-}${CLAUDE_CODE_OAUTH_TOKEN:-}" ]] \
        || { echo "ERROR: no LLM gateway credentials in the environment" >&2; exit 1; }
    busy="$(rocm-smi --showpidgpus 2>/dev/null | awk '/is using [1-9]/ {grab=1; next} grab && /^[0-9 ]+$/ {print; grab=0}' | tr ' ' '\n' | sort -u)"
    for i in "${!ARMS[@]}"; do
        gpu=$(( GPU_BASE + (i + REPEAT + ROTATE) % ${#ARMS[@]} ))
        if [[ "${ALLOW_BUSY:-0}" != "1" ]] && grep -qx "$gpu" <<< "$busy"; then
            echo "ERROR: GPU $gpu is in use by another process (rocm-smi --showpidgpus); ALLOW_BUSY=1 overrides" >&2
            exit 1
        fi
    done
    mkdir -p "$LOG_DIR"
fi

record() {  # one JSON line per event; all values are plain tokens
    printf '{"event":"%s","phase":"%s","task":"%s","repeat":%s,"arm":"%s","gpu":%s,"run_suffix":"%s","time":"%s"%s}\n' \
        "$1" "$PHASE" "$TASK" "$REPEAT" "$2" "$3" "$4" "$(date -u +%FT%TZ)" "${5:-}" >> "$LEDGER"
}

pids=()
for i in "${!ARMS[@]}"; do
    arm="${ARMS[$i]}"
    gpu=$(( GPU_BASE + (i + REPEAT + ROTATE) % ${#ARMS[@]} ))
    tag="${PHASE}-${TASK##*/}-r${REPEAT}-${arm}"
    env_args=()
    for name in "${UNSET[@]}"; do env_args+=(-u "$name"); done
    # HOME lives in this arm's own checkout (mounted at /workspace) so the agent
    # session transcripts survive the container for the leakage audit.
    # HOME points the runner at an empty Claude state (ROOT/claude-home): a node's
    # own ~/.claude can carry MCP servers, instructions or skills into the agent.
    env_args+=(AKA_VISIBLE_GPU="$gpu" AKA_WORKER_ID="$arm" AGENT_HOME_ISOLATION=1
               AKA_CONTAINER_HOME="/workspace/wiki_ab_results/home-$tag" AKA_CACHE_SUFFIX="$tag"
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 AKA_NODE_PREFIX="$ROOT/agent-cli"
               HOME="$ROOT/claude-home"
               NODE_EXTRA_CA_CERTS=/workspace/wiki_ab/ca-certificates.crt
               SSL_CERT_FILE=/workspace/wiki_ab/ca-certificates.crt
               REQUESTS_CA_BUNDLE=/workspace/wiki_ab/ca-certificates.crt)
    case "$arm" in
        kf*) hl_pin="$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['arms'][sys.argv[2]].get('hyperloom', 'hyperloom'))" "$HERE/manifest.yaml" "$arm")"
             env_args+=(HYPERLOOM_ROOT="$ROOT/src/${hl_pin//_/-}") ;;
        geak*) env_args+=(GEAK_HOME="$ROOT/src/geak") ;;
    esac
    wiki_pin="$(python3 -c "import sys, yaml; v = yaml.safe_load(open(sys.argv[1]))['arms'][sys.argv[2]]['knowledge_wiki']
print(v if isinstance(v, str) else 'wiki' if v else '')" "$HERE/manifest.yaml" "$arm")"
    [[ -n "$wiki_pin" ]] && env_args+=(AKA_KNOWLEDGE_WIKI_ROOT="$ROOT/src/infinity-${wiki_pin//_/-}")
    command=(make docker-run CONFIG="wiki_ab/configs/$PHASE/$SLUG/$arm.yaml" RUN_ARGS="--run-suffix $tag")
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'GPU %s  %-9s cd %s && env %s %s\n' "$gpu" "$arm" "$ROOT/aka-$arm" "${env_args[*]}" "${command[*]}"
        continue
    fi
    mkdir -p "$ROOT/aka-$arm/wiki_ab_results"
    (
        record start "$arm" "$gpu" "$tag"
        status=0
        (cd "$ROOT/aka-$arm" && env "${env_args[@]}" "${command[@]}") > "$LOG_DIR/$arm.log" 2>&1 || status=$?
        record end "$arm" "$gpu" "$tag" ",\"exit_code\":$status"
    ) &
    pids+=("$!")
done

if [[ "$DRY_RUN" == "0" ]]; then
    echo "launched ${#pids[@]} arms for $PHASE $TASK r$REPEAT; logs in $LOG_DIR"
    wait "${pids[@]}"
    echo "finished $PHASE $TASK r$REPEAT; see $LEDGER"
    python3 "$HERE/make_record.py" "$ROOT" --phase "$PHASE" || echo "WARN: the $PHASE test record was not updated" >&2
fi
