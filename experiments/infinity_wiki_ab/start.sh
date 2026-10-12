#!/usr/bin/env bash
# Start a phase on a deployed machine, with the LLM gateway credentials from a local env file.
#   start.sh --host USER@HOST --root DIR --env-file FILE [--key FILE] pilot [--task T] [--repeat N] [--gpu-base B]
#   start.sh --host USER@HOST --root DIR --env-file FILE [--key FILE] main
# The env file's values travel over ssh stdin only; nothing is printed or put in argv.
# Pilot defaults: the manifest's pilot task, the next unused repeat, the first free GPU pair.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="" ROOT="" KEY="" ENV_FILE="" PHASE="" TASK="" REPEAT="" GPU_BASE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --root) ROOT="$2"; shift 2 ;;
        --key) KEY="$2"; shift 2 ;;
        --env-file) ENV_FILE="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --repeat) REPEAT="$2"; shift 2 ;;
        --gpu-base) GPU_BASE="$2"; shift 2 ;;
        pilot|main) PHASE="$1"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$HOST" && -n "$ROOT" && -n "$ENV_FILE" && -n "$PHASE" ]] \
    || { echo "usage: start.sh --host USER@HOST --root DIR --env-file FILE [--key FILE] pilot|main [...]" >&2; exit 2; }
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 ${KEY:+-i "$KEY"})

names="$(grep -oE '^(export[[:space:]]+)?[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE" | sed -E 's/^export[[:space:]]+//; s/=$//')"
grep -qx ANTHROPIC_BASE_URL <<< "$names" || { echo "ERROR: $ENV_FILE sets no ANTHROPIC_BASE_URL" >&2; exit 1; }
grep -qxE 'ANTHROPIC_API_KEY|ANTHROPIC_AUTH_TOKEN' <<< "$names" \
    || { echo "ERROR: $ENV_FILE sets neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN" >&2; exit 1; }
echo "credentials from $ENV_FILE: $(tr '\n' ' ' <<< "$names")"

if [[ "$PHASE" == "pilot" ]]; then
    TASK="${TASK:-$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['phases']['pilot']['tasks'][0])" "$HERE/manifest.yaml")}"
    # The launch itself refuses busy GPUs; this only picks a pair that is free now.
    picked="$("${SSH[@]}" "$HOST" bash -s -- "$ROOT" "$TASK" <<'EOF'
root="$1" task="$2"
repeat="$(python3 -c "import json, sys
used = {json.loads(l)['repeat'] for l in open(sys.argv[1]) if json.loads(l).get('phase') == 'pilot' and json.loads(l).get('task') == sys.argv[2]}
print(max(used) + 1 if used else 0)" "$root/invocations.jsonl" "$task" 2>/dev/null || echo 0)"
busy=" $(rocm-smi --showpidgpus 2>/dev/null | awk '/is using [1-9]/ {grab=1; next} grab && /^[0-9 ]+$/ {print; grab=0}' | tr '\n' ' ') "
count="$(rocm-smi --showid 2>/dev/null | grep -c 'GPU\[[0-9]*\].*Device ID')"
for (( base = 0; base + 1 < count; base += 2 )); do
    [[ "$busy" != *" $base "* && "$busy" != *" $((base + 1)) "* ]] && { echo "$repeat $base"; exit 0; }
done
echo "$repeat none"
EOF
)"
    REPEAT="${REPEAT:-${picked%% *}}"
    GPU_BASE="${GPU_BASE:-${picked##* }}"
    [[ "$GPU_BASE" != none ]] || { echo "ERROR: no free pair of adjacent GPUs on $HOST" >&2; exit 1; }
    slug="${TASK//\//__}"
    command="ARMS=\"\$(tr '\n' ' ' < $ROOT/dist/arms.txt)\" setsid nohup bash $ROOT/dist/experiment/launch_wave.sh $ROOT pilot $TASK $REPEAT $GPU_BASE > $ROOT/logs/pilot-$slug-r$REPEAT.launch.log 2>&1 < /dev/null &"
    echo "pilot: $TASK repeat $REPEAT on GPUs $GPU_BASE-$((GPU_BASE + 1))"
else
    command="setsid nohup bash $ROOT/dist/experiment/run_main.sh $ROOT > $ROOT/logs/main.log 2>&1 < /dev/null &"
    echo "main: every main task as a pair per wave; log $ROOT/logs/main.log"
fi

{ printf 'set -a\n'; cat "$ENV_FILE"; printf '\nset +a\nmkdir -p %s/logs\n' "$ROOT"; printf '%s\n' "$command"; echo "sleep 15; tail -n 3 $ROOT/invocations.jsonl"; } \
    | "${SSH[@]}" "$HOST" bash -s
