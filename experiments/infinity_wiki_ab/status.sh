#!/usr/bin/env bash
# Report a phase's runs on a deployed machine, regenerate its test record, and optionally copy it back.
#   status.sh --host USER@HOST --root DIR [--key FILE] [--phase pilot|main] [--env-file FILE] [--fetch LOCAL_DIR]
# --env-file contributes only the gateway's host name, so the audit does not flag it.
set -euo pipefail

HOST="" ROOT="" KEY="" PHASE="pilot" ENV_FILE="" FETCH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --root) ROOT="$2"; shift 2 ;;
        --key) KEY="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --env-file) ENV_FILE="$2"; shift 2 ;;
        --fetch) FETCH="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$HOST" && -n "$ROOT" ]] || { echo "usage: status.sh --host USER@HOST --root DIR [--phase P] [--env-file F] [--fetch DIR]" >&2; exit 2; }
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 ${KEY:+-i "$KEY"})
gateway=""
if [[ -n "$ENV_FILE" ]]; then
    gateway="$(grep -E '^(export[[:space:]]+)?ANTHROPIC_BASE_URL=' "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d "\"'" \
        | python3 -c "import sys, urllib.parse; print(urllib.parse.urlparse(sys.stdin.read().strip()).hostname or '')")"
fi

"${SSH[@]}" "$HOST" bash -s -- "$ROOT" "$PHASE" "$gateway" <<'EOF'
root="$1" phase="$2" gateway="$3"
python3 - "$root" "$phase" <<'PY'
import json, pathlib, re, sys
root, phase = pathlib.Path(sys.argv[1]), sys.argv[2]
path = root / "invocations.jsonl"
ledger = [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
starts = {r["run_suffix"]: r for r in ledger if r.get("phase") == phase and r["event"] == "start"}
ends = {r["run_suffix"]: r for r in ledger if r.get("phase") == phase and r["event"] == "end"}
marker = re.compile(r"Iteration \d+|\[KEEP\]|\[REVERT|\[prepare\]|Knowledge base:|FORGE_RESULT")
for suffix, run in starts.items():
    end = ends.get(suffix)
    state = f"finished exit={end['exit_code']} {end['time']}" if end else f"running since {run['time']}"
    log = root / "logs" / phase / run["task"].replace("/", "__") / f"r{run['repeat']}" / f"{run['arm']}.log"
    lines = [l for l in log.read_text(errors="replace").splitlines() if marker.search(l)] if log.is_file() else []
    latest = lines[-1].split("[FORGE]")[-1].strip()[:140] if lines else ""
    calls = sum(1 for log_path in root.glob(f"aka-{run['arm']}/workspace_*/run_*_{suffix}/*-forge-*/knowledge_wiki_calls.jsonl")
                for line in log_path.read_text().splitlines() if line.strip())
    print(f"{suffix}: gpu {run['gpu']}, {state}, knowledge base calls {calls}\n    {latest}")
print(f"{len(ends)}/{len(starts)} {phase} runs finished")
PY
ANTHROPIC_BASE_URL="${gateway:+https://$gateway}" python3 "$root/dist/experiment/make_record.py" "$root" --phase "$phase"
EOF

if [[ -n "$FETCH" ]]; then
    mkdir -p "$FETCH"
    rsync -a -e "${SSH[*]}" "$HOST:$ROOT/results/$PHASE-record.md" "$FETCH/"
    echo "record copied to $FETCH/$PHASE-record.md"
fi
