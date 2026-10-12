#!/usr/bin/env bash
# Set up a GPU machine for the experiment from the three local repositories.
#   deploy.sh --host USER@HOST --root DIR [--key FILE] [--arms "kf kf_wiki"] [--dry-run]
# The repositories sit side by side in one directory: AgentKernelArena (this checkout,
# on infinity-wiki-AB-test and clean), Hyperloom and infinity-wiki. HYPERLOOM_DIR and
# WIKI_DIR override the last two. The machine needs no GitHub access.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AKA_DIR="$(git -C "$HERE" rev-parse --show-toplevel)"
WORKSPACE="$(dirname "$AKA_DIR")"
HYPERLOOM_DIR="${HYPERLOOM_DIR:-$WORKSPACE/Hyperloom}"
WIKI_DIR="${WIKI_DIR:-$WORKSPACE/infinity-wiki}"
HOST="" ROOT="" KEY="" ARMS="kf kf_wiki" DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --root) ROOT="$2"; shift 2 ;;
        --key) KEY="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$HOST" && -n "$ROOT" ]] || { echo "usage: deploy.sh --host USER@HOST --root DIR [--key FILE] [--arms ARMS] [--dry-run]" >&2; exit 2; }
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 ${KEY:+-i "$KEY"})
pin() { python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['pins'][sys.argv[2]]['commit'])" "$HERE/manifest.yaml" "$1"; }

echo "== local repositories under $WORKSPACE"
for repo in "$AKA_DIR" "$HYPERLOOM_DIR" "$WIKI_DIR"; do
    [[ -d "$repo/.git" || -f "$repo/.git" ]] || { echo "ERROR: $repo is not a git checkout" >&2; exit 1; }
done
[[ "$(git -C "$AKA_DIR" branch --show-current)" == "infinity-wiki-AB-test" ]] \
    || echo "WARN: $AKA_DIR is not on infinity-wiki-AB-test; its HEAD is what gets pinned"
for spec in "hyperloom:$HYPERLOOM_DIR" "wiki:$WIKI_DIR"; do
    name="${spec%%:*}" repo="${spec#*:}" commit="$(pin "${spec%%:*}")"
    if ! git -C "$repo" cat-file -e "$commit^{commit}" 2>/dev/null; then
        echo "  $name: fetching $commit into $repo"
        git -C "$repo" fetch -q origin "$commit" 2>/dev/null || git -C "$repo" fetch -q --all
    fi
    git -C "$repo" cat-file -e "$commit^{commit}" || { echo "ERROR: $repo lacks the $name pin $commit" >&2; exit 1; }
    echo "  $name ${commit:0:12} found in $repo"
done
echo "  aka $(git -C "$AKA_DIR" rev-parse --short=12 HEAD) (HEAD of $AKA_DIR)"

echo "== pack"
ARMS="$ARMS" HYPERLOOM_SRC="$HYPERLOOM_DIR" WIKI_SRC="$WIKI_DIR" bash "$HERE/pack.sh" | tail -n 4

echo "== remote preflight on $HOST"
preflight='
set -u
root="$1"; fail=0
arch="$(rocminfo 2>/dev/null | grep -o -m1 "gfx[0-9a-f]*" || true)"
echo "gpu arch: ${arch:-unknown}"; [[ "$arch" == gfx942 ]] || { echo "FAIL: this experiment needs MI300X (gfx942)"; fail=1; }
echo "gpus: $(rocm-smi --showid 2>/dev/null | grep -c "GPU\[[0-9]*\].*Device ID" || echo 0)"
busy="$(rocm-smi --showpidgpus 2>/dev/null | awk "/is using [1-9]/ {grab=1; next} grab && /^[0-9 ]+\$/ {print; grab=0}" | tr " " "\n" | sort -un | tr "\n" " ")"
echo "gpus in use by other processes: ${busy:-none}"
docker info >/dev/null 2>&1 && echo "docker: ok" || { echo "FAIL: docker is not usable by $(id -un)"; fail=1; }
python3 -c "import yaml" 2>/dev/null && echo "python3 + PyYAML: ok" || { echo "FAIL: python3 with PyYAML is required"; fail=1; }
[[ -s /etc/ssl/certs/ca-certificates.crt ]] && echo "CA bundle: ok" || { echo "FAIL: /etc/ssl/certs/ca-certificates.crt is missing"; fail=1; }
mkdir -p "$root" && echo "free disk under $root: $(df -BG --output=avail "$root" | tail -1 | tr -d " ")"
if [[ -f "$root/invocations.jsonl" ]]; then
    running="$(python3 -c "import json,sys
s,e=set(),set()
for l in open(sys.argv[1]):
    r=json.loads(l); (s if r[\"event\"]==\"start\" else e).add(r[\"run_suffix\"])
print(\" \".join(sorted(s-e)))" "$root/invocations.jsonl")"
    [[ -z "$running" ]] || { echo "FAIL: runs still in progress, which use these checkouts: $running"; fail=1; }
fi
echo "other containers: $(docker ps --format "{{.Names}}" 2>/dev/null | wc -l)"
exit $fail
'
if [[ "$DRY_RUN" == "1" ]]; then
    echo "(dry run) ${SSH[*]} $HOST bash -s -- $ROOT  < preflight"
    echo "(dry run) rsync -a -e '${SSH[*]}' $HERE/dist/ $HOST:$ROOT/dist/"
    echo "(dry run) ${SSH[*]} $HOST bash $ROOT/dist/experiment/setup_node.sh $ROOT --provision"
    exit 0
fi
"${SSH[@]}" "$HOST" bash -s -- "$ROOT" <<< "$preflight" || { echo "ERROR: remote preflight failed" >&2; exit 1; }

echo "== copy dist/ to $HOST:$ROOT/dist/"
rsync -a -e "${SSH[*]}" "$HERE/dist/" "$HOST:$ROOT/dist/"

echo "== set up and provision (first run pulls the runtime image and installs the agent CLI)"
"${SSH[@]}" "$HOST" bash "$ROOT/dist/experiment/setup_node.sh" "$ROOT" --provision < /dev/null
"${SSH[@]}" "$HOST" bash "$ROOT/dist/experiment/setup_node.sh" "$ROOT" --check < /dev/null | tail -n 12
echo "deployed: $(tr '\n' ' ' < "$HERE/dist/pins.txt")to $HOST:$ROOT"
