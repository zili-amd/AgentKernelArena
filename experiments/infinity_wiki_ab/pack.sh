#!/usr/bin/env bash
# Build dist/: one git bundle per pinned repository plus this experiment's files.
# Run it where those repositories are reachable; a GPU node without GitHub
# credentials then needs only dist/.
#   pack.sh                      ARMS="kf kf_wiki" by default
# AgentKernelArena is pinned to the clean HEAD of the checkout this script is in.
# Hyperloom, the wiki and GEAK come from manifest.yaml pins; *_SRC may name a local
# checkout or another remote holding the same commit.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AKA_SRC="$(git -C "$HERE" rev-parse --show-toplevel)"
DIST="${DIST:-$HERE/dist}"
HYPERLOOM_SRC="${HYPERLOOM_SRC:-https://github.com/AMD-AGI/Hyperloom.git}"
WIKI_SRC="${WIKI_SRC:-https://github.com/AMD-BRAIN-Internal/infinity-wiki.git}"
GEAK_SRC="${GEAK_SRC:-https://github.com/AMD-AGI/GEAK.git}"
read -r -a ARMS <<< "${ARMS:-kf kf_wiki}"

pin() { python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['pins'][sys.argv[2]]['commit'])" "$HERE/manifest.yaml" "$1"; }

[[ -z "$(git -C "$AKA_SRC" status --porcelain --untracked-files=no)" ]] \
    || { echo "ERROR: commit or stash the changes in $AKA_SRC first; the pin is its HEAD" >&2; exit 1; }

# Bundle `commit` from `repo` under the branch name `ab-pin`, via a scratch mirror
# so that no ref is created in the source repository. A mirror also carries the
# source's remote-tracking refs, where a local checkout often holds the pin.
bundle() {
    local name="$1" repo="$2" commit="$3" scratch
    scratch="$(mktemp -d "$DIST/.scratch-XXXXXX")"
    git clone -q --mirror "$repo" "$scratch/repo.git"
    git -C "$scratch/repo.git" cat-file -e "$commit^{commit}" || { echo "missing $commit in $repo" >&2; exit 1; }
    git -C "$scratch/repo.git" branch -q -f ab-pin "$commit"
    git -C "$scratch/repo.git" bundle create -q "$DIST/$name.bundle" ab-pin
    git -C "$scratch/repo.git" bundle verify -q "$DIST/$name.bundle" >/dev/null
    rm -rf "$scratch"
    printf '%s %s\n' "$name" "$commit" >> "$DIST/pins.txt"
}

mkdir -p "$DIST"
rm -f "$DIST"/*.bundle "$DIST/pins.txt"
rm -rf "$DIST/experiment"
bundle aka "$AKA_SRC" "$(git -C "$AKA_SRC" rev-parse HEAD)"
bundle hyperloom "$HYPERLOOM_SRC" "$(pin hyperloom)"
bundle wiki "$WIKI_SRC" "$(pin wiki)"
if [[ " ${ARMS[*]} " == *" geak"* ]]; then
    bundle geak "$GEAK_SRC" "$(pin geak)"
fi

python3 "$HERE/make_configs.py" --out "$DIST/experiment/configs" >/dev/null
cp "$HERE"/*.yaml "$HERE"/*.sh "$HERE"/*.py "$HERE/TEST_RECORD_TEMPLATE.md" "$HERE/PROTOCOL.md" "$DIST/experiment/"
printf '%s\n' "${ARMS[@]}" > "$DIST/arms.txt"
du -sh "$DIST"/*.bundle
cat "$DIST/pins.txt"
echo "copy to the node: rsync -a $DIST/ <node>:<root>/dist/ && ssh <node> bash <root>/dist/experiment/setup_node.sh <root>"
