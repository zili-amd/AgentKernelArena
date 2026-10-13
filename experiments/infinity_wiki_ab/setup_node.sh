#!/usr/bin/env bash
# Unpack dist/ on the GPU node into ROOT. Safe to rerun: existing checkouts are
# fast-forwarded to their pins from the bundles and verified, never recreated.
#   setup_node.sh ROOT              clone and verify every checkout
#   setup_node.sh ROOT --provision  also install the forge engine / GEAK SDK in each arm's checkout
#   setup_node.sh ROOT --check      print docker, GPU and process state (read-only)
# The arms are the ones pack.sh was run for (dist/arms.txt).
set -euo pipefail

ROOT="$(realpath -m "${1:?usage: setup_node.sh ROOT [--provision|--check]}")"
MODE="${2:-}"
DIST="$ROOT/dist"
mapfile -t ARMS < "$DIST/arms.txt"
NODE_VERSION=v22.23.3
CLAUDE_CODE_VERSION=2.1.296

pin() { awk -v n="$1" '$1 == n {print $2}' "$DIST/pins.txt"; }

checkout() {
    local name="$1" dest="$2" want
    want="$(pin "$name")"
    if [[ ! -d "$dest/.git" ]]; then
        git clone -q -b ab-pin "$DIST/$name.bundle" "$dest"
    elif [[ "$(git -C "$dest" rev-parse HEAD)" != "$want" ]]; then
        git -C "$dest" pull -q --ff-only "$DIST/$name.bundle" ab-pin
    fi
    [[ "$(git -C "$dest" rev-parse HEAD)" == "$want" ]] || { echo "ERROR: $dest is not at $want" >&2; exit 1; }
    [[ -z "$(git -C "$dest" status --porcelain --untracked-files=no)" ]] || { echo "ERROR: $dest has local changes" >&2; exit 1; }
    echo "ok $dest @ ${want:0:12}"
}

if [[ "$MODE" == "--check" ]]; then
    docker --version
    rocm-smi --showid || true
    rocm-smi --showuse || true
    echo "== GPU processes (must be empty before a launch)"
    rocm-smi --showpids || true
    docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
    exit 0
fi

mkdir -p "$ROOT/src" "$ROOT/logs" "$ROOT/held_out"
checkout hyperloom "$ROOT/src/hyperloom"
# The pins, not the bundles present, say what this pack needs: dist/ can hold an older pack's files.
[[ -n "$(pin geak)" ]] && checkout geak "$ROOT/src/geak"
# Pin wiki goes to src/infinity-wiki, pin wiki_branch to src/infinity-wiki-branch; launch_wave.sh maps arms the same way.
for name in $(awk '$1 ~ /^wiki/ {print $1}' "$DIST/pins.txt"); do
    checkout "$name" "$ROOT/src/infinity-${name//_/-}"
done
# Each checkout gets the experiment files under wiki_ab/ (configs, manifest and
# tools) because a container only sees its own checkout. Results stay outside it.
for arm in "${ARMS[@]}"; do
    checkout aka "$ROOT/aka-$arm"
    rm -rf "$ROOT/aka-$arm/wiki_ab"
    cp -r "$DIST/experiment" "$ROOT/aka-$arm/wiki_ab"
    # The runtime image does not trust the private CA that signs the LLM gateway;
    # launch_wave.sh points NODE_EXTRA_CA_CERTS and friends at the host's bundle.
    cp /etc/ssl/certs/ca-certificates.crt "$ROOT/aka-$arm/wiki_ab/ca-certificates.crt"
    for ignored in wiki_ab/ wiki_ab_results/; do
        grep -qx "$ignored" "$ROOT/aka-$arm/.git/info/exclude" \
            || echo "$ignored" >> "$ROOT/aka-$arm/.git/info/exclude"
    done
done

if [[ "$MODE" == "--provision" ]]; then
    # The runtime image AKA selects for gfx942 at the pinned commit.
    image="$(sed -nE 's/^DEFAULT_DOCKER_IMAGE_GFX942="\$\{AKA_DOCKER_IMAGE_GFX942:-([^}]+)\}"/\1/p' \
        "$ROOT/aka-${ARMS[0]}/src/scripts/docker_benchmark.sh")"
    docker image inspect "$image" >/dev/null 2>&1 || docker pull -q "$image"
    echo "image $image $(docker image inspect "$image" --format '{{.Id}}')"
    # Pinned agent CLI in a private Node prefix (mounted by AKA via AKA_NODE_PREFIX).
    if [[ ! -x "$ROOT/agent-cli/bin/claude" ]]; then
        node_file="node-$NODE_VERSION-linux-x64.tar.xz"
        mkdir -p "$ROOT/agent-cli" "$ROOT/tmp"
        curl -fsSL -o "$ROOT/tmp/$node_file" "https://nodejs.org/dist/$NODE_VERSION/$node_file"
        (cd "$ROOT/tmp" && curl -fsSL "https://nodejs.org/dist/$NODE_VERSION/SHASUMS256.txt" \
            | grep " $node_file\$" | sha256sum -c -)
        tar -xJf "$ROOT/tmp/$node_file" -C "$ROOT/agent-cli" --strip-components=1
        PATH="$ROOT/agent-cli/bin:$PATH" npm install -g --prefix "$ROOT/agent-cli" --no-fund --no-audit \
            "@anthropic-ai/claude-code@$CLAUDE_CODE_VERSION"
    fi
    echo "agent cli $("$ROOT/agent-cli/bin/claude" --version)"
    # Empty Claude state for the runner to mount instead of the node's own ~/.claude.
    mkdir -p "$ROOT/claude-home/.claude"
    [[ -f "$ROOT/claude-home/.claude.json" ]] || echo '{"hasCompletedOnboarding": true}' > "$ROOT/claude-home/.claude.json"
    for arm in "${ARMS[@]}"; do
        case "$arm" in
            kf|kf_wiki) (cd "$ROOT/aka-$arm" && HYPERLOOM_ROOT="$ROOT/src/hyperloom" make docker-setup-forge) ;;
            geak|geak_wiki) (cd "$ROOT/aka-$arm" && GEAK_HOME="$ROOT/src/geak" make docker-setup-geak) ;;
        esac
    done
fi
echo "node ready under $ROOT"
