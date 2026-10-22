#!/usr/bin/env bash
# Provision a Spur (MI355X) allocation for the mxfp4 phases, after setup_node.sh ROOT has made the checkouts on the
# shared home and the agent CLI is in ROOT/agent-cli. Runs on the allocated node: Docker images are per node.
#   spur_provision.sh ROOT
# Builds the non-root Aiter image (the runs are the Spur user, not root) and installs each arm's forge engine in it.
set -euo pipefail

ROOT="$(realpath -m "${1:?usage: spur_provision.sh ROOT}")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mapfile -t ARMS < "$ROOT/dist/arms.txt"
BASE_IMAGE="$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['phases']['mi300_aiter']['docker_image'])" "$HERE/manifest.yaml")"
IMAGE="$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['phases']['mxfp4_seed']['docker_image'])" "$HERE/manifest.yaml")"
GPU="$(cut -d, -f1 <<< "${SPUR_JOB_GPUS:-${SLURM_JOB_GPUS:-0}}")"

docker info >/dev/null 2>&1 || { echo "ERROR: docker is not usable on $(hostname)" >&2; exit 1; }
[[ -x "$ROOT/agent-cli/bin/claude" ]] || { echo "ERROR: no agent CLI in $ROOT/agent-cli" >&2; exit 1; }
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull -q "$BASE_IMAGE"
docker build -q --build-arg BASE="$BASE_IMAGE" -t "$IMAGE" - < "$HERE/aiter-nonroot.Dockerfile"
echo "image $IMAGE $(docker image inspect "$IMAGE" --format '{{.Id}}') from $BASE_IMAGE"

export AKA_GPU_ARCH=gfx950 AKA_SKIP_DEV_MEM=1 AKA_DOCKER_IMAGE="$IMAGE" AKA_VISIBLE_GPU="$GPU"
for arm in "${ARMS[@]}"; do
    hl="$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['arms'][sys.argv[2]].get('hyperloom', 'hyperloom'))" "$HERE/manifest.yaml" "$arm")"
    (cd "$ROOT/aka-$arm" && HYPERLOOM_ROOT="$ROOT/src/${hl//_/-}" make docker-setup-forge)
done
echo "provisioned $(hostname) for ${ARMS[*]}"
