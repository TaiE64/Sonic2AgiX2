#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
A2A_DIR=$(cd -- "$DEMO_DIR/.." && pwd)
REPO_DIR=$(cd -- "$A2A_DIR/.." && pwd)

POLICY_ONNX=${1:-}
PLANNER_ONNX=${2:-}
BUNDLE_OUTPUT=${3:-$REPO_DIR/walking_demo_v21_s12000.tar.gz}

if [[ -z "$POLICY_ONNX" || -z "$PLANNER_ONNX" ]]; then
    echo "usage: $0 <model_step_012000_g1.onnx> <planner_sonic.onnx> [output.tar.gz]" >&2
    exit 2
fi
for artifact in "$POLICY_ONNX" "$PLANNER_ONNX"; do
    if [[ ! -f "$artifact" ]]; then
        echo "missing model: $artifact" >&2
        exit 1
    fi
done

POLICY_SHA=$(sha256sum "$POLICY_ONNX" | cut -d' ' -f1)
PLANNER_SHA=$(sha256sum "$PLANNER_ONNX" | cut -d' ' -f1)
if [[ "$POLICY_SHA" != "9112bba8a90d97ae26cccaabd81678af749f7e83fdff241a7857844e0e3dad86" ]]; then
    echo "unexpected V21 step-12000 policy checksum: $POLICY_SHA" >&2
    exit 1
fi
if [[ "$PLANNER_SHA" != "39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea" ]]; then
    echo "unexpected SONIC planner checksum: $PLANNER_SHA" >&2
    exit 1
fi

BUNDLE_STAGE=$(mktemp -d)
trap 'rm -rf -- "$BUNDLE_STAGE"' EXIT
PACKAGE_DIR=$BUNDLE_STAGE/walking_demo
mkdir -p "$PACKAGE_DIR/Any2Any/sim2sim/models" "$PACKAGE_DIR/GMR/assets"

cp "$A2A_DIR/kinematic_alignment.py" "$PACKAGE_DIR/Any2Any/"
cp "$DEMO_DIR/planner_teleop_x2.py" "$DEMO_DIR/planner_lib.py" \
   "$DEMO_DIR/mujoco_sim2sim_x2.py" "$DEMO_DIR/onnx_block_order.py" \
   "$DEMO_DIR/runtime_constants.json" "$DEMO_DIR/requirements-walking.txt" \
   "$DEMO_DIR/run_walking_demo.sh" "$DEMO_DIR/README.md" \
   "$PACKAGE_DIR/Any2Any/sim2sim/"
cp -a "$DEMO_DIR/licenses" "$PACKAGE_DIR/Any2Any/sim2sim/"
cp -a "$REPO_DIR/GMR/assets/agibot_x2" "$PACKAGE_DIR/GMR/assets/"
cp "$REPO_DIR/GMR/LICENSE" "$PACKAGE_DIR/GMR/"
cp "$POLICY_ONNX" "$PACKAGE_DIR/Any2Any/sim2sim/models/model_step_012000_g1.onnx"
cp "$PLANNER_ONNX" "$PACKAGE_DIR/Any2Any/sim2sim/models/planner_sonic.onnx"

tar -C "$BUNDLE_STAGE" -czf "$BUNDLE_OUTPUT" walking_demo
echo "created: $BUNDLE_OUTPUT"
echo "recipient: cd walking_demo/Any2Any/sim2sim && python3 -m venv .venv && . .venv/bin/activate"
echo "           pip install -r requirements-walking.txt && ./run_walking_demo.sh --speed 0.5"
