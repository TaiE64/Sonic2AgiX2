#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEMO_PYTHON=${DEMO_PYTHON:-python3}

exec env -u MUJOCO_GL "$DEMO_PYTHON" "$DEMO_DIR/planner_teleop_x2.py" "$@"
