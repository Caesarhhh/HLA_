#!/usr/bin/env bash
set -euo pipefail

# Run directly on an already allocated GPU node. This performs no Slurm action.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MODEL_VARIANT=hla
exec bash "$SCRIPT_DIR/train_1p3b_100b_from_scratch_common.sh" "$@"
