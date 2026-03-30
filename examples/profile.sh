#!/bin/bash
# Profile an ASTER example with rocprofv3 ATT tracing.
#
# Usage:
#   examples/profile.sh examples/02_vector_add/run.py [extra args...]
#
# Environment variables:
#   ROCPROFV3  - path to rocprofv3 (default: auto-detected via which)
#   ITERATIONS - kernel launch iterations (default: 5, profiles 3rd)

set -e

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: activate your venv first (source .aster/bin/activate)"
    exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
ROCPROFV3="${ROCPROFV3:-$(which rocprofv3 2>/dev/null || true)}"
if [ -z "$ROCPROFV3" ]; then
    echo "Error: rocprofv3 not found. Set ROCPROFV3=/path/to/rocprofv3"
    exit 1
fi

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $(basename "$0") <example/run.py> [extra args...]"
    echo ""
    echo "Examples:"
    echo "  examples/profile.sh examples/02_vector_add/run.py"
    echo "  examples/profile.sh examples/05_mfma/run.py --print-asm"
    exit 1
fi

PY_SCRIPT="$1"
shift

if [ ! -f "$PY_SCRIPT" ]; then
    echo "Error: $PY_SCRIPT not found"
    exit 1
fi

EXAMPLE_NAME="$(basename "$(dirname "$PY_SCRIPT")")"
TRACE_DIR="trace_$(hostname)_${EXAMPLE_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "=== Profiling: $EXAMPLE_NAME ==="
echo "  trace: $TRACE_DIR"

"$ROCPROFV3" \
    --kernel-iteration-range 3 \
    --kernel-include-regex ".*kernel.*" \
    --att \
    --att-activity 10 \
    -d "$TRACE_DIR" \
    -- \
    "$PYTHON_BIN" "$PY_SCRIPT" "$@"

echo ""
echo "Done. Trace: $(pwd)/$TRACE_DIR"
