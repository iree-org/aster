# ASTER Examples

Bottom-up examples oriented towards folks interested in low-level hardware programming and co-design.
Each example shows MLIR IR, assembly output, and GPU execution with numpy verification (where applicable) and ATT trace / performance counters.


## Setup

Minimal setup on an MI300X machine:

```bash
# 1. Install uv (Python package manager) if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# 3. Install ASTER
uv pip install aster-mlir-hip # [ --find-links /tmp ]

# 4. Install ROCm runtime for GPU execution
uv pip install -r requirements-amd-gfx94X.txt --prerelease=allow

# 5. Run an example
python examples/01_hello_asm/run.py
```

**4. (Optional) Install [rocprof-compute-viewer](https://github.com/ROCm/rocprof-compute-viewer/) for trace visualization.**
Download a pre-built binary from the [releases page](https://github.com/ROCm/rocprof-compute-viewer/releases),
or build from source (requires Qt 6).

## Running

```bash
# Single example
python examples/01_hello_asm/run.py

# All examples
for d in examples/*/; do echo "=== $d ===" && python "$d/run.py"; done
```

### CLI flags (all compiling examples)

```bash
# Print IR after each compiler pass (see what each pass does)
python examples/01_hello_asm/run.py --print-ir-after-all

# Print assembly via compiler diagnostics
python examples/01_hello_asm/run.py --print-asm
```

Environment variables also work: `ASTER_PRINT_IR_AFTER_ALL=1`,
`ASTER_PRINT_ASM=1`, `ASTER_PRINT_DIR=/tmp/debug`.

### Profiling with rocprofv3

```bash
# ATT trace for any example
examples/profile.sh examples/01_hello_asm/run.py
```

## Structure

Each example directory contains:
- `README.md` -- terse explanation
- `run.py` -- executable script (IR + ASM + optional execution)
- `kernel.mlir` -- hand-written MLIR (when applicable)

Shared utilities: `examples/common.py`
