# GEMM Benchmarks

A suite for comparing the performance of different GEMM backends on AMD GPUs. Supports six backends: [aiter](#aiter), [HipBLAS](#hipblas), [Inductor](#inductor), [IREE](#iree), [rocBLAS](#rocblas), and [Triton](#triton).

## Setup

```bash
cd benchmarks/
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a `.venv/` virtual environment, installs Python dependencies (PyTorch for ROCm, Triton, IREE Turbine, pytest, tabulate), clones the aiter source tree into `aiter_src/`, and installs its native extensions.

If `--rocm-lib` is not provided, the script auto-detects the ROCm library directory by checking `$ROCM_PATH/lib`, `/opt/rocm/lib`, and `/opt/rocm-7.2.1/lib`, in that order.

## Running benchmarks

Always activate the virtual environment first:

```bash
source .venv/bin/activate
```

Run all backends on a given problem size:

```bash
python run_all.py -m 8192 -n 8192 -k 8192
```

Run specific backends:

```bash
python run_all.py -m 4096 -n 4096 -k 4096 --backends triton rocblas
```

Save results to JSON:

```bash
python run_all.py -m 8192 -n 8192 -k 8192 --output-json results.json
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `-m`, `-n`, `-k` | — | Matrix dimensions (required) |
| `--dtype` | `f16` | Data type: `f16`, `bf16`, `f32` |
| `--num-its`, `-i` | `10` | Number of timed iterations |
| `--warmup`, `-w` | `5` | Number of warm-up iterations |
| `--backends` | all | Space-separated list of backends to run |
| `--output-json PATH` | — | Write results + config to a JSON file |
| `--print-asm` | — | Dump backend assembly (Triton, IREE) |
| `--print-mlir` | — | Dump Torch-MLIR IR before IREE compilation |
| `--seed` | `42` | Random seed for input matrices |
| `--rocblas-bench PATH` | `rocblas-bench` | Path to rocblas-bench executable |
| `--hipblas-bench PATH` | `hipblaslt-bench` | Path to hipblaslt-bench executable |

### Output

Results are printed as a table with milliseconds per iteration and TFLOP/s for each backend. When `--output-json` is provided, results include the full problem config and per-backend timing.

## Backends

### aiter

[`aiter_bench/bench_gemm.py`](aiter_bench/bench_gemm.py) — Wraps AMD AI Tensor Engine's `TunedGemm.tgemm.mm()` API. Requires f16 or bf16 input (f32 is not supported). The aiter source tree is cloned into `aiter_src/` during setup and used as a fallback import path.

### HipBLAS

[`hipblas/bench_gemm.py`](hipblas/bench_gemm.py) — Subprocess wrapper around the `hipblaslt-bench` CLI (note: requires `hipblaslt-bench`, not `hipblas-bench`). Parses the "Winner:" section of its output to extract the best algorithm's timing.

**Requirement:** `hipblaslt-bench` must be in PATH or specified via `--hipblas-bench=<path>`.

### Inductor

[`inductor_bench/bench_gemm.py`](inductor_bench/bench_gemm.py) — Uses `torch.compile` with `backend="inductor"` and `mode="max-autotune"`. The first call triggers autotuning; subsequent calls measure steady-state performance.

### IREE

[`iree_bench/bench_gemm.py`](iree_bench/bench_gemm.py) — Ahead-of-time compilation via iree-turbine. Compiles the GEMM to a `.vmfb` artifact, then benchmarks it. Supports `--print-mlir` to inspect Torch-MLIR IR and `--print-asm` to inspect the compiled assembly.

### rocBLAS

[`rocblas/bench_gemm.py`](rocblas/bench_gemm.py) — Subprocess wrapper around the `rocblas-bench` CLI. Parses CSV output for timing (reported in microseconds or as GFLOP/s).

**Requirement:** `rocblas-bench` must be in PATH or specified via `--rocblas-bench=<path>`.

### Triton

[`triton_bench/bench_gemm.py`](triton_bench/bench_gemm.py) — Triton kernel with AMD-specific autotuning. Explores tile shapes (BLOCK_M/N/K up to 256), pipeline depth, K-packing, `waves_per_eu`, and `matrix_instr_nonkdim`. Supports `--print-asm` to dump the AMDGCN ISA.

## Structure

```
benchmarks/
├── common/             # Shared CLI, GEMMConfig, and profiling utilities
├── aiter_bench/        # aiter backend
├── aiter_src/          # aiter source tree (cloned by setup.sh, not tracked)
├── hipblas/            # HipBLAS backend
├── inductor_bench/     # Torch Inductor backend
├── iree_bench/         # IREE Turbine backend
├── rocblas/            # rocBLAS backend
├── triton_bench/       # Triton backend
├── requirements.txt    # Python dependencies
├── run_all.py          # Unified benchmark runner
└── setup.sh            # Environment setup script
```
