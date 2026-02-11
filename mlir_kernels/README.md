# MLIR Kernels

This directory contains reusable MLIR kernel definitions for AMDGCN targets.

## Terminology

MLIR kernels are higher-level APIs that operate on low-level quantities. To help
disambiguate between terms that are often overused, we choose the following
naming conventions and stick to them rigorously:
- A **size** is a high-level dimension of the problem at a particular level of
API (e.g. problem size, tile size, mma size).
- A **trip count** is a bound (often related to a loop) induced by a size.
- An **index** is a variable that evolves in units of **elements** for a particular
level of API, within ranges defined by **trip counts**. At a high-level, an
element may be e.g. a block tile. At a low-level, an element may be a `dwordx2`
representing a `4xf16`.
- A **position** is a constant displacement in units of **elements**, that
comes from a higher-level API (i.e. from above). This is typically a quantity
derived from an **index** higher up in the loop nesting structure. There is a
an analogy with affine dialect dims (**index**) and symbols (**position**). A
**position** can never become an **index** going down in the API call hierarchy /
deeper in the loop nest hierarchy. An **index** is expected to be part of
expressions that produce **positions** got lower levels of the call hierarchy /
deeper loops in the loop nest hierarchy.
- An **offset** is a byte displacement in some memory address space and requires
manipulating the `sizeof` an **element**.
- A **pointer** provides a base memory address for **offsets**.


## Organization

Kernels are organized by functionality:

- `copy-1d-dwordx4.mlir` - 1D parallel copy kernel using dwordx4 (16 bytes per thread)

## Structure

Each kernel file follows this pattern:

1. **Helper functions** - Reusable functions for loading arguments, computing offsets, etc.
2. **Load/Store functions** - Separated operations that communicate via memref for SROA optimization
3. **Kernel entry point** - Main kernel that composes the helper functions

## Usage

These kernels are designed to be:
- **Composable** - Functions can be mixed and matched
- **Optimizable** - Use memref for SROA to eliminate intermediate allocations
- **Testable** - Each kernel has corresponding integration tests in `test/`

## Testing

Kernel tests are located in the `test/` subdirectory. Tests use the shared test utilities
from `test/integration/test_utils.py` which provides:
- `compile_mlir_file_to_asm()` - Compile MLIR to assembly
- `execute_kernel_and_verify()` - Execute kernels on GPU and verify results
- `DEFAULT_SROA_PASS_PIPELINE` - Standard pass pipeline for kernel compilation
- `format_throughput_stats()` - Format performance statistics (from `mlir_kernels.benchmarks.benchmark_utils`)

To run kernel tests:
```bash
(cd build && ninja install) && pytest ./mlir_kernels/test/
```

## Benchmarking

Benchmark scripts in `benchmarks/` provide a way to benchmark multiple kernel configurations
in parallel. They compile kernels using multiple processes and execute them across available GPUs.

To run all benchmarks via pytest:
```bash
(cd build && ninja install) && pytest ./mlir_kernels/benchmarks/
```

To run a specific benchmark directly:
```bash
(cd build && ninja install) && python ./mlir_kernels/benchmarks/benchmark_copy_1d.py
```

The benchmark scripts:
- Compile all kernel configurations in parallel using multiple CPU cores
- Execute kernels across available GPUs (round-robin distribution)
- Verify correctness (can be skipped with `--skip-test`)
- Print performance statistics for each configuration

## Nanobenchmarks

Nanobenchmarks in `nanobenchmarks/` test specific low-level operations (LDS reads/writes, global loads, etc.).

To run all nanobenchmarks via pytest:
```bash
(cd build && ninja install) && pytest ./mlir_kernels/nanobenchmarks/
```

## Pass Pipeline

Kernels in this directory are compiled with the DEFAULT_SROA_PASS_PIPELINE defined
in `test/integration/test_utils.py`.
This ensures memref operations are properly eliminated before register allocation.

## Test Utilities

The `test/integration/test_utils.py` module provides reusable utilities for testing MLIR kernels.
These utilities can be imported from any test file:

```python
from integration.test_utils import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
    DEFAULT_SROA_PASS_PIPELINE,
)
from mlir_kernels.benchmarks.benchmark_utils import format_throughput_stats
```
