#!/bin/bash
set -ex

# Minimal smoke checks before kicking off benchmarks
python ./mlir_kernels/test/test_gemm_dword4_mxnxk_16x16x16_f16f16f32.py --m 64 --n 32 --k 16 --m-tile 32 --n-tile 32 --k-tile 16 -W 2 --mlir-filename gemm_sched_dword4_mxnxk_16x16x16_f16f16f32.mlir
python ./mlir_kernels/test/test_gemm_dword4_mxnxk_16x16x16_f16f16f32.py --m 128 --n 128 --k 128 --m-tile 32 --n-tile 32 --k-tile 16 -W 2 --mlir-filename gemm_sched_dword4_mxnxk_16x16x16_f16f16f32.mlir

python ./mlir_kernels/benchmarks/benchmark_copy_1d.py
python ./mlir_kernels/benchmarks/benchmark_mfma_dword4_mxnxk_16x16x16_f16f16f32.py
python ./mlir_kernels/benchmarks/benchmark_gemm_dword4_mxnxk_16x16x16_f16f16f32.py
python ./mlir_kernels/benchmarks/benchmark_gemm_dword4_mxnxk_16x16x16_f16f16f32.py --mlir-filename gemm_sched_dword4_mxnxk_16x16x16_f16f16f32.mlir
