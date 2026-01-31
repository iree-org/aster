#!/bin/bash
set -ex

python ./mlir_kernels/benchmarks/benchmark_copy_1d.py
python ./mlir_kernels/benchmarks/benchmark_mfma_dword4_mxnxk_16x16x16_f16f16f32.py
python ./mlir_kernels/benchmarks/benchmark_gemm_dword4_mxnxk_16x16x16_f16f16f32.py
python ./mlir_kernels/benchmarks/benchmark_gemm_dword4_mxnxk_16x16x16_f16f16f32.py --mlir-filename gemm_sched_1wave_dword4_mxnxk_16x16x16_f16f16f32.mlir --num-waves 1
