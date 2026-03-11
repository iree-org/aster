"""Benchmark: Weak-scaling TFLOPS sweep for constexpr GEMM (16x16x16 MFMA + dwordx4).

Phase 1: Parallel compilation (MLIR -> HSACO) across all configs.
Phase 2: Parallel GPU execution with round-robin across available GPUs,
         each config in its own subprocess for crash isolation.

Sweep axes: load_type (flat/buffer) x a_path (lds/direct).
By default sweeps all implemented (a_path, load_type) combos.

Usage (sweep):
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --full-sweep
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --use-buffer   # buffer only
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --use-flat     # flat only
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --direct-a     # direct-A only
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --num-gpus 8 --compile-workers 16

Usage (single config):
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py \
        --m-wg 38 --n-wg 32 --m-waves 2 --n-waves 2 \
        --m-tiles-wg 4 --n-tiles-wg 4 --k-tiles 1 --stages 2 --k-scaling-factor 128
    ... --use-flat      # flat memory ops (default)
    ... --use-buffer    # buffer memory ops
    ... --direct-a      # A via bpermute (LDS bypass)

Usage (compile only / execute pre-compiled HSACO):
    ... --compile-only --hsaco /tmp/output.hsaco
    ... --hsaco /tmp/output.hsaco
"""

# IMPORTANT: Top configs to run by default. If non-empty, only these labels are run
# unless --full-sweep is passed. Empty list = full sweep (need to populate after first sweep).
# Label suffix scheme: _flat, _buf (LDS path), _direct_flat, _direct_buf (direct-A path).
_TOP_K_BASE = [
    "m9728xn8192xk8192_wg38x32_w2x2_twg16x16x1_s2_flat",
    "m9728xn8192xk8192_wg38x32_w2x2_twg16x16x1_s2_buf",
    "m4864xn4096xk8192_wg19x16_w2x2_twg16x16x1_s2_flat",
    "m12160xn6144xk8192_wg38x32_w2x2_twg20x12x1_s2_flat",
    "m12160xn6144xk8192_wg38x32_w2x2_twg20x12x1_s2_buf",
    "m3648xn5120xk8192_wg38x32_w2x2_twg6x10x1_s2_flat",
    "m7296xn10240xk8192_wg38x32_w2x2_twg12x20x1_s2_flat",
    "m4864xn4096xk4096_wg38x32_w2x2_twg8x8x1_s2_buf",
    "m3648xn4096xk8192_wg19x16_w2x2_twg12x16x1_s2_flat",
    "m3648xn5120xk8192_wg38x32_w2x2_twg6x10x1_s2_buf",
    "m4864xn4096xk4096_wg38x32_w2x2_twg8x8x1_s2_flat",
    "m4864xn3072xk8192_wg19x16_w2x2_twg16x12x1_s2_buf",
    "m3648xn5120xk4096_wg19x16_w2x2_twg12x20x1_s2_flat",
    "m7296xn10240xk8192_wg38x32_w2x2_twg12x20x1_s2_buf",
    "m12160xn6144xk4096_wg38x32_w2x2_twg20x12x1_s2_flat",
    "m4864xn4096xk4096_wg19x16_w2x2_twg16x16x1_s2_flat",
    "m9728xn6144xk8192_wg38x32_w2x2_twg16x12x1_s2_flat",
    "m3648xn5120xk4096_wg38x32_w2x2_twg6x10x1_s2_flat",
    "m9728xn6144xk8192_wg38x32_w2x2_twg16x12x1_s2_buf",
    "m3648xn5120xk8192_wg19x16_w2x2_twg12x20x1_s2_buf",
]

# Known-broken configs: add labels here to skip them during the sweep.
KNOWN_BROKEN = [
    "m4864xn4096xk4096_wg38x32_w4x4_twg8x8x2_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn4096xk4096_wg19x16_w3x2_twg12x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn3072xk4096_wg19x16_w3x2_twg15x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn4096xk4096_wg19x16_w3x2_twg15x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn5120xk4096_wg19x16_w3x2_twg12x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn4096xk4096_wg19x16_w3x4_twg12x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn3072xk4096_wg19x16_w4x4_twg12x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn3072xk4096_wg19x16_w3x4_twg15x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn5120xk4096_wg19x16_w3x4_twg12x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn4096xk4096_wg19x16_w3x4_twg15x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn3072xk4096_wg19x16_w4x4_twg16x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn4096xk4096_wg19x16_w4x4_twg12x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn5120xk4096_wg19x16_w4x4_twg12x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn4096xk4096_wg19x16_w4x4_twg16x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m6080xn3072xk4096_wg19x16_w4x4_twg20x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn10240xk4096_wg38x32_w3x2_twg6x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m5472xn8192xk4096_wg38x32_w3x2_twg9x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m5472xn10240xk4096_wg38x32_w3x2_twg9x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn8192xk4096_wg38x32_w3x2_twg12x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn10240xk4096_wg38x32_w3x2_twg12x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn6144xk4096_wg38x32_w3x2_twg15x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn8192xk4096_wg38x32_w3x2_twg15x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m5472xn10240xk4096_wg38x32_w3x4_twg9x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn8192xk4096_wg38x32_w3x4_twg12x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn10240xk4096_wg38x32_w3x4_twg12x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn6144xk4096_wg38x32_w3x4_twg15x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn8192xk4096_wg38x32_w3x4_twg15x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn4096xk8192_wg38x32_w4x4_twg8x8x2_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn8192xk4096_wg38x32_w4x4_twg8x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn10240xk4096_wg38x32_w4x4_twg8x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn6144xk4096_wg38x32_w4x4_twg12x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9728xn4096xk4096_wg38x32_w4x4_twg16x8x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn8192xk4096_wg38x32_w4x4_twg12x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn10240xk4096_wg38x32_w4x4_twg12x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9728xn6144xk4096_wg38x32_w4x4_twg16x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m12160xn4096xk4096_wg38x32_w4x4_twg20x8x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9728xn8192xk4096_wg38x32_w4x4_twg16x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m12160xn6144xk4096_wg38x32_w4x4_twg20x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn4096xk8192_wg19x16_w3x2_twg12x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn5120xk8192_wg19x16_w3x2_twg12x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn3072xk8192_wg19x16_w3x2_twg15x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn4096xk8192_wg19x16_w3x2_twg15x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn4096xk8192_wg19x16_w3x4_twg12x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn3072xk8192_wg19x16_w3x4_twg15x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn5120xk8192_wg19x16_w3x4_twg12x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn3072xk8192_wg19x16_w4x4_twg12x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn4096xk8192_wg19x16_w3x4_twg15x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn4096xk8192_wg19x16_w4x4_twg12x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn5120xk8192_wg19x16_w4x4_twg12x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn3072xk8192_wg19x16_w4x4_twg16x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn4096xk8192_wg19x16_w4x4_twg16x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m6080xn3072xk8192_wg19x16_w4x4_twg20x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn10240xk8192_wg38x32_w3x2_twg6x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m5472xn8192xk8192_wg38x32_w3x2_twg9x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m5472xn10240xk8192_wg38x32_w3x2_twg9x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn8192xk8192_wg38x32_w3x2_twg12x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn10240xk8192_wg38x32_w3x2_twg12x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn6144xk8192_wg38x32_w3x2_twg15x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn8192xk8192_wg38x32_w3x2_twg15x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m5472xn10240xk8192_wg38x32_w3x4_twg9x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn8192xk8192_wg38x32_w3x4_twg12x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn10240xk8192_wg38x32_w3x4_twg12x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn6144xk8192_wg38x32_w3x4_twg15x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn8192xk8192_wg38x32_w3x4_twg15x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn4096xk16384_wg38x32_w4x4_twg8x8x2_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn8192xk8192_wg38x32_w4x4_twg8x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn10240xk8192_wg38x32_w4x4_twg8x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn6144xk8192_wg38x32_w4x4_twg12x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9728xn4096xk8192_wg38x32_w4x4_twg16x8x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn8192xk8192_wg38x32_w4x4_twg12x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn10240xk8192_wg38x32_w4x4_twg12x20x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9728xn6144xk8192_wg38x32_w4x4_twg16x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m12160xn4096xk8192_wg38x32_w4x4_twg20x8x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9728xn8192xk8192_wg38x32_w4x4_twg16x16x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m12160xn6144xk8192_wg38x32_w4x4_twg20x12x1_s2_buf",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn4096xk4096_wg38x32_w4x4_twg8x8x2_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn4096xk4096_wg19x16_w3x2_twg12x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn5120xk4096_wg19x16_w3x2_twg12x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn3072xk4096_wg19x16_w3x2_twg15x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn4096xk4096_wg19x16_w3x2_twg15x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn4096xk4096_wg19x16_w3x4_twg12x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn5120xk4096_wg19x16_w3x4_twg12x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn3072xk4096_wg19x16_w3x4_twg15x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn3072xk4096_wg19x16_w4x4_twg12x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn4096xk4096_wg19x16_w3x4_twg15x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn4096xk4096_wg19x16_w4x4_twg12x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn3072xk4096_wg19x16_w4x4_twg16x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn5120xk4096_wg19x16_w4x4_twg12x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn4096xk4096_wg19x16_w4x4_twg16x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m6080xn3072xk4096_wg19x16_w4x4_twg20x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn10240xk4096_wg38x32_w3x2_twg6x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m5472xn8192xk4096_wg38x32_w3x2_twg9x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m5472xn10240xk4096_wg38x32_w3x2_twg9x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn8192xk4096_wg38x32_w3x2_twg12x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn10240xk4096_wg38x32_w3x2_twg12x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn6144xk4096_wg38x32_w3x2_twg15x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn8192xk4096_wg38x32_w3x2_twg15x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m5472xn10240xk4096_wg38x32_w3x4_twg9x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn8192xk4096_wg38x32_w3x4_twg12x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn10240xk4096_wg38x32_w3x4_twg12x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn6144xk4096_wg38x32_w3x4_twg15x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn8192xk4096_wg38x32_w3x4_twg15x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn4096xk8192_wg38x32_w4x4_twg8x8x2_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn8192xk4096_wg38x32_w4x4_twg8x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn10240xk4096_wg38x32_w4x4_twg8x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn6144xk4096_wg38x32_w4x4_twg12x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9728xn4096xk4096_wg38x32_w4x4_twg16x8x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn8192xk4096_wg38x32_w4x4_twg12x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn10240xk4096_wg38x32_w4x4_twg12x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9728xn6144xk4096_wg38x32_w4x4_twg16x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m12160xn4096xk4096_wg38x32_w4x4_twg20x8x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9728xn8192xk4096_wg38x32_w4x4_twg16x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m12160xn6144xk4096_wg38x32_w4x4_twg20x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn4096xk8192_wg19x16_w3x2_twg12x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn5120xk8192_wg19x16_w3x2_twg12x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn3072xk8192_wg19x16_w3x2_twg15x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn4096xk8192_wg19x16_w3x2_twg15x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn4096xk8192_wg19x16_w3x4_twg12x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn5120xk8192_wg19x16_w3x4_twg12x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn3072xk8192_wg19x16_w3x4_twg15x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn3072xk8192_wg19x16_w4x4_twg12x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4560xn4096xk8192_wg19x16_w3x4_twg15x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn4096xk8192_wg19x16_w4x4_twg12x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn5120xk8192_wg19x16_w4x4_twg12x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn3072xk8192_wg19x16_w4x4_twg16x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn4096xk8192_wg19x16_w4x4_twg16x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m6080xn3072xk8192_wg19x16_w4x4_twg20x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m3648xn10240xk8192_wg38x32_w3x2_twg6x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m5472xn8192xk8192_wg38x32_w3x2_twg9x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m5472xn10240xk8192_wg38x32_w3x2_twg9x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn8192xk8192_wg38x32_w3x2_twg12x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn10240xk8192_wg38x32_w3x2_twg12x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn6144xk8192_wg38x32_w3x2_twg15x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn8192xk8192_wg38x32_w3x2_twg15x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m5472xn10240xk8192_wg38x32_w3x4_twg9x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn8192xk8192_wg38x32_w3x4_twg12x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn10240xk8192_wg38x32_w3x4_twg12x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn6144xk8192_wg38x32_w3x4_twg15x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9120xn8192xk8192_wg38x32_w3x4_twg15x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn4096xk16384_wg38x32_w4x4_twg8x8x2_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn8192xk8192_wg38x32_w4x4_twg8x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m4864xn10240xk8192_wg38x32_w4x4_twg8x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn6144xk8192_wg38x32_w4x4_twg12x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9728xn4096xk8192_wg38x32_w4x4_twg16x8x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn8192xk8192_wg38x32_w4x4_twg12x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m7296xn10240xk8192_wg38x32_w4x4_twg12x20x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9728xn6144xk8192_wg38x32_w4x4_twg16x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m9728xn8192xk8192_wg38x32_w4x4_twg16x16x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m12160xn4096xk8192_wg38x32_w4x4_twg20x8x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
    "m12160xn6144xk8192_wg38x32_w4x4_twg20x12x1_s2_flat",  # exit 1: HIP error at /home/nico/aster/python/lib/RuntimeModule.cpp:182 - invalid
]

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from test_perf_001_gemm_fp16_weak_scaled import (
    MLIR_FILES,
    WeakScaleConfig,
    compile_gemm,
    execute_gemm_hsaco,
)
from bench_harness import (
    add_sweep_cli_args,
    add_single_cli_args,
    bench_perf_sweep,
    run_single,
    NUM_ITERATIONS,
)

# Sweep grid -- 16x16 MFMA with dwordx4: 4 VGPRs per C tile (vs 16 for 32x32).
# More tiles feasible per wave, so wider multiples than 32x32 variant.
STAGE_CONFIGS = [2, 3, 4, 5]
WAVE_CONFIGS = [(2, 2), (3, 2), (3, 4), (4, 4)]
# Per-workgroup tile counts. Per-wave tiles derived as m_tiles_wg // m_waves.
# Max 1-5x multiples: 4 VGPRs per C tile allows more tiles per wave.
_MULTIPLES = range(1, 6)
_K_TILES_RANGE = range(1, 4)
_tile_wg_pairs = {
    (mw * mm, nw * nm)
    for (mw, nw), mm, nm in itertools.product(WAVE_CONFIGS, _MULTIPLES, _MULTIPLES)
}
TILE_WG_CONFIGS = sorted((m, n, k) for m, n in _tile_wg_pairs for k in _K_TILES_RANGE)
WG_GRIDS = [(19, 16), (38, 32)]
# K = k_scaling_factor * k_tiles * 32 (each 16x32 transfer tile = 32 K elements).
K_SCALING_FACTORS = [64, 128, 256]
SKIP_FIRST_N_CONFIGS = 0


MIN_DIM = 3000  # Skip configs where M, N, or K < 3000


def _make_label_suffix(a_path, load_type):
    """Build label suffix from a_path and load_type, e.g. '_flat', '_buf', '_direct_flat'."""
    lt = "buf" if load_type == "buffer" else "flat"
    return f"_direct_{lt}" if a_path == "direct" else f"_{lt}"


def _generate_configs(variants=None):
    """Generate the full sweep grid, filtering for divisibility and minimum dimensions.

    Args:
        variants: list of (a_path, load_type) tuples to sweep.
            Defaults to all implemented combos from MLIR_FILES.
    """
    if variants is None:
        variants = list(MLIR_FILES.keys())
    configs = []
    for a_path, load_type in variants:
        if (a_path, load_type) not in MLIR_FILES:
            continue
        suffix = _make_label_suffix(a_path, load_type)
        for k_factor in K_SCALING_FACTORS:
            for m_wg, n_wg in WG_GRIDS:
                for m_w, n_w in WAVE_CONFIGS:
                    for m_twg, n_twg, k_t in TILE_WG_CONFIGS:
                        if m_twg % m_w != 0 or n_twg % n_w != 0:
                            continue
                        for stages in STAGE_CONFIGS:
                            k = k_factor * k_t * 32
                            cfg = WeakScaleConfig(
                                m_wg,
                                n_wg,
                                m_w,
                                n_w,
                                m_twg,
                                n_twg,
                                k_t,
                                stages,
                                k,
                                load_type=load_type,
                                a_path=a_path,
                                _label_suffix=suffix,
                            )
                            if (
                                cfg.m_dim < MIN_DIM
                                or cfg.n_dim < MIN_DIM
                                or cfg.k < MIN_DIM
                            ):
                                continue
                            configs.append(cfg)
    return configs


def _repro_cmd(cfg, num_iterations):
    """Return a CLI command to reproduce a single config."""
    k_factor = cfg.k // (cfg.k_tiles * 32)
    buf_flag = " --use-buffer" if cfg.use_buffer else " --use-flat"
    direct_flag = " --direct-a" if cfg.direct_a else ""
    return (
        f"python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py"
        f" --m-wg {cfg.m_wg} --n-wg {cfg.n_wg}"
        f" --m-waves {cfg.m_waves} --n-waves {cfg.n_waves}"
        f" --m-tiles-wg {cfg.m_tiles_wg} --n-tiles-wg {cfg.n_tiles_wg} --k-tiles {cfg.k_tiles}"
        f" --stages {cfg.num_stages} --k-scaling-factor {k_factor}"
        f"{buf_flag}{direct_flag}"
        f" --iterations {num_iterations}"
    )


def _cfg_to_cli_args(cfg):
    """Serialize config to CLI args for subprocess invocation."""
    k_factor = cfg.k // (cfg.k_tiles * 32)
    args = [
        "--m-wg",
        str(cfg.m_wg),
        "--n-wg",
        str(cfg.n_wg),
        "--m-waves",
        str(cfg.m_waves),
        "--n-waves",
        str(cfg.n_waves),
        "--m-tiles-wg",
        str(cfg.m_tiles_wg),
        "--n-tiles-wg",
        str(cfg.n_tiles_wg),
        "--k-tiles",
        str(cfg.k_tiles),
        "--stages",
        str(cfg.num_stages),
        "--k-scaling-factor",
        str(k_factor),
    ]
    args.append("--use-buffer" if cfg.use_buffer else "--use-flat")
    if cfg.direct_a:
        args.append("--direct-a")
    return args


def _make_config_from_args(args, load_type, a_path):
    """Construct a WeakScaleConfig from parsed CLI args."""
    k = args.k_scaling_factor * args.k_tiles * 32
    suffix = _make_label_suffix(a_path, load_type)
    return WeakScaleConfig(
        args.m_wg,
        args.n_wg,
        args.m_waves,
        args.n_waves,
        args.m_tiles_wg,
        args.n_tiles_wg,
        args.k_tiles,
        args.stages,
        k,
        load_type=load_type,
        a_path=a_path,
        _label_suffix=suffix,
    )


def _compile_fn(cfg, output_hsaco_path, **kwargs):
    """Compile wrapper -- cfg carries load_type and a_path."""
    return compile_gemm(cfg, output_hsaco_path, **kwargs)


CORRECTNESS_K = 128  # Small K for fast compile+execute correctness checks.
CORRECTNESS_TOP_N = 20  # Number of top configs to verify after a sweep.


def verify_top_configs(results, num_configs=CORRECTNESS_TOP_N):
    """Phase 3: Verify correctness of the top N configs from the sweep.

    Recompiles each config at K=128 (fast), executes, and checks against numpy.
    """
    import tempfile
    import numpy as np

    if not results:
        return
    top = results[:num_configs]
    print(
        f"\n--- Phase 3: Correctness verification (top {len(top)} configs, K={CORRECTNESS_K}) ---"
    )
    sys.stdout.flush()

    passed = 0
    failed_labels = []
    for rank, (cfg, ms, tflops, pct) in enumerate(top, 1):
        small_cfg = WeakScaleConfig(
            cfg.m_wg,
            cfg.n_wg,
            cfg.m_waves,
            cfg.n_waves,
            cfg.m_tiles_wg,
            cfg.n_tiles_wg,
            cfg.k_tiles,
            cfg.num_stages,
            CORRECTNESS_K,
            load_type=cfg.load_type,
            a_path=cfg.a_path,
            _label_suffix=cfg._label_suffix,
        )
        tag = f"[{rank}/{len(top)}] {cfg.label}"
        try:
            np.random.seed(42)
            A = (np.random.randn(small_cfg.m_dim, small_cfg.k) * 0.1).astype(np.float16)
            B = (np.random.randn(small_cfg.n_dim, small_cfg.k) * 0.1).astype(np.float16)
            with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
                compile_gemm(small_cfg, tmp.name)
                C_output, _ = execute_gemm_hsaco(
                    small_cfg, tmp.name, 1, A, B, skip_gpu_check=True
                )
            expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
            np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
            passed += 1
            print(f"  PASS  {tag}")
        except Exception as e:
            failed_labels.append(cfg.label)
            err_line = str(e).split("\n")[0][:120]
            print(f"  FAIL  {tag}: {err_line}")
        sys.stdout.flush()

    print(f"\nCorrectness: {passed}/{len(top)} passed", end="")
    if failed_labels:
        print(f", {len(failed_labels)} FAILED:")
        for label in failed_labels:
            print(f"  {label}")
    else:
        print(" -- all correct")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weak-scaled 16x16+dwordx4 GEMM benchmark: sweep or single-config repro",
    )
    add_sweep_cli_args(parser)
    # Single-config args
    parser.add_argument("--m-wg", type=int, help="Workgroups along M")
    parser.add_argument("--n-wg", type=int, help="Workgroups along N")
    parser.add_argument("--m-waves", type=int, help="Waves per WG along M")
    parser.add_argument("--n-waves", type=int, help="Waves per WG along N")
    parser.add_argument("--m-tiles-wg", type=int, help="Tiles per workgroup along M")
    parser.add_argument("--n-tiles-wg", type=int, help="Tiles per workgroup along N")
    parser.add_argument("--k-tiles", type=int, help="Tiles per wave along K")
    parser.add_argument("--stages", type=int, help="Pipeline stages")
    parser.add_argument(
        "--k-scaling-factor",
        type=int,
        help="K scaling factor (K = factor * k_tiles * 32, each 16x32 tile = 32 K elements)",
    )
    add_single_cli_args(parser)
    buf_group = parser.add_mutually_exclusive_group()
    buf_group.add_argument(
        "--use-buffer",
        action="store_true",
        help="Sweep buffer_load/buffer_store only",
    )
    buf_group.add_argument(
        "--use-flat",
        action="store_true",
        help="Sweep global_load/global_store only",
    )
    parser.add_argument(
        "--direct-a",
        action="store_true",
        help="A operand via bpermute (LDS bypass) instead of LDS",
    )

    args = parser.parse_args()

    # Determine a_path
    a_path = "direct" if args.direct_a else "lds"

    # Determine load_type variants to sweep.
    if args.use_buffer:
        load_types = ["buffer"]
    elif args.use_flat:
        load_types = ["flat"]
    else:
        load_types = ["flat", "buffer"]

    # Build (a_path, load_type) variant list.
    # In sweep mode without --direct-a, sweep all implemented combos.
    # With --direct-a, sweep only direct combos.
    if args.full_sweep or args.sweep:
        if args.direct_a:
            variants = [(a_path, lt) for lt in load_types]
        else:
            # Sweep all a_path values for each load_type
            variants = [(ap, lt) for lt in load_types for ap in ["lds", "direct"]]
    else:
        variants = [(a_path, lt) for lt in load_types]

    # Filter to implemented combos
    variants = [(ap, lt) for ap, lt in variants if (ap, lt) in MLIR_FILES]

    # For single-config mode
    load_type = "buffer" if args.use_buffer else "flat"

    # TOP_K labels include suffix -- filter to selected variants.
    variant_suffixes = {_make_label_suffix(ap, lt) for ap, lt in variants}
    top_k_to_run = [
        label
        for label in _TOP_K_BASE
        if any(label.endswith(s) for s in variant_suffixes)
    ]

    if args.full_sweep or args.sweep:
        variant_str = ", ".join(f"{ap}/{lt}" for ap, lt in variants)
        print(f"Variants: {variant_str}")
        results = bench_perf_sweep(
            configs=_generate_configs(variants),
            compile_fn=_compile_fn,
            cfg_to_cli_args=_cfg_to_cli_args,
            repro_cmd_fn=_repro_cmd,
            script_path=__file__,
            top_k_to_run=top_k_to_run,
            known_broken=KNOWN_BROKEN,
            skip_first_n=SKIP_FIRST_N_CONFIGS,
            full_sweep=args.full_sweep,
            num_gpus=args.num_gpus,
            compile_workers=args.compile_workers,
        )
        verify_top_configs(results)
    else:
        required = [
            "m_wg",
            "n_wg",
            "m_waves",
            "n_waves",
            "m_tiles_wg",
            "n_tiles_wg",
            "k_tiles",
            "stages",
            "k_scaling_factor",
        ]
        missing = [a for a in required if getattr(args, a) is None]
        if missing:
            flags = ", ".join(f"--{a.replace('_', '-')}" for a in missing)
            parser.error(f"Single-config mode requires: {flags}")
        run_single(
            _make_config_from_args(args, load_type, a_path),
            compile_gemm,
            args,
            execute_fn=execute_gemm_hsaco,
        )
