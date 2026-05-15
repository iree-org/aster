# Tooling for basic sweep, update and regression checks

## Explore a set of sizes for a bench

Sizes are passed on the CLI via one or more `--size MxNxK` flags
(required, no default; repeat for multiple):

```
python contrib/kittens/test/bench/tools/perf_explore.py \
    --bench bench_perf_102_gemm_python_multitile_directb_cdna3 \
    --mcpu gfx942 \
    --size 2432x4096x4096 \
    --size 4864x4096x4096 \
    --compile-workers 128 \
    --compile-sample 1000
```

## Check for regression (recompiles and runs)
```
python contrib/kittens/test/bench/tools/perf_dashboard.py \
    --mcpu gfx942 --bench bench_perf_102_gemm_python_multitile_directb_cdna3
```

## Checks against a measurements file
```
python contrib/kittens/test/bench/tools/perf_dashboard.py \
    --measurements /tmp/today.json \
    --mcpu gfx942 --bench bench_perf_102_gemm_python_multitile_directb_cdna3
```

## Updates the best known fomr a measurements file
```
python contrib/kittens/test/bench/tools/perf_best_known_update.py \
    --input /tmp/today.json \
    --mcpu gfx942 \
    --bench bench_perf_102_gemm_python_multitile_directb_cdna3 \
    --apply
```
