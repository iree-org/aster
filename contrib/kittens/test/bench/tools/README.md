# Tooling for basic sweep, update and regression checks

## Explore the SIZES at the top of the bench_perf_102_gemm_python_multitile_directb_cdna3.py file
First, upste the SIZES entry in contrib/kittens/test/bench/tools/perf_explore.py.

Then:
```
python contrib/kittens/test/bench/tools/perf_explore.py \
    --bench bench_perf_102_gemm_python_multitile_directb_cdna3 \
    --mcpu gfx942 \
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
