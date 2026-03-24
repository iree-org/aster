# contrib/mlir-air

linalg op -> AMDGCN assembly via transform dialect tiling + library.

See `README_devs.md` (repo root) for ASTER setup, build, and test instructions.

## Build

```bash
( 
  cd build && \
  cmake .. -DASTER_ENABLE_MLIR_AIR=ON && \
  ninja install
)
```

## Run the Linalg -> ASM test

```bash
mlir-air-opt contrib/mlir-air/test/linalg-to-amdgcn.mlir \
  --transform-interpreter --canonicalize \
  --convert-linalg-to-amdgcn \
  --amdgcn-preload-library="library-paths=mlir_kernels/library/common/register-init.mlir,mlir_kernels/library/common/indexing.mlir,mlir_kernels/library/common/indexing_ptr.mlir,mlir_kernels/library/common/futures.mlir,contrib/kittens/library/compute_16x16_f16.mlir,contrib/kittens/library/global_16x64_b.mlir,contrib/kittens/library/lds_16x64_b.mlir,contrib/kittens/library/lds_mfma_16x64_b.mlir" \
  --inline --symbol-dce --canonicalize \
  --mlir-air-to-asm \
| aster-translate --mlir-to-asm
```

## Run the e2e execution test

On a machine with aster properly configured executing GPU code + the mlir-air
specific build steps above:
```
pytest contrib/mlir-air/test/integration/test_linalg_matmul_e2e.py
```
