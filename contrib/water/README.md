# Water 💧

```
 cmake -B build -DMLIR_DIR=/path/to/llvm/build/lib/cmake/mlir
 cmake --build build --target water-opt

 ./build/bin/water-opt matmul.mlir   --water-wave-asterize='pipeline-strategy=2'   --allow-unregistered-dialect   --water-wave-aster-lowering="\
     library-file=contrib/kittens/test/gemm_16x32_f16_k_loop_helpers.mlir \
     pipeline-strategy=2 \
     library-paths=mlir_kernels/library/common/register-init.mlir,\
 mlir_kernels/library/common/indexing.mlir,\
 mlir_kernels/library/common/simple-copies.mlir,\
 mlir_kernels/library/common/copies.mlir,\
 mlir_kernels/library/common/multi-tile-copies.mlir,\
 mlir_kernels/library/common/futures.mlir,\
 mlir_kernels/library/common/indexing_ptr.mlir,\
 contrib/kittens/library/global_16x64_b.mlir,\
 contrib/kittens/library/lds_16x64_b.mlir,\
 contrib/kittens/library/lds_mfma_16x64_b.mlir,\
 contrib/kittens/library/compute_16x16_f16.mlir \
     lcm-unroll=true num-vgprs=128 epilogue-peeling=false" --water-wave-translate-to-asm
```
