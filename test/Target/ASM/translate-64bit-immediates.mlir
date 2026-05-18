// RUN: not aster-translate --split-input-file %s --mlir-to-asm 2>&1 | FileCheck %s

// Non-inline f64 constant on an MFMA f64 instruction is rejected.
// CHECK: constant operand 2 has unsupported 64-bit float type

amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k {
  ^bb0:
    %a0 = amdgcn.alloca : !amdgcn.agpr<0>
    %a1 = amdgcn.alloca : !amdgcn.agpr<1>
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %v1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %dst = amdgcn.make_register_range %a0, %a1 : !amdgcn.agpr<0>, !amdgcn.agpr<1>
    %src = amdgcn.make_register_range %v0, %v1 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
    %c = arith.constant 3.14159265 : f64
    amdgcn.v_mfma_f64_4x4x4_4b_f64 outs(%dst) ins(%src, %src, %c) : outs(!amdgcn.agpr<[0 : 2]>) ins(!amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[0 : 2]>, f64)
    amdgcn.end_kernel
  }
}

// -----

// i64 constant outside the inline literal range on s_ashr_i64 is rejected.
// CHECK: constant operand 0 has unsupported 64-bit integer type

amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k {
  ^bb0:
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %scc = amdgcn.alloca : !amdgcn.scc<0>
    %dst = amdgcn.make_register_range %s0, %s1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %c = arith.constant 100 : i64
    %shift = arith.constant 1 : i32
    amdgcn.s_ashr_i64 outs(%dst, %scc) ins(%c, %shift) : outs(!amdgcn.sgpr<[0 : 2]>, !amdgcn.scc<0>) ins(i64, i32)
    amdgcn.end_kernel
  }
}
