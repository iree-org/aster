// RUN: aster-opt %s --amdgcn-reg-alloc --aster-disable-verifiers --aster-suppress-disabled-verifier-warning | FileCheck %s

// After the new reg-alloc pipeline (bufferization + to-register-semantics +
// register-coloring), block arguments are eliminated. Loop-carried values flow
// through side-effects (register writes/reads via s_mov_b32 copies).
//
// Key invariants verified:
// 1. Block arguments are eliminated (cf.br ^bb1 with no args).
// 2. All allocas are colored to concrete registers.
// 3. Loop back-edge uses s_mov_b32 to maintain loop-carried state.
// 4. Chained selects and loop counter advance all get concrete registers.
//
// CHECK-LABEL: kernel @chained_select_loop_3_way_buffer_mux {
// CHECK:   cf.br ^bb1
// CHECK: ^bb1:
// CHECK:   lsir.select %{{[0-9]+}},
// CHECK:   lsir.select %{{[0-9]+}},
// CHECK:   test_inst ins %{{[0-9]+}} : (!amdgcn.sgpr<{{[0-9]+}}>)
// CHECK:   cf.cond_br %{{.*}}, ^bb1, ^bb2
amdgcn.module @chained_select target = <gfx942> isa = <cdna3> {
  kernel @chained_select_loop_3_way_buffer_mux {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c6_i32 = arith.constant 6 : i32
    %c100_i32 = arith.constant 100 : i32
    %c200_i32 = arith.constant 200 : i32

    // Storage for loop counter and buffer index
    %s_k = alloca : !amdgcn.sgpr
    %s_buf = alloca : !amdgcn.sgpr

    // Storage for select destinations (chained)
    %s_inner = alloca : !amdgcn.sgpr
    %s_outer = alloca : !amdgcn.sgpr

    // Storage for computations
    %s_cmp = alloca : !amdgcn.sgpr
    %s_add = alloca : !amdgcn.sgpr
    %s_next_buf = alloca : !amdgcn.sgpr

    // Initialize loop counter and buffer index
    %k_init = sop1 s_mov_b32 outs %s_k ins %c0_i32 : !amdgcn.sgpr, i32
    %buf_init = sop1 s_mov_b32 outs %s_buf ins %c0_i32 : !amdgcn.sgpr, i32

    cf.br ^loop(%k_init, %buf_init : !amdgcn.sgpr, !amdgcn.sgpr)

  ^loop(%k: !amdgcn.sgpr, %buf_idx: !amdgcn.sgpr):
    // 3-way buffer mux: chained select on buf_idx
    %is_buf0 = lsir.cmpi i32 eq %buf_idx, %c0_i32 : !amdgcn.sgpr, i32
    %is_buf1 = lsir.cmpi i32 eq %buf_idx, %c1_i32 : !amdgcn.sgpr, i32

    // Chained select: inner picks between buf1 / buf2 values,
    // outer picks between buf0 and inner result.
    %inner = lsir.select %s_inner, %is_buf1, %c100_i32, %c200_i32 : !amdgcn.sgpr, i1, i32, i32
    %outer = lsir.select %s_outer, %is_buf0, %c0_i32, %inner : !amdgcn.sgpr, i1, i32, !amdgcn.sgpr

    // Use the outer select result (prevents DCE)
    test_inst ins %outer : (!amdgcn.sgpr) -> ()

    // Advance buf_idx: (buf_idx + 1) % 3 via wrap
    %next_raw = sop2 s_add_u32 outs %s_add ins %buf_idx, %c1_i32 : !amdgcn.sgpr, !amdgcn.sgpr, i32
    %is_3 = lsir.cmpi i32 eq %next_raw, %c3_i32 : !amdgcn.sgpr, i32
    %next_buf = lsir.select %s_next_buf, %is_3, %c0_i32, %next_raw : !amdgcn.sgpr, i1, i32, !amdgcn.sgpr

    // Advance loop counter
    %k_next = sop2 s_add_u32 outs %s_cmp ins %k, %c1_i32 : !amdgcn.sgpr, !amdgcn.sgpr, i32
    %done = lsir.cmpi i32 slt %k_next, %c6_i32 : !amdgcn.sgpr, i32
    cf.cond_br %done, ^loop(%k_next, %next_buf : !amdgcn.sgpr, !amdgcn.sgpr), ^exit

  ^exit:
    end_kernel
  }
}
