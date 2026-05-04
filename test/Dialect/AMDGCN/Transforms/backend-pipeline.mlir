// RUN: aster-opt %s --amdgcn-backend --split-input-file | FileCheck %s

// CHECK-LABEL: kernel @two_vgpr_alloc
// CHECK-DAG:     %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:         v_mov_b32 outs(%[[V0]])
// CHECK:         v_mov_b32 outs(%[[V1]])
// CHECK:         test_inst ins %[[V0]], %[[V1]]
// CHECK-NOT:     alloca : !amdgcn.vgpr{{$}}
// CHECK-NOT:     amdgcn.wait
amdgcn.module @test_two_vgpr target = <gfx942> {
  amdgcn.kernel @two_vgpr_alloc {
    %a = amdgcn.alloca : !amdgcn.vgpr
    %b = amdgcn.alloca : !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %a_val = amdgcn.v_mov_b32 outs(%a) ins(%c0) : outs(!amdgcn.vgpr) ins(i32)
    %b_val = amdgcn.v_mov_b32 outs(%b) ins(%c1) : outs(!amdgcn.vgpr) ins(i32)
    amdgcn.test_inst ins %a_val, %b_val : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    amdgcn.end_kernel
  }
}

// -----

// CHECK-LABEL: kernel @load_then_store
// CHECK-DAG:     alloca : !amdgcn.sgpr<
// CHECK-DAG:     alloca : !amdgcn.vgpr<0>
// CHECK:         s_load_dwordx2
// CHECK:         global_load_dword
// CHECK:         s_waitcnt vmcnt = 0
// CHECK:         global_store_dword
// CHECK:         end_kernel
// CHECK-NOT:     amdgcn.wait
// CHECK-NOT:     alloca : !amdgcn.vgpr{{$}}
// CHECK-NOT:     load_arg
amdgcn.module @test_load_store target = <gfx942> {
  amdgcn.kernel @load_then_store arguments <[
      #amdgcn.buffer_arg<address_space = generic, access = read_only>,
      #amdgcn.buffer_arg<address_space = generic>
    ]> {
    %src = amdgcn.load_arg 0 : !amdgcn.sgpr<[? : ? + 2]>
    %dst = amdgcn.load_arg 1 : !amdgcn.sgpr<[? : ? + 2]>
    %data = amdgcn.alloca : !amdgcn.vgpr
    %c0_i32_mig1 = arith.constant 0 : i32
    %voff = amdgcn.alloca : !amdgcn.vgpr
    %data_val, %tok = amdgcn.global_load_dword dest %data addr %src offset d(%voff) + c(%c0_i32_mig1) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
    amdgcn.wait deps %tok : !amdgcn.read_token<flat>
    %voff2 = amdgcn.alloca : !amdgcn.vgpr
    %wtok = amdgcn.global_store_dword data %data_val addr %dst offset d(%voff2) + c(%c0_i32_mig1) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.end_kernel
  }
}
