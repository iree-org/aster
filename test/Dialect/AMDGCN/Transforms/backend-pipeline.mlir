// RUN: aster-opt %s --amdgcn-backend --split-input-file | FileCheck %s

// CHECK-LABEL: kernel @two_vgpr_alloc
// CHECK-DAG:     %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32> %[[V0]]
// CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32> %[[V1]]
// CHECK:         test_inst ins %[[V0]], %[[V1]]
// CHECK-NOT:     alloca : !amdgcn.vgpr{{$}}
// CHECK-NOT:     amdgcn.wait
amdgcn.module @test_two_vgpr target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @two_vgpr_alloc {
    %a = amdgcn.alloca : !amdgcn.vgpr
    %b = amdgcn.alloca : !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %a_val = amdgcn.vop1.vop1 <v_mov_b32_e32> %a, %c0 : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
    %b_val = amdgcn.vop1.vop1 <v_mov_b32_e32> %b, %c1 : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
    amdgcn.test_inst ins %a_val, %b_val : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    amdgcn.end_kernel
  }
}

// -----

// CHECK-LABEL: kernel @load_then_store
// CHECK-DAG:     alloca : !amdgcn.sgpr<
// CHECK-DAG:     alloca : !amdgcn.vgpr<0>
// CHECK:         load s_load_dwordx2
// CHECK:         load global_load_dword
// CHECK:         amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
// CHECK:         store global_store_dword
// CHECK:         end_kernel
// CHECK-NOT:     amdgcn.wait
// CHECK-NOT:     alloca : !amdgcn.vgpr{{$}}
// CHECK-NOT:     load_arg
amdgcn.module @test_load_store target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @load_then_store arguments <[
      #amdgcn.buffer_arg<address_space = generic, access = read_only>,
      #amdgcn.buffer_arg<address_space = generic>
    ]> {
    %src = amdgcn.load_arg 0 : !amdgcn.sgpr<[? : ? + 2]>
    %dst = amdgcn.load_arg 1 : !amdgcn.sgpr<[? : ? + 2]>
    %data = amdgcn.alloca : !amdgcn.vgpr
    %data_val, %tok = amdgcn.load global_load_dword dest %data addr %src : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    amdgcn.wait deps %tok : !amdgcn.read_token<flat>
    %wtok = amdgcn.store global_store_dword data %data_val addr %dst : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
    amdgcn.end_kernel
  }
}
