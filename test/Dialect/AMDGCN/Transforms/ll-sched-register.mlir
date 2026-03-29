// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.kernel(amdgcn-low-level-scheduler{register-semantics=true}))" | FileCheck %s

// CHECK-LABEL:   amdgcn.kernel @i1_serialize_cmpi_select {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[CMPI_0:.*]] = lsir.cmpi i32 slt %[[VAL_0]], %[[CONSTANT_0]] : !amdgcn.vgpr<?>, i32
// CHECK:           lsir.select %[[VAL_1]], %[[CMPI_0]], %[[VAL_2]], %[[VAL_0]] : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[CMPI_1:.*]] = lsir.cmpi i32 slt %[[VAL_2]], %[[CONSTANT_0]] : !amdgcn.vgpr<?>, i32
// CHECK:           lsir.select %[[VAL_3]], %[[CMPI_1]], %[[VAL_0]], %[[VAL_2]] : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @i1_serialize_cmpi_select {
  %c0_i32 = arith.constant 0 : i32
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = alloca : !amdgcn.vgpr<?>
  %4 = lsir.cmpi i32 slt %0, %c0_i32 : !amdgcn.vgpr<?>, i32
  lsir.select %1, %4, %2, %0 : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %5 = lsir.cmpi i32 slt %2, %c0_i32 : !amdgcn.vgpr<?>, i32
  lsir.select %3, %5, %0, %2 : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  end_kernel
}

// CHECK-LABEL:   amdgcn.kernel @i1_serialize_three_chains {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[CMPI_0:.*]] = lsir.cmpi i32 slt %[[VAL_0]], %[[CONSTANT_0]] : !amdgcn.vgpr<?>, i32
// CHECK:           lsir.select %[[VAL_1]], %[[CMPI_0]], %[[VAL_2]], %[[VAL_0]] : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[CMPI_1:.*]] = lsir.cmpi i32 slt %[[VAL_2]], %[[CONSTANT_0]] : !amdgcn.vgpr<?>, i32
// CHECK:           lsir.select %[[VAL_3]], %[[CMPI_1]], %[[VAL_0]], %[[VAL_2]] : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[CMPI_2:.*]] = lsir.cmpi i32 slt %[[VAL_4]], %[[CONSTANT_0]] : !amdgcn.vgpr<?>, i32
// CHECK:           lsir.select %[[VAL_5]], %[[CMPI_2]], %[[VAL_0]], %[[VAL_4]] : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @i1_serialize_three_chains {
  %c0_i32 = arith.constant 0 : i32
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = alloca : !amdgcn.vgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = lsir.cmpi i32 slt %0, %c0_i32 : !amdgcn.vgpr<?>, i32
  lsir.select %1, %6, %2, %0 : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %7 = lsir.cmpi i32 slt %2, %c0_i32 : !amdgcn.vgpr<?>, i32
  lsir.select %3, %7, %0, %2 : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %8 = lsir.cmpi i32 slt %4, %c0_i32 : !amdgcn.vgpr<?>, i32
  lsir.select %5, %8, %0, %4 : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  end_kernel
}

// CHECK-LABEL:   amdgcn.kernel @i1_serialize_fanout {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[CMPI_0:.*]] = lsir.cmpi i32 slt %[[VAL_0]], %[[CONSTANT_0]] : !amdgcn.vgpr<?>, i32
// CHECK:           lsir.select %[[VAL_1]], %[[CMPI_0]], %[[VAL_2]], %[[VAL_0]] : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           lsir.select %[[VAL_3]], %[[CMPI_0]], %[[VAL_4]], %[[VAL_0]] : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[CMPI_1:.*]] = lsir.cmpi i32 slt %[[VAL_2]], %[[CONSTANT_0]] : !amdgcn.vgpr<?>, i32
// CHECK:           lsir.select %[[VAL_5]], %[[CMPI_1]], %[[VAL_0]], %[[VAL_2]] : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @i1_serialize_fanout {
  %c0_i32 = arith.constant 0 : i32
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = alloca : !amdgcn.vgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = lsir.cmpi i32 slt %0, %c0_i32 : !amdgcn.vgpr<?>, i32
  lsir.select %1, %6, %2, %0 : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  lsir.select %3, %6, %4, %0 : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %7 = lsir.cmpi i32 slt %2, %c0_i32 : !amdgcn.vgpr<?>, i32
  lsir.select %5, %7, %0, %2 : !amdgcn.vgpr<?>, i1, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  end_kernel
}

// CHECK-LABEL:   amdgcn.kernel @group_valu_salu {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.sgpr<?>
// CHECK:           %[[VAL_5:.*]] = alloca : !amdgcn.sgpr<?>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[VAL_0]], %[[VAL_1]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:           sop1 s_mov_b32 outs %[[VAL_4]] ins %[[CONSTANT_1]] : !amdgcn.sgpr<?>, i32
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[VAL_2]], %[[VAL_3]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:           sop1 s_mov_b32 outs %[[VAL_5]] ins %[[CONSTANT_0]] : !amdgcn.sgpr<?>, i32
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @group_valu_salu {
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = alloca : !amdgcn.vgpr<?>
  %4 = alloca : !amdgcn.sgpr<?>
  %5 = alloca : !amdgcn.sgpr<?>
  amdgcn.vop1.vop1 <v_mov_b32_e32> %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  sop1 s_mov_b32 outs %4 ins %c0_i32 : !amdgcn.sgpr<?>, i32
  amdgcn.vop1.vop1 <v_mov_b32_e32> %2, %3 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  sop1 s_mov_b32 outs %5 ins %c1_i32 : !amdgcn.sgpr<?>, i32
  end_kernel
}

// CHECK-LABEL:   amdgcn.kernel @respect_data_deps {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 42 : i32
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<?>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[VAL_0]], %[[VAL_2]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:           sop1 s_mov_b32 outs %[[VAL_3]] ins %[[CONSTANT_0]] : !amdgcn.sgpr<?>, i32
// CHECK:           vop2 v_add_u32 outs %[[VAL_1]] ins %[[VAL_0]], %[[VAL_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @respect_data_deps {
  %c42_i32 = arith.constant 42 : i32
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  amdgcn.vop1.vop1 <v_mov_b32_e32> %0, %2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  sop1 s_mov_b32 outs %3 ins %c42_i32 : !amdgcn.sgpr<?>, i32
  vop2 v_add_u32 outs %1 ins %0, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  end_kernel
}

// CHECK-LABEL:   amdgcn.kernel @vmem_addr_load_interleave {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 3072 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2048 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1024 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<?>
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<?>
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_6:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_7:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_8:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_9:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_10:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_11:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_12:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_13:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_14:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_15:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_16:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_17:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_18:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_19:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_20:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_21:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_22:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_23:.*]] = make_register_range %[[VAL_1]], %[[VAL_2]] : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
// CHECK:           %[[VAL_24:.*]] = make_register_range %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_25:.*]] = make_register_range %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_26:.*]] = make_register_range %[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_14]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_27:.*]] = make_register_range %[[VAL_15]], %[[VAL_16]], %[[VAL_17]], %[[VAL_18]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           vop2 v_add_u32 outs %[[VAL_19]] ins %[[CONSTANT_3]], %[[VAL_0]] : !amdgcn.vgpr<?>, i32, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_28:.*]] = load global_load_dwordx4 dest %[[VAL_24]] addr %[[VAL_23]] offset d(%[[VAL_19]]) : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr<?>) -> !amdgcn.read_token<flat>
// CHECK:           vop2 v_add_u32 outs %[[VAL_20]] ins %[[CONSTANT_2]], %[[VAL_0]] : !amdgcn.vgpr<?>, i32, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_29:.*]] = load global_load_dwordx4 dest %[[VAL_25]] addr %[[VAL_23]] offset d(%[[VAL_20]]) : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr<?>) -> !amdgcn.read_token<flat>
// CHECK:           vop2 v_add_u32 outs %[[VAL_21]] ins %[[CONSTANT_1]], %[[VAL_0]] : !amdgcn.vgpr<?>, i32, !amdgcn.vgpr<?>
// CHECK:           vop2 v_add_u32 outs %[[VAL_22]] ins %[[CONSTANT_0]], %[[VAL_0]] : !amdgcn.vgpr<?>, i32, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_30:.*]] = load global_load_dwordx4 dest %[[VAL_26]] addr %[[VAL_23]] offset d(%[[VAL_21]]) : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr<?>) -> !amdgcn.read_token<flat>
// CHECK:           %[[VAL_31:.*]] = load global_load_dwordx4 dest %[[VAL_27]] addr %[[VAL_23]] offset d(%[[VAL_22]]) : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr<?>) -> !amdgcn.read_token<flat>
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @vmem_addr_load_interleave {
  %c3072_i32 = arith.constant 3072 : i32
  %c2048_i32 = arith.constant 2048 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.sgpr<?>
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = make_register_range %1, %2 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = alloca : !amdgcn.vgpr<?>
  %8 = make_register_range %4, %5, %6, %7 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %9 = alloca : !amdgcn.vgpr<?>
  %10 = alloca : !amdgcn.vgpr<?>
  %11 = alloca : !amdgcn.vgpr<?>
  %12 = alloca : !amdgcn.vgpr<?>
  %13 = make_register_range %9, %10, %11, %12 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %14 = alloca : !amdgcn.vgpr<?>
  %15 = alloca : !amdgcn.vgpr<?>
  %16 = alloca : !amdgcn.vgpr<?>
  %17 = alloca : !amdgcn.vgpr<?>
  %18 = make_register_range %14, %15, %16, %17 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %19 = alloca : !amdgcn.vgpr<?>
  %20 = alloca : !amdgcn.vgpr<?>
  %21 = alloca : !amdgcn.vgpr<?>
  %22 = alloca : !amdgcn.vgpr<?>
  %23 = make_register_range %19, %20, %21, %22 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %24 = alloca : !amdgcn.vgpr<?>
  %25 = alloca : !amdgcn.vgpr<?>
  %26 = alloca : !amdgcn.vgpr<?>
  %27 = alloca : !amdgcn.vgpr<?>
  vop2 v_add_u32 outs %24 ins %c0_i32, %0 : !amdgcn.vgpr<?>, i32, !amdgcn.vgpr<?>
  vop2 v_add_u32 outs %25 ins %c1024_i32, %0 : !amdgcn.vgpr<?>, i32, !amdgcn.vgpr<?>
  vop2 v_add_u32 outs %26 ins %c2048_i32, %0 : !amdgcn.vgpr<?>, i32, !amdgcn.vgpr<?>
  vop2 v_add_u32 outs %27 ins %c3072_i32, %0 : !amdgcn.vgpr<?>, i32, !amdgcn.vgpr<?>
  %token = load global_load_dwordx4 dest %8 addr %3 offset d(%24) : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr<?>) -> !amdgcn.read_token<flat>
  %token_0 = load global_load_dwordx4 dest %13 addr %3 offset d(%25) : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr<?>) -> !amdgcn.read_token<flat>
  %token_1 = load global_load_dwordx4 dest %18 addr %3 offset d(%26) : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr<?>) -> !amdgcn.read_token<flat>
  %token_2 = load global_load_dwordx4 dest %23 addr %3 offset d(%27) : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr<?>) -> !amdgcn.read_token<flat>
  end_kernel
}

// CHECK-LABEL:   amdgcn.kernel @barrier_separates_lds {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 8 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_6:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_7:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_8:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_9:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_10:.*]] = make_register_range %[[VAL_2]], %[[VAL_3]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_11:.*]] = make_register_range %[[VAL_4]], %[[VAL_5]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_12:.*]] = make_register_range %[[VAL_6]], %[[VAL_7]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_13:.*]] = make_register_range %[[VAL_8]], %[[VAL_9]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_14:.*]] = store ds_write_b64 data %[[VAL_10]] addr %[[VAL_0]] offset c(%[[CONSTANT_1]]) : ins(!amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<?>, i32) -> !amdgcn.write_token<shared>
// CHECK:           %[[VAL_15:.*]] = store ds_write_b64 data %[[VAL_11]] addr %[[VAL_1]] offset c(%[[CONSTANT_1]]) : ins(!amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<?>, i32) -> !amdgcn.write_token<shared>
// CHECK:           amdgcn.sopp.sopp <s_barrier>
// CHECK:           %[[VAL_16:.*]] = load ds_read_b64 dest %[[VAL_12]] addr %[[VAL_0]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<?>, i32) -> !amdgcn.read_token<shared>
// CHECK:           %[[VAL_17:.*]] = load ds_read_b64 dest %[[VAL_13]] addr %[[VAL_1]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<?>, i32) -> !amdgcn.read_token<shared>
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @barrier_separates_lds {
  %c8_i32 = arith.constant 8 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = alloca : !amdgcn.vgpr<?>
  %4 = make_register_range %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = make_register_range %5, %6 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %8 = alloca : !amdgcn.vgpr<?>
  %9 = alloca : !amdgcn.vgpr<?>
  %10 = make_register_range %8, %9 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %11 = alloca : !amdgcn.vgpr<?>
  %12 = alloca : !amdgcn.vgpr<?>
  %13 = make_register_range %11, %12 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %14 = store ds_write_b64 data %4 addr %0 offset c(%c0_i32) : ins(!amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<?>, i32) -> !amdgcn.write_token<shared>
  %15 = store ds_write_b64 data %7 addr %1 offset c(%c0_i32) : ins(!amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<?>, i32) -> !amdgcn.write_token<shared>
  amdgcn.sopp.sopp <s_barrier>
  %token = load ds_read_b64 dest %10 addr %0 offset c(%c8_i32) : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<?>, i32) -> !amdgcn.read_token<shared>
  %token_0 = load ds_read_b64 dest %13 addr %1 offset c(%c8_i32) : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<?>, i32) -> !amdgcn.read_token<shared>
  end_kernel
}

// CHECK-LABEL:   amdgcn.kernel @lds_ops_ordered {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 8 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_6:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_7:.*]] = make_register_range %[[VAL_1]], %[[VAL_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_8:.*]] = make_register_range %[[VAL_3]], %[[VAL_4]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_9:.*]] = make_register_range %[[VAL_5]], %[[VAL_6]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_10:.*]] = store ds_write_b64 data %[[VAL_7]] addr %[[VAL_0]] offset c(%[[CONSTANT_1]]) : ins(!amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<?>, i32) -> !amdgcn.write_token<shared>
// CHECK:           %[[VAL_11:.*]] = load ds_read_b64 dest %[[VAL_9]] addr %[[VAL_0]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<?>, i32) -> !amdgcn.read_token<shared>
// CHECK:           %[[VAL_12:.*]] = store ds_write_b64 data %[[VAL_8]] addr %[[VAL_0]] offset c(%[[CONSTANT_1]]) : ins(!amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<?>, i32) -> !amdgcn.write_token<shared>
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @lds_ops_ordered {
  %c8_i32 = arith.constant 8 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = make_register_range %4, %5 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %7 = alloca : !amdgcn.vgpr<?>
  %8 = alloca : !amdgcn.vgpr<?>
  %9 = make_register_range %7, %8 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %10 = store ds_write_b64 data %3 addr %0 offset c(%c0_i32) : ins(!amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<?>, i32) -> !amdgcn.write_token<shared>
  %token = load ds_read_b64 dest %9 addr %0 offset c(%c8_i32) : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<?>, i32) -> !amdgcn.read_token<shared>
  %11 = store ds_write_b64 data %6 addr %0 offset c(%c0_i32) : ins(!amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<?>, i32) -> !amdgcn.write_token<shared>
  end_kernel
}

// CHECK-LABEL:   amdgcn.kernel @lgkm_wait_gates_ds_read {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_6:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_7:.*]] = make_register_range %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_8:.*]] = make_register_range %[[VAL_6]], %[[VAL_4]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[VAL_3]], %[[VAL_4]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[VAL_6]], %[[VAL_3]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[VAL_4]], %[[VAL_3]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:           %[[VAL_9:.*]] = store ds_write_b64 data %[[VAL_8]] addr %[[VAL_5]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<?>, i32) -> !amdgcn.write_token<shared>
// CHECK:           wait deps %[[VAL_9]] : !amdgcn.write_token<shared>
// CHECK:           %[[VAL_10:.*]] = load ds_read_b64 dest %[[VAL_7]] addr %[[VAL_2]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<?>, i32) -> !amdgcn.read_token<shared>
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @lgkm_wait_gates_ds_read {
  %c0_i32 = arith.constant 0 : i32
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = alloca : !amdgcn.vgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  amdgcn.vop1.vop1 <v_mov_b32_e32> %3, %4 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  %6 = alloca : !amdgcn.vgpr<?>
  amdgcn.vop1.vop1 <v_mov_b32_e32> %6, %3 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  amdgcn.vop1.vop1 <v_mov_b32_e32> %4, %3 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  %7 = make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %8 = make_register_range %6, %4 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %9 = store ds_write_b64 data %8 addr %5 offset c(%c0_i32) : ins(!amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<?>, i32) -> !amdgcn.write_token<shared>
  wait deps %9 : !amdgcn.write_token<shared>
  %token = load ds_read_b64 dest %7 addr %2 offset c(%c0_i32) : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<?>, i32) -> !amdgcn.read_token<shared>
  end_kernel
}

// CHECK-LABEL:   amdgcn.kernel @vmem_ops_ordered {
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.sgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<?>
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_6:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_7:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_8:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_9:.*]] = make_register_range %[[VAL_0]], %[[VAL_1]] : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
// CHECK:           %[[VAL_10:.*]] = make_register_range %[[VAL_2]] : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_11:.*]] = make_register_range %[[VAL_3]] : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_12:.*]] = make_register_range %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_13:.*]] = store global_store_dword data %[[VAL_10]] addr %[[VAL_9]] : ins(!amdgcn.vgpr<?>, !amdgcn.sgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           %[[VAL_14:.*]] = load global_load_dwordx4 dest %[[VAL_12]] addr %[[VAL_9]] offset d(%[[VAL_8]]) : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr<?>) -> !amdgcn.read_token<flat>
// CHECK:           %[[VAL_15:.*]] = store global_store_dword data %[[VAL_11]] addr %[[VAL_9]] : ins(!amdgcn.vgpr<?>, !amdgcn.sgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @vmem_ops_ordered {
  %0 = alloca : !amdgcn.sgpr<?>
  %1 = alloca : !amdgcn.sgpr<?>
  %2 = make_register_range %0, %1 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.vgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = alloca : !amdgcn.vgpr<?>
  %8 = alloca : !amdgcn.vgpr<?>
  %9 = make_register_range %5, %6, %7, %8 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %10 = alloca : !amdgcn.vgpr<?>
  %11 = make_register_range %3 : !amdgcn.vgpr<?>
  %12 = make_register_range %4 : !amdgcn.vgpr<?>
  %13 = store global_store_dword data %11 addr %2 : ins(!amdgcn.vgpr<?>, !amdgcn.sgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  %token = load global_load_dwordx4 dest %9 addr %2 offset d(%10) : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr<?>) -> !amdgcn.read_token<flat>
  %14 = store global_store_dword data %12 addr %2 : ins(!amdgcn.vgpr<?>, !amdgcn.sgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  end_kernel
}
