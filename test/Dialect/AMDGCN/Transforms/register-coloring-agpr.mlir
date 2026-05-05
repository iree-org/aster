// RUN: aster-opt %s --amdgcn-register-coloring --cse --split-input-file | FileCheck %s

// CHECK-LABEL: amdgcn.kernel @agpr_no_interference {
// CHECK:         %[[A0:.*]] = alloca : !amdgcn.agpr<0>
// CHECK:         test_inst outs %[[A0]] : (!amdgcn.agpr<0>) -> ()
// CHECK:         test_inst outs %[[A0]] : (!amdgcn.agpr<0>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @agpr_no_interference {
  %0 = alloca : !amdgcn.agpr<?>
  %1 = alloca : !amdgcn.agpr<?>
  test_inst outs %0 : (!amdgcn.agpr<?>) -> ()
  test_inst outs %1 : (!amdgcn.agpr<?>) -> ()
  end_kernel
}

// -----

// CHECK-LABEL: amdgcn.kernel @agpr_interference {
// CHECK-DAG:     %[[A0:.*]] = alloca : !amdgcn.agpr<0>
// CHECK-DAG:     %[[A1:.*]] = alloca : !amdgcn.agpr<1>
// CHECK:         test_inst outs %[[A0]] : (!amdgcn.agpr<0>) -> ()
// CHECK:         test_inst ins %[[A0]] : (!amdgcn.agpr<0>) -> ()
// CHECK:         test_inst outs %[[A1]] : (!amdgcn.agpr<1>) -> ()
// CHECK:         test_inst ins %[[A0]], %[[A1]] : (!amdgcn.agpr<0>, !amdgcn.agpr<1>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @agpr_interference {
  %0 = alloca : !amdgcn.agpr<?>
  %1 = alloca : !amdgcn.agpr<?>
  test_inst outs %0 : (!amdgcn.agpr<?>) -> ()
  test_inst ins %0 : (!amdgcn.agpr<?>) -> ()
  test_inst outs %1 : (!amdgcn.agpr<?>) -> ()
  test_inst ins %0, %1 : (!amdgcn.agpr<?>, !amdgcn.agpr<?>) -> ()
  end_kernel
}

// -----

// CHECK-LABEL: amdgcn.kernel @agpr_range_allocation {
// CHECK-DAG:     %[[A0:.*]] = alloca : !amdgcn.agpr<0>
// CHECK-DAG:     %[[A1:.*]] = alloca : !amdgcn.agpr<1>
// CHECK-DAG:     %[[A2:.*]] = alloca : !amdgcn.agpr<2>
// CHECK-DAG:     %[[A3:.*]] = alloca : !amdgcn.agpr<3>
// CHECK:         %[[RANGE:.*]] = make_register_range %[[A0]], %[[A1]], %[[A2]], %[[A3]] : !amdgcn.agpr<0>, !amdgcn.agpr<1>, !amdgcn.agpr<2>, !amdgcn.agpr<3>
// CHECK:         test_inst ins %[[RANGE]] : (!amdgcn.agpr<[0 : 4]>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @agpr_range_allocation {
  %0 = alloca : !amdgcn.agpr<?>
  %1 = alloca : !amdgcn.agpr<?>
  %2 = alloca : !amdgcn.agpr<?>
  %3 = alloca : !amdgcn.agpr<?>
  %r = make_register_range %0, %1, %2, %3
      : !amdgcn.agpr<?>, !amdgcn.agpr<?>, !amdgcn.agpr<?>, !amdgcn.agpr<?>
  test_inst ins %r : (!amdgcn.agpr<[? : ? + 4]>) -> ()
  end_kernel
}

// -----

// No other VGPRs allocated, so scratch is v0.
// CHECK-LABEL: amdgcn.kernel @agpr_copy {
// CHECK-DAG:     %[[A0:.*]] = alloca : !amdgcn.agpr<0>
// CHECK-DAG:     %[[A1:.*]] = alloca : !amdgcn.agpr<1>
// CHECK:         test_inst outs %[[A0]] : (!amdgcn.agpr<0>) -> ()
// CHECK:         %[[VTMP:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:         v_accvgpr_read outs(%[[VTMP]]) ins(%[[A1]])
// CHECK:         v_accvgpr_write outs(%[[A0]]) ins(%[[VTMP]])
// CHECK:         test_inst ins %[[A1]] : (!amdgcn.agpr<1>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @agpr_copy {
  %0 = alloca : !amdgcn.agpr<?>
  %1 = alloca : !amdgcn.agpr<?>
  test_inst outs %0 : (!amdgcn.agpr<?>) -> ()
  lsir.copy %0, %1 : !amdgcn.agpr<?>, !amdgcn.agpr<?>
  test_inst ins %1 : (!amdgcn.agpr<?>) -> ()
  end_kernel
}

// -----

// With VGPRs v0..v2 in use, the scratch should be v3.
// CHECK-LABEL: amdgcn.kernel @agpr_copy_with_vgprs {
// CHECK-DAG:     %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:     %[[V2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:     %[[A0:.*]] = alloca : !amdgcn.agpr<0>
// CHECK-DAG:     %[[A1:.*]] = alloca : !amdgcn.agpr<1>
// CHECK:         %[[VTMP:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:         v_accvgpr_read outs(%[[VTMP]]) ins(%[[A1]])
// CHECK:         v_accvgpr_write outs(%[[A0]]) ins(%[[VTMP]])
// CHECK:         end_kernel
amdgcn.kernel @agpr_copy_with_vgprs {
  %v0 = alloca : !amdgcn.vgpr<?>
  %v1 = alloca : !amdgcn.vgpr<?>
  %v2 = alloca : !amdgcn.vgpr<?>
  %a0 = alloca : !amdgcn.agpr<?>
  %a1 = alloca : !amdgcn.agpr<?>
  test_inst outs %v0 : (!amdgcn.vgpr<?>) -> ()
  test_inst outs %v1 : (!amdgcn.vgpr<?>) -> ()
  test_inst outs %v2 : (!amdgcn.vgpr<?>) -> ()
  test_inst outs %a0 : (!amdgcn.agpr<?>) -> ()
  lsir.copy %a0, %a1 : !amdgcn.agpr<?>, !amdgcn.agpr<?>
  test_inst ins %v0, %v1, %v2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  test_inst ins %a1 : (!amdgcn.agpr<?>) -> ()
  end_kernel
}

// -----

// CSE merges non-interfering allocas.
// CHECK-LABEL: amdgcn.kernel @mixed_vgpr_agpr_independent {
// CHECK-DAG:     %[[V:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[A:.*]] = alloca : !amdgcn.agpr<0>
// CHECK:         test_inst outs %[[V]] : (!amdgcn.vgpr<0>) -> ()
// CHECK:         test_inst outs %[[A]] : (!amdgcn.agpr<0>) -> ()
// CHECK:         test_inst ins %[[V]], %[[A]] : (!amdgcn.vgpr<0>, !amdgcn.agpr<0>) -> ()
// CHECK:         test_inst outs %[[V]] : (!amdgcn.vgpr<0>) -> ()
// CHECK:         test_inst outs %[[A]] : (!amdgcn.agpr<0>) -> ()
// CHECK:         test_inst ins %[[V]], %[[A]] : (!amdgcn.vgpr<0>, !amdgcn.agpr<0>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @mixed_vgpr_agpr_independent {
  %v0 = alloca : !amdgcn.vgpr<?>
  %v1 = alloca : !amdgcn.vgpr<?>
  %a0 = alloca : !amdgcn.agpr<?>
  %a1 = alloca : !amdgcn.agpr<?>
  test_inst outs %v0 : (!amdgcn.vgpr<?>) -> ()
  test_inst outs %a0 : (!amdgcn.agpr<?>) -> ()
  test_inst ins %v0, %a0 : (!amdgcn.vgpr<?>, !amdgcn.agpr<?>) -> ()
  test_inst outs %v1 : (!amdgcn.vgpr<?>) -> ()
  test_inst outs %a1 : (!amdgcn.agpr<?>) -> ()
  test_inst ins %v1, %a1 : (!amdgcn.vgpr<?>, !amdgcn.agpr<?>) -> ()
  end_kernel
}

// -----

// CHECK-LABEL: amdgcn.kernel @explicit_accvgpr_write_imm {
// CHECK-DAG:     %[[A:.*]] = alloca : !amdgcn.agpr<0>
// CHECK-DAG:     %[[V:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:         v_accvgpr_write outs(%[[A]]) ins(%c42_i32) : outs(!amdgcn.agpr<0>) ins(i32)
// CHECK:         v_accvgpr_read outs(%[[V]]) ins(%[[A]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.agpr<0>)
// CHECK:         test_inst ins %[[V]] : (!amdgcn.vgpr<0>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @explicit_accvgpr_write_imm {
  %a = alloca : !amdgcn.agpr<?>
  %v = alloca : !amdgcn.vgpr<?>
  %c42 = arith.constant 42 : i32
  amdgcn.v_accvgpr_write outs(%a) ins(%c42) : outs(!amdgcn.agpr<?>) ins(i32)
  amdgcn.v_accvgpr_read outs(%v) ins(%a) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.agpr<?>)
  test_inst ins %v : (!amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----

// CHECK-LABEL: amdgcn.kernel @explicit_accvgpr_write_from_vgpr {
// CHECK-DAG:     %[[V:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[A:.*]] = alloca : !amdgcn.agpr<0>
// CHECK:         v_mov_b32 outs(%[[V]]) ins(%c99_i32) : outs(!amdgcn.vgpr<0>) ins(i32)
// CHECK:         v_accvgpr_write outs(%[[A]]) ins(%[[V]]) : outs(!amdgcn.agpr<0>) ins(!amdgcn.vgpr<0>)
// CHECK:         v_accvgpr_read outs(%[[V]]) ins(%[[A]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.agpr<0>)
// CHECK:         test_inst ins %[[V]] : (!amdgcn.vgpr<0>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @explicit_accvgpr_write_from_vgpr {
  %v_src = alloca : !amdgcn.vgpr<?>
  %a = alloca : !amdgcn.agpr<?>
  %v_dst = alloca : !amdgcn.vgpr<?>
  %c99 = arith.constant 99 : i32
  amdgcn.v_mov_b32 outs(%v_src) ins(%c99) : outs(!amdgcn.vgpr<?>) ins(i32)
  amdgcn.v_accvgpr_write outs(%a) ins(%v_src) : outs(!amdgcn.agpr<?>) ins(!amdgcn.vgpr<?>)
  amdgcn.v_accvgpr_read outs(%v_dst) ins(%a) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.agpr<?>)
  test_inst ins %v_dst : (!amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----

// Must use explicit v_accvgpr_write_b32 instead.
// CHECK-LABEL: amdgcn.kernel @vgpr_to_agpr_copy_rejected {
// CHECK-DAG:     %[[V:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[A:.*]] = alloca : !amdgcn.agpr<0>
// CHECK:         test_inst outs %[[V]] : (!amdgcn.vgpr<0>) -> ()
// CHECK:         lsir.copy %[[V]], %[[A]] : !amdgcn.vgpr<0>, !amdgcn.agpr<0>
// CHECK:         test_inst ins %[[A]] : (!amdgcn.agpr<0>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @vgpr_to_agpr_copy_rejected {
  %v = alloca : !amdgcn.vgpr<?>
  %a = alloca : !amdgcn.agpr<?>
  test_inst outs %v : (!amdgcn.vgpr<?>) -> ()
  lsir.copy %v, %a : !amdgcn.vgpr<?>, !amdgcn.agpr<?>
  test_inst ins %a : (!amdgcn.agpr<?>) -> ()
  end_kernel
}

// -----

// Must use explicit v_accvgpr_read_b32 instead.
// CHECK-LABEL: amdgcn.kernel @agpr_to_vgpr_copy_rejected {
// CHECK-DAG:     %[[A:.*]] = alloca : !amdgcn.agpr<0>
// CHECK-DAG:     %[[V:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:         test_inst outs %[[A]] : (!amdgcn.agpr<0>) -> ()
// CHECK:         lsir.copy %[[A]], %[[V]] : !amdgcn.agpr<0>, !amdgcn.vgpr<0>
// CHECK:         test_inst ins %[[V]] : (!amdgcn.vgpr<0>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @agpr_to_vgpr_copy_rejected {
  %a = alloca : !amdgcn.agpr<?>
  %v = alloca : !amdgcn.vgpr<?>
  test_inst outs %a : (!amdgcn.agpr<?>) -> ()
  lsir.copy %a, %v : !amdgcn.agpr<?>, !amdgcn.vgpr<?>
  test_inst ins %v : (!amdgcn.vgpr<?>) -> ()
  end_kernel
}
