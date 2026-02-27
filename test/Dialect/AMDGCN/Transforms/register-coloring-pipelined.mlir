// RUN: aster-opt --amdgcn-register-coloring --cse --split-input-file %s | FileCheck %s

func.func private @rand() -> i1
// CHECK-LABEL:   func.func @two_stage_load_basic() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_2]] : (!amdgcn.vgpr<2>) -> ()
// CHECK:           %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_2]] : (!amdgcn.vgpr<2>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
func.func @two_stage_load_basic() {
  %0 = func.call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %3 addr %5 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb1
  amdgcn.test_inst ins %4 : (!amdgcn.vgpr<?>) -> ()
  %token_0 = amdgcn.load global_load_dword dest %3 addr %5 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.cond_br %0, ^bb1, ^bb2
^bb2:  // pred: ^bb1
  amdgcn.test_inst ins %4 : (!amdgcn.vgpr<?>) -> ()
  return
}

// -----
func.func private @rand() -> i1
// CHECK-LABEL:   func.func @two_independent_loads() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<3>
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<3>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_2]], %[[ALLOCA_3]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
// CHECK:           %[[LOAD_2:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[LOAD_3:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<3>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_2]], %[[ALLOCA_3]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
func.func @two_independent_loads() {
  %0 = func.call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.alloca : !amdgcn.vgpr<?>
  %6 = amdgcn.alloca : !amdgcn.vgpr<?>
  %7 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %3 addr %7 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  %token_0 = amdgcn.load global_load_dword dest %5 addr %7 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  lsir.copy %6, %5 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb1
  amdgcn.test_inst ins %4, %6 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  %token_1 = amdgcn.load global_load_dword dest %3 addr %7 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  %token_2 = amdgcn.load global_load_dword dest %5 addr %7 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  lsir.copy %6, %5 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.cond_br %0, ^bb1, ^bb2
^bb2:  // pred: ^bb1
  amdgcn.test_inst ins %4, %6 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  return
}

// -----
func.func private @rand() -> i1
// CHECK-LABEL:   func.func @two_stage_load_dwordx2() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<3>
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dwordx2 dest %[[MAKE_REGISTER_RANGE_1]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<[2 : 4]>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           amdgcn.test_inst ins %[[MAKE_REGISTER_RANGE_1]] : (!amdgcn.vgpr<[2 : 4]>) -> ()
// CHECK:           %[[LOAD_1:.*]] = amdgcn.load global_load_dwordx2 dest %[[MAKE_REGISTER_RANGE_1]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<[2 : 4]>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           amdgcn.test_inst ins %[[MAKE_REGISTER_RANGE_1]] : (!amdgcn.vgpr<[2 : 4]>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
func.func @two_stage_load_dwordx2() {
  %0 = func.call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.alloca : !amdgcn.vgpr<?>
  %6 = amdgcn.alloca : !amdgcn.vgpr<?>
  %7 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %8 = amdgcn.make_register_range %3, %4 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %9 = amdgcn.make_register_range %5, %6 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dwordx2 dest %8 addr %7 : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %9, %8 : !amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<[? : ? + 2]>
  cf.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb1
  amdgcn.test_inst ins %9 : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
  %token_0 = amdgcn.load global_load_dwordx2 dest %8 addr %7 : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %9, %8 : !amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<[? : ? + 2]>
  cf.cond_br %0, ^bb1, ^bb2
^bb2:  // pred: ^bb1
  amdgcn.test_inst ins %9 : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
  return
}

// -----
func.func private @rand() -> i1
// CHECK-LABEL:   func.func @three_stage_load_compute_store() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<3>
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<4>
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_3]], %[[ALLOCA_2]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>) -> ()
// CHECK:           %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.test_inst outs %[[ALLOCA_4]] ins %[[ALLOCA_3]] : (!amdgcn.vgpr<4>, !amdgcn.vgpr<3>) -> ()
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_3]], %[[ALLOCA_2]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>) -> ()
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_4]] : (!amdgcn.vgpr<4>) -> ()
// CHECK:           %[[LOAD_2:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.test_inst outs %[[ALLOCA_4]] ins %[[ALLOCA_3]] : (!amdgcn.vgpr<4>, !amdgcn.vgpr<3>) -> ()
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_3]], %[[ALLOCA_2]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>) -> ()
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_4]] : (!amdgcn.vgpr<4>) -> ()
// CHECK:           amdgcn.test_inst outs %[[ALLOCA_4]] ins %[[ALLOCA_3]] : (!amdgcn.vgpr<4>, !amdgcn.vgpr<3>) -> ()
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_4]] : (!amdgcn.vgpr<4>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
func.func @three_stage_load_compute_store() {
  %0 = func.call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.alloca : !amdgcn.vgpr<?>
  %6 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %3 addr %6 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token_0 = amdgcn.load global_load_dword dest %3 addr %6 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.test_inst outs %5 ins %4 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb1
  amdgcn.test_inst ins %5 : (!amdgcn.vgpr<?>) -> ()
  %token_1 = amdgcn.load global_load_dword dest %3 addr %6 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.test_inst outs %5 ins %4 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.cond_br %0, ^bb1, ^bb2
^bb2:  // pred: ^bb1
  amdgcn.test_inst ins %5 : (!amdgcn.vgpr<?>) -> ()
  amdgcn.test_inst outs %5 ins %4 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  amdgcn.test_inst ins %5 : (!amdgcn.vgpr<?>) -> ()
  return
}

// -----
func.func private @rand() -> i1
// CHECK-LABEL:   func.func @pipeline_copy_interferes() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<3>
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_3]], %[[ALLOCA_2]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>) -> ()
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_3]], %[[ALLOCA_2]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>) -> ()
// CHECK:           %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_3]], %[[ALLOCA_2]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>) -> ()
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
func.func @pipeline_copy_interferes() {
  %0 = func.call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %3 addr %5 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb1
  amdgcn.test_inst ins %4, %3 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  %token_0 = amdgcn.load global_load_dword dest %3 addr %5 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.cond_br %0, ^bb1, ^bb2
^bb2:  // pred: ^bb1
  return
}

// -----
func.func private @rand() -> i1
// CHECK-LABEL:   func.func @mixed_coalesce_and_no_coalesce() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<3>
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<4>
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<3>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_4]], %[[ALLOCA_3]] : (!amdgcn.vgpr<4>, !amdgcn.vgpr<3>) -> ()
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_2]] : (!amdgcn.vgpr<2>) -> ()
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_4]], %[[ALLOCA_3]] : (!amdgcn.vgpr<4>, !amdgcn.vgpr<3>) -> ()
// CHECK:           %[[LOAD_2:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[LOAD_3:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<3>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_4]], %[[ALLOCA_3]] : (!amdgcn.vgpr<4>, !amdgcn.vgpr<3>) -> ()
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
func.func @mixed_coalesce_and_no_coalesce() {
  %0 = func.call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.alloca : !amdgcn.vgpr<?>
  %6 = amdgcn.alloca : !amdgcn.vgpr<?>
  %7 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %3 addr %7 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  %token_0 = amdgcn.load global_load_dword dest %5 addr %7 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  lsir.copy %6, %5 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb1
  amdgcn.test_inst ins %4 : (!amdgcn.vgpr<?>) -> ()
  amdgcn.test_inst ins %6, %5 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  %token_1 = amdgcn.load global_load_dword dest %3 addr %7 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  %token_2 = amdgcn.load global_load_dword dest %5 addr %7 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  lsir.copy %6, %5 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.cond_br %0, ^bb1, ^bb2
^bb2:  // pred: ^bb1
  return
}

// -----
func.func private @rand() -> i1
// CHECK-LABEL:   func.func @pipeline_chain_copies() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_2]] : (!amdgcn.vgpr<2>) -> ()
// CHECK:           %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_2]] : (!amdgcn.vgpr<2>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
func.func @pipeline_chain_copies() {
  %0 = func.call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.alloca : !amdgcn.vgpr<?>
  %6 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %3 addr %6 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  lsir.copy %5, %4 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb1
  amdgcn.test_inst ins %5 : (!amdgcn.vgpr<?>) -> ()
  %token_0 = amdgcn.load global_load_dword dest %3 addr %6 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  lsir.copy %5, %4 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.cond_br %0, ^bb1, ^bb2
^bb2:  // pred: ^bb1
  amdgcn.test_inst ins %5 : (!amdgcn.vgpr<?>) -> ()
  return
}

// -----
func.func private @rand() -> i1
// CHECK-LABEL:   func.func @two_stage_load_dwordx4() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<3>
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<4>
// CHECK:           %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr<5>
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_4]], %[[ALLOCA_5]] : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]], %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dwordx4 dest %[[MAKE_REGISTER_RANGE_1]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<[0 : 4]>) ins(!amdgcn.vgpr<[4 : 6]>) -> !amdgcn.read_token<flat>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           amdgcn.test_inst ins %[[MAKE_REGISTER_RANGE_1]] : (!amdgcn.vgpr<[0 : 4]>) -> ()
// CHECK:           %[[LOAD_1:.*]] = amdgcn.load global_load_dwordx4 dest %[[MAKE_REGISTER_RANGE_1]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<[0 : 4]>) ins(!amdgcn.vgpr<[4 : 6]>) -> !amdgcn.read_token<flat>
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           amdgcn.test_inst ins %[[MAKE_REGISTER_RANGE_1]] : (!amdgcn.vgpr<[0 : 4]>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
func.func @two_stage_load_dwordx4() {
  %0 = func.call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.alloca : !amdgcn.vgpr<?>
  %6 = amdgcn.alloca : !amdgcn.vgpr<?>
  %7 = amdgcn.alloca : !amdgcn.vgpr<?>
  %8 = amdgcn.alloca : !amdgcn.vgpr<?>
  %9 = amdgcn.alloca : !amdgcn.vgpr<?>
  %10 = amdgcn.alloca : !amdgcn.vgpr<?>
  %11 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %12 = amdgcn.make_register_range %3, %4, %5, %6 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %13 = amdgcn.make_register_range %7, %8, %9, %10 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dwordx4 dest %12 addr %11 : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %13, %12 : !amdgcn.vgpr<[? : ? + 4]>, !amdgcn.vgpr<[? : ? + 4]>
  cf.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb1
  amdgcn.test_inst ins %13 : (!amdgcn.vgpr<[? : ? + 4]>) -> ()
  %token_0 = amdgcn.load global_load_dwordx4 dest %12 addr %11 : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %13, %12 : !amdgcn.vgpr<[? : ? + 4]>, !amdgcn.vgpr<[? : ? + 4]>
  cf.cond_br %0, ^bb1, ^bb2
^bb2:  // pred: ^bb1
  amdgcn.test_inst ins %13 : (!amdgcn.vgpr<[? : ? + 4]>) -> ()
  return
}

// -----
func.func private @rand() -> i1
// CHECK-LABEL:   func.func @two_loads_to_compute() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<3>
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<3>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           amdgcn.test_inst outs %[[ALLOCA_2]] ins %[[ALLOCA_2]], %[[ALLOCA_3]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
// CHECK:           %[[LOAD_2:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[LOAD_3:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<3>) ins(!amdgcn.vgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           amdgcn.test_inst outs %[[ALLOCA_2]] ins %[[ALLOCA_2]], %[[ALLOCA_3]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
// CHECK:           return
// CHECK:         }
func.func @two_loads_to_compute() {
  %0 = func.call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.alloca : !amdgcn.vgpr<?>
  %6 = amdgcn.alloca : !amdgcn.vgpr<?>
  %7 = amdgcn.alloca : !amdgcn.vgpr<?>
  %8 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %3 addr %8 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  %token_0 = amdgcn.load global_load_dword dest %5 addr %8 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  lsir.copy %6, %5 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb1
  amdgcn.test_inst outs %7 ins %4, %6 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  %token_1 = amdgcn.load global_load_dword dest %3 addr %8 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  %token_2 = amdgcn.load global_load_dword dest %5 addr %8 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %4, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  lsir.copy %6, %5 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.cond_br %0, ^bb1, ^bb2
^bb2:  // pred: ^bb1
  amdgcn.test_inst outs %7 ins %4, %6 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  return
}
