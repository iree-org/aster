// RUN: aster-opt %s --amdgcn-wait-insertion | FileCheck %s

// CHECK-LABEL:   func.func @simple_load_store() {
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_1]], %[[ALLOCA_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_0]], %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[LOAD_1:.*]] = memref.load %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_1]] : !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @simple_load_store() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %3 addr %2 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  %4 = amdgcn.store global_store_dword data %3 addr %2 : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  return
}

// CHECK-LABEL:   func.func @load_then_vop() {
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_3]], %[[ALLOCA_4]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_1]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_0]], %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[LOAD_1:.*]] = memref.load %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_1]] : !amdgcn.read_token<flat>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_2]], %[[ALLOCA_1]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_2]] : (!amdgcn.vgpr<?>) -> ()
// CHECK:           return
// CHECK:         }
func.func @load_then_vop() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.make_register_range %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %0 addr %4 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.vop1.vop1 <v_mov_b32_e32> %1, %0 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  amdgcn.test_inst ins %1 : (!amdgcn.vgpr<?>) -> ()
  return
}

// CHECK-LABEL:   func.func @multiple_loads_multiple_consumers() {
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[ALLOCA_1:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 8 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_4]], %[[ALLOCA_5]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] offset c(%[[CONSTANT_2]]) : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>, i32) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_0]], %[[ALLOCA_1]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] offset c(%[[CONSTANT_1]]) : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>, i32) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_1]], %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[LOAD_2:.*]] = memref.load %[[ALLOCA_1]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_2]] : !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] offset c(%[[CONSTANT_2]]) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>, i32) -> !amdgcn.write_token<flat>
// CHECK:           %[[LOAD_3:.*]] = memref.load %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_3]] : !amdgcn.read_token<flat>
// CHECK:           %[[STORE_1:.*]] = amdgcn.store global_store_dword data %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>, i32) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @multiple_loads_multiple_consumers() {
  %c8_i32 = arith.constant 8 : i32
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.make_register_range %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %0 addr %4 offset c(%c0_i32) : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>, i32) -> !amdgcn.read_token<flat>
  %token_0 = amdgcn.load global_load_dword dest %1 addr %4 offset c(%c4_i32) : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>, i32) -> !amdgcn.read_token<flat>
  %5 = amdgcn.store global_store_dword data %0 addr %4 offset c(%c0_i32) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>, i32) -> !amdgcn.write_token<flat>
  %6 = amdgcn.store global_store_dword data %1 addr %4 offset c(%c8_i32) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>, i32) -> !amdgcn.write_token<flat>
  return
}

// CHECK-LABEL:   func.func @load_with_existing_wait() {
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_1]], %[[ALLOCA_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_0]], %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_0]] : !amdgcn.read_token<flat>
// CHECK:           %[[LOAD_1:.*]] = memref.load %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_1]] : !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @load_with_existing_wait() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %3 addr %2 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token : !amdgcn.read_token<flat>
  %4 = amdgcn.store global_store_dword data %3 addr %2 : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  return
}

// CHECK-LABEL:   func.func @load_in_branch_consumed_in_merge(
// CHECK-SAME:      %[[ARG0:.*]]: i1) {
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[ALLOCA_1:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_6:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_7:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_6]], %[[ALLOCA_7]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           cf.cond_br %[[ARG0]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_4]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_0]], %[[ALLOCA_1]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[LOAD_1:.*]] = memref.load %[[ALLOCA_1]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_1]] : !amdgcn.read_token<flat>
// CHECK:           lsir.copy %[[ALLOCA_2]], %[[ALLOCA_4]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[LOAD_2:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_5]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_2]], %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[LOAD_3:.*]] = memref.load %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_3]] : !amdgcn.read_token<flat>
// CHECK:           lsir.copy %[[ALLOCA_2]], %[[ALLOCA_5]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           lsir.copy %[[ALLOCA_3]], %[[ALLOCA_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @load_in_branch_consumed_in_merge(%arg0: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.alloca : !amdgcn.vgpr<?>
  %6 = amdgcn.make_register_range %4, %5 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %token = amdgcn.load global_load_dword dest %2 addr %6 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %0, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %token_0 = amdgcn.load global_load_dword dest %3 addr %6 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  lsir.copy %0, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  lsir.copy %1, %0 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %7 = amdgcn.store global_store_dword data %1 addr %6 : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  return
}

// CHECK-LABEL:   func.func @no_load_consumption() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_1]], %[[ALLOCA_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ALLOCA_0]] addr %[[MAKE_REGISTER_RANGE_0]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @no_load_consumption() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %4 = amdgcn.store global_store_dword data %0 addr %3 : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  return
}

// CHECK-LABEL:   func.func @load_in_loop() {
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           cf.br ^bb1(%[[CONSTANT_2]] : index)
// CHECK:         ^bb1(%[[VAL_0:.*]]: index):
// CHECK:           %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_1]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_1]], %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[LOAD_2:.*]] = memref.load %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_2]] : !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ALLOCA_1]] addr %[[MAKE_REGISTER_RANGE_0]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi ult, %[[ADDI_0]], %[[CONSTANT_0]] : index
// CHECK:           cf.cond_br %[[CMPI_0]], ^bb1(%[[ADDI_0]] : index), ^bb2
// CHECK:         ^bb2:
// CHECK:           return
// CHECK:         }
func.func @load_in_loop() {
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb1(%c0 : index)
^bb1(%4: index):  // 2 preds: ^bb0, ^bb1
  %token = amdgcn.load global_load_dword dest %0 addr %3 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  %5 = amdgcn.store global_store_dword data %0 addr %3 : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  %6 = arith.addi %4, %c1 : index
  %7 = arith.cmpi ult, %6, %c4 : index
  cf.cond_br %7, ^bb1(%6 : index), ^bb2
^bb2:  // pred: ^bb1
  return
}

// CHECK-LABEL:   func.func @ds_load_store() {
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<!amdgcn.read_token<shared>>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load ds_read_b32 dest %[[ALLOCA_1]] addr %[[ALLOCA_2]] : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<?>) -> !amdgcn.read_token<shared>
// CHECK:           memref.store %[[LOAD_0]], %[[ALLOCA_0]][] : memref<!amdgcn.read_token<shared>>
// CHECK:           %[[LOAD_1:.*]] = memref.load %[[ALLOCA_0]][] : memref<!amdgcn.read_token<shared>>
// CHECK:           amdgcn.wait deps %[[LOAD_1]] : !amdgcn.read_token<shared>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b32 data %[[ALLOCA_1]] addr %[[ALLOCA_2]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.write_token<shared>
// CHECK:           return
// CHECK:         }
func.func @ds_load_store() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %token = amdgcn.load ds_read_b32 dest %0 addr %1 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<?>) -> !amdgcn.read_token<shared>
  %2 = amdgcn.store ds_write_b32 data %0 addr %1 : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.write_token<shared>
  return
}

// CHECK-LABEL:   func.func @one_load_two_consumers() {
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK-NOT:       memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[MRR:.*]] = amdgcn.make_register_range %[[ALLOCA_3]], %[[ALLOCA_4]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_1]] addr %[[MRR]] : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_0]], %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK-NOT:       memref.store
// CHECK:           %{{.*}} = amdgcn.store global_store_dword data %[[ALLOCA_1]] addr %[[MRR]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           lsir.copy %[[ALLOCA_2]], %[[ALLOCA_1]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           return
// CHECK:         }
func.func @one_load_two_consumers() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %addr = amdgcn.make_register_range %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %0 addr %addr
      : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  %w = amdgcn.store global_store_dword data %0 addr %addr
      : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  lsir.copy %1, %0 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  return
}

// CHECK-LABEL:   func.func @load_killed_before_consumer() {
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[MRR:.*]] = amdgcn.make_register_range %[[ALLOCA_3]], %[[ALLOCA_4]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_1]] addr %[[MRR]] : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_0]], %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[TOK_0:.*]] = memref.load %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[TOK_0]] : !amdgcn.read_token<flat>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_1]], %[[ALLOCA_2]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK-NOT:       amdgcn.wait
// CHECK:           %{{.*}} = amdgcn.store global_store_dword data %[[ALLOCA_1]] addr %[[MRR]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @load_killed_before_consumer() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %addr = amdgcn.make_register_range %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %0 addr %addr
      : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.vop1.vop1 <v_mov_b32_e32> %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  %w = amdgcn.store global_store_dword data %0 addr %addr
      : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  return
}

// CHECK-LABEL:   func.func @load_before_branch_consumed_in_both(
//  CHECK-SAME:       %[[ARG0:.*]]: i1
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK-NOT:       memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[MRR:.*]] = amdgcn.make_register_range %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_1]] addr %[[MRR]] : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_0]], %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           cf.cond_br %[[ARG0]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[TOK_0:.*]] = memref.load %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[TOK_0]] : !amdgcn.read_token<flat>
// CHECK:           %{{.*}} = amdgcn.store global_store_dword data %[[ALLOCA_1]] addr %[[MRR]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[TOK_1:.*]] = memref.load %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[TOK_1]] : !amdgcn.read_token<flat>
// CHECK:           %{{.*}} = amdgcn.store global_store_dword data %[[ALLOCA_1]] addr %[[MRR]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           return
// CHECK:         }
func.func @load_before_branch_consumed_in_both(%cond: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %addr = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %0 addr %addr
      : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %w0 = amdgcn.store global_store_dword data %0 addr %addr
      : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  cf.br ^bb3
^bb2:
  %w1 = amdgcn.store global_store_dword data %0 addr %addr
      : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  cf.br ^bb3
^bb3:
  return
}

// CHECK-LABEL:   func.func @load_before_branch_consumed_in_single(
// CHECK-SAME:      %[[ARG0:.*]]: i1) {
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_1]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_0]], %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           cf.cond_br %[[ARG0]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[LOAD_1:.*]] = memref.load %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_1]] : !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ALLOCA_1]] addr %[[MAKE_REGISTER_RANGE_0]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           %[[LOAD_2:.*]] = memref.load %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_2]] : !amdgcn.read_token<flat>
// CHECK:           %[[STORE_1:.*]] = amdgcn.store global_store_dword data %[[ALLOCA_1]] addr %[[MAKE_REGISTER_RANGE_0]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @load_before_branch_consumed_in_single(%cond: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %addr = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %0 addr %addr
      : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %w0 = amdgcn.store global_store_dword data %0 addr %addr
      : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  cf.br ^bb3
^bb2:
  cf.br ^bb3
^bb3:
  %w1 = amdgcn.store global_store_dword data %0 addr %addr
      : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  return
}

// CHECK-LABEL:   func.func @load_before_branch_consumed_in_single_clobbered(
// CHECK-SAME:      %[[ARG0:.*]]: i1) {
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[ALLOCA_1:.*]] = memref.alloca() : memref<!amdgcn.read_token<flat>>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_3]], %[[ALLOCA_4]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_0]], %[[ALLOCA_1]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           cf.cond_br %[[ARG0]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[LOAD_1:.*]] = memref.load %[[ALLOCA_1]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_1]] : !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[LOAD_2:.*]] = memref.load %[[ALLOCA_1]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_2]] : !amdgcn.read_token<flat>
// CHECK:           %[[LOAD_3:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           memref.store %[[LOAD_3]], %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           %[[LOAD_4:.*]] = memref.load %[[ALLOCA_0]][] : memref<!amdgcn.read_token<flat>>
// CHECK:           amdgcn.wait deps %[[LOAD_4]] : !amdgcn.read_token<flat>
// CHECK:           %[[STORE_1:.*]] = amdgcn.store global_store_dword data %[[ALLOCA_2]] addr %[[MAKE_REGISTER_RANGE_0]] : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @load_before_branch_consumed_in_single_clobbered(%cond: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %addr = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %0 addr %addr
      : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %w0 = amdgcn.store global_store_dword data %0 addr %addr
      : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  cf.br ^bb3
^bb2:
  %token_2 = amdgcn.load global_load_dword dest %0 addr %addr
      : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  cf.br ^bb3
^bb3:
  %w1 = amdgcn.store global_store_dword data %0 addr %addr
      : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  return
}
