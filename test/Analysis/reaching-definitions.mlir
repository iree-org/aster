// RUN: aster-opt %s --test-reaching-definitions=only-loads=false --split-input-file 2>&1 | FileCheck %s
// RUN: aster-opt %s --test-reaching-definitions=only-loads=true --split-input-file 2>&1 | FileCheck %s --check-prefix=CHECK-LOAD
// RUN: aster-opt %s --test-reaching-definitions="only-loads=true kill-consumed-loads=true" --split-input-file 2>&1 | FileCheck %s --check-prefix=CHECK-LOAD
// RUN: aster-opt %s --test-reaching-definitions="only-loads=true kill-consumed-loads=true" --split-input-file 2>&1 | FileCheck %s --check-prefix=CHECK-LOAD-CONSUMED

func.func private @rand() -> i1
// CHECK-LABEL: Function: diamond_reaching
// CHECK-LOAD-LABEL: Function: diamond_reaching
// CHECK:  Operation: `%{{.*}} = func.call @rand() : () -> i1`
// CHECK:    results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:    results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:    results: [4 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [5 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [6 = `%{{.*}}`]Op: %{{.*}} = func.call @rand() : () -> i1
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {2 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {2 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {2 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {5 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: cf.br ^bb3
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {2 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {5 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {2 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {6 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: cf.br ^bb3
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {2 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {6 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {2 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {5 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:      {6 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: func.return
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {2 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()<0>}
// CHECK:      {5 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:      {6 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK-LOAD-NOT: \{{{.*}}\}
func.func @diamond_reaching() {
  %0 = call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.sgpr<?>
  %4 = amdgcn.alloca : !amdgcn.sgpr<?>
  %5 = amdgcn.alloca : !amdgcn.vgpr<?>
  %6 = amdgcn.alloca : !amdgcn.vgpr<?>
  amdgcn.test_inst outs %1 ins %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()
  amdgcn.test_inst outs %2 ins %4 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  amdgcn.test_inst outs %5 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  cf.br ^bb3
^bb2:  // pred: ^bb0
  amdgcn.test_inst outs %6 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  cf.br ^bb3
^bb3:  // preds: ^bb1, ^bb2
  amdgcn.test_inst ins %5, %6 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  return
}

// CHECK-LABEL: Function: multi_def
// CHECK-LOAD-LABEL: Function: multi_def
// CHECK:  Operation: `%{{.*}} = func.call @rand() : () -> i1`
// CHECK:    results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [2 = `%{{.*}}`]Op: %{{.*}} = func.call @rand() : () -> i1
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: cf.br ^bb2
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: func.return
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:      {1 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()<0>}
// CHECK:    ]
// CHECK-LOAD-NOT: \{{{.*}}\}
func.func @multi_def() {
  %0 = call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  amdgcn.test_inst outs %1 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  amdgcn.test_inst outs %1 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  cf.br ^bb2
^bb2:  // preds: ^bb0, ^bb1
  amdgcn.test_inst ins %1, %2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  return
}

// CHECK-LABEL: Function: load
// CHECK:  Operation: `%{{.*}} = func.call @rand() : () -> i1`
// CHECK:    results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [4 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:    results: [5 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:    results: [6 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = arith.constant 0 : i32`
// CHECK:    results: [7 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>`
// CHECK:    results: [8 = `%{{.*}}`]Op: %{{.*}} = func.call @rand() : () -> i1
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = arith.constant 0 : i32
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:    ]
// CHECK:  Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:    ]
// CHECK:  Op: amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()<0>}
// CHECK:      {4 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: cf.br ^bb2
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()<0>}
// CHECK:      {4 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: func.return
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:      {3 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()<0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()<0>}
// CHECK:    ]
// CHECK-LOAD-LABEL: Function: load
// CHECK-LOAD:  Op: %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:    ]
// CHECK-LOAD:  Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:    ]
// CHECK-LOAD:  Op: amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
// CHECK-LOAD:    REACHING DEFS AFTER: []
// CHECK-LOAD:  Op: cf.br ^bb2
// CHECK-LOAD:    REACHING DEFS AFTER: []
// CHECK-LOAD:  Op: func.return
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:    ]
func.func @load() {
  %0 = call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %6 = amdgcn.make_register_range %3, %4 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %c0_i32_mig1 = arith.constant 0 : i32
  %token = amdgcn.global_load_dwordx2 dest %6 addr %5 offset c(%c0_i32_mig1) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  amdgcn.test_inst outs %6 : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
  cf.br ^bb2
^bb2:  // preds: ^bb0, ^bb1
  return
}

// CHECK-LABEL: Function: no_load_kill
// CHECK:  Operation: `%{{.*}} = func.call @rand() : () -> i1`
// CHECK:    results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:    results: [4 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:    results: [5 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:    results: [6 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = arith.constant 0 : i32`
// CHECK:    results: [7 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>`
// CHECK:    results: [8 = `%{{.*}}`]Op: %{{.*}} = func.call @rand() : () -> i1
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: %{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:    REACHING DEFS AFTER: []
// CHECK:  Op: amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()<0>}
// CHECK:      {4 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: %{{.*}} = arith.constant 0 : i32
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()<0>}
// CHECK:      {4 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()<0>}
// CHECK:    ]
// CHECK:  Op: %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:    ]
// CHECK:  Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:    ]
// CHECK:  Op: cf.br ^bb2
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:    ]
// CHECK:  Op: func.return
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK:    ]
// CHECK-LOAD-LABEL: Function: no_load_kill
// CHECK-LOAD:  Op: %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:    ]
// CHECK-LOAD:  Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:    ]
// CHECK-LOAD:  Op: cf.br ^bb2
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:    ]
// CHECK-LOAD:  Op: func.return
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dwordx2 dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:    ]
// CHECK-LOAD:  Op: func.func @no_load_kill() {...}
// CHECK-LOAD:    REACHING DEFS AFTER: []
func.func @no_load_kill() {
  %0 = call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %6 = amdgcn.make_register_range %3, %4 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst outs %6 : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
  %c0_i32_mig2 = arith.constant 0 : i32
  %token = amdgcn.global_load_dwordx2 dest %6 addr %5 offset c(%c0_i32_mig2) : outs(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  cf.br ^bb2
^bb2:  // preds: ^bb0, ^bb1
  return
}

// -----

// CHECK-LOAD-CONSUMED-LABEL: Function: one_load_two_consumers
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {0 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dword
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
func.func @one_load_two_consumers() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %addr = amdgcn.make_register_range %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %c0_i32_mig3 = arith.constant 0 : i32
  %token = amdgcn.global_load_dword dest %0 addr %addr offset c(%c0_i32_mig3) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig1 = arith.constant 0 : i32
  %w = amdgcn.global_store_dword data %0 addr %addr offset c(%c0_i32_mig1) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
  lsir.copy %1, %0 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  return
}

// -----

// CHECK-LOAD-CONSUMED-LABEL: Function: load_before_branch_consumed_in_both
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dword
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dword
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
func.func @load_before_branch_consumed_in_both(%cond: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %addr = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %c0_i32_mig4 = arith.constant 0 : i32
  %token = amdgcn.global_load_dword dest %0 addr %addr offset c(%c0_i32_mig4) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %c0_i32_mig2 = arith.constant 0 : i32
  %w0 = amdgcn.global_store_dword data %0 addr %addr offset c(%c0_i32_mig2) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
  cf.br ^bb3
^bb2:
  %c0_i32_mig3 = arith.constant 0 : i32
  %w1 = amdgcn.global_store_dword data %0 addr %addr offset c(%c0_i32_mig3) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
  cf.br ^bb3
^bb3:
  return
}

// -----

// CHECK-LOAD-CONSUMED-LABEL: Function: load_before_branch_consumed_in_single
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dword
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dword
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
func.func @load_before_branch_consumed_in_single(%cond: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %addr = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %c0_i32_mig5 = arith.constant 0 : i32
  %token = amdgcn.global_load_dword dest %0 addr %addr offset c(%c0_i32_mig5) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %c0_i32_mig4 = arith.constant 0 : i32
  %w0 = amdgcn.global_store_dword data %0 addr %addr offset c(%c0_i32_mig4) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
  cf.br ^bb3
^bb2:
  cf.br ^bb3
^bb3:
  %w1 = amdgcn.global_store_dword data %0 addr %addr offset c(%c0_i32_mig5) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
  return
}


// -----

// CHECK-LOAD-CONSUMED-LABEL: Function: load_before_branch_consumed_in_single_clobbered
// CHECK-LOAD-CONSUMED:  Op: %[[TOK_0:.*]] = amdgcn.global_load_dword dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %[[TOK_0:.*]] = amdgcn.global_load_dword
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
// CHECK-LOAD-CONSUMED:  Op: %[[TOK_1:.*]] = amdgcn.global_load_dword dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %[[TOK_0]] = amdgcn.global_load_dword
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %[[TOK_1]] = amdgcn.global_load_dword
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:     {1 = `%{{.*}}`, %[[TOK_1]] = amdgcn.global_load_dword
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
func.func @load_before_branch_consumed_in_single_clobbered(%cond: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %addr = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %c0_i32_mig6 = arith.constant 0 : i32
  %token = amdgcn.global_load_dword dest %0 addr %addr offset c(%c0_i32_mig6) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %w0 = amdgcn.global_store_dword data %0 addr %addr offset c(%c0_i32_mig6) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
  cf.br ^bb3
^bb2:
  %c0_i32_mig7 = arith.constant 0 : i32
  %token_2 = amdgcn.global_load_dword dest %0 addr %addr offset c(%c0_i32_mig7) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  cf.br ^bb3
^bb3:
  %c0_i32_mig7b = arith.constant 0 : i32
  %w1 = amdgcn.global_store_dword data %0 addr %addr offset c(%c0_i32_mig7b) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
  return
}

// -----

// CHECK-LOAD-CONSUMED-LABEL: Function: load_in_branch_consumed_in_merge
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dword
// CHECK-LOAD-CONSUMED:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dword
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
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
  %c0_i32_mig8 = arith.constant 0 : i32
  %token = amdgcn.global_load_dword dest %2 addr %6 offset c(%c0_i32_mig8) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %c0_i32_mig9 = arith.constant 0 : i32
  %token_0 = amdgcn.global_load_dword dest %2 addr %6 offset c(%c0_i32_mig9) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  %c0_i32_mig8b = arith.constant 0 : i32
  %7 = amdgcn.global_store_dword data %2 addr %6 offset c(%c0_i32_mig8b) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
  return
}

// -----

// CHECK-LOAD-CONSUMED-LABEL: Function: load_in_loop
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.global_load_dword dest %{{.*}} addr %{{.*}} offset c(%{{.*}}) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: []
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: [
// CHECK-LOAD-CONSUMED:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dword
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.global_load_dword
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
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
  %c0_i32_mig10 = arith.constant 0 : i32
  %token = amdgcn.global_load_dword dest %0 addr %3 offset c(%c0_i32_mig10) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig9 = arith.constant 0 : i32
  %5 = amdgcn.global_store_dword data %0 addr %3 offset c(%c0_i32_mig9) : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
  %6 = arith.addi %4, %c1 : index
  %7 = arith.cmpi ult, %6, %c4 : index
  cf.cond_br %7, ^bb1(%6 : index), ^bb2
^bb2:  // pred: ^bb1
  return
}
