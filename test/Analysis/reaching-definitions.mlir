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
// CHECK:  Operation: `%{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>`
// CHECK:    results: [7 = `%{{.*}}`]Op: %{{.*}} = func.call @rand() : () -> i1
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
// CHECK:  Op: %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK:    ]
// CHECK:  Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
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
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK:      {3 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()<0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()<0>}
// CHECK:    ]
// CHECK-LOAD-LABEL: Function: load
// CHECK-LOAD:  Op: %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:    ]
// CHECK-LOAD:  Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:    ]
// CHECK-LOAD:  Op: amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
// CHECK-LOAD:    REACHING DEFS AFTER: []
// CHECK-LOAD:  Op: cf.br ^bb2
// CHECK-LOAD:    REACHING DEFS AFTER: []
// CHECK-LOAD:  Op: func.return
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:    ]
func.func @load() {
  %0 = call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %6 = amdgcn.make_register_range %3, %4 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dwordx2 dest %6 addr %5 : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
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
// CHECK:  Operation: `%{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>`
// CHECK:    results: [7 = `%{{.*}}`]Op: %{{.*}} = func.call @rand() : () -> i1
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
// CHECK:  Op: %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK:    ]
// CHECK:  Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK:    ]
// CHECK:  Op: cf.br ^bb2
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK:    ]
// CHECK:  Op: func.return
// CHECK:    REACHING DEFS AFTER: [
// CHECK:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK:    ]
// CHECK-LOAD-LABEL: Function: no_load_kill
// CHECK-LOAD:  Op: %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:    ]
// CHECK-LOAD:  Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:    ]
// CHECK-LOAD:  Op: cf.br ^bb2
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:    ]
// CHECK-LOAD:  Op: func.return
// CHECK-LOAD:    REACHING DEFS AFTER: [
// CHECK-LOAD:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
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
  %token = amdgcn.load global_load_dwordx2 dest %6 addr %5 : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  cf.br ^bb2
^bb2:  // preds: ^bb0, ^bb1
  return
}

// -----

// CHECK-LOAD-CONSUMED-LABEL: Function: one_load_two_consumers
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {0 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
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

// -----

// CHECK-LOAD-CONSUMED-LABEL: Function: load_before_branch_consumed_in_both
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
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

// -----

// CHECK-LOAD-CONSUMED-LABEL: Function: load_before_branch_consumed_in_single
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
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


// -----

// CHECK-LOAD-CONSUMED-LABEL: Function: load_before_branch_consumed_in_single_clobbered
// CHECK-LOAD-CONSUMED:  Op: %[[TOK_0:.*]] = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %[[TOK_0]] = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
// CHECK-LOAD-CONSUMED:  Op: %[[TOK_1:.*]] = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %[[TOK_0]] = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %[[TOK_1]] = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {1 = `%{{.*}}`, %[[TOK_1]] = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: []
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

// -----

// CHECK-LOAD-CONSUMED-LABEL: Function: load_in_branch_consumed_in_merge
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD-CONSUMED:      {3 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
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
  %token = amdgcn.load global_load_dword dest %2 addr %6 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %token_0 = amdgcn.load global_load_dword dest %2 addr %6 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  %7 = amdgcn.store global_store_dword data %2 addr %6 : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  return
}

// -----

// CHECK-LOAD-CONSUMED-LABEL: Function: load_in_loop
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: []
// CHECK-LOAD-CONSUMED:    REACHING DEFS AFTER: [
// CHECK-LOAD-CONSUMED:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
// CHECK-LOAD-CONSUMED:    ]
// CHECK-LOAD-CONSUMED:  Op: %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
// CHECK-LOAD-CONSUMED:    REACHING DEFS BEFORE: [
// CHECK-LOAD-CONSUMED:      {4 = `%{{.*}}`, %{{.*}} = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat><0>}
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
  %token = amdgcn.load global_load_dword dest %0 addr %3 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  %5 = amdgcn.store global_store_dword data %0 addr %3 : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  %6 = arith.addi %4, %c1 : index
  %7 = arith.cmpi ult, %6, %c4 : index
  cf.cond_br %7, ^bb1(%6 : index), ^bb2
^bb2:  // pred: ^bb1
  return
}
