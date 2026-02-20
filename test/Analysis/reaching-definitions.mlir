// RUN: aster-opt %s --test-reaching-definitions=only-loads=false 2>&1 | FileCheck %s
// RUN: aster-opt %s --test-reaching-definitions=only-loads=true 2>&1 | FileCheck %s --check-prefix=CHECK-LOAD

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
