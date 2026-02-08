// RUN: aster-opt --split-input-file --test-amdgcn-liveness-analysis %s | FileCheck %s

// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [0 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [1 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [2 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [3 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [4 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [5 = `%{{.*}}`]
// CHECK: Op: module {...}
// CHECK:   LIVE BEFORE: []
// CHECK-LABEL: no_interference_mixed
// CHECK: Op: amdgcn.kernel @no_interference_mixed {...}
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`, 3 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [3 = `%{{.*}}`]
// CHECK: Op: amdgcn.end_kernel
// CHECK:   LIVE BEFORE: []
amdgcn.kernel @no_interference_mixed {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = test_inst outs %0 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  %5 = test_inst outs %1 ins %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  end_kernel
}

// -----
// CHECK-LABEL: SSA map
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [0 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [1 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [2 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [3 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [4 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [5 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [6 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [7 = `%{{.*}}`]
// CHECK: Op: module {...}
// CHECK:   LIVE BEFORE: []
// CHECK-LABEL: interference_mixed_all_live
// CHECK: Op: amdgcn.kernel @interference_mixed_all_live {...}
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`, 3 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`, 3 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`, 3 = `%{{.*}}`, 4 = `%{{.*}}`, 5 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 3 = `%{{.*}}`, 4 = `%{{.*}}`, 5 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`, 5 = `%{{.*}}`]
// CHECK: Op: amdgcn.end_kernel
// CHECK:   LIVE BEFORE: []
amdgcn.kernel @interference_mixed_all_live {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = test_inst outs %0 ins %2, %4 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %7 = test_inst outs %1 ins %3, %5 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  test_inst ins %0, %1, %4, %5 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL: SSA map
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [0 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [1 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [2 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [3 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [4 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [5 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [6 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [7 = `%{{.*}}`]
// CHECK: Op: module {...}
// CHECK:   LIVE BEFORE: []
// CHECK-LABEL: interference_mixed_with_reuse
// CHECK: Op: amdgcn.kernel @interference_mixed_with_reuse {...}
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`, 3 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`, 3 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`, 3 = `%{{.*}}`, 4 = `%{{.*}}`, 5 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 3 = `%{{.*}}`, 5 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`]
// CHECK: Op: amdgcn.end_kernel
// CHECK:   LIVE BEFORE: []
amdgcn.kernel @interference_mixed_with_reuse {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = test_inst outs %0 ins %2, %4 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %7 = test_inst outs %1 ins %3, %5 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  test_inst ins %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL: SSA map
// CHECK-LABEL: func.call @rand()
// CHECK:   results: [0 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [1 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [2 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [3 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [4 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [5 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [6 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [7 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [8 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [9 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [10 = `%{{.*}}`]
// CHECK: Op: module {...}
// CHECK:   LIVE BEFORE: []
// CHECK-LABEL: rand
// CHECK: Op: func.func private @rand() -> i1
// CHECK:   LIVE BEFORE: []
// CHECK-LABEL: interference_cf
// CHECK: Op: amdgcn.kernel @interference_cf {...}
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = func.call @rand() : () -> i1
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: [3 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [3 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [3 = `%{{.*}}`, 4 = `%{{.*}}`, 5 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [3 = `%{{.*}}`, 4 = `%{{.*}}`, 5 = `%{{.*}}`, 6 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 4 = `%{{.*}}`, 5 = `%{{.*}}`, 6 = `%{{.*}}`]
// CHECK: Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 2 = `%{{.*}}`, 5 = `%{{.*}}`, 6 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 6 = `%{{.*}}`]
// CHECK: Op: cf.br ^bb3
// CHECK:   LIVE BEFORE: [5 = `%{{.*}}`, 6 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`, 5 = `%{{.*}}`]
// CHECK: Op: cf.br ^bb3
// CHECK:   LIVE BEFORE: [5 = `%{{.*}}`, 6 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:   LIVE BEFORE: [5 = `%{{.*}}`, 6 = `%{{.*}}`]
// CHECK: Op: amdgcn.end_kernel
// CHECK:   LIVE BEFORE: []
func.func private @rand() -> i1
amdgcn.kernel @interference_cf {
  %0 = func.call @rand() : () -> i1
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.sgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  %8 = test_inst outs %2 ins %4 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // CHECK: pred: ^bb0
  %9 = test_inst outs %5 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  cf.br ^bb3
^bb2:  // CHECK: pred: ^bb0
  %10 = test_inst outs %6 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  cf.br ^bb3
^bb3:  // CHECK: 2 preds: ^bb1, ^bb2
  test_inst ins %5, %6 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL: SSA map
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [0 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [1 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [2 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [3 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [4 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [5 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [6 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [7 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:   results: [8 = `%{{.*}}`]
// CHECK: Op: module {...}
// CHECK:   LIVE BEFORE: []
// CHECK-LABEL: test_make_range_liveness
// CHECK: Op: amdgcn.kernel @test_make_range_liveness {...}
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr_range<[? : ? + 2]>, !amdgcn.vgpr<?>) -> ()
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: amdgcn.end_kernel
// CHECK:   LIVE BEFORE: []
amdgcn.kernel @test_make_range_liveness {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %3 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = test_inst outs %4 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = test_inst outs %6 ins %4 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %8 = make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  test_inst ins %8, %4 : (!amdgcn.vgpr_range<[? : ? + 2]>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL: SSA map
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [0 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [1 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [2 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [3 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [4 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [5 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [6 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [7 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:   results: [8 = `%{{.*}}`]
// CHECK: Op: module {...}
// CHECK:   LIVE BEFORE: []
// CHECK-LABEL: test_make_range_liveness_1
// CHECK: Op: amdgcn.kernel @test_make_range_liveness_1 {...}
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr_range<[? : ? + 2]>, !amdgcn.vgpr<?>) -> ()
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: amdgcn.end_kernel
// CHECK:   LIVE BEFORE: []
amdgcn.kernel @test_make_range_liveness_1 {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %3 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = test_inst outs %4 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = test_inst outs %6 ins %4 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %8 = make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  test_inst ins %8, %4 : (!amdgcn.vgpr_range<[? : ? + 2]>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL: SSA map
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [0 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [1 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [2 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [3 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [4 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [5 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [6 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [7 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:   results: [8 = `%{{.*}}`]
// CHECK: Op: module {...}
// CHECK:   LIVE BEFORE: []
// CHECK-LABEL: test_make_range_liveness_2
// CHECK: Op: amdgcn.kernel @test_make_range_liveness_2 {...}
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}} : (!amdgcn.vgpr<?>) -> ()
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}} : (!amdgcn.vgpr_range<[? : ? + 2]>) -> ()
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`]
// CHECK: Op: amdgcn.end_kernel
// CHECK:   LIVE BEFORE: []
amdgcn.kernel @test_make_range_liveness_2 {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %3 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = test_inst outs %4 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = test_inst outs %6 ins %4 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  test_inst ins %4 : (!amdgcn.vgpr<?>) -> ()
  %8 = make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  test_inst ins %8 : (!amdgcn.vgpr_range<[? : ? + 2]>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL: SSA map
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [0 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [1 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [2 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [3 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [4 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [5 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [6 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [7 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:   results: [8 = `%{{.*}}`]
// CHECK: Op: module {...}
// CHECK:   LIVE BEFORE: []
// CHECK-LABEL: test_make_range_liveness_3
// CHECK: Op: amdgcn.kernel @test_make_range_liveness_3 {...}
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}} : (!amdgcn.vgpr_range<[? : ? + 2]>) -> ()
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}} : (!amdgcn.vgpr<?>) -> ()
// CHECK:   LIVE BEFORE: [4 = `%{{.*}}`]
// CHECK: Op: amdgcn.end_kernel
// CHECK:   LIVE BEFORE: []
amdgcn.kernel @test_make_range_liveness_3 {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %3 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = test_inst outs %4 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = test_inst outs %6 ins %4 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %8 = make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  test_inst ins %8 : (!amdgcn.vgpr_range<[? : ? + 2]>) -> ()
  test_inst ins %4 : (!amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL: SSA map
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [0 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [1 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [2 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [3 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [4 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [5 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [6 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [7 = `%{{.*}}`]
// CHECK: Op: module {...}
// CHECK:   LIVE BEFORE: []
// CHECK-LABEL: reg_interference
// CHECK: Op: amdgcn.kernel @reg_interference {...}
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
// CHECK:   LIVE BEFORE: [0 = `%{{.*}}`, 1 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`, 3 = `%{{.*}}`]
// CHECK: Op: amdgcn.reg_interference %{{.*}}, %{{.*}}, %{{.*}} : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: [4 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
// CHECK:   LIVE BEFORE: [4 = `%{{.*}}`, 5 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: [6 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
// CHECK:   LIVE BEFORE: [6 = `%{{.*}}`, 7 = `%{{.*}}`]
// CHECK: Op: amdgcn.reg_interference %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: amdgcn.end_kernel
// CHECK:   LIVE BEFORE: []
amdgcn.kernel @reg_interference {
  %0 = alloca : !amdgcn.sgpr<?>
  %1 = alloca : !amdgcn.sgpr<?>
  test_inst ins %0, %1 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  test_inst ins %2, %3 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  reg_interference %0, %2, %3 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.sgpr<?>
  %5 = alloca : !amdgcn.sgpr<?>
  test_inst ins %4, %5 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  %6 = alloca : !amdgcn.sgpr<?>
  %7 = alloca : !amdgcn.sgpr<?>
  test_inst ins %6, %7 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  reg_interference %4, %1, %3, %7 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  end_kernel
}

// -----
// CHECK-LABEL: SSA map
// CHECK: `%{{.*}} = arith.constant 0 : i32`
// CHECK:   results: [0 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [1 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [2 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [3 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [4 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [5 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [6 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [7 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [8 = `%{{.*}}`]
// CHECK: `%{{.*}} = lsir.cmpi i32 eq %{{.*}}, %{{.*}} : !amdgcn.sgpr<?>, i32`
// CHECK:   results: [9 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [10 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [11 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [12 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [13 = `%{{.*}}`]
// CHECK: `%{{.*}} = lsir.copy %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:   results: [14 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [15 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [16 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [17 = `%{{.*}}`]
// CHECK: `%{{.*}} = lsir.copy %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:   results: [18 = `%{{.*}}`]
// CHECK: Op: module {...}
// CHECK:   LIVE BEFORE: []
// CHECK-LABEL: phi_coalescing_2
// CHECK: Op: amdgcn.kernel @phi_coalescing_2 {...}
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = arith.constant 0 : i32
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: [3 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [3 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [3 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [3 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 3 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = lsir.cmpi i32 eq %{{.*}}, %{{.*}} : !amdgcn.sgpr<?>, i32
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 2 = `%{{.*}}`, 3 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 2 = `%{{.*}}`]
// CHECK: Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 2 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = lsir.copy %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [12 = `%{{.*}}`]
// CHECK: Op: cf.br ^bb3
// CHECK:   LIVE BEFORE: [10 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [2 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = lsir.copy %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [16 = `%{{.*}}`]
// CHECK: Op: cf.br ^bb3
// CHECK:   LIVE BEFORE: [10 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}} : (!amdgcn.vgpr<?>) -> ()
// CHECK:   LIVE BEFORE: [10 = `%{{.*}}`]
// CHECK: Op: amdgcn.end_kernel
// CHECK:   LIVE BEFORE: []
amdgcn.kernel @phi_coalescing_2 {
  %c0_i32 = arith.constant 0 : i32
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = test_inst outs %0 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  %8 = lsir.cmpi i32 eq %2, %c0_i32 : !amdgcn.sgpr<?>, i32
  %9 = alloca : !amdgcn.vgpr<?>
  cf.cond_br %8, ^bb1, ^bb2
^bb1:  // CHECK: pred: ^bb0
  %10 = test_inst outs %4 ins %0 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %11 = alloca : !amdgcn.vgpr<?>
  %12 = test_inst outs %11 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %13 = lsir.copy %9, %11 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb3
^bb2:  // CHECK: pred: ^bb0
  %14 = test_inst outs %5 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %15 = alloca : !amdgcn.vgpr<?>
  %16 = test_inst outs %15 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %17 = lsir.copy %9, %15 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb3
^bb3:  // CHECK: 2 preds: ^bb1, ^bb2
  test_inst ins %9 : (!amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL: SSA map
// CHECK: `%{{.*}} = arith.constant 0 : i32`
// CHECK:   results: [0 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [1 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [2 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [3 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:   results: [4 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [5 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>`
// CHECK:   results: [6 = `%{{.*}}`]
// CHECK: `%{{.*}} = lsir.cmpi i32 eq %{{.*}}, %{{.*}} : !amdgcn.sgpr<?>, i32`
// CHECK:   results: [7 = `%{{.*}}`]
// CHECK: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:   results: [8 = `%{{.*}}`]
// CHECK: `%{{.*}} = lsir.copy %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:   results: [9 = `%{{.*}}`]
// CHECK: `%{{.*}} = lsir.copy %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:   results: [10 = `%{{.*}}`]
// CHECK: Op: module {...}
// CHECK:   LIVE BEFORE: []
// CHECK-LABEL: phi_coalescing_3
// CHECK: Op: amdgcn.kernel @phi_coalescing_3 {...}
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = arith.constant 0 : i32
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: []
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:   LIVE BEFORE: [3 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [3 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 3 = `%{{.*}}`, 4 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = lsir.cmpi i32 eq %{{.*}}, %{{.*}} : !amdgcn.sgpr<?>, i32
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 2 = `%{{.*}}`, 3 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 2 = `%{{.*}}`]
// CHECK: Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 2 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = lsir.copy %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 2 = `%{{.*}}`]
// CHECK: Op: cf.br ^bb3
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 2 = `%{{.*}}`, 8 = `%{{.*}}`]
// CHECK: Op: %{{.*}} = lsir.copy %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 2 = `%{{.*}}`]
// CHECK: Op: cf.br ^bb3
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 2 = `%{{.*}}`, 8 = `%{{.*}}`]
// CHECK: Op: amdgcn.test_inst ins %{{.*}}, %{{.*}}, %{{.*}} : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:   LIVE BEFORE: [1 = `%{{.*}}`, 2 = `%{{.*}}`, 8 = `%{{.*}}`]
// CHECK: Op: amdgcn.end_kernel
// CHECK:   LIVE BEFORE: []
amdgcn.kernel @phi_coalescing_3 {
  %c0_i32 = arith.constant 0 : i32
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = test_inst outs %0 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  %5 = test_inst outs %1 ins %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  %6 = lsir.cmpi i32 eq %2, %c0_i32 : !amdgcn.sgpr<?>, i32
  %7 = alloca : !amdgcn.vgpr<?>
  cf.cond_br %6, ^bb1, ^bb2
^bb1:  // CHECK: pred: ^bb0
  %8 = lsir.copy %7, %0 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb3
^bb2:  // CHECK: pred: ^bb0
  %9 = lsir.copy %7, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  cf.br ^bb3
^bb3:  // CHECK: 2 preds: ^bb1, ^bb2
  test_inst ins %7, %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}
