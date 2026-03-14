// RUN: aster-opt %s --split-input-file --aster-scf-rotate | FileCheck %s

!vgpr = !amdgcn.vgpr

// CHECK-LABEL: func.func @basic_rotation
//       CHECK:   scf.for
//  CHECK-NEXT:     amdgcn.test_inst{{.*}}ins %{{.*}} {was_rotate}
//       CHECK:     amdgcn.test_inst{{.*}}
//       CHECK:     scf.yield

func.func @basic_rotation(%ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s0 = amdgcn.alloca : !vgpr
  %s_out = amdgcn.alloca : !vgpr
  %init = amdgcn.test_inst outs %s0 : (!vgpr) -> !vgpr

  scf.for %k = %c0 to %ub step %c1 iter_args(%acc = %init) -> !vgpr {
    %s_data = amdgcn.alloca : !vgpr
    %data = amdgcn.test_inst outs %s_data : (!vgpr) -> !vgpr

    // Rotate-head op: uses %acc (block arg) and %s_out (outer def).
    // No in-block dependencies -> moves to block front.
    %new_acc = amdgcn.test_inst outs %s_out ins %acc
        {sched.rotate_head, was_rotate} : (!vgpr, !vgpr) -> !vgpr

    scf.yield %new_acc : !vgpr
  }

  return
}

// -----

!vgpr_1 = !amdgcn.vgpr

// CHECK-LABEL: func.func @no_rotate_head
//       CHECK: scf.for
//       CHECK:   amdgcn.test_inst{{.*}}
//       CHECK:   scf.yield
func.func @no_rotate_head(%ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s0 = amdgcn.alloca : !vgpr_1
  %init = amdgcn.test_inst outs %s0 : (!vgpr_1) -> !vgpr_1

  scf.for %k = %c0 to %ub step %c1 iter_args(%acc = %init) -> !vgpr_1 {
    %s1 = amdgcn.alloca : !vgpr_1
    %v = amdgcn.test_inst outs %s1 ins %acc : (!vgpr_1, !vgpr_1) -> !vgpr_1
    scf.yield %v : !vgpr_1
  }

  return
}

// -----

!vgpr_2 = !amdgcn.vgpr

// CHECK-LABEL: func.func @rotate_chain
//       CHECK:   scf.for
//  CHECK-NEXT:     amdgcn.test_inst{{.*}}{first}
//  CHECK-NEXT:     amdgcn.test_inst{{.*}}{second}
//       CHECK:     amdgcn.test_inst{{.*}}{rest}
//       CHECK:     scf.yield
func.func @rotate_chain(%ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s0 = amdgcn.alloca : !vgpr_2
  %s1 = amdgcn.alloca : !vgpr_2
  %s2 = amdgcn.alloca : !vgpr_2
  %init = amdgcn.test_inst outs %s0 : (!vgpr_2) -> !vgpr_2

  scf.for %k = %c0 to %ub step %c1 iter_args(%acc = %init) -> !vgpr_2 {
    // Rest op (should end up after rotate_head ops).
    %s_rest = amdgcn.alloca : !vgpr_2
    %rest = amdgcn.test_inst outs %s_rest {rest} : (!vgpr_2) -> !vgpr_2

    // First rotate_head: no in-block deps.
    %v1 = amdgcn.test_inst outs %s1 ins %acc
        {sched.rotate_head, first} : (!vgpr_2, !vgpr_2) -> !vgpr_2

    // Second rotate_head: depends on first (another rotate_head -> OK).
    %v2 = amdgcn.test_inst outs %s2 ins %v1
        {sched.rotate_head, second} : (!vgpr_2, !vgpr_2) -> !vgpr_2

    scf.yield %v2 : !vgpr_2
  }

  return
}

// -----

!vgpr_3 = !amdgcn.vgpr

// CHECK-LABEL: func.func @pull_dep
//       CHECK:   scf.for
//       The alloca and dep op get co-moved before the rotate_head op.
//  CHECK-NEXT:     amdgcn.alloca
//  CHECK-NEXT:     amdgcn.test_inst{{.*}}{dep}
//  CHECK-NEXT:     amdgcn.test_inst{{.*}}{was_rotate}
//       CHECK:     amdgcn.test_inst{{.*}}{rest}
//       CHECK:     scf.yield

func.func @pull_dep(%ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s0 = amdgcn.alloca : !vgpr_3
  %s1 = amdgcn.alloca : !vgpr_3
  %init = amdgcn.test_inst outs %s0 : (!vgpr_3) -> !vgpr_3

  scf.for %k = %c0 to %ub step %c1 iter_args(%acc = %init) -> !vgpr_3 {
    // Rest op (should end up after the rotated group).
    %s_rest = amdgcn.alloca : !vgpr_3
    %rest = amdgcn.test_inst outs %s_rest {rest} : (!vgpr_3) -> !vgpr_3

    // In-block dep of the rotate_head op (e.g., ds_read).
    %s_data = amdgcn.alloca : !vgpr_3
    %data = amdgcn.test_inst outs %s_data {dep} : (!vgpr_3) -> !vgpr_3

    // rotate_head op depends on %data -> %data and its alloca get co-moved.
    %new_acc = amdgcn.test_inst outs %s1 ins %acc, %data
        {sched.rotate_head, was_rotate} : (!vgpr_3, !vgpr_3, !vgpr_3) -> !vgpr_3

    scf.yield %new_acc : !vgpr_3
  }

  return
}

// -----

!vgpr_4 = !amdgcn.vgpr

// CHECK-LABEL: func.func @transitive_dep_chain
//       CHECK:   scf.for
//  CHECK-NEXT:     amdgcn.alloca
//  CHECK-NEXT:     amdgcn.test_inst{{.*}}{dep_A}
//  CHECK-NEXT:     amdgcn.alloca
//  CHECK-NEXT:     amdgcn.test_inst{{.*}}{dep_B}
//  CHECK-NEXT:     amdgcn.test_inst{{.*}}{was_rotate}
//       CHECK:     amdgcn.test_inst{{.*}}{rest}
//       CHECK:     scf.yield

func.func @transitive_dep_chain(%ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s0 = amdgcn.alloca : !vgpr_4
  %s_out = amdgcn.alloca : !vgpr_4
  %init = amdgcn.test_inst outs %s0 : (!vgpr_4) -> !vgpr_4

  scf.for %k = %c0 to %ub step %c1 iter_args(%acc = %init) -> !vgpr_4 {
    // Rest op.
    %s_rest = amdgcn.alloca : !vgpr_4
    %rest = amdgcn.test_inst outs %s_rest {rest} : (!vgpr_4) -> !vgpr_4

    // A: no in-block deps.
    %s_a = amdgcn.alloca : !vgpr_4
    %a = amdgcn.test_inst outs %s_a {dep_A} : (!vgpr_4) -> !vgpr_4

    // B: depends on A.
    %s_b = amdgcn.alloca : !vgpr_4
    %b = amdgcn.test_inst outs %s_b ins %a {dep_B} : (!vgpr_4, !vgpr_4) -> !vgpr_4

    // rotate_head depends on B (transitively on A).
    %new_acc = amdgcn.test_inst outs %s_out ins %acc, %b
        {sched.rotate_head, was_rotate} : (!vgpr_4, !vgpr_4, !vgpr_4) -> !vgpr_4

    scf.yield %new_acc : !vgpr_4
  }

  return
}
