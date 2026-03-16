// RUN: aster-opt %s --split-input-file --aster-cfg-simplification | FileCheck %s

// CHECK-LABEL: func.func @basic_block_merge
//   CHECK-NOT: cf.br
//       CHECK: return

func.func @basic_block_merge() {
^entry:
  cf.br ^exit
^exit:
  return
}

// -----

// CHECK-LABEL: func.func @chain_merge
//   CHECK-NOT: cf.br
//       CHECK: return

func.func @chain_merge() {
^entry:
  cf.br ^middle
^middle:
  cf.br ^exit
^exit:
  return
}

// -----

// CHECK-LABEL: func.func @merge_preserves_ops
//   CHECK-NOT: cf.br
//       CHECK: arith.addi
//       CHECK: return

func.func @merge_preserves_ops(%x: i32, %y: i32) -> i32 {
  cf.br ^work
^work:
  %z = arith.addi %x, %y : i32
  return %z : i32
}

// -----

// CHECK-LABEL: func.func @no_merge_multi_succ
//       CHECK: cf.cond_br
// CHECK-COUNT-2: return

func.func @no_merge_multi_succ(%cond: i1) {
  cf.cond_br %cond, ^taken, ^not_taken
^taken:
  return
^not_taken:
  return
}

// -----

// CHECK-LABEL: func.func @empty_block_elim_both
//   CHECK-NOT: cf.br
//       CHECK: cf.cond_br %{{.*}}, ^bb1, ^bb1
//       CHECK: ^bb1:
//  CHECK-NEXT: return

func.func @empty_block_elim_both(%cond: i1) {
  cf.cond_br %cond, ^a, ^b
^a:
  cf.br ^exit
^b:
  cf.br ^exit
^exit:
  return
}

// -----

// CHECK-LABEL: func.func @interaction_merge_then_elim
//   CHECK-NOT: cf.br
//       CHECK: cf.cond_br %{{.*}}, ^bb1, ^bb1
//       CHECK: ^bb1:
//  CHECK-NEXT: return

func.func @interaction_merge_then_elim(%cond: i1) {
  cf.br ^dispatch
^dispatch:
  cf.cond_br %cond, ^left, ^right
^left:
  cf.br ^exit
^right:
  cf.br ^exit
^exit:
  return
}

// -----

// CHECK-LABEL: func.func @t1_forwarded_args
//   CHECK-NOT: cf.br
//       CHECK: arith.addi %arg0, %arg1
//       CHECK: return

func.func @t1_forwarded_args(%x: i32, %y: i32) -> i32 {
  cf.br ^work(%x, %y : i32, i32)
^work(%a: i32, %b: i32):
  %z = arith.addi %a, %b : i32
  return %z : i32
}

// -----

// CHECK-LABEL: func.func @t2_forwarded_args
//   CHECK-NOT: cf.br
//       CHECK: cf.cond_br %arg0, ^bb1(%arg1 : i32), ^bb1(%arg2 : i32)
//       CHECK: ^bb1(
//  CHECK-NEXT: return

func.func @t2_forwarded_args(%cond: i1, %x: i32, %y: i32) -> i32 {
  cf.cond_br %cond, ^pass(%x : i32), ^exit(%y : i32)
^pass(%a: i32):
  cf.br ^exit(%a : i32)
^exit(%v: i32):
  return %v : i32
}

// -----

// CHECK-LABEL: func.func @t2_forwarded_outer_val
//   CHECK-NOT: cf.br
//       CHECK: cf.cond_br %arg0, ^bb1(%arg2 : i32), ^bb1(%arg1 : i32)
//       CHECK: ^bb1(
//  CHECK-NEXT: return

func.func @t2_forwarded_outer_val(%cond: i1, %x: i32, %outer: i32) -> i32 {
  cf.cond_br %cond, ^pass(%x : i32), ^exit(%x : i32)
^pass(%a: i32):
  cf.br ^exit(%outer : i32)
^exit(%v: i32):
  return %v : i32
}
