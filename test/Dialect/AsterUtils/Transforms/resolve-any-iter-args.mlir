// RUN: aster-opt --aster-resolve-any-iter-args %s | FileCheck %s

// CHECK-LABEL: func.func @basic_any_iter_arg
// CHECK-SAME:    %[[INIT:.*]]: i32
// CHECK:         %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ARG:.*]] = %[[INIT]]) -> (i32)
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ARG]], %[[ARG]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
// CHECK:         return %[[FOR]] : i32
func.func @basic_any_iter_arg(%init: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init_any = aster_utils.to_any %init : i32
  %result_any = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %init_any) -> (!aster_utils.any) {
    %val = aster_utils.from_any %arg : i32
    %next = arith.addi %val, %val : i32
    %next_any = aster_utils.to_any %next : i32
    scf.yield %next_any : !aster_utils.any
  }
  %result = aster_utils.from_any %result_any : i32
  return %result : i32
}

// CHECK-LABEL: func.func @mixed_any_and_concrete
// CHECK-SAME:    %[[S:.*]]: i32, %[[V:.*]]: f32
// CHECK:         scf.for {{.*}} iter_args(%[[A:.*]] = %[[S]], %[[B:.*]] = %[[V]]) -> (i32, f32)
// CHECK:           scf.yield %[[A]], %[[B]] : i32, f32
func.func @mixed_any_and_concrete(%s: i32, %v: f32) -> (i32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %s_any = aster_utils.to_any %s : i32
  %result:2 = scf.for %i = %c0 to %c10 step %c1 iter_args(%a = %s_any, %b = %v) -> (!aster_utils.any, f32) {
    %a_val = aster_utils.from_any %a : i32
    %a_any = aster_utils.to_any %a_val : i32
    scf.yield %a_any, %b : !aster_utils.any, f32
  }
  %r0 = aster_utils.from_any %result#0 : i32
  return %r0, %result#1 : i32, f32
}

// CHECK-LABEL: func.func @no_any_unchanged
// CHECK:         scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32)
// CHECK-NOT:     aster_utils.any
func.func @no_any_unchanged(%init: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%v = %init) -> (i32) {
    scf.yield %v : i32
  }
  return %result : i32
}

// CHECK-LABEL: func.func @rotation_pattern
// CHECK-SAME:    %[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32
// CHECK:         scf.for {{.*}} iter_args(%[[X:.*]] = %[[C]], %[[Y:.*]] = %[[B]], %[[Z:.*]] = %[[A]])
// CHECK-SAME:      -> (i32, i32, i32)
// CHECK:           %[[NEW:.*]] = arith.addi %[[Z]], %[[Z]] : i32
// CHECK:           scf.yield %[[NEW]], %[[X]], %[[Y]] : i32, i32, i32
func.func @rotation_pattern(%a: i32, %b: i32, %c: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %a_any = aster_utils.to_any %a : i32
  %b_any = aster_utils.to_any %b : i32
  %c_any = aster_utils.to_any %c : i32
  // 3-deep rotation buffer: consume oldest (arg2), shift others, produce new
  %result:3 = scf.for %i = %c0 to %c10 step %c1
      iter_args(%x = %c_any, %y = %b_any, %z = %a_any) -> (!aster_utils.any, !aster_utils.any, !aster_utils.any) {
    %consumed = aster_utils.from_any %z : i32
    %new_val = arith.addi %consumed, %consumed : i32
    %new_any = aster_utils.to_any %new_val : i32
    // Rotate: new -> x, old x -> y, old y -> z (consumed and dropped)
    scf.yield %new_any, %x, %y : !aster_utils.any, !aster_utils.any, !aster_utils.any
  }
  %final = aster_utils.from_any %result#2 : i32
  return %final : i32
}

// CHECK-LABEL: func.func @unresolvable_any
// CHECK:         scf.for {{.*}} iter_args({{.*}}) -> (!aster_utils.any)
func.func @unresolvable_any(%init_any: !aster_utils.any) -> !aster_utils.any {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %init_any) -> (!aster_utils.any) {
    scf.yield %arg : !aster_utils.any
  }
  return %result : !aster_utils.any
}

// CHECK-LABEL: func.func @nested_both_any
// CHECK-SAME:    %[[INIT:.*]]: i32
// CHECK:         scf.for {{.*}} iter_args(%[[O:.*]] = %[[INIT]]) -> (i32)
// CHECK:           scf.for {{.*}} iter_args(%[[I:.*]] = %[[O]]) -> (i32)
// CHECK:             %[[NEXT:.*]] = arith.addi %[[I]], %[[I]] : i32
// CHECK:             scf.yield %[[NEXT]] : i32
// CHECK:           scf.yield
func.func @nested_both_any(%init: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init_any = aster_utils.to_any %init : i32
  %outer = scf.for %i = %c0 to %c10 step %c1 iter_args(%oarg = %init_any) -> (!aster_utils.any) {
    %oval = aster_utils.from_any %oarg : i32
    %oval_any = aster_utils.to_any %oval : i32
    %inner = scf.for %j = %c0 to %c10 step %c1 iter_args(%iarg = %oval_any) -> (!aster_utils.any) {
      %ival = aster_utils.from_any %iarg : i32
      %next = arith.addi %ival, %ival : i32
      %next_any = aster_utils.to_any %next : i32
      scf.yield %next_any : !aster_utils.any
    }
    %inner_val = aster_utils.from_any %inner : i32
    %inner_any = aster_utils.to_any %inner_val : i32
    scf.yield %inner_any : !aster_utils.any
  }
  %result = aster_utils.from_any %outer : i32
  return %result : i32
}

// CHECK-LABEL: func.func @multi_type
// CHECK-SAME:    %[[SI:.*]]: i32, %[[SF:.*]]: f32
// CHECK:         scf.for {{.*}} iter_args(%[[A:.*]] = %[[SI]], %[[B:.*]] = %[[SF]]) -> (i32, f32)
// CHECK:           %[[AI:.*]] = arith.addi %[[A]], %[[A]] : i32
// CHECK:           %[[BF:.*]] = arith.addf %[[B]], %[[B]] : f32
// CHECK:           scf.yield %[[AI]], %[[BF]] : i32, f32
func.func @multi_type(%si: i32, %sf: f32) -> (i32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %si_any = aster_utils.to_any %si : i32
  %sf_any = aster_utils.to_any %sf : f32
  %result:2 = scf.for %i = %c0 to %c10 step %c1 iter_args(%a = %si_any, %b = %sf_any) -> (!aster_utils.any, !aster_utils.any) {
    %ai = aster_utils.from_any %a : i32
    %bf = aster_utils.from_any %b : f32
    %ai2 = arith.addi %ai, %ai : i32
    %bf2 = arith.addf %bf, %bf : f32
    %ai2_any = aster_utils.to_any %ai2 : i32
    %bf2_any = aster_utils.to_any %bf2 : f32
    scf.yield %ai2_any, %bf2_any : !aster_utils.any, !aster_utils.any
  }
  %ri = aster_utils.from_any %result#0 : i32
  %rf = aster_utils.from_any %result#1 : f32
  return %ri, %rf : i32, f32
}

// CHECK-LABEL: func.func @type_mismatch
// CHECK:         scf.for {{.*}} iter_args({{.*}}) -> (!aster_utils.any)
func.func @type_mismatch(%init: i32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init_any = aster_utils.to_any %init : i32
  %result_any = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %init_any) -> (!aster_utils.any) {
    %val = aster_utils.from_any %arg : f32
    %next = arith.addf %val, %val : f32
    %next_any = aster_utils.to_any %next : f32
    scf.yield %next_any : !aster_utils.any
  }
  %result = aster_utils.from_any %result_any : f32
  return %result : f32
}

// CHECK-LABEL: func.func @partial_resolve
// CHECK-SAME:    %[[I:.*]]: i32, %[[O:.*]]: !aster_utils.any
// CHECK:         scf.for {{.*}} iter_args(%[[A:.*]] = %[[I]], %[[B:.*]] = %[[O]]) -> (i32, !aster_utils.any)
func.func @partial_resolve(%init: i32, %opaque: !aster_utils.any) -> (i32, !aster_utils.any) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init_any = aster_utils.to_any %init : i32
  %result:2 = scf.for %i = %c0 to %c10 step %c1 iter_args(%a = %init_any, %b = %opaque) -> (!aster_utils.any, !aster_utils.any) {
    %aval = aster_utils.from_any %a : i32
    %aval_any = aster_utils.to_any %aval : i32
    scf.yield %aval_any, %b : !aster_utils.any, !aster_utils.any
  }
  %ra = aster_utils.from_any %result#0 : i32
  return %ra, %result#1 : i32, !aster_utils.any
}

// CHECK-LABEL: func.func @multi_from_any_same_type
// CHECK-SAME:    %[[INIT:.*]]: i32
// CHECK:         scf.for {{.*}} iter_args(%[[A:.*]] = %[[INIT]]) -> (i32)
// CHECK:           %[[SUM:.*]] = arith.addi %[[A]], %[[A]] : i32
// CHECK:           scf.yield %[[SUM]] : i32
func.func @multi_from_any_same_type(%init: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init_any = aster_utils.to_any %init : i32
  %result_any = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %init_any) -> (!aster_utils.any) {
    %v1 = aster_utils.from_any %arg : i32
    %v2 = aster_utils.from_any %arg : i32
    %sum = arith.addi %v1, %v2 : i32
    %sum_any = aster_utils.to_any %sum : i32
    scf.yield %sum_any : !aster_utils.any
  }
  %result = aster_utils.from_any %result_any : i32
  return %result : i32
}

// CHECK-LABEL: func.func @multi_from_any_conflict
// CHECK:         scf.for {{.*}} iter_args({{.*}}) -> (!aster_utils.any)
func.func @multi_from_any_conflict(%init: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init_any = aster_utils.to_any %init : i32
  %result_any = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %init_any) -> (!aster_utils.any) {
    %vi = aster_utils.from_any %arg : i32
    %vf = aster_utils.from_any %arg : f32
    %vi_any = aster_utils.to_any %vi : i32
    scf.yield %vi_any : !aster_utils.any
  }
  %result = aster_utils.from_any %result_any : i32
  return %result : i32
}

// ===----------------------------------------------------------------------===//
// CF block argument tests
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @cf_basic_single_pred
// CHECK-SAME:    %[[VAL:.*]]: i32
// CHECK:         cf.br ^bb1(%[[VAL]] : i32)
// CHECK:       ^bb1(%[[ARG:.*]]: i32):
// CHECK:         return %[[ARG]] : i32
func.func @cf_basic_single_pred(%val: i32) -> i32 {
  %any = aster_utils.to_any %val : i32
  cf.br ^bb1(%any : !aster_utils.any)
^bb1(%arg: !aster_utils.any):
  %result = aster_utils.from_any %arg : i32
  return %result : i32
}

// CHECK-LABEL: func.func @cf_multi_pred_same_type
// CHECK-SAME:    %[[COND:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32
// CHECK:         cf.cond_br %[[COND]], ^bb1(%[[A]] : i32), ^bb1(%[[B]] : i32)
// CHECK:       ^bb1(%[[ARG:.*]]: i32):
// CHECK:         return %[[ARG]] : i32
func.func @cf_multi_pred_same_type(%cond: i1, %a: i32, %b: i32) -> i32 {
  %a_any = aster_utils.to_any %a : i32
  %b_any = aster_utils.to_any %b : i32
  cf.cond_br %cond, ^bb1(%a_any : !aster_utils.any), ^bb1(%b_any : !aster_utils.any)
^bb1(%arg: !aster_utils.any):
  %result = aster_utils.from_any %arg : i32
  return %result : i32
}

// CHECK-LABEL: func.func @cf_multi_pred_type_mismatch
// CHECK:       ^bb1(%{{.*}}: i32):
// CHECK:       ^bb2(%{{.*}}: !aster_utils.any):
func.func @cf_multi_pred_type_mismatch(%cond: i1, %a: i32, %b: f32) -> !aster_utils.any {
  %a_any = aster_utils.to_any %a : i32
  %b_any = aster_utils.to_any %b : f32
  cf.cond_br %cond, ^bb1(%a_any : !aster_utils.any), ^bb2(%b_any : !aster_utils.any)
^bb1(%arg1: !aster_utils.any):
  // This one could resolve (single pred, consistent i32).
  %v1 = aster_utils.from_any %arg1 : i32
  %v1_any = aster_utils.to_any %v1 : i32
  cf.br ^bb2(%v1_any : !aster_utils.any)
^bb2(%arg2: !aster_utils.any):
  // Two preds: bb0 sends f32, bb1 sends i32. Must NOT resolve.
  return %arg2 : !aster_utils.any
}

// CHECK-LABEL: func.func @cf_loop
// CHECK-SAME:    %[[COND:.*]]: i1, %[[INIT:.*]]: i32
// CHECK:         cf.br ^bb1(%[[INIT]] : i32)
// CHECK:       ^bb1(%[[ARG:.*]]: i32):
// CHECK:         %[[NEXT:.*]] = arith.addi %[[ARG]], %[[ARG]] : i32
// CHECK:         cf.cond_br %[[COND]], ^bb1(%[[NEXT]] : i32), ^bb2
func.func @cf_loop(%cond: i1, %init: i32) -> i32 {
  %init_any = aster_utils.to_any %init : i32
  cf.br ^header(%init_any : !aster_utils.any)
^header(%arg: !aster_utils.any):
  %val = aster_utils.from_any %arg : i32
  %next = arith.addi %val, %val : i32
  %next_any = aster_utils.to_any %next : i32
  cf.cond_br %cond, ^header(%next_any : !aster_utils.any), ^exit
^exit:
  return %next : i32
}

// CHECK-LABEL: func.func @cf_passthrough_chain
// CHECK-SAME:    %[[VAL:.*]]: i32
// CHECK:         cf.br ^bb1(%[[VAL]] : i32)
// CHECK:       ^bb1(%[[A:.*]]: i32):
// CHECK:         cf.br ^bb2(%[[A]] : i32)
// CHECK:       ^bb2(%[[B:.*]]: i32):
// CHECK:         return %[[B]] : i32
func.func @cf_passthrough_chain(%val: i32) -> i32 {
  %any = aster_utils.to_any %val : i32
  cf.br ^bb1(%any : !aster_utils.any)
^bb1(%a: !aster_utils.any):
  cf.br ^bb2(%a : !aster_utils.any)
^bb2(%b: !aster_utils.any):
  %result = aster_utils.from_any %b : i32
  return %result : i32
}

// CHECK-LABEL: func.func @cf_unresolvable
// CHECK:       ^bb1(%{{.*}}: !aster_utils.any):
func.func @cf_unresolvable(%opaque: !aster_utils.any) -> !aster_utils.any {
  cf.br ^bb1(%opaque : !aster_utils.any)
^bb1(%arg: !aster_utils.any):
  return %arg : !aster_utils.any
}

// CHECK-LABEL: func.func @cf_entry_block_untouched
// CHECK-SAME:    %[[ARG:.*]]: !aster_utils.any
// CHECK:         %[[VAL:.*]] = aster_utils.from_any %[[ARG]] : i32
// CHECK:         return %[[VAL]] : i32
func.func @cf_entry_block_untouched(%arg: !aster_utils.any) -> i32 {
  %val = aster_utils.from_any %arg : i32
  return %val : i32
}

// CHECK-LABEL: func.func @cf_different_targets
// CHECK-SAME:    %[[COND:.*]]: i1, %[[I:.*]]: i32, %[[F:.*]]: f32
// CHECK:         cf.cond_br %[[COND]], ^bb1(%[[I]] : i32), ^bb2(%[[F]] : f32)
// CHECK:       ^bb1(%[[A:.*]]: i32):
// CHECK-NEXT:    return %[[A]] : i32
// CHECK:       ^bb2(%[[B:.*]]: f32):
// CHECK:         %[[CAST:.*]] = arith.fptosi %[[B]] : f32 to i32
// CHECK:         return %[[CAST]] : i32
func.func @cf_different_targets(%cond: i1, %i: i32, %f: f32) -> i32 {
  %i_any = aster_utils.to_any %i : i32
  %f_any = aster_utils.to_any %f : f32
  cf.cond_br %cond, ^bb1(%i_any : !aster_utils.any), ^bb2(%f_any : !aster_utils.any)
^bb1(%a: !aster_utils.any):
  %ai = aster_utils.from_any %a : i32
  return %ai : i32
^bb2(%b: !aster_utils.any):
  %bf = aster_utils.from_any %b : f32
  %cast = arith.fptosi %bf : f32 to i32
  return %cast : i32
}

// CHECK-LABEL: func.func @cf_fixpoint_cycle
// CHECK-SAME:    %[[COND:.*]]: i1, %[[INIT:.*]]: i32
// CHECK:         cf.br ^bb1(%[[INIT]] : i32)
// CHECK:       ^bb1(%[[A:.*]]: i32):
// CHECK:         cf.br ^bb2(%[[A]] : i32)
// CHECK:       ^bb2(%[[B:.*]]: i32):
// CHECK:         %[[NEXT:.*]] = arith.addi %[[B]], %[[B]] : i32
// CHECK:         cf.cond_br %[[COND]], ^bb1(%[[NEXT]] : i32), ^bb3
func.func @cf_fixpoint_cycle(%cond: i1, %init: i32) -> i32 {
  %init_any = aster_utils.to_any %init : i32
  cf.br ^bb1(%init_any : !aster_utils.any)
^bb1(%a: !aster_utils.any):
  cf.br ^bb2(%a : !aster_utils.any)
^bb2(%b: !aster_utils.any):
  %val = aster_utils.from_any %b : i32
  %next = arith.addi %val, %val : i32
  %next_any = aster_utils.to_any %next : i32
  cf.cond_br %cond, ^bb1(%next_any : !aster_utils.any), ^exit
^exit:
  return %next : i32
}

// CHECK-LABEL: func.func @cf_and_scf_mixed
// CHECK-SAME:    %[[INIT:.*]]: i32
// CHECK:         cf.br ^bb1(%[[INIT]] : i32)
// CHECK:       ^bb1(%[[ARG:.*]]: i32):
// CHECK:         scf.for {{.*}} iter_args(%[[ITER:.*]] = %[[ARG]]) -> (i32)
func.func @cf_and_scf_mixed(%init: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init_any = aster_utils.to_any %init : i32
  cf.br ^bb1(%init_any : !aster_utils.any)
^bb1(%arg: !aster_utils.any):
  %val = aster_utils.from_any %arg : i32
  %val_any = aster_utils.to_any %val : i32
  %result_any = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter = %val_any) -> (!aster_utils.any) {
    %v = aster_utils.from_any %iter : i32
    %next = arith.addi %v, %v : i32
    %next_any = aster_utils.to_any %next : i32
    scf.yield %next_any : !aster_utils.any
  }
  %result = aster_utils.from_any %result_any : i32
  return %result : i32
}

// CHECK-LABEL: func.func @cf_multi_arg_partial
// CHECK-SAME:    %[[A:.*]]: i32, %[[B:.*]]: !aster_utils.any
// CHECK:         cf.br ^bb1(%[[A]], %[[B]] : i32, !aster_utils.any)
// CHECK:       ^bb1(%[[X:.*]]: i32, %[[Y:.*]]: !aster_utils.any):
func.func @cf_multi_arg_partial(%a: i32, %b: !aster_utils.any) -> i32 {
  %a_any = aster_utils.to_any %a : i32
  cf.br ^bb1(%a_any, %b : !aster_utils.any, !aster_utils.any)
^bb1(%x: !aster_utils.any, %y: !aster_utils.any):
  %xi = aster_utils.from_any %x : i32
  return %xi : i32
}
