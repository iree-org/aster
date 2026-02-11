// RUN: aster-opt %s --aster-destructure-struct-iter-args | FileCheck %s

// CHECK-LABEL: func.func @basic_struct_iter_arg
// CHECK-SAME:    %[[INIT:.*]]: !aster_utils.struct<a: i32, b: f32>
// CHECK:         %[[FIELDS:.*]]:2 = aster_utils.struct_extract %[[INIT]] ["a", "b"]
// CHECK:         %[[FOR:.*]]:2 = scf.for {{.*}} iter_args(%[[A:.*]] = %[[FIELDS]]#0, %[[B:.*]] = %[[FIELDS]]#1) -> (i32, f32)
// CHECK:           %[[S:.*]] = aster_utils.struct_create(%[[A]], %[[B]])
// CHECK:           %[[EX:.*]]:2 = aster_utils.struct_extract %[[S]] ["a", "b"]
// CHECK:           scf.yield %[[EX]]#0, %[[EX]]#1 : i32, f32
// CHECK:         %[[RESULT:.*]] = aster_utils.struct_create(%[[FOR]]#0, %[[FOR]]#1)
// CHECK:         return %[[RESULT]]
func.func @basic_struct_iter_arg(%init: !aster_utils.struct<a: i32, b: f32>) -> !aster_utils.struct<a: i32, b: f32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%s = %init) -> (!aster_utils.struct<a: i32, b: f32>) {
    scf.yield %s : !aster_utils.struct<a: i32, b: f32>
  }
  return %result : !aster_utils.struct<a: i32, b: f32>
}

// CHECK-LABEL: func.func @mixed_iter_args
// CHECK-SAME:    %[[S:.*]]: !aster_utils.struct<x: index, y: index>, %[[V:.*]]: i32
// CHECK:         %[[FIELDS:.*]]:2 = aster_utils.struct_extract %[[S]] ["x", "y"]
// CHECK:         %[[FOR:.*]]:4 = scf.for {{.*}} iter_args(%{{.*}} = %[[V]], %{{.*}} = %[[FIELDS]]#0, %{{.*}} = %[[FIELDS]]#1, %{{.*}} = %[[V]]) -> (i32, index, index, i32)
func.func @mixed_iter_args(%s: !aster_utils.struct<x: index, y: index>, %v: i32)
    -> (i32, !aster_utils.struct<x: index, y: index>, i32)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %result:3 = scf.for %i = %c0 to %c10 step %c1 iter_args(%vi = %v, %si = %s, %vi2 = %v)
      -> (i32, !aster_utils.struct<x: index, y: index>, i32)
  {
    scf.yield %vi, %si, %vi2 : i32, !aster_utils.struct<x: index, y: index>, i32
  }
  return %result#0, %result#1, %result#2 : i32, !aster_utils.struct<x: index, y: index>, i32
}

// CHECK-LABEL: func.func @no_struct_unchanged
// CHECK:         scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32)
// CHECK-NOT:     aster_utils.struct
func.func @no_struct_unchanged(%init: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%v = %init) -> (i32) {
    scf.yield %v : i32
  }
  return %result : i32
}
