// RUN: aster-opt --pass-pipeline="builtin.module(func.func(test-lds-interference-graph))" %s 2>&1 | FileCheck %s

// CHECK: graph LDSInterferenceGraph {
// CHECK:   0 [label="0: %{{.*}} (size=128, align=16)"];
// CHECK:   1 [label="1: %{{.*}} (size=256, align=16)"];
// CHECK:   2 [label="2: %{{.*}} (size=512, align=16)"];
// CHECK:   0 -- 1;
// CHECK:   0 -- 2;
// CHECK: }

func.func @test(%arg0: index, %arg1: index) {
  %0 = amdgcn.alloc_lds 128
  %1 = amdgcn.alloc_lds 256
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %2 = amdgcn.get_lds_offset %0 : i32
  %3 = amdgcn.get_lds_offset %1 : i32
  %4:2 = scf.for %arg2 = %arg0 to %arg1 step %c1 iter_args(%arg3 = %2, %arg4 = %3) -> (i32, i32) {
    scf.yield %arg4, %arg3 : i32, i32
  }
  amdgcn.dealloc_lds %1
  %5 = amdgcn.alloc_lds 512
  %6 = amdgcn.get_lds_offset %5 : index
  return
}
