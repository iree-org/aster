// RUN: aster-opt %s --aster-codegen | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<
  !ptr.ptr<#amdgcn.addr_space<local, read_write>> = #ptr.spec<size = 32, abi = 32, preferred = 32>,
  !ptr.ptr<#amdgcn.addr_space<global, read_write>> = #ptr.spec<size = 64, abi = 64, preferred = 64>>} {

// CHECK-LABEL: func.func @test_minui(
// CHECK-SAME:    %[[A:.*]]: !amdgcn.vgpr, %[[B:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:         %[[DST:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:         %[[CMP_DST:.*]] = lsir.alloca : !amdgcn.vcc
// CHECK:         %[[CMP:.*]] = lsir.cmpi i32 ult %[[CMP_DST]], %[[A]], %[[B]]
// CHECK:         lsir.select %[[DST]], %[[CMP]], %[[A]], %[[B]]
func.func @test_minui(%a: i32, %b: i32) -> i32
    attributes {abi = (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr} {
  %r = arith.minui %a, %b : i32
  return {abi = (!amdgcn.vgpr) -> !amdgcn.vgpr} %r : i32
}

// CHECK-LABEL: func.func @test_maxui(
// CHECK-SAME:    %[[A:.*]]: !amdgcn.vgpr, %[[B:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:         %[[DST:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:         lsir.maxui i32 %[[DST]], %[[A]], %[[B]]
func.func @test_maxui(%a: i32, %b: i32) -> i32
    attributes {abi = (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr} {
  %r = arith.maxui %a, %b : i32
  return {abi = (!amdgcn.vgpr) -> !amdgcn.vgpr} %r : i32
}

// CHECK-LABEL: func.func @test_minsi(
// CHECK-SAME:    %[[A:.*]]: !amdgcn.vgpr, %[[B:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:         %[[DST:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:         %[[CMP_DST:.*]] = lsir.alloca : !amdgcn.vcc
// CHECK:         %[[CMP:.*]] = lsir.cmpi i32 slt %[[CMP_DST]], %[[A]], %[[B]]
// CHECK:         lsir.select %[[DST]], %[[CMP]], %[[A]], %[[B]]
func.func @test_minsi(%a: i32, %b: i32) -> i32
    attributes {abi = (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr} {
  %r = arith.minsi %a, %b : i32
  return {abi = (!amdgcn.vgpr) -> !amdgcn.vgpr} %r : i32
}

// CHECK-LABEL: func.func @test_maxsi(
// CHECK-SAME:    %[[A:.*]]: !amdgcn.vgpr, %[[B:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:         %[[DST:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:         lsir.maxsi i32 %[[DST]], %[[A]], %[[B]]
func.func @test_maxsi(%a: i32, %b: i32) -> i32
    attributes {abi = (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr} {
  %r = arith.maxsi %a, %b : i32
  return {abi = (!amdgcn.vgpr) -> !amdgcn.vgpr} %r : i32
}

}
