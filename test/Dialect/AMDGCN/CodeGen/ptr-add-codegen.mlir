// RUN: aster-opt %s --aster-codegen | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!ptr.ptr<#amdgcn.addr_space<local, read_write>> = #ptr.spec<size = 32, abi = 32, preferred = 32>, !ptr.ptr<#amdgcn.addr_space<global, read_write>> = #ptr.spec<size = 64, abi = 64, preferred = 64>>} {
// CHECK-LABEL:   func.func @test_ptr_add_constant_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 16 : i32
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] c_off = 16 : !amdgcn.vgpr
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.vgpr
// CHECK:         }
  func.func @test_ptr_add_constant_offset(%arg0: !ptr.ptr<#amdgcn.addr_space<local, read_write>>) -> !ptr.ptr<#amdgcn.addr_space<local, read_write>> attributes {abi = (!amdgcn.vgpr) -> !amdgcn.vgpr} {
    %c16_i32 = arith.constant 16 : i32
    %0 = ptr.ptr_add %arg0, %c16_i32 : !ptr.ptr<#amdgcn.addr_space<local, read_write>>, i32
    return {abi = (!amdgcn.vgpr) -> !amdgcn.vgpr} %0 : !ptr.ptr<#amdgcn.addr_space<local, read_write>>
  }
// CHECK-LABEL:   func.func @test_ptr_add_dynamic_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] d_off = %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.vgpr
// CHECK:         }
  func.func @test_ptr_add_dynamic_vgpr(%arg0: !ptr.ptr<#amdgcn.addr_space<local, read_write>>, %arg1: i32) -> !ptr.ptr<#amdgcn.addr_space<local, read_write>> attributes {abi = (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr} {
    %0 = ptr.ptr_add %arg0, %arg1 : !ptr.ptr<#amdgcn.addr_space<local, read_write>>, i32
    return {abi = (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr} %0 : !ptr.ptr<#amdgcn.addr_space<local, read_write>>
  }
// CHECK-LABEL:   func.func @test_ptr_add_dynamic_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] u_off = %[[ARG1]] : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
  func.func @test_ptr_add_dynamic_sgpr(%arg0: !ptr.ptr<#amdgcn.addr_space<global, read_write>>, %arg1: i32) -> !ptr.ptr<#amdgcn.addr_space<global, read_write>> attributes {abi = (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr) -> !amdgcn.sgpr<[? + 2]>} {
    %0 = ptr.ptr_add %arg0, %arg1 : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
    lsir.reg_constraint %0 {kind = #amdgcn.reg_kind<SGPR>} : !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    return {abi = (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr) -> !amdgcn.sgpr<[? + 2]>} %0 : !ptr.ptr<#amdgcn.addr_space<global, read_write>>
  }
// CHECK-LABEL:   func.func @test_ptr_add_constant_offset_global(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 32 : i32
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] c_off = 32 : !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
  func.func @test_ptr_add_constant_offset_global(%arg0: !ptr.ptr<#amdgcn.addr_space<global, read_write>>) -> !ptr.ptr<#amdgcn.addr_space<global, read_write>> attributes {abi = (!amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>} {
    %c32_i32 = arith.constant 32 : i32
    %0 = ptr.ptr_add %arg0, %c32_i32 : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
    return {abi = (!amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>} %0 : !ptr.ptr<#amdgcn.addr_space<global, read_write>>
  }
// CHECK-LABEL:   func.func @test_ptr_add_dynamic_vgpr_global(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] d_off = %[[ARG1]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
  func.func @test_ptr_add_dynamic_vgpr_global(%arg0: !ptr.ptr<#amdgcn.addr_space<global, read_write>>, %arg1: i32) -> !ptr.ptr<#amdgcn.addr_space<global, read_write>> attributes {abi = (!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]>} {
    %0 = ptr.ptr_add %arg0, %arg1 : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
    return {abi = (!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]>} %0 : !ptr.ptr<#amdgcn.addr_space<global, read_write>>
  }
// CHECK-LABEL:   func.func @test_load(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] d_off = %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load ds_read_b32 dest %[[ALLOCA_0]] addr %[[PTR_ADD_0]] : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
  func.func @test_load(%arg0: !ptr.ptr<#amdgcn.addr_space<local, read_write>>, %arg1: i32) -> !amdgcn.vgpr attributes {abi = (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr} {
    %0 = ptr.ptr_add %arg0, %arg1 : !ptr.ptr<#amdgcn.addr_space<local, read_write>>, i32
    %1 = lsir.to_reg %0 : !ptr.ptr<#amdgcn.addr_space<local, read_write>> -> !amdgcn.vgpr
    %2 = amdgcn.alloca : !amdgcn.vgpr
    %dest_res, %token = amdgcn.load ds_read_b32 dest %2 addr %1 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
    return %dest_res : !amdgcn.vgpr
  }
}
