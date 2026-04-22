// RUN: water-opt %s | FileCheck %s

// CHECK: #wave.symbol<"A">
func.func private @attr() attributes { test.foo = #wave.symbol<"A"> }

// CHECK: !wave.tensor<[@A, @B] of bf16>
func.func private @foo() -> !wave.tensor<[@A, @B] of bf16>

// CHECK: !wave.tensor<any of bf16>
func.func private @unspecified_tensor() -> !wave.tensor<any of bf16>

// CHECK: !wave.tensor<any of i32, <global>>
func.func private @address_space() -> !wave.tensor<any of i32, <global>>

// CHECK: !wave.tensor<any of i32, <global>>
func.func private @address_space_full() -> !wave.tensor<any of i32, #wave.address_space<global>>

// CHECK: !wave.tensor<any of i8>
func.func private @address_space_default() -> !wave.tensor<any of i8, <unspecified>>

// CHECK-LABEL: @dims_in_expr_list
// CHECK-SAME:  #wave.expr_list<[](d0, d1, d2) -> (d0, d1 + d2, d0 + 42)>
func.func private @dims_in_expr_list() attributes { wave_test.index = #wave.expr_list<[](d0, d1, d2) -> (d0, d1 + d2, d0 + 42)>}

// CHECK-LABEL: @dims_and_symbols_in_expr_list
// CHECK-SAME:  #wave.expr_list<[#wave.symbol<"A">, #wave.symbol<"B">](d0, d1) -> (d1 + A, B + 42, d0)>
func.func private @dims_and_symbols_in_expr_list() attributes {
  wave_test.index = #wave.expr_list<[#wave.symbol<"A">, #wave.symbol<"B">](d0, d1) -> (A + d1, B + 42, d0)>
}

// CHECK-LABEL: @unused_dims_in_expr_list
// CHECK-SAME:  #wave.expr_list<[](d0, d1) -> (d1)>
func.func private @unused_dims_in_expr_list() attributes {
  wave_test.index = #wave.expr_list<[](d0, d1) -> (d1)>
}

// Unlike symbol names, dimension names are not preserved.
// CHECK-LABEL: @custom_dim_names_in_expr_list
// CHECK-SAME:  #wave.expr_list<[](d0, d1, d2, d3) -> (d0, d1 + d2, 42)>
func.func private @custom_dim_names_in_expr_list() attributes {
  wave_test.index = #wave.expr_list<[](A, B, C, D) -> (A, B + C, 42)>
}

// CHECK-LABEL: @empty_dim_names_in_expr_list
// CHECK-SAME:  #wave.expr_list<[] -> (42)>
func.func private @empty_dim_names_in_expr_list() attributes {
  wave_test.index = #wave.expr_list<[]() -> (42)>
}
