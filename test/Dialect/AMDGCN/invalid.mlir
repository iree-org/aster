// RUN: aster-opt %s --verify-diagnostics --split-input-file --allow-unregistered-dialect

func.func @mixed_relocatable_registers() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  // expected-error@+1 {{expected all operand types to be of the same kind}}
  %3 = amdgcn.make_register_range %0, %1, %2 : !amdgcn.vgpr<1>, !amdgcn.vgpr, !amdgcn.vgpr
  return
}

// -----

func.func @duplicate_registers() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.vgpr<1>
  %2 = amdgcn.alloca : !amdgcn.vgpr<2>
  // expected-error@+1 {{duplicate register found: 1}}
  %3 = amdgcn.make_register_range %0, %1, %2 : !amdgcn.vgpr<1>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>
  return
}

// -----

func.func @non_contiguous_range() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.vgpr<5>
  %2 = amdgcn.alloca : !amdgcn.vgpr<2>
  // expected-error@+1 {{missing register in range: 3}}
  %3 = amdgcn.make_register_range %0, %1, %2 : !amdgcn.vgpr<1>, !amdgcn.vgpr<5>, !amdgcn.vgpr<2>
  return
}

// -----

func.func @mixed_registers() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.agpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  // expected-error@+1 {{expected all operand types to be of the same kind}}
  %3 = amdgcn.make_register_range %0, %1, %2 : !amdgcn.vgpr<1>, !amdgcn.agpr, !amdgcn.vgpr
  return
}

// -----

func.func @mixed_registers() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  // expected-note@+1 {{prior use here}}
  %3 = amdgcn.make_register_range %0, %1, %2 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  // expected-error@+1 {{expects different type than prior uses: '!amdgcn.vgpr_range<[? + 4]>' vs '!amdgcn.vgpr_range<[? + 3]>'}}
  %4 = "test_op"(%3) : (!amdgcn.vgpr_range<[? + 4]>) -> ()
  return
}

// -----

func.func @split_non_range_type() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  // expected-error@+1 {{expected register range type}}
  %1, %2 = amdgcn.split_register_range %0 : !amdgcn.vgpr<1>
  return
}

// -----

func.func @split_range_into_wrong_count() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<0>
  %1 = amdgcn.alloca : !amdgcn.vgpr<1>
  %2 = amdgcn.alloca : !amdgcn.vgpr<2>
  %3 = amdgcn.make_register_range %0, %1, %2 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>
  // expected-error@+1 {{operation defines 3 results but was provided 4 to bind}}
  %4, %5, %6, %7 = amdgcn.split_register_range %3 : !amdgcn.vgpr_range<[0 : 3]>
  return
}

// -----

func.func @add(%arg0: !amdgcn.vgpr, %arg1: !amdgcn.vgpr, %arg2: !amdgcn.vgpr, %arg3: !amdgcn.sgpr_range<[? + 2]>) -> !amdgcn.vgpr {
  // expected-error@+1 {{expected `carry_out` to not be present}}
  %add, %0 = amdgcn.vop.add v_add_i32 outs %arg0 carry_out = %arg3 ins %arg1, %arg2 : !amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, !amdgcn.vgpr
  return %add : !amdgcn.vgpr
}
