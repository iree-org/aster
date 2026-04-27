// RUN: aster-opt %s --split-input-file --verify-diagnostics

normalform.module @value_vgpr [#amdgcn.no_value_semantic_registers] {
  func.func @f() {
    // expected-error @below {{normal form violation: register types with value semantics are disallowed but found}}
    %0 = amdgcn.alloca : !amdgcn.vgpr
    return
  }
}

// -----

normalform.module @value_sgpr [#amdgcn.no_value_semantic_registers] {
  func.func @f() {
    // expected-error @below {{normal form violation: register types with value semantics are disallowed but found}}
    %0 = amdgcn.alloca : !amdgcn.sgpr
    return
  }
}

// -----

normalform.module @value_in_func_arg [#amdgcn.no_value_semantic_registers] {
  // expected-error @below {{normal form violation: register types with value semantics are disallowed but found}}
  func.func @f(%arg: !amdgcn.vgpr) {
    return
  }
}

// -----

normalform.module @value_in_func_result [#amdgcn.no_value_semantic_registers] {
  // expected-error @below {{normal form violation: register types with value semantics are disallowed but found}}
  func.func @f() -> !amdgcn.vgpr {
    %0 = amdgcn.alloca : !amdgcn.vgpr
    return %0 : !amdgcn.vgpr
  }
}

// -----

// by_val_arg type is ABI metadata, not a value-semantic register in the body
amdgcn.module @by_val_arg_metadata target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test arguments <[
    #amdgcn.by_val_arg<size = 4, type = !amdgcn.vgpr>
  ]> attributes {
    normal_forms = [#amdgcn.no_value_semantic_registers],
    shared_memory_size = 0 : i32
  } {
    amdgcn.end_kernel
  }
}
