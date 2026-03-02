// RUN: aster-opt %s --split-input-file --verify-diagnostics

//-----------------------------------------------------------------------------
// Test dialect normal form attribute tests (single attribute).
//-----------------------------------------------------------------------------

// no_index_types: index function argument.
normalform.module @no_index_types_arg [#aster_test.no_index_types] {
  // expected-error @below {{normal form prohibits index types}}
  func.func @f(%arg: index) {
    return
  }
}

// -----

// no_index_types: index result type.
normalform.module @no_index_types_result [#aster_test.no_index_types] {
  // expected-error @below {{normal form prohibits index types}}
  func.func @f() -> index {
    %0 = arith.constant 0 : index
    return %0 : index
  }
}

// -----

// no_index_types: index-typed block argument in nested region (scf.for iter_arg).
// This tests that block arguments in nested regions are verified for type
// constraints. The error is triggered by the arith.constant result type, but
// the scf.for iter_arg and induction variable block arguments are also checked.
normalform.module @no_index_types_nested_block_arg [#aster_test.no_index_types] {
  func.func @f() {
    // expected-error @below {{normal form prohibits index types}}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%iter = %c0) -> (index) {
      scf.yield %iter : index
    }
    return
  }
}

// -----

// no_invalid_ops: division operation.
normalform.module @no_invalid_ops [#aster_test.no_invalid_ops] {
  func.func @f(%a: f32, %b: f32) -> f32 {
    // expected-error @below {{normal form prohibits division operations}}
    %0 = arith.divf %a, %b : f32
    return %0 : f32
  }
}

// -----

// no_invalid_attrs: string attribute with value "invalid".
normalform.module @no_invalid_attrs [#aster_test.no_invalid_attrs] {
  // expected-error @below {{normal form prohibits 'invalid' string attribute values}}
  func.func @f() attributes {foo = "invalid"} {
    return
  }
}

// -----

//-----------------------------------------------------------------------------
// Test dialect normal form attribute tests (multiple attributes).
//-----------------------------------------------------------------------------

// Multiple test attributes: violation on no_invalid_ops.
normalform.module @multi_attrs_invalid_op [#aster_test.no_index_types, #aster_test.no_invalid_ops] {
  func.func @f(%a: i32, %b: i32) -> i32 {
    // expected-error @below {{normal form prohibits division operations}}
    %0 = arith.divsi %a, %b : i32
    return %0 : i32
  }
}

// -----

// Multiple test attributes: only no_invalid_attrs violation present.
normalform.module @multi_attrs_invalid_attr [#aster_test.no_index_types, #aster_test.no_invalid_attrs] {
  // expected-error @below {{normal form prohibits 'invalid' string attribute values}}
  func.func @f() attributes {x = "invalid"} {
    return
  }
}

// -----

// Multiple test attributes: only no_index_types violation present.
normalform.module @multi_attrs_index_type [#aster_test.no_invalid_ops, #aster_test.no_index_types] {
  // expected-error @below {{normal form prohibits index types}}
  func.func @f(%arg: index) {
    return
  }
}

// -----

// All three attributes: violation on no_invalid_ops.
normalform.module @all_attrs_invalid_op [#aster_test.no_index_types, #aster_test.no_invalid_ops, #aster_test.no_invalid_attrs] {
  func.func @f(%a: i32, %b: i32) -> i32 {
    // expected-error @below {{normal form prohibits division operations}}
    %0 = arith.divui %a, %b : i32
    return %0 : i32
  }
}

// -----

// All three attributes: violation on no_invalid_attrs only.
normalform.module @all_attrs_invalid_attr [#aster_test.no_index_types, #aster_test.no_invalid_ops, #aster_test.no_invalid_attrs] {
  // expected-error @below {{normal form prohibits 'invalid' string attribute values}}
  func.func @f() attributes {x = "invalid"} {
    return
  }
}

// -----

//-----------------------------------------------------------------------------
// Duplicate attribute rejection.
//-----------------------------------------------------------------------------

// Duplicate normal form attributes are rejected.
// expected-error @below {{contains duplicate normal form attribute}}
normalform.module @duplicate [#aster_test.no_index_types, #aster_test.no_index_types] {
  func.func @f() {
    return
  }
}
