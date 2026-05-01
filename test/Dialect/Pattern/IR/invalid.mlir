// RUN: aster-opt %s -split-input-file -verify-diagnostics

// Non-field op in fields region.

// expected-error @+1 {{fields region must only contain pattern.field ops}}
pattern.rewrite_pattern @bad_fields benefit(1) op(@Op) fields {
  %c = arith.constant 42 : i32
  pattern.yield %c : i32
} body {
  %cond = arith.constant true
  pattern.action %cond {
    pattern.yield
  }
}

// -----

// Yield operand count mismatch with field count.

// expected-error @+1 {{yield in fields region must have the same number of operands as field declarations}}
pattern.rewrite_pattern @yield_mismatch benefit(1) op(@Op) fields {
  %f0 = pattern.field @a : i32
  %f1 = pattern.field @b : i32
  pattern.yield %f0 : i32
} body {
  %cond = arith.constant true
  pattern.action %cond {
    pattern.yield
  }
}

// -----

// Yield operand type mismatch.

// expected-error @+1 {{yield operand #0 type mismatch with field @a}}
pattern.rewrite_pattern @type_mismatch benefit(1) op(@Op) fields {
  %f0 = pattern.field @a : i32
  %f1 = pattern.field @b : i64
  pattern.yield %f1, %f0 : i64, i32
} body {
  %cond = arith.constant true
  pattern.action %cond {
    pattern.yield
  }
}

// -----

// Action yield must have no operands.

pattern.rewrite_pattern @action_yield_args benefit(1) op(@Op) body {
  %cond = arith.constant true
  // expected-error @+1 {{action body yield must have no operands}}
  pattern.action %cond {
    %c = arith.constant 1 : i32
    pattern.yield %c : i32
  }
}
