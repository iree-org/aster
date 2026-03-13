// RUN: aster-opt %s --verify-diagnostics --split-input-file

func.func @min_type_mismatch(%arg0: i8) -> i8 {
  // expected-error@+1 {{static min type mismatch: expected 'i8', got 'i32'}}
  %0 = "aster_utils.assume_range"(%arg0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_max = 44 : i8, static_min = -44 : i32}> : (i8) -> i8
  return %0 : i8
}

func.func @max_type_mismatch(%arg0: i8) -> i8 {
  // expected-error@+1 {{static max type mismatch: expected 'i8', got 'i32'}}
  %0 = "aster_utils.assume_range"(%arg0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_max = 44 : i32, static_min = -44 : i8}> : (i8) -> i8
  return %0 : i8
}

// -----

func.func @addi_one_operand(%a: index) -> index {
  // expected-error@+1 {{requires at least 2 operands, but got 1}}
  %0 = aster_utils.addi %a : index
  return %0 : index
}

// -----

func.func @muli_one_operand(%a: index) -> index {
  // expected-error@+1 {{requires at least 2 operands, but got 1}}
  %0 = aster_utils.muli %a : index
  return %0 : index
}
