//===- PIROps.td - PIR operations --------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PIR operations
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_PIR_IR_PIROPS_TD
#define ASTER_DIALECT_PIR_IR_PIROPS_TD

include "aster/Dialect/PIR/IR/PIRDialect.td"
include "aster/Dialect/PIR/IR/PIRTypes.td"
include "aster/Interfaces/InstOpInterface.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/// Helper class to ensure that all specified operands have the same underlying
/// type.
class AllUnderlyingTypesMatch<list<string> names> :
  AllMatchSameOperatorTrait<names,
    "::mlir::aster::pir::getUnderlyingType($_self.getType())",
    "underlying type"
  >;

/// Helper class to ensure that all specified operands have the same underlying
/// register type.
class AllRegTypesMatch<list<string> names> :
  AllMatchSameOperatorTrait<names,
    "llvm::cast<pir::TypedRegisterType>($_self.getType()).getReg()",
    "underlying register type"
  >;

/// Helper class for binary operations with integer or register operands.
class BinaryOp<string mnemonic, list<Trait> traits = []> :
  PIR_Op<mnemonic, traits # [
    AllTypesMatch<["dst", "result"]>,
    AllUnderlyingTypesMatch<["dst", "lhs", "rhs"]>,
    Pure
  ]> {
  let arguments = (ins
    IntRegType:$dst,
    IntOrRegType:$lhs,
    IntOrRegType:$rhs
  );
  let results = (outs IntRegType:$result);
  let assemblyFormat = [{
    $dst `,` $lhs `,` $rhs attr-dict
    `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getOperation()->getOpOperands().take_front(1);
    }
    /// Get the instruction input operands.
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getOperation()->getOpOperands().drop_front(1);
    }
  }];
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

def PIR_AllocaOp : PIR_Op<"alloca"> {
  let summary = "PIR alloca operation";
  let description = [{
    Allocates a register-like type.
  }];
  let arguments = (ins);
  let results = (outs
    Res<AnyRegType, "", [
      MemAlloc<AutomaticAllocationScopeResource, 0, FullEffect>
    ]>:$result
  );
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

def PIR_AddIOp : BinaryOp<"addi"> {
  let summary = "Integer addition operation";
  let description = [{
    The `pir.addi` operation performs integer addition on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.addi %dst, %lhs, %rhs : !amgcn.sgpr, !amgcn.sgpr, !amgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

def PIR_AndIOp : BinaryOp<"andi"> {
  let summary = "Integer bitwise AND operation";
  let description = [{
    The `pir.andi` operation performs bitwise AND on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.andi %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

def PIR_CmpIOp : PIR_Op<"cmpi", [
    Pure, AllTypesMatch<["dst", "result"]>,
    AllUnderlyingTypesMatch<["lhs", "rhs"]>
  ]> {
  let summary = "Integer comparison operation";
  let description = [{
    The `pir.cmpi` operation compares two integer operands according to the
    specified predicate and produces an i1 result.

    Example:
    ```mlir
    %result = pir.cmpi i32 eq %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
  let arguments = (ins
    Arith_CmpIPredicateAttr:$predicate,
    IntRegType:$dst,
    IntOrRegType:$lhs,
    IntOrRegType:$rhs
  );
  let results = (outs IntRegType:$result);
  let assemblyFormat = [{
    $predicate $dst `,` $lhs `,` $rhs attr-dict
    `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
  let extraClassDeclaration = [{
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getDstMutable();
    }
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getOperation()->getOpOperands().drop_front(2);
    }
  }];
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

def PIR_DivSIOp : BinaryOp<"divsi"> {
  let summary = "Signed integer division operation";
  let description = [{
    The `pir.divsi` operation performs signed integer division on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.divsi %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

def PIR_DivUIOp : BinaryOp<"divui"> {
  let summary = "Unsigned integer division operation";
  let description = [{
    The `pir.divui` operation performs unsigned integer division on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.divui %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

def PIR_ExtSIOp : PIR_Op<"extsi", [
    Pure, AllTypesMatch<["dst", "result"]>,
  ]> {
  let summary = "Sign extension operation";
  let description = [{
    The `pir.extsi` operation sign-extends an integer value to a wider type.

    Example:
    ```mlir
    %result = pir.extsi i32 from i16 %dst, %value : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
  let arguments = (ins
    IntRegType:$dst,
    IntOrRegType:$value
  );
  let results = (outs IntRegType:$result);
  let assemblyFormat = [{
    $dst `,` $value attr-dict `:` type($dst) `,` type($value)
  }];
  let extraClassDeclaration = [{
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getDstMutable();
    }
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getValueMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

def PIR_ExtUIOp : PIR_Op<"extui", [
    Pure, AllTypesMatch<["dst", "result"]>,
  ]> {
  let summary = "Zero extension operation";
  let description = [{
    The `pir.extui` operation zero-extends an integer value to a wider type.

    Example:
    ```mlir
    %result = pir.extui %dst, %value : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
  let arguments = (ins
    IntRegType:$dst,
    IntOrRegType:$value
  );
  let results = (outs IntRegType:$result);
  let assemblyFormat = [{
    $dst `,` $value attr-dict `:` type($dst) `,` type($value)
  }];
  let extraClassDeclaration = [{
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getDstMutable();
    }
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getValueMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// MovOp
//===----------------------------------------------------------------------===//

def PIR_MovOp : PIR_Op<"mov", [
    Pure, AllTypesMatch<["dst", "result"]>,
    AllUnderlyingTypesMatch<["dst", "value"]>
  ]> {
  let summary = "Move operation";
  let description = [{
    The `pir.mov` operation moves a value from one register to another.

    Example:
    ```mlir
    %result = pir.mov %dst, %value : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
  let arguments = (ins
    AnyRegType:$dst,
    AnyRegType:$value
  );
  let results = (outs AnyRegType:$result);
  let assemblyFormat = [{
    $dst `,` $value attr-dict `:` type($dst) `,` type($value)
  }];
  let extraClassDeclaration = [{
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getDstMutable();
    }
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getValueMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

def PIR_MulIOp : BinaryOp<"muli"> {
  let summary = "Integer multiplication operation";
  let description = [{
    The `pir.muli` operation performs integer multiplication on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.muli %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

def PIR_OrIOp : BinaryOp<"ori"> {
  let summary = "Integer bitwise OR operation";
  let description = [{
    The `pir.ori` operation performs bitwise OR on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.ori %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RemSIOp
//===----------------------------------------------------------------------===//

def PIR_RemSIOp : BinaryOp<"remsi"> {
  let summary = "Signed integer remainder operation";
  let description = [{
    The `pir.remsi` operation computes the signed remainder of integer division
    on two operands of `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.remsi %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

def PIR_RemUIOp : BinaryOp<"remui"> {
  let summary = "Unsigned integer remainder operation";
  let description = [{
    The `pir.remui` operation computes the unsigned remainder of integer division
    on two operands of `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.remui %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ShLIOp
//===----------------------------------------------------------------------===//

def PIR_ShLIOp : PIR_Op<"shli", [
    Pure, AllTypesMatch<["dst", "result"]>,
    AllElementTypesMatch<["dst", "lhs"]>
  ]> {
  let summary = "Shift left operation";
  let description = [{
    The `pir.shli` operation performs logical left shift on the first operand
    by the amount specified in the second operand.

    Example:
    ```mlir
    %result = pir.shli %dst, %value, %amount : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
  let arguments = (ins
    IntRegType:$dst,
    IntOrRegType:$lhs,
    IntOrRegType:$rhs
  );
  let results = (outs IntRegType:$result);
  let assemblyFormat = [{
    $dst `,` $lhs `,` $rhs attr-dict
    `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
  let extraClassDeclaration = [{
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getDstMutable();
    }
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getOperation()->getOpOperands().drop_front(1);
    }
  }];
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

def PIR_ShRSIOp : PIR_Op<"shrsi", [
    Pure, AllTypesMatch<["dst", "result"]>,
    AllElementTypesMatch<["dst", "lhs"]>
  ]> {
  let summary = "Arithmetic shift right operation";
  let description = [{
    The `pir.shrsi` operation performs arithmetic (signed) right shift on the first
    operand by the amount specified in the second operand.

    Example:
    ```mlir
    %result = pir.shrsi %dst, %value, %amount : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
  let arguments = (ins
    IntRegType:$dst,
    IntOrRegType:$lhs,
    IntOrRegType:$rhs
  );
  let results = (outs IntRegType:$result);
  let assemblyFormat = [{
    $dst `,` $lhs `,` $rhs attr-dict
    `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
  let extraClassDeclaration = [{
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getDstMutable();
    }
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getOperation()->getOpOperands().drop_front(1);
    }
  }];
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

def PIR_ShRUIOp : PIR_Op<"shrui", [
    Pure, AllTypesMatch<["dst", "result"]>,
    AllElementTypesMatch<["dst", "lhs"]>
  ]> {
  let summary = "Logical shift right operation";
  let description = [{
    The `pir.shrui` operation performs logical (unsigned) right shift on the first
    operand by the amount specified in the second operand.

    Example:
    ```mlir
    %result = pir.shrui i32 %dst, %value, %amount : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
  let arguments = (ins
    IntRegType:$dst,
    IntOrRegType:$lhs,
    IntOrRegType:$rhs
  );
  let results = (outs IntRegType:$result);
  let assemblyFormat = [{
    $dst `,` $lhs `,` $rhs attr-dict
    `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
  let extraClassDeclaration = [{
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getDstMutable();
    }
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getOperation()->getOpOperands().drop_front(1);
    }
  }];
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

def PIR_SubIOp : BinaryOp<"subi"> {
  let summary = "Integer subtraction operation";
  let description = [{
    The `pir.subi` operation performs integer subtraction on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.subi %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

def PIR_TruncIOp : PIR_Op<"trunci", [
    Pure, AllTypesMatch<["dst", "result"]>,
  ]> {
  let summary = "Integer truncation operation";
  let description = [{
    The `pir.trunci` operation truncates an integer value to a narrower type.

    Example:
    ```mlir
    %result = pir.trunci %dst, %value : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
  let arguments = (ins
    IntRegType:$dst,
    IntOrRegType:$value
  );
  let results = (outs IntRegType:$result);
  let assemblyFormat = [{
    $dst `,` $value attr-dict `:` type($dst) `,` type($value)
  }];
  let extraClassDeclaration = [{
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getDstMutable();
    }
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getValueMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// TypeCastOp
//===----------------------------------------------------------------------===//

def PIR_TypeCastOp : PIR_Op<"type_cast", [
    Pure, AllTypesMatch<["dst", "result"]>,
    AllRegTypesMatch<["dst", "value"]>
  ]> {
  let summary = "Move operation";
  let description = [{
    The `pir.mov` operation moves a value from one register to another.

    Example:
    ```mlir
    %result = pir.mov %dst, %value : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
    ```
  }];
  let arguments = (ins
    AnyRegType:$dst,
    AnyRegType:$value
  );
  let results = (outs AnyRegType:$result);
  let assemblyFormat = [{
    $dst `,` $value attr-dict `:` type($dst) `,` type($value)
  }];
  let extraClassDeclaration = [{
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getDstMutable();
    }
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getValueMutable();
    }
  }];
  let hasFolder = 1;
}

#endif // ASTER_DIALECT_PIR_IR_PIROPS_TD
