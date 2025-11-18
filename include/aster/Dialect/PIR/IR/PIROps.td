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
include "aster/Dialect/PIR/IR/PIRAttrs.td"
include "aster/Dialect/PIR/IR/PIRTypes.td"
include "aster/Interfaces/InstOpInterface.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/// Helper class for binary operations with integer or register operands.
class BinaryIOp<string mnemonic, list<Trait> traits = []> :
  PIR_Op<mnemonic, traits # [
    AllTypesMatch<["dst", "result"]>,
    InstOpInterface, Pure
  ]> {
  let arguments = (ins
    AnyIntTypeAttr:$semantics,
    RegType:$dst,
    IntOrRegType:$lhs,
    IntOrRegType:$rhs
  );
  let results = (outs RegType:$result);
  let assemblyFormat = [{
    $semantics $dst `,` $lhs `,` $rhs attr-dict
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

/// Helper class for binary operations with float or register operands.
class BinaryFOp<string mnemonic, list<Trait> traits = []> :
  PIR_Op<mnemonic, traits # [
    AllTypesMatch<["dst", "result"]>,
    InstOpInterface, Pure
  ]> {
  let arguments = (ins
    AnyFloatTypeAttr:$semantics,
    RegType:$dst,
    FloatOrRegType:$lhs,
    FloatOrRegType:$rhs
  );
  let results = (outs RegType:$result);
  let assemblyFormat = [{
    $semantics $dst `,` $lhs `,` $rhs attr-dict
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

/// Helper class cast-like operations.
class CastOp<string mnemonic, Attr tgtAttr,
             Attr srcAttr, Type srcTy, list<Trait> traits = []> :
  PIR_Op<mnemonic, traits # [
    AllTypesMatch<["dst", "result"]>, InstOpInterface, Pure,
  ]> {
  let arguments = (ins
    tgtAttr:$tgt_type,
    srcAttr:$src_type,
    RegType:$dst,
    srcTy:$value
  );
  let results = (outs RegType:$result);
  let assemblyFormat = [{
    $tgt_type `from` $src_type $dst `,` $value attr-dict `:` type($dst) `,` type($value)
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getDstMutable();
    }
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getValueMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

def PIR_AddFOp : BinaryFOp<"addf"> {
  let summary = "Floating point addition operation";
  let description = [{
    The `pir.addf` operation performs floating point addition on two operands.

    Example:
    ```mlir
    %result = pir.addf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

def PIR_AddIOp : BinaryIOp<"addi"> {
  let summary = "Integer addition operation";
  let description = [{
    The `pir.addi` operation performs integer addition on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.addi i32 %dst, %lhs, %rhs : !amgcn.sgpr, !amgcn.sgpr, !amgcn.sgpr
    ```
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
    Res<RegType, "", [
      MemAlloc<AutomaticAllocationScopeResource, 0, FullEffect>
    ]>:$result
  );
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

def PIR_AndIOp : BinaryIOp<"andi"> {
  let summary = "Integer bitwise AND operation";
  let description = [{
    The `pir.andi` operation performs bitwise AND on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.andi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// AssumeNoaliasOp
//===----------------------------------------------------------------------===//

def PIR_AssumeNoaliasOp : PIR_Op<"assume_noalias", [
    Pure,
    SameVariadicOperandSize,
    // SameVariadicResultSize
    ]> {
  let summary = "Assume no aliasing between multiple values";
  let description = [{
    The `pir.assume_noalias` operation takes multiple operands and returns
    results of the same types. It asserts to MemoryDependenceAnalysis that
    the returned values do not alias with each other, eliminating unnecessary
    synchronization.

    This is useful for:
    - SGPRs/VGPRs holding pointers known to point to non-overlapping memory
    - Values from different sources known to be independent
    - Optimization hints for memory dependence analysis

    Example:
    ```mlir
    %ptr1_noalias, %ptr2_noalias = pir.assume_noalias %ptr1, %ptr2
      : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{ $operands attr-dict `:` functional-type(operands, results) }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AssumeRangeOp
//===----------------------------------------------------------------------===//

def PIR_AssumeRangeOp : PIR_Op<"assume_range", [
    Pure, AllTypesMatch<["input", "result"]>,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "PIR assume int range";
  let arguments = (ins
    AnySignlessIntegerOrIndex:$input,
    OptionalAttr<IndexAttr>:$min,
    OptionalAttr<IndexAttr>:$max
  );
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let assemblyFormat = [{
    $input (`min` $min^)? (`max` $max^)? attr-dict `:` type($input)
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AssumeUniform
//===----------------------------------------------------------------------===//

def PIR_AssumeUniformOp : PIR_Op<"assume_uniform", [
    Pure, AllTypesMatch<["input", "result"]>,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRangesFromOptional"]>
  ]> {
  let summary = "PIR assume uniform value";
  let arguments = (ins
    AnyType:$input
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// BlockDimOp
//===----------------------------------------------------------------------===//

def PIR_BlockDimOp : PIR_Op<"block_dim", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "PIR block dimension operation";
  let description = [{
    Returns the block (workgroup) ID in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = "$dim attr-dict";
}

//===----------------------------------------------------------------------===//
// BlockIdOp
//===----------------------------------------------------------------------===//

def PIR_BlockIdOp : PIR_Op<"block_id", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "PIR block ID operation";
  let description = [{
    Returns the block (workgroup) ID in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = "$dim attr-dict";
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

def PIR_CmpFOp : PIR_Op<"cmpf", [
    Pure, AllTypesMatch<["dst", "result"]>, InstOpInterface
  ]> {
  let summary = "Floating point comparison operation";
  let description = [{
    The `pir.cmpf` operation compares two floating point values.

    Example:
    ```mlir
    %result = pir.cmpf f32 olt %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
  let arguments = (ins
    AnyFloatTypeAttr:$semantics,
    Arith_CmpFPredicateAttr:$predicate,
    RegType:$dst,
    FloatOrRegType:$lhs,
    FloatOrRegType:$rhs
  );
  let results = (outs RegType:$result);
  let assemblyFormat = [{
    $semantics $predicate $dst `,` $lhs `,` $rhs attr-dict
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
// CmpIOp
//===----------------------------------------------------------------------===//

def PIR_CmpIOp : PIR_Op<"cmpi", [
    Pure, AllTypesMatch<["dst", "result"]>, InstOpInterface
  ]> {
  let summary = "Integer comparison operation";
  let description = [{
    The `pir.cmpi` operation compares two integer operands according to the
    specified predicate and produces an i1 result.

    Example:
    ```mlir
    %result = pir.cmpi i32 eq %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
  let arguments = (ins
    AnyIntTypeAttr:$semantics,
    Arith_CmpIPredicateAttr:$predicate,
    RegType:$dst,
    IntOrRegType:$lhs,
    IntOrRegType:$rhs
  );
  let results = (outs RegType:$result);
  let assemblyFormat = [{
    $semantics $predicate $dst `,` $lhs `,` $rhs attr-dict
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
// DivFOp
//===----------------------------------------------------------------------===//

def PIR_DivFOp : BinaryFOp<"divf"> {
  let summary = "Floating point division operation";
  let description = [{
    The `pir.divf` operation performs floating point division.

    Example:
    ```mlir
    %result = pir.divf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

def PIR_DivSIOp : BinaryIOp<"divsi"> {
  let summary = "Signed integer division operation";
  let description = [{
    The `pir.divsi` operation performs signed integer division on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.divsi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

def PIR_DivUIOp : BinaryIOp<"divui"> {
  let summary = "Unsigned integer division operation";
  let description = [{
    The `pir.divui` operation performs unsigned integer division on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.divui i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ExtFOp
//===----------------------------------------------------------------------===//

def PIR_ExtFOp : CastOp<"extf", AnyFloatTypeAttr, AnyFloatTypeAttr, FloatOrRegType> {
  let summary = "Floating point extension operation";
  let description = [{
    The `pir.extf` operation extends a floating point value to a wider type.

    Example:
    ```mlir
    %result = pir.extf f64 from f32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

def PIR_ExtSIOp : CastOp<"extsi", AnyIntTypeAttr, AnyIntTypeAttr, IntOrRegType> {
  let summary = "Sign extension operation";
  let description = [{
    The `pir.extsi` operation sign-extends an integer value to a wider type.

    Example:
    ```mlir
    %result = pir.extsi i32 from i16 %dst, %value : !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

def PIR_ExtUIOp : CastOp<"extui", AnyIntTypeAttr, AnyIntTypeAttr, IntOrRegType> {
  let summary = "Zero extension operation";
  let description = [{
    The `pir.extui` operation zero-extends an integer value to a wider type.

    Example:
    ```mlir
    %result = pir.extui i32 from i16 %dst, %value : !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// FPToSIOp
//===----------------------------------------------------------------------===//

def PIR_FPToSIOp : CastOp<"fptosi", AnyIntTypeAttr, AnyFloatTypeAttr, FloatOrRegType> {
  let summary = "Floating point to signed integer conversion";
  let description = [{
    The `pir.fptosi` operation converts a floating point value to a signed integer.

    Example:
    ```mlir
    %result = pir.fptosi i32 from f32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// FPToUIOp
//===----------------------------------------------------------------------===//

def PIR_FPToUIOp : CastOp<"fptoui", AnyIntTypeAttr, AnyFloatTypeAttr, FloatOrRegType> {
  let summary = "Floating point to unsigned integer conversion";
  let description = [{
    The `pir.fptoui` operation converts a floating point value to an unsigned integer.

    Example:
    ```mlir
    %result = pir.fptoui i32 from f32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// FromRegOp
//===----------------------------------------------------------------------===//

def PIR_FromRegOp : PIR_Op<"from_reg", [Pure]> {
  let summary = "Convert register type to non-register value";
  let description = [{
    The `pir.from_reg` operation converts a register type to a non-register value.

    Example:
    ```mlir
    %value = pir.from_reg %reg : !amdgcn.sgpr -> i32
    ```
  }];
  let arguments = (ins RegisterTypeInterface:$input);
  let results = (outs NotReg:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// GridDimOp
//===----------------------------------------------------------------------===//

def PIR_GridDimOp : PIR_Op<"grid_dim", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "PIR grid dimension operation";
  let description = [{
    Returns the grid dimension in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = "$dim attr-dict";
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def PIR_LoadOp : PIR_Op<"load", [
    AllTypesMatch<["dst", "dst_res"]>, InstOpInterface, DependentOpInterface
  ]> {
  let summary = "Load operation";
  let description = [{
    The `pir.load` operation loads a value from memory.

    Example:
    ```mlir
    %result = pir.load #oir.ms<#gpu.address_space<global>, rw> %dst, %src, %offset
      : !amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, i32
    ```
  }];
  let arguments = (ins
    MemorySpaceAttrInterface:$memory_space,
    RegType:$dst,
    // This is not a 1-1 mapping as the op also has an offset and const_offset
    // operands.
    Arg<RegType, "the reference to load from",
                 [MemRead]>:$addr,
    IntOrRegType:$offset,
    I32:$const_offset,
    Variadic<LoadOrStoreToken>:$dependencies
  );
  let results = (outs RegType:$dst_res, LoadToken:$token);
  let assemblyFormat = [{
    $memory_space $dst `,` $addr  `,` $offset `,` $const_offset
    (`dependencies` $dependencies^)? attr-dict
    `:` type($dst) `,` type($addr) `,` type($offset) `,` type($const_offset)
    (`,` type($dependencies)^)?
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
      return getOperation()->getOpOperands().slice(1, 3);
    }
    /// Get the instruction result ranges.
    ResultRange getInstResults() {
      return getResults().slice(0, 1);
    }
    //===------------------------------------------------------------------===//
    // DependentOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the output result representing the dependency.
    Value getOutDependency() {
      return getToken();
    }
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MaximumFOp
//===----------------------------------------------------------------------===//

def PIR_MaximumFOp : BinaryFOp<"maximumf"> {
  let summary = "Floating point maximum operation";
  let description = [{
    The `pir.maximumf` operation returns the maximum of two floating point values.

    Example:
    ```mlir
    %result = pir.maximumf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

def PIR_MaxSIOp : BinaryIOp<"maxsi"> {
  let summary = "Signed integer maximum operation";
  let description = [{
    The `pir.maxsi` operation returns the maximum of two signed integers.

    Example:
    ```mlir
    %result = pir.maxsi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

def PIR_MaxUIOp : BinaryIOp<"maxui"> {
  let summary = "Unsigned integer maximum operation";
  let description = [{
    The `pir.maxui` operation returns the maximum of two unsigned integers.

    Example:
    ```mlir
    %result = pir.maxui i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MinimumFOp
//===----------------------------------------------------------------------===//

def PIR_MinimumFOp : BinaryFOp<"minimumf"> {
  let summary = "Floating point minimum operation";
  let description = [{
    The `pir.minimumf` operation returns the minimum of two floating point values.

    Example:
    ```mlir
    %result = pir.minimumf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MovOp
//===----------------------------------------------------------------------===//

def PIR_MovOp : PIR_Op<"mov", [
    Pure, AllTypesMatch<["dst", "result"]>, InstOpInterface
  ]> {
  let summary = "Move operation";
  let description = [{
    The `pir.mov` operation moves a value from one register to another.

    Example:
    ```mlir
    %result = pir.mov %dst, %value : !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
  let arguments = (ins
    RegType:$dst,
    IntFloatOrRegType:$value
  );
  let results = (outs RegType:$result);
  let assemblyFormat = [{
    $dst `,` $value attr-dict `:` type($dst) `,` type($value)
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getDstMutable();
    }
    /// Get the instruction input operands.
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getValueMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

def PIR_MulFOp : BinaryFOp<"mulf"> {
  let summary = "Floating point multiplication operation";
  let description = [{
    The `pir.mulf` operation performs floating point multiplication.

    Example:
    ```mlir
    %result = pir.mulf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

def PIR_MulIOp : BinaryIOp<"muli"> {
  let summary = "Integer multiplication operation";
  let description = [{
    The `pir.muli` operation performs integer multiplication on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.muli i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// NegFOp
//===----------------------------------------------------------------------===//

def PIR_NegFOp : PIR_Op<"negf", [
    Pure, AllTypesMatch<["dst", "result"]>, InstOpInterface
  ]> {
  let summary = "Floating point negation operation";
  let description = [{
    The `pir.negf` operation negates a floating point value.

    Example:
    ```mlir
    %result = pir.negf f32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
  let arguments = (ins
    AnyFloatTypeAttr:$semantics,
    RegType:$dst,
    FloatOrRegType:$value
  );
  let results = (outs RegType:$result);
  let assemblyFormat = [{
    $semantics $dst `,` $value attr-dict `:` type($dst) `,` type($value)
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
// OrIOp
//===----------------------------------------------------------------------===//

def PIR_OrIOp : BinaryIOp<"ori"> {
  let summary = "Integer bitwise OR operation";
  let description = [{
    The `pir.ori` operation performs bitwise OR on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.ori i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RegCastOp
//===----------------------------------------------------------------------===//

def PIR_RegCastOp : PIR_Op<"reg_cast", [Pure]> {
  let summary = "PIR register cast operation";
  let arguments = (ins
    RegisterTypeInterface:$src
  );
  let results = (outs RegisterTypeInterface:$tgt);
  let assemblyFormat = [{
    $src attr-dict `:` type($src) `->` type($tgt)
  }];
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// RegConstraintOp
//===----------------------------------------------------------------------===//

def PIR_RegConstraintOp : PIR_Op<"reg_constraint"> {
  let summary = "Apply register kind constraint to value";
  let description = [{
    The `pir.reg_constraint` operation applies a register kind constraint to
    a value, indicating the preferred or required register class.

    Example:
    ```mlir
    pir.reg_constraint %value {kind = #amdgcn.reg_kind<sgpr>} : i32
    ```
  }];
  let arguments = (ins AnyType:$input, AnyAttr:$kind);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// RemSIOp
//===----------------------------------------------------------------------===//

def PIR_RemSIOp : BinaryIOp<"remsi"> {
  let summary = "Signed integer remainder operation";
  let description = [{
    The `pir.remsi` operation computes the signed remainder of integer division
    on two operands of `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.remsi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

def PIR_RemUIOp : BinaryIOp<"remui"> {
  let summary = "Unsigned integer remainder operation";
  let description = [{
    The `pir.remui` operation computes the unsigned remainder of integer division
    on two operands of `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.remui i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def PIR_SelectOp : PIR_Op<"select", [
    Pure, AllTypesMatch<["dst", "result"]>, InstOpInterface
  ]> {
  let summary = "Select operation";
  let description = [{
    The `pir.select` operation selects between two values based on a condition.

    Example:
    ```mlir
    %result = pir.select %dst, %cond, %true_val, %false_val : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
  let arguments = (ins
    RegType:$dst,
    IntFloatOrRegType:$condition,
    IntFloatOrRegType:$true_value,
    IntFloatOrRegType:$false_value
  );
  let results = (outs RegType:$result);
  let assemblyFormat = [{
    $dst `,` $condition `,` $true_value `,` $false_value attr-dict
    `:` type($dst) `,` type($condition) `,` type($true_value) `,` type($false_value)
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
// ShLIOp
//===----------------------------------------------------------------------===//

def PIR_ShLIOp : BinaryIOp<"shli"> {
  let summary = "Shift left operation";
  let description = [{
    The `pir.shli` operation performs logical left shift on the first operand
    by the amount specified in the second operand.

    Example:
    ```mlir
    %result = pir.shli i32 %dst, %value, %amount : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

def PIR_ShRSIOp : BinaryIOp<"shrsi"> {
  let summary = "Arithmetic shift right operation";
  let description = [{
    The `pir.shrsi` operation performs arithmetic (signed) right shift on the first
    operand by the amount specified in the second operand.

    Example:
    ```mlir
    %result = pir.shrsi i32 %dst, %value, %amount : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

def PIR_ShRUIOp : BinaryIOp<"shrui"> {
  let summary = "Logical shift right operation";
  let description = [{
    The `pir.shrui` operation performs logical (unsigned) right shift on the first
    operand by the amount specified in the second operand.

    Example:
    ```mlir
    %result = pir.shrui i32 %dst, %value, %amount : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SIToFPOp
//===----------------------------------------------------------------------===//

def PIR_SIToFPOp : CastOp<"sitofp", AnyFloatTypeAttr, AnyIntTypeAttr, IntOrRegType> {
  let summary = "Signed integer to floating point conversion";
  let description = [{
    The `pir.sitofp` operation converts a signed integer to a floating point value.

    Example:
    ```mlir
    %result = pir.sitofp f32 from i32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def PIR_StoreOp : PIR_Op<"store", [
    InstOpInterface,
    DependentOpInterface
  ]> {
  let summary = "Store operation";
  let description = [{
    The `pir.store` operation stores a value to memory.

    Example:
    ```mlir
    %token = pir.store #oir.ms<#gpu.global, rw> %value, %src, %offset
      : !amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, i32
    ```
  }];
  let arguments = (ins
    MemorySpaceAttrInterface:$memory_space,
    RegType:$value,
    // This is not a 1-1 mapping as the op also has an offset and const_offset
    // operands.
    Arg<IntOrRegType, "the reference to load from",
                      [MemWrite]>:$addr,
    IntOrRegType:$offset,
    I32:$const_offset,
    Variadic<LoadOrStoreToken>:$dependencies
  );
  let results = (outs StoreToken:$token);
  let assemblyFormat = [{
    $memory_space $value `,` $addr  `,` $offset `,` $const_offset
    (`dependencies` $dependencies^)? attr-dict
    `:` type($value) `,` type($addr) `,` type($offset) `,` type($const_offset)
    (`,` type($dependencies)^)?
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return {};
    }
    /// Get the instruction input operands.
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getOperation()->getOpOperands().slice(0, 4);
    }
    /// Get the instruction result ranges.
    ResultRange getInstResults() {
      return getOperation()->getResults().slice(0, 0);
    }
    //===------------------------------------------------------------------===//
    // DependentOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the output result representing the dependency.
    Value getOutDependency() {
      return getToken();
    }
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

def PIR_SubFOp : BinaryFOp<"subf"> {
  let summary = "Floating point subtraction operation";
  let description = [{
    The `pir.subf` operation performs floating point subtraction.

    Example:
    ```mlir
    %result = pir.subf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

def PIR_SubIOp : BinaryIOp<"subi"> {
  let summary = "Integer subtraction operation";
  let description = [{
    The `pir.subi` operation performs integer subtraction on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = pir.subi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

def PIR_ThreadIdOp : PIR_Op<"thread_id", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "PIR thread ID operation";
  let description = [{
    Returns the thread ID in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = "$dim attr-dict";
}

//===----------------------------------------------------------------------===//
// TimingStartOp
//===----------------------------------------------------------------------===//

def PIR_TimingStartOp : PIR_Op<"timing_start"> {
  let summary = "Start timing measurement";
  let description = [{
    The `pir.timing_start` operation reads the current time using s_memtime
    and returns it as an SGPR range. This operation does not perform any
    stores, allowing the measurement interval to start immediately after
    this operation.

    Example:
    ```mlir
    %start_time = pir.timing_start : !amdgcn.sgpr_range<[? + 2]>
    ```
  }];
  let arguments = (ins);
  let results = (outs RegisterTypeInterface:$start_time);
  let assemblyFormat = "attr-dict `:` type($start_time)";
}

//===----------------------------------------------------------------------===//
// TimingStopOp
//===----------------------------------------------------------------------===//

def PIR_TimingStopOp : PIR_Op<"timing_stop"> {
  let summary = "Stop timing measurement and store timestamps";
  let description = [{
    The `pir.timing_stop` operation:
    1. Reads the current time using s_memtime (end time)
    2. Stores both the start time (from timing_start) and end time to the
       provided buffer pointers using smem_store

    The stores are performed outside the [start, stop) measurement interval,
    ensuring they are not counted in the cycle count.

    Example:
    ```mlir
    pir.timing_stop %start_time, %start_buffer_ptr, %end_buffer_ptr
      : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>
    ```
  }];
  let arguments = (ins
    RegisterTypeInterface:$start_time,
    RegisterTypeInterface:$start_buffer_ptr,
    RegisterTypeInterface:$end_buffer_ptr
  );
  let results = (outs);
  let assemblyFormat = [{
    $start_time `,` $start_buffer_ptr `,` $end_buffer_ptr attr-dict
    `:` type($start_time) `,` type($start_buffer_ptr) `,` type($end_buffer_ptr)
  }];
}

//===----------------------------------------------------------------------===//
// ToRegOp
//===----------------------------------------------------------------------===//

def PIR_ToRegOp : PIR_Op<"to_reg", [Pure]> {
  let summary = "Convert non-register value to register type";
  let description = [{
    The `pir.to_reg` operation converts a non-register value to a register type.

    Example:
    ```mlir
    %reg = pir.to_reg %value : i32 -> !amdgcn.sgpr
    ```
  }];
  let arguments = (ins NotReg:$input);
  let results = (outs RegisterTypeInterface:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// TruncFOp
//===----------------------------------------------------------------------===//

def PIR_TruncFOp : CastOp<"truncf", AnyFloatTypeAttr, AnyFloatTypeAttr, FloatOrRegType> {
  let summary = "Floating point truncation operation";
  let description = [{
    The `pir.truncf` operation truncates a floating point value to a narrower type.

    Example:
    ```mlir
    %result = pir.truncf f32 from f64 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

def PIR_TruncIOp : CastOp<"trunci", AnyIntTypeAttr, AnyIntTypeAttr, IntOrRegType> {
  let summary = "Integer truncation operation";
  let description = [{
    The `pir.trunci` operation truncates an integer value to a narrower type.

    Example:
    ```mlir
    %result = pir.trunci i32 from i16 %dst, %value : !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// UIToFPOp
//===----------------------------------------------------------------------===//

def PIR_UIToFPOp : CastOp<"uitofp", AnyFloatTypeAttr, AnyIntTypeAttr, IntOrRegType> {
  let summary = "Unsigned integer to floating point conversion";
  let description = [{
    The `pir.uitofp` operation converts an unsigned integer to a floating point value.

    Example:
    ```mlir
    %result = pir.uitofp f32 from i32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

def DependenciesType : Variadic<TokenDependencyTypeInterface> {
  let minSize = 1;
}

def PIR_WaitOp : PIR_Op<"wait", [DependentOpInterface]> {
  let summary = "Wait operation";
  let description = [{
    The `pir.wait` waits for one or more operations to complete.

    Example:
    ```mlir
    pir.wait %d0, %d1, %d2 : !pir.store_token, !pir.store_token, !pir.load_token
    ```
  }];
  let arguments = (ins
    DependenciesType:$dependencies
  );
  let assemblyFormat = [{
    $dependencies attr-dict `:` type($dependencies)
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // DependentOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the output result representing the dependency.
    Value getOutDependency() {
      return {};
    }
  }];
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

def PIR_XOrIOp : BinaryIOp<"xori"> {
  let summary = "Integer bitwise XOR operation";
  let description = [{
    The `pir.xori` operation performs bitwise XOR on two integer operands.

    Example:
    ```mlir
    %result = pir.xori i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

#endif // ASTER_DIALECT_PIR_IR_PIROPS_TD
