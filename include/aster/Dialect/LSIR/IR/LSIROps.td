//===- LSIROps.td - LSIR operations ------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LSIR operations
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LSIR_IR_LSIROPS_TD
#define ASTER_DIALECT_LSIR_IR_LSIROPS_TD

include "aster/Dialect/LSIR/IR/LSIRDialect.td"
include "aster/Dialect/LSIR/IR/LSIRAttrs.td"
include "aster/Dialect/LSIR/IR/LSIRTypes.td"
include "aster/Interfaces/InstOpInterface.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td"
include "aster/IR/InstBase.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/// Helper class for binary operations with integer or register operands.
class BinaryIOp<string mnemonic, list<Trait> traits = []> :
  InstOperation<LSIR_Dialect, mnemonic, traits # [
    AllTypesMatch<["dst", "dst_res"]>,
    InstOpInterface, InstInferType, PureInst
  ]> {
  let leadingArguments = (ins AnyIntTypeAttr:$semantics);
  let outputs = (ins RegType:$dst);
  let inputs = (ins IntOrRegType:$lhs, IntOrRegType:$rhs);
  let assemblyFormat = [{
    $semantics $dst `,` $lhs `,` $rhs attr-dict
    `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
  let genInstVerifier = 0;
}

/// Helper class for binary operations with float or register operands.
class BinaryFOp<string mnemonic, list<Trait> traits = []> :
  LSIR_Op<mnemonic, traits # [
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
    OperandRange getInstOuts() {
      return getAsOperandRange(getOperation()->getOpOperands().take_front(1));
    }
    /// Get the instruction input operands.
    OperandRange getInstIns() {
      return getAsOperandRange(getOperation()->getOpOperands().drop_front(1));
    }
  }];
}

/// Helper class cast-like operations.
class CastOp<string mnemonic, Attr tgtAttr,
             Attr srcAttr, Type srcTy, list<Trait> traits = []> :
  LSIR_Op<mnemonic, traits # [
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
    OperandRange getInstOuts() {
      return getAsOperandRange(getDstMutable());
    }
    OperandRange getInstIns() {
      return getAsOperandRange(getValueMutable());
    }
  }];
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

def LSIR_AddFOp : BinaryFOp<"addf"> {
  let summary = "Floating point addition operation";
  let description = [{
    The `lsir.addf` operation performs floating point addition on two operands.

    Example:
    ```mlir
    %result = lsir.addf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

def LSIR_AddIOp : BinaryIOp<"addi"> {
  let summary = "Integer addition operation";
  let description = [{
    The `lsir.addi` operation performs integer addition on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = lsir.addi i32 %dst, %lhs, %rhs : !amgcn.sgpr, !amgcn.sgpr, !amgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

def LSIR_AllocaOp : LSIR_Op<"alloca"> {
  let summary = "LSIR alloca operation";
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

def LSIR_AndIOp : BinaryIOp<"andi"> {
  let summary = "Integer bitwise AND operation";
  let description = [{
    The `lsir.andi` operation performs bitwise AND on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = lsir.andi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// AssumeNoaliasOp
//===----------------------------------------------------------------------===//

def LSIR_AssumeNoaliasOp : LSIR_Op<"assume_noalias", [
    Pure,
    SameVariadicOperandSize,
    // SameVariadicResultSize
    ]> {
  let summary = "Assume no aliasing between multiple values";
  let description = [{
    The `lsir.assume_noalias` operation takes multiple operands and returns
    results of the same types. It asserts to MemoryDependenceAnalysis that
    the returned values do not alias with each other, eliminating unnecessary
    synchronization.

    This is useful for:
    - SGPRs/VGPRs holding pointers known to point to non-overlapping memory
    - Values from different sources known to be independent
    - Optimization hints for memory dependence analysis

    Example:
    ```mlir
    %ptr1_noalias, %ptr2_noalias = lsir.assume_noalias %ptr1, %ptr2
      : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{ $operands attr-dict `:` functional-type(operands, results) }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

def LSIR_CmpFOp : LSIR_Op<"cmpf", [InstOpInterface, Pure]> {
  let summary = "Floating point comparison operation";
  let description = [{
    The `lsir.cmpf` operation compares two floating point values and returns i1.
    The `i1` return value is kept late in the pipeline and is only translated to
    SCC after register allocation, together with cf branch operations.

    Example:
    ```mlir
    %result = lsir.cmpf f32 olt %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
  let arguments = (ins
    AnyFloatTypeAttr:$semantics,
    Arith_CmpFPredicateAttr:$predicate,
    FloatOrRegType:$lhs,
    FloatOrRegType:$rhs
  );
  let results = (outs I1:$result);
  let assemblyFormat = [{
    $semantics $predicate $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Returns false; comparison ops produce i1 (SCC) not DPS format.
    bool isDPSInstruction() { return false; }
    /// Get the instruction output operands.
    OperandRange getInstOuts() {
      return getAsOperandRange(getOperation()->getOpOperands().slice(0, 0));
    }
    /// Get the instruction input operands.
    OperandRange getInstIns() {
      return getAsOperandRange(getOperation()->getOpOperands());
    }
    /// Get the instruction results.
    ResultRange getInstResults() {
      return getOperation()->getResults().slice(0, 0);
    }
  }];
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

def LSIR_CmpIOp : LSIR_Op<"cmpi", [InstOpInterface, Pure]> {
  let summary = "Integer comparison operation";
  let description = [{
    The `lsir.cmpi` operation compares two integer operands according to the
    specified predicate and returns i1.
    The `i1` return value is kept late in the pipeline and is only translated to
    SCC after register allocation, together with cf branch operations.

    Example:
    ```mlir
    %result = lsir.cmpi i32 eq %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
  let arguments = (ins
    AnyIntTypeAttr:$semantics,
    Arith_CmpIPredicateAttr:$predicate,
    IntOrRegType:$lhs,
    IntOrRegType:$rhs
  );
  let results = (outs I1:$result);
  let assemblyFormat = [{
    $semantics $predicate $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Returns false; comparison ops produce i1 (SCC) not DPS format.
    bool isDPSInstruction() { return false; }
    /// Get the instruction output operands.
    OperandRange getInstOuts() {
      return getAsOperandRange(getOperation()->getOpOperands().slice(0, 0));
    }
    /// Get the instruction input operands.
    OperandRange getInstIns() {
      return getAsOperandRange(getOperation()->getOpOperands());
    }
    /// Get the instruction results.
    ResultRange getInstResults() {
      return getOperation()->getResults().slice(0, 0);
    }
  }];
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

def LSIR_CopyOp : LSIR_Op<"copy", [
    AllTypesMatch<["target", "result"]>, InstOpInterface,
    ConditionallySpeculatable, MemoryEffectsOpInterface
  ]> {
  let summary = "Copy operation";
  let description = [{
    The `lsir.copy` operation copies a value from a source register to a target
    register. The target and result types must match.

    Example:
    ```mlir
    %result = lsir.copy %target, %source : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
  let arguments = (ins RegType:$target, RegType:$source);
  let results = (outs RegType:$result);
  let assemblyFormat = [{
    $target `,` $source attr-dict `:` type($target) `,` type($source)
  }];
  let hasCanonicalizeMethod = 1;
  let extraClassDeclaration = [{
    /// Returns the speculatability of the operation.
    Speculation::Speculatability getSpeculatability() {
      return aster::detail::getInstSpeculatabilityImpl(*this);
    }

    /// Returns the memory effects of the operation.
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
            &effects) {
      return aster::detail::getInstEffectsImpl(*this, effects);
    }

    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    OperandRange getInstOuts() {
      return getAsOperandRange(getOperation()->getOpOperands().take_front(1));
    }
    /// Get the instruction input operands.
    OperandRange getInstIns() {
      return getAsOperandRange(getOperation()->getOpOperands().drop_front(1));
    }
  }];
}

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

def LSIR_DivFOp : BinaryFOp<"divf"> {
  let summary = "Floating point division operation";
  let description = [{
    The `lsir.divf` operation performs floating point division.

    Example:
    ```mlir
    %result = lsir.divf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

def LSIR_DivSIOp : BinaryIOp<"divsi"> {
  let summary = "Signed integer division operation";
  let description = [{
    The `lsir.divsi` operation performs signed integer division on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = lsir.divsi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

def LSIR_DivUIOp : BinaryIOp<"divui"> {
  let summary = "Unsigned integer division operation";
  let description = [{
    The `lsir.divui` operation performs unsigned integer division on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = lsir.divui i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ExtFOp
//===----------------------------------------------------------------------===//

def LSIR_ExtFOp : CastOp<"extf", AnyFloatTypeAttr, AnyFloatTypeAttr, FloatOrRegType> {
  let summary = "Floating point extension operation";
  let description = [{
    The `lsir.extf` operation extends a floating point value to a wider type.

    Example:
    ```mlir
    %result = lsir.extf f64 from f32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

def LSIR_ExtSIOp : CastOp<"extsi", AnyIntTypeAttr, AnyIntTypeAttr, IntOrRegType> {
  let summary = "Sign extension operation";
  let description = [{
    The `lsir.extsi` operation sign-extends an integer value to a wider type.

    Example:
    ```mlir
    %result = lsir.extsi i32 from i16 %dst, %value : !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

def LSIR_ExtUIOp : CastOp<"extui", AnyIntTypeAttr, AnyIntTypeAttr, IntOrRegType> {
  let summary = "Zero extension operation";
  let description = [{
    The `lsir.extui` operation zero-extends an integer value to a wider type.

    Example:
    ```mlir
    %result = lsir.extui i32 from i16 %dst, %value : !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// FPToSIOp
//===----------------------------------------------------------------------===//

def LSIR_FPToSIOp : CastOp<"fptosi", AnyIntTypeAttr, AnyFloatTypeAttr, FloatOrRegType> {
  let summary = "Floating point to signed integer conversion";
  let description = [{
    The `lsir.fptosi` operation converts a floating point value to a signed integer.

    Example:
    ```mlir
    %result = lsir.fptosi i32 from f32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// FPToUIOp
//===----------------------------------------------------------------------===//

def LSIR_FPToUIOp : CastOp<"fptoui", AnyIntTypeAttr, AnyFloatTypeAttr, FloatOrRegType> {
  let summary = "Floating point to unsigned integer conversion";
  let description = [{
    The `lsir.fptoui` operation converts a floating point value to an unsigned integer.

    Example:
    ```mlir
    %result = lsir.fptoui i32 from f32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// FromRegOp
//===----------------------------------------------------------------------===//

def LSIR_FromRegOp : LSIR_Op<"from_reg", [Pure]> {
  let summary = "Convert register type to non-register value";
  let description = [{
    The `lsir.from_reg` operation converts a register type to a non-register value.

    Example:
    ```mlir
    %value = lsir.from_reg %reg : !amdgcn.sgpr -> i32
    ```
  }];
  let arguments = (ins RegisterTypeInterface:$input);
  let results = (outs NotReg:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def LSIR_LoadOp : LSIR_Op<"load", [
    AllTypesMatch<["dst", "dst_res"]>, InstOpInterface, DependentOpInterface
  ]> {
  let summary = "Load operation";
  let description = [{
    The `lsir.load` operation loads a value from memory.

    Example:
    ```mlir
    %result = lsir.load #oir.ms<#gpu.address_space<global>, rw> %dst, %src, %offset
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
    OperandRange getInstOuts() {
      return getAsOperandRange(getOperation()->getOpOperands().take_front(1));
    }
    /// Get the instruction input operands.
    OperandRange getInstIns() {
      return getAsOperandRange(getOperation()->getOpOperands().slice(1, 3));
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

def LSIR_MaximumFOp : BinaryFOp<"maximumf"> {
  let summary = "Floating point maximum operation";
  let description = [{
    The `lsir.maximumf` operation returns the maximum of two floating point values.

    Example:
    ```mlir
    %result = lsir.maximumf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

def LSIR_MaxSIOp : BinaryIOp<"maxsi"> {
  let summary = "Signed integer maximum operation";
  let description = [{
    The `lsir.maxsi` operation returns the maximum of two signed integers.

    Example:
    ```mlir
    %result = lsir.maxsi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

def LSIR_MaxUIOp : BinaryIOp<"maxui"> {
  let summary = "Unsigned integer maximum operation";
  let description = [{
    The `lsir.maxui` operation returns the maximum of two unsigned integers.

    Example:
    ```mlir
    %result = lsir.maxui i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MinimumFOp
//===----------------------------------------------------------------------===//

def LSIR_MinimumFOp : BinaryFOp<"minimumf"> {
  let summary = "Floating point minimum operation";
  let description = [{
    The `lsir.minimumf` operation returns the minimum of two floating point values.

    Example:
    ```mlir
    %result = lsir.minimumf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MovOp
//===----------------------------------------------------------------------===//

def LSIR_MovOp : LSIR_Op<"mov", [
    Pure, AllTypesMatch<["dst", "result"]>, InstOpInterface
  ]> {
  let summary = "Move operation";
  let description = [{
    The `lsir.mov` operation moves a value from one register to another.

    Example:
    ```mlir
    %result = lsir.mov %dst, %value : !amdgcn.sgpr, !amdgcn.sgpr
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
    OperandRange getInstOuts() {
      return getAsOperandRange(getDstMutable());
    }
    /// Get the instruction input operands.
    OperandRange getInstIns() {
      return getAsOperandRange(getValueMutable());
    }
  }];
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

def LSIR_MulFOp : BinaryFOp<"mulf"> {
  let summary = "Floating point multiplication operation";
  let description = [{
    The `lsir.mulf` operation performs floating point multiplication.

    Example:
    ```mlir
    %result = lsir.mulf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

def LSIR_MulIOp : BinaryIOp<"muli"> {
  let summary = "Integer multiplication operation";
  let description = [{
    The `lsir.muli` operation performs integer multiplication on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = lsir.muli i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// NegFOp
//===----------------------------------------------------------------------===//

def LSIR_NegFOp : LSIR_Op<"negf", [
    Pure, AllTypesMatch<["dst", "result"]>, InstOpInterface
  ]> {
  let summary = "Floating point negation operation";
  let description = [{
    The `lsir.negf` operation negates a floating point value.

    Example:
    ```mlir
    %result = lsir.negf f32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
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
    OperandRange getInstOuts() {
      return getAsOperandRange(getOperation()->getOpOperands().take_front(1));
    }
    /// Get the instruction input operands.
    OperandRange getInstIns() {
      return getAsOperandRange(getOperation()->getOpOperands().drop_front(1));
    }
  }];
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

def LSIR_OrIOp : BinaryIOp<"ori"> {
  let summary = "Integer bitwise OR operation";
  let description = [{
    The `lsir.ori` operation performs bitwise OR on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = lsir.ori i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RegCastOp
//===----------------------------------------------------------------------===//

def LSIR_RegCastOp : LSIR_Op<"reg_cast", [Pure]> {
  let summary = "LSIR register cast operation";
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

def LSIR_RegConstraintOp : LSIR_Op<"reg_constraint"> {
  let summary = "Apply register kind constraint to value";
  let description = [{
    The `lsir.reg_constraint` operation applies a register kind constraint to
    a value, indicating the preferred or required register class.

    Example:
    ```mlir
    lsir.reg_constraint %value {kind = #amdgcn.reg_kind<sgpr>} : i32
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

def LSIR_RemSIOp : BinaryIOp<"remsi"> {
  let summary = "Signed integer remainder operation";
  let description = [{
    The `lsir.remsi` operation computes the signed remainder of integer division
    on two operands of `IntOrRegType`.

    Example:
    ```mlir
    %result = lsir.remsi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

def LSIR_RemUIOp : BinaryIOp<"remui"> {
  let summary = "Unsigned integer remainder operation";
  let description = [{
    The `lsir.remui` operation computes the unsigned remainder of integer division
    on two operands of `IntOrRegType`.

    Example:
    ```mlir
    %result = lsir.remui i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def LSIR_SelectOp : LSIR_Op<"select", [
    AllTypesMatch<["dst", "result"]>, InstOpInterface,
    ConditionallySpeculatable, MemoryEffectsOpInterface
  ]> {
  let summary = "Select operation";
  let description = [{
    The `lsir.select` operation selects between two values based on a condition.
    The condition can be either a register type or i1 (from lsir.cmpi/cmpf).

    When the condition is i1, LegalizeCF fuses the cmpi + select into
    s_cmp + s_cselect_b32. When the condition is a register, it lowers to
    a conditional move instruction.

    Example:
    ```mlir
    %result = lsir.select %dst, %cond, %tv, %fv
      : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    %result = lsir.select %dst, %i1_cond, %tv, %fv
      : !amdgcn.sgpr, i1, i32, i32
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
    /// Returns the speculatability of the operation.
    Speculation::Speculatability getSpeculatability() {
      return aster::detail::getInstSpeculatabilityImpl(*this);
    }

    /// Returns the memory effects of the operation.
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
            &effects) {
      return aster::detail::getInstEffectsImpl(*this, effects);
    }

    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    OperandRange getInstOuts() {
      return getAsOperandRange(getOperation()->getOpOperands().take_front(1));
    }
    /// Get the instruction input operands.
    OperandRange getInstIns() {
      return getAsOperandRange(getOperation()->getOpOperands().drop_front(1));
    }
  }];
}

//===----------------------------------------------------------------------===//
// ShLIOp
//===----------------------------------------------------------------------===//

def LSIR_ShLIOp : BinaryIOp<"shli"> {
  let summary = "Shift left operation";
  let description = [{
    The `lsir.shli` operation performs logical left shift on the first operand
    by the amount specified in the second operand.

    Example:
    ```mlir
    %result = lsir.shli i32 %dst, %value, %amount : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

def LSIR_ShRSIOp : BinaryIOp<"shrsi"> {
  let summary = "Arithmetic shift right operation";
  let description = [{
    The `lsir.shrsi` operation performs arithmetic (signed) right shift on the first
    operand by the amount specified in the second operand.

    Example:
    ```mlir
    %result = lsir.shrsi i32 %dst, %value, %amount : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

def LSIR_ShRUIOp : BinaryIOp<"shrui"> {
  let summary = "Logical shift right operation";
  let description = [{
    The `lsir.shrui` operation performs logical (unsigned) right shift on the first
    operand by the amount specified in the second operand.

    Example:
    ```mlir
    %result = lsir.shrui i32 %dst, %value, %amount : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SIToFPOp
//===----------------------------------------------------------------------===//

def LSIR_SIToFPOp : CastOp<"sitofp", AnyFloatTypeAttr, AnyIntTypeAttr, IntOrRegType> {
  let summary = "Signed integer to floating point conversion";
  let description = [{
    The `lsir.sitofp` operation converts a signed integer to a floating point value.

    Example:
    ```mlir
    %result = lsir.sitofp f32 from i32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def LSIR_StoreOp : LSIR_Op<"store", [
    InstOpInterface,
    DependentOpInterface
  ]> {
  let summary = "Store operation";
  let description = [{
    The `lsir.store` operation stores a value to memory.

    Example:
    ```mlir
    %token = lsir.store #oir.ms<#gpu.global, rw> %value, %src, %offset
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
    OperandRange getInstOuts() {
      return getAsOperandRange({});
    }
    /// Get the instruction input operands.
    OperandRange getInstIns() {
      return getAsOperandRange(getOperation()->getOpOperands().slice(0, 4));
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

def LSIR_SubFOp : BinaryFOp<"subf"> {
  let summary = "Floating point subtraction operation";
  let description = [{
    The `lsir.subf` operation performs floating point subtraction.

    Example:
    ```mlir
    %result = lsir.subf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

def LSIR_SubIOp : BinaryIOp<"subi"> {
  let summary = "Integer subtraction operation";
  let description = [{
    The `lsir.subi` operation performs integer subtraction on two operands of
    `IntOrRegType`.

    Example:
    ```mlir
    %result = lsir.subi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// TimingStartOp
//===----------------------------------------------------------------------===//

def LSIR_TimingStartOp : LSIR_Op<"timing_start"> {
  let summary = "Start timing measurement";
  let description = [{
    The `lsir.timing_start` operation reads the current time using s_memtime
    and returns it as an SGPR range. This operation does not perform any
    stores, allowing the measurement interval to start immediately after
    this operation.

    Example:
    ```mlir
    %start_time = lsir.timing_start : !amdgcn.sgpr_range<[? + 2]>
    ```
  }];
  let arguments = (ins);
  let results = (outs RegisterTypeInterface:$start_time);
  let assemblyFormat = "attr-dict `:` type($start_time)";
}

//===----------------------------------------------------------------------===//
// TimingStopOp
//===----------------------------------------------------------------------===//

def LSIR_TimingStopOp : LSIR_Op<"timing_stop"> {
  let summary = "Stop timing measurement and store timestamps";
  let description = [{
    The `lsir.timing_stop` operation:
    1. Reads the current time using s_memtime (end time)
    2. Stores both the start time (from timing_start) and end time to the
       provided buffer pointers using smem_store

    The stores are performed outside the [start, stop) measurement interval,
    ensuring they are not counted in the cycle count.

    Example:
    ```mlir
    lsir.timing_stop %start_time, %start_buffer_ptr, %end_buffer_ptr
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

def LSIR_ToRegOp : LSIR_Op<"to_reg", [Pure]> {
  let summary = "Convert non-register value to register type";
  let description = [{
    The `lsir.to_reg` operation converts a non-register value to a register type.

    Example:
    ```mlir
    %reg = lsir.to_reg %value : i32 -> !amdgcn.sgpr
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

def LSIR_TruncFOp : CastOp<"truncf", AnyFloatTypeAttr, AnyFloatTypeAttr, FloatOrRegType> {
  let summary = "Floating point truncation operation";
  let description = [{
    The `lsir.truncf` operation truncates a floating point value to a narrower type.

    Example:
    ```mlir
    %result = lsir.truncf f32 from f64 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

def LSIR_TruncIOp : CastOp<"trunci", AnyIntTypeAttr, AnyIntTypeAttr, IntOrRegType> {
  let summary = "Integer truncation operation";
  let description = [{
    The `lsir.trunci` operation truncates an integer value to a narrower type.

    Example:
    ```mlir
    %result = lsir.trunci i32 from i16 %dst, %value : !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// UIToFPOp
//===----------------------------------------------------------------------===//

def LSIR_UIToFPOp : CastOp<"uitofp", AnyFloatTypeAttr, AnyIntTypeAttr, IntOrRegType> {
  let summary = "Unsigned integer to floating point conversion";
  let description = [{
    The `lsir.uitofp` operation converts an unsigned integer to a floating point value.

    Example:
    ```mlir
    %result = lsir.uitofp f32 from i32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

def DependenciesType : Variadic<TokenDependencyTypeInterface> {
  let minSize = 1;
}

def LSIR_WaitOp : LSIR_Op<"wait", [DependentOpInterface]> {
  let summary = "Wait operation";
  let description = [{
    The `lsir.wait` waits for one or more operations to complete.

    Example:
    ```mlir
    lsir.wait %d0, %d1, %d2 : !lsir.store_token, !lsir.store_token, !lsir.load_token
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

def LSIR_XOrIOp : BinaryIOp<"xori"> {
  let summary = "Integer bitwise XOR operation";
  let description = [{
    The `lsir.xori` operation performs bitwise XOR on two integer operands.

    Example:
    ```mlir
    %result = lsir.xori i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    ```
  }];
}

#endif // ASTER_DIALECT_LSIR_IR_LSIROPS_TD
