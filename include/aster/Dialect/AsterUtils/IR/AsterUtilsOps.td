//===- AsterUtilsOps.td - AsterUtils operations ------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AsterUtils operations
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSOPS_TD
#define ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSOPS_TD

include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.td"
include "aster/Dialect/AsterUtils/IR/AsterUtilsAttrs.td"
include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.td"
include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AssumeRangeOp
//===----------------------------------------------------------------------===//

def AsterUtils_AssumeRangeOp : AsterUtils_Op<"assume_range", [
    Pure, AllTypesMatch<["input", "result"]>,
    AttrSizedOperandSegments,
    OptionalTypesMatchWith<"dynamic_min type matches input type",
                           "input", "dynamic_min", "$_self">,
    OptionalTypesMatchWith<"dynamic_max type matches input type",
                           "input", "dynamic_max", "$_self">,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "AsterUtils assume int range";
  let description = [{
    The `aster_utils.assume_range` operation specifies that the input value
    is within a given range. The range can be specified either statically
    (via attributes) or dynamically (via SSA values).

    Example:
    ```mlir
    // Static range: value is in [0, 1024)
    %0 = aster_utils.assume_range %x min 0 max 1024 : i64

    // Dynamic min, static max:
    %1 = aster_utils.assume_range %x min %min_val max 1024 : i64

    // Fully dynamic range:
    %2 = aster_utils.assume_range %x min %min_val max %max_val : i64
    ```
  }];
  let arguments = (ins
    AnySignlessIntegerOrIndex:$input,
    Optional<AnySignlessIntegerOrIndex>:$dynamic_min,
    Optional<AnySignlessIntegerOrIndex>:$dynamic_max,
    OptionalAttr<IndexAttr>:$static_min,
    OptionalAttr<IndexAttr>:$static_max
  );
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let assemblyFormat = [{
    $input
    custom<AssumeRangeBound>("\"min\"", $dynamic_min, $static_min)
    custom<AssumeRangeBound>("\"max\"", $dynamic_max, $static_max)
    attr-dict `:` type($input)
  }];
  let extraClassDeclaration = [{
    /// Get the minimum bound as an OpFoldResult (either static or dynamic).
    OpFoldResult getMin() {
      if (getDynamicMin())
        return getDynamicMin();
      if (getStaticMin())
        return getStaticMinAttr();
      return nullptr;
    }

    /// Get the maximum bound as an OpFoldResult (either static or dynamic).
    OpFoldResult getMax() {
      if (getDynamicMax())
        return getDynamicMax();
      if (getStaticMax())
        return getStaticMaxAttr();
      return nullptr;
    }

    /// Returns true if the minimum bound is statically known.
    bool hasStaticMin() {
      return !getDynamicMin() && getStaticMin().has_value();
    }

    /// Returns true if the maximum bound is statically known.
    bool hasStaticMax() {
      return !getDynamicMax() && getStaticMax().has_value();
    }
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AssumeUniform
//===----------------------------------------------------------------------===//

def AsterUtils_AssumeUniformOp : AsterUtils_Op<"assume_uniform", [
    Pure, AllTypesMatch<["input", "result"]>,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRangesFromOptional"]>
  ]> {
  let summary = "AsterUtils assume uniform value";
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

def AsterUtils_BlockDimOp : AsterUtils_Op<"block_dim", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "AsterUtils block dimension operation";
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

def AsterUtils_BlockIdOp : AsterUtils_Op<"block_id", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "AsterUtils block ID operation";
  let description = [{
    Returns the block (workgroup) ID in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = "$dim attr-dict";
}

//===----------------------------------------------------------------------===//
// ExecuteRegionOp
//===----------------------------------------------------------------------===//

def AsterUtils_ExecuteRegionOp : AsterUtils_Op<"execute_region", [
    AffineScope, AutomaticAllocationScope, NoRegionArguments,
    DeclareOpInterfaceMethods<RegionBranchOpInterface, [
      "getSuccessorInputs"
    ]>,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"aster_utils::YieldOp">,
  ]> {
  let summary = "Executes its region once";
  let description = [{
    The `aster_utils.execute_region` operation is similar to `scf.execute_region`,
    executing its single region exactly once. It can be used to isolate a region
    of code for optimization or analysis purposes.
    The main difference with `scf.execute_region` is that this operation is
    also an affine scope, and contains a single block.
  }];
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    attr-dict-with-keyword (`:` type($results)^)? $region
  }];
}

//===----------------------------------------------------------------------===//
// FromAnyOp
//===----------------------------------------------------------------------===//

def AsterUtils_FromAnyOp : AsterUtils_Op<"from_any", [Pure]> {
  let summary = "Convert an AnyType to a concrete type";
  let description = [{
    The `from_any` operation converts a value of AsterUtils AnyType to a
    concrete type. This is the inverse of `to_any` and is used to recover
    the original type from a type-erased value.

    Example:
    ```mlir
    %value = aster_utils.from_any %any : i32
    ```
  }];
  let arguments = (ins AsterUtils_AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($result)
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// GridDimOp
//===----------------------------------------------------------------------===//

def AsterUtils_GridDimOp : AsterUtils_Op<"grid_dim", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "AsterUtils grid dimension operation";
  let description = [{
    Returns the grid dimension in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = "$dim attr-dict";
}

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

def AsterUtils_ThreadIdOp : AsterUtils_Op<"thread_id", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "AsterUtils thread ID operation";
  let description = [{
    Returns the thread ID in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = "$dim attr-dict";
}

//===----------------------------------------------------------------------===//
// ToAnyOp
//===----------------------------------------------------------------------===//

def AsterUtils_ToAnyOp : AsterUtils_Op<"to_any", [Pure]> {
  let summary = "Convert a value to AnyType";
  let description = [{
    The `to_any` operation converts any value to the AsterUtils AnyType.
    This is useful for type-erasing values that need to be stored or passed
    through generic interfaces.

    Example:
    ```mlir
    %any = aster_utils.to_any %value : i32
    ```
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AsterUtils_AnyType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// PtrAddOp
//===----------------------------------------------------------------------===//

def AsterUtils_PtrAddOp : AsterUtils_Op<"ptr_add", [
    Pure,
    AllTypesMatch<["ptr", "result"]>,
    OptionalTypesMatchWith<"uniform_offset type matches offset type",
                           "offset", "uniform_offset",
                           "$_self">
  ]> {
  let summary = "Add offsets to a pointer";
  let description = [{
    The `ptr_add` operation computes a new pointer by adding offsets to an
    existing pointer. The operation supports multiple offset components:
    - `offset`: A dynamic (potentially non-uniform) offset
    - `uniform_offset`: An optional uniform offset (same across threads)
    - `const_offset`: A constant offset attribute (defaults to 0)

    The final address is computed as:
      result = ptr + offset + uniform_offset + const_offset

    All offset operands must be signless integer types.

    Example:
    ```mlir
    %new_ptr = aster_utils.ptr_add %ptr, %offset : !ptr.ptr<#ptr.generic_space>, i64
    %new_ptr = aster_utils.ptr_add %ptr, %offset, %uniform : !ptr.ptr<#ptr.generic_space>, i64
    %new_ptr = aster_utils.ptr_add %ptr, %offset const_offset = 16 : !ptr.ptr<#ptr.generic_space>, i64
    ```
  }];
  let arguments = (ins
    Ptr_PtrType:$ptr,
    AnySignlessInteger:$offset,
    Optional<AnySignlessInteger>:$uniform_offset,
    DefaultValuedAttr<I64Attr, "0">:$const_offset
  );
  let results = (outs Ptr_PtrType:$result);
  let assemblyFormat = [{
    $ptr `,` $offset (`,` $uniform_offset^)? (`const_offset` `=` $const_offset^)?
    attr-dict `:` type($ptr) `,` type($offset)
  }];
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

def AsterUtils_StructCreateOp : AsterUtils_Op<"struct_create", [Pure]> {
  let summary = "Create a struct value from field values";
  let description = [{
    The `struct_create` operation creates a struct value from a list of field
    values. The number and types of the values must match the struct type's
    field definitions.

    Example:
    ```mlir
    %s = aster_utils.struct_create(%x, %y) : (i32, f32) -> !aster_utils.struct<x: i32, y: f32>
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$fields);
  let results = (outs AsterUtils_Struct:$result);
  let assemblyFormat = [{
    `(` $fields `)` attr-dict `:` functional-type($fields, $result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

def AsterUtils_StructExtractOp : AsterUtils_Op<"struct_extract", [Pure]> {
  let summary = "Extract fields from a struct value";
  let description = [{
    The `struct_extract` operation extracts one or more field values from a
    struct by name. All field names must exist in the struct type.

    Example:
    ```mlir
    %x = aster_utils.struct_extract %s["x"] : !aster_utils.struct<x: i32, y: f32> -> i32
    %x, %y = aster_utils.struct_extract %s["x", "y"] : !aster_utils.struct<x: i32, y: f32> -> i32, f32
    ```

    Canonicalization: If the input is a `struct_create`, the extract is folded
    to directly use the corresponding operands of the create operation.
  }];
  let arguments = (ins
    AsterUtils_Struct:$input,
    StrArrayAttr:$fieldNames
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $input $fieldNames attr-dict `:` type($input) `->` type($results)
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def AsterUtils_YieldOp : AsterUtils_Op<"yield", [
    Pure, ReturnLike, Terminator, ParentOneOf<["ExecuteRegionOp"]>
  ]> {
  let summary = "Yield operation";
  let description = [{
    The `aster_utils.yield` operation yields control-flow and values to the
    parent operation, similar to `scf.yield`.
  }];
  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /**/ }]>];
  let assemblyFormat = [{
    ($results^)?  attr-dict (`:` type($results)^)?
  }];
}

#endif // ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSOPS_TD
