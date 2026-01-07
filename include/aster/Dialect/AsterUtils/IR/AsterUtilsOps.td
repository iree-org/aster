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
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AssumeRangeOp
//===----------------------------------------------------------------------===//

def AsterUtils_AssumeRangeOp : AsterUtils_Op<"assume_range", [
    Pure, AllTypesMatch<["input", "result"]>,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "AsterUtils assume int range";
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
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
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
