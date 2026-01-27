//===- AsterUtilsTypes.td - AsterUtils types ---------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSTYPES_TD
#define ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSTYPES_TD

include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// AsterUtils types
//===----------------------------------------------------------------------===//

def AsterUtils_AnyType :
  AsterUtils_Type<"AnyType", "any", [MemRefElementTypeInterface]> {
  let description = "Any type";
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

def AsterUtils_StructType :
  AsterUtils_Type<"Struct", "struct"> {
  let summary = "Composite type with named fields";
  let description = [{
    A struct type represents a composite value with named fields. Each field
    has a name (string) and a type. Fields are accessed by name.

    Syntax:
    ```mlir
    !aster_utils.struct<x: i32, y: f32>
    !aster_utils.struct<a: tensor<4xf16>, b: tensor<4xf16>>
    ```

    The struct type is immutable and uniqued by its field names and types.
  }];

  let parameters = (ins
    ArrayRefParameter<"mlir::StringAttr", "field names">:$fieldNames,
    ArrayRefParameter<"mlir::Type", "field types">:$fieldTypes
  );

  let extraClassDeclaration = [{
    /// Get the number of fields in the struct.
    size_t getNumFields() const { return getFieldNames().size(); }

    /// Get the type of a field by index.
    mlir::Type getFieldType(size_t index) const {
      return getFieldTypes()[index];
    }

    /// Get the name of a field by index.
    mlir::StringAttr getFieldName(size_t index) const {
      return getFieldNames()[index];
    }

    /// Get the index of a field by name. Returns std::nullopt if not found.
    std::optional<size_t> getFieldIndex(mlir::StringAttr name) const;
    std::optional<size_t> getFieldIndex(llvm::StringRef name) const;

    /// Get the type of a field by name. Returns nullptr if not found.
    mlir::Type getFieldTypeByName(mlir::StringAttr name) const;
    mlir::Type getFieldTypeByName(llvm::StringRef name) const;
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

/// Type constraint for AsterUtils struct types.
def AsterUtils_Struct : Type<
  CPred<"::llvm::isa<::mlir::aster::aster_utils::StructType>($_self)">,
  "aster_utils struct type">;

#endif // ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSTYPES_TD
