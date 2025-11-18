//===- LSIRTypes.td - LSIR types ---------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LSIR_IR_LSIRTYPES_TD
#define ASTER_DIALECT_LSIR_IR_LSIRTYPES_TD

include "aster/Dialect/LSIR/IR/LSIRDialect.td"
include "aster/Interfaces/DependentOpInterface.td"
include "aster/Interfaces/RegisterType.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// LSIR types
//===----------------------------------------------------------------------===//

def LoadToken : LSIR_Type<"LoadToken", "load_token",  [
    MemRefElementTypeInterface,
    TokenDependencyTypeInterface
  ]> {
  let summary = "Load token type";
  let description = [{
    A type representing a load operation token.
  }];
}

def StoreToken : LSIR_Type<"StoreToken", "store_token", [
    MemRefElementTypeInterface,
    TokenDependencyTypeInterface
  ]> {
  let summary = "Store token type";
  let description = [{
    A type representing a store operation token.
  }];
}

/// Type predicate for load or store token types.
def LoadOrStoreToken : AnyTypeOf<[LoadToken, StoreToken]> {
  let summary = "Load or store token";
}

//===----------------------------------------------------------------------===//
// LSIR helper type predicates
//===----------------------------------------------------------------------===//

/// Register operand type.
defvar RegType = RegisterTypeInterface;

/// Type predicate for int-like or int-register types.
def IntOrRegType : AnyTypeOf<[RegType, AnySignlessInteger]> {
  let summary = "Integer or register";
}

/// Type predicate for float-like or float-register types.
def FloatOrRegType : AnyTypeOf<[RegType, AnyFloat]> {
  let summary = "Float or register";
}

/// Type predicate for int-like or int-register types.
def IntFloatOrRegType : AnyTypeOf<[RegType, AnySignlessInteger, AnyFloat]> {
  let summary = "Integer, float or register";
}

/// Not a register type.
def NotReg : Type<Neg<RegType.predicate>, "Not a register type">;

/// Type attribute for any integer type.
def AnyIntTypeAttr : TypeAttrBase<"IntegerType", "any integer type attribute">;

/// Type attribute for any float type.
def AnyFloatTypeAttr : TypeAttrBase<"FloatType", "any float type attribute">;

#endif // ASTER_DIALECT_LSIR_IR_LSIRTYPES_TD
