//===- PIRTypes.td - PIR types -----------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_PIR_IR_PIRTYPES_TD
#define ASTER_DIALECT_PIR_IR_PIRTYPES_TD

include "aster/Dialect/PIR/IR/PIRDialect.td"
include "aster/Interfaces/RegisterType.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// PIR types
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TypedRegister
//===----------------------------------------------------------------------===//

def TypedRegister : PIR_Type<"TypedRegister", "reg", [
    RegisterTypeInterface, MemRefElementTypeInterface
  ]> {
  let summary = "Typed register";
  let parameters = (ins "::mlir::Type":$type,
                        "::mlir::aster::RegisterTypeInterface":$reg);
  let assemblyFormat = "`<` $type `:` $reg `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    /// Returns true if the register is relocatable.
    bool isRelocatable() const { return getReg().isRelocatable(); }
    bool isRegisterRange() const { return getReg().isRegisterRange(); }
    RegisterRange getAsRange() const {
      return getReg().getAsRange();
    }
    int64_t getRegisterKindAsInt() const {
      return getReg().getRegisterKindAsInt();
    }
    Attribute getRegisterKind() const {
      return getReg().getRegisterKind();
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      return TypedRegisterType::get(getType(), getReg().cloneRegisterType(range));
    }
    RegisterTypeInterface cloneRegisterType(Register reg) const {
      return TypedRegisterType::get(getType(), getReg().cloneRegisterType(reg));
    }
  }];
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$type,
      "RegisterTypeInterface":$regTy), [{
      return $_get(type.getContext(), type, regTy);
    }]>,
  ];
}

/// Type predicate for TypedRegister of specific element types.
class RegTypeOf<list<Type> types, string descr = "register type"> :
  Type<And<[TypedRegister.predicate,
            Concat<"[](::mlir::Type ty) { return ",
              SubstLeaves<"$_self", "ty",
              AnyTypeOf<types>.predicate>,
              "; }(::llvm::cast<::mlir::aster::pir::TypedRegisterType>($_self).getType())">]>,
        descr # " of " # AnyTypeOf<types>.summary # " values", "::mlir::aster::pir::TypedRegisterType"> {
  let cppType = "::mlir::aster::pir::TypedRegisterType";
}

/// Integer Register Type
def IntRegType : RegTypeOf<[AnySignlessInteger]> {
  let summary = "Integer Register";
}

/// Type predicate for int-like or int-register types.
def IntOrRegType : AnyTypeOf<[AnySignlessInteger, IntRegType]> {
  let summary = "Integer or int-register";
}

/// Any Register Type
def AnyRegType : RegTypeOf<[AnyType]> {
  let summary = "Any Register";
}

#endif // ASTER_DIALECT_PIR_IR_PIRTYPES_TD
