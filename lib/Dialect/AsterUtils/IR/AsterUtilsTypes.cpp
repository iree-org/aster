//===- AsterUtilsTypes.cpp - AsterUtils types -------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

void AsterUtilsDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

std::optional<size_t> StructType::getFieldIndex(StringAttr name) const {
  ArrayRef<StringAttr> names = getFieldNames();
  for (size_t i = 0, e = names.size(); i < e; ++i) {
    if (names[i] == name)
      return i;
  }
  return std::nullopt;
}

std::optional<size_t> StructType::getFieldIndex(StringRef name) const {
  ArrayRef<StringAttr> names = getFieldNames();
  for (size_t i = 0, e = names.size(); i < e; ++i) {
    if (names[i].getValue() == name)
      return i;
  }
  return std::nullopt;
}

Type StructType::getFieldTypeByName(StringAttr name) const {
  if (auto index = getFieldIndex(name))
    return getFieldTypes()[*index];
  return nullptr;
}

Type StructType::getFieldTypeByName(StringRef name) const {
  if (auto index = getFieldIndex(name))
    return getFieldTypes()[*index];
  return nullptr;
}

LogicalResult StructType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<StringAttr> fieldNames,
                                 ArrayRef<Type> fieldTypes) {
  // Check that the number of names matches the number of types.
  if (fieldNames.size() != fieldTypes.size()) {
    return emitError() << "struct type must have the same number of field "
                          "names and types, got "
                       << fieldNames.size() << " names and "
                       << fieldTypes.size() << " types";
  }

  // Check for duplicate field names.
  llvm::StringSet<> seenNames;
  for (StringAttr name : fieldNames) {
    if (!seenNames.insert(name.getValue()).second) {
      return emitError() << "struct type has duplicate field name: '"
                         << name.getValue() << "'";
    }
  }

  return success();
}

/// Parse a struct type.
/// struct-type ::= `struct` `<` (field-name `:` type (`,` field-name `:`
/// type)*)? `>`
Type StructType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<StringAttr> fieldNames;
  SmallVector<Type> fieldTypes;

  // Handle empty struct.
  if (succeeded(parser.parseOptionalGreater()))
    return get(parser.getContext(), fieldNames, fieldTypes);

  // Parse fields.
  do {
    // Parse field name.
    std::string fieldName;
    if (parser.parseKeywordOrString(&fieldName))
      return Type();

    // Parse colon.
    if (parser.parseColon())
      return Type();

    // Parse field type.
    Type fieldType;
    if (parser.parseType(fieldType))
      return Type();

    fieldNames.push_back(StringAttr::get(parser.getContext(), fieldName));
    fieldTypes.push_back(fieldType);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseGreater())
    return Type();

  return get(parser.getContext(), fieldNames, fieldTypes);
}

/// Print a struct type.
void StructType::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleaveComma(
      llvm::zip(getFieldNames(), getFieldTypes()), printer, [&](auto pair) {
        printer << std::get<0>(pair).getValue() << ": " << std::get<1>(pair);
      });
  printer << ">";
}

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.cpp.inc"
