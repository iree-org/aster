//===- ParsePrintUtils.cpp - Parse/Print Utilities -----------------C++ -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements shared helpers for parsing and printing instruction
// operand segments in the MLIR assembly format.
//
//===----------------------------------------------------------------------===//

#include "aster/IR/ParsePrintUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"

using namespace mlir;
using namespace mlir::aster;

//===----------------------------------------------------------------------===//
// Parsing helpers
//===----------------------------------------------------------------------===//

/// Parses a single plain operand named `name`.
static OptionalParseResult
parsePlainOperand(OpAsmParser &parser, StringRef name,
                  SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                  int32_t &count) {
  OpAsmParser::UnresolvedOperand operand;
  if (parser.parseOperand(operand))
    return failure();
  count = 1;
  operands.push_back(operand);
  return success();
}

/// Parses an optional operand: `name` `=` operand.
static OptionalParseResult
parseOptionalOperand(OpAsmParser &parser, StringRef name,
                     SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                     int32_t &count) {
  if (failed(parser.parseOptionalKeyword(name)))
    return std::nullopt;
  OpAsmParser::UnresolvedOperand operand;
  if (parser.parseEqual() || parser.parseOperand(operand))
    return failure();
  count = 1;
  operands.push_back(operand);
  return success();
}

/// Parses a variadic operand list: `name` `=` `[` operand (`,` operand)* `]`.
static OptionalParseResult
parseVariadicOperand(OpAsmParser &parser, StringRef name,
                     SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                     int32_t &count) {
  if (llvm::failed(parser.parseOptionalKeyword(name)))
    return std::nullopt;
  SmallVector<OpAsmParser::UnresolvedOperand> varOperands;
  if (parser.parseEqual() ||
      parser.parseOperandList(varOperands, OpAsmParser::Delimiter::Square))
    return failure();
  count = static_cast<int32_t>(varOperands.size());
  llvm::append_range(operands, varOperands);
  return success();
}

/// Parse an operand with the given name and kind. Count is populated with the
/// number of operands parsed.
static OptionalParseResult
parseOperand(OpAsmParser &parser,
             SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
             StringRef name, ODSOperandKind kind, int32_t &count) {
  count = 0;
  switch (kind) {
  case ODSOperandKind::Plain:
    return parsePlainOperand(parser, name, operands, count);
  case ODSOperandKind::Optional:
    return parseOptionalOperand(parser, name, operands, count);
  case ODSOperandKind::Variadic:
    return parseVariadicOperand(parser, name, operands, count);
  };
}

LogicalResult mlir::aster::parseInstOperands(
    OpAsmParser &parser, StringRef prefix,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    ArrayRef<StringRef> argNames, ArrayRef<ODSOperandKind> argKinds,
    MutableArrayRef<int32_t> segmentSizes) {
  assert(argNames.size() == argKinds.size() &&
         "names and kinds must have equal size");
  assert(argNames.size() == segmentSizes.size() &&
         "names and segmentSizes must have equal size");
  if (argNames.empty())
    return success();

  // Parse the prefix.
  llvm::SMLoc prefixLoc = parser.getCurrentLocation();
  if (parser.parseKeyword(prefix) || parser.parseLParen())
    return failure();

  // Flag indicating whether an operand requires a comma. This is set the first
  // time an operand is parsed.
  bool requiresComma = false;
  // Flag indicating whether a comma is the last parsed token.
  bool activeComma = false;
  // Flag indicating whether the current operand is optional.
  bool isOpt = false;

  // Helper function to parse a comma.
  auto parseComma = [&]() -> ParseResult {
    if (!requiresComma || activeComma)
      return success();
    if (isOpt) {
      activeComma = succeeded(parser.parseOptionalComma());
      return success();
    }
    if (parser.parseComma())
      return failure();
    activeComma = true;
    return success();
  };

  for (auto &&[name, kind, segmentSize] :
       llvm::zip_equal(argNames, argKinds, segmentSizes)) {
    segmentSize = 0;
    isOpt = kind != ODSOperandKind::Plain;

    // Maybe parse a comma.
    if (parseComma())
      return failure();

    // Skip if the operand is optional and no comma was parsed.
    if (requiresComma && !activeComma && isOpt)
      continue;

    // Parse the operand.
    OptionalParseResult result =
        parseOperand(parser, operands, name, kind, segmentSize);
    if (result.has_value() && failed(result.value()))
      return failure();
    requiresComma |= result.has_value();
    if (activeComma)
      activeComma = !result.has_value();
  }

  // Check if there is an unexpected comma.
  if (activeComma) {
    return parser.emitError(prefixLoc)
           << "found unexpected comma after last operand";
  }
  return parser.parseRParen();
}

//===----------------------------------------------------------------------===//
// Printing helpers
//===----------------------------------------------------------------------===//

/// Prints a single operand.
static void printPlainOperand(OpAsmPrinter &printer, Value operand) {
  printer.printOperand(operand);
}

/// Prints an optional operand with its keyword prefix (if present).
static bool printOptionalOperand(OpAsmPrinter &printer, StringRef name,
                                 OperandRange operands) {
  if (operands.empty())
    return false;
  printer << name << " = ";
  printer.printOperand(operands.front());
  return true;
}

/// Prints a variadic operand list: `name [op0, op1, ...]`.
static void printVariadicOperand(OpAsmPrinter &printer, StringRef name,
                                 OperandRange operands) {
  printer << name << " = [";
  llvm::interleaveComma(operands, printer,
                        [&](Value operand) { printer.printOperand(operand); });
  printer << ']';
}

void mlir::aster::printInstOperands(OpAsmPrinter &printer, StringRef prefix,
                                    OperandRange operands,
                                    ArrayRef<StringRef> argNames,
                                    ArrayRef<ODSOperandKind> argKinds,
                                    ArrayRef<int32_t> segmentSizes) {
  assert(argNames.size() == argKinds.size() &&
         "names and kinds must have equal size");
  assert(argNames.size() == segmentSizes.size() &&
         "names and segment sizes must have equal size");
  if (argNames.empty())
    return;
  printer << ' ' << prefix << '(';

  int64_t offset = 0;
  bool needsComma = false;
  for (auto [name, kind, size] :
       llvm::zip_equal(argNames, argKinds, segmentSizes)) {
    // Skip absent operands.
    if (size == 0)
      continue;

    OperandRange segment = operands.slice(offset, size);
    offset += size;
    if (needsComma)
      printer << ", ";

    // Print the operands.
    switch (kind) {
    case ODSOperandKind::Plain:
      printPlainOperand(printer, segment.front());
      break;
    case ODSOperandKind::Optional:
      printOptionalOperand(printer, name, segment);
      break;
    case ODSOperandKind::Variadic:
      printVariadicOperand(printer, name, segment);
      break;
    }
    needsComma = true;
  }
  printer << ')';
}

//===----------------------------------------------------------------------===//
// Type parsing helpers
//===----------------------------------------------------------------------===//

/// Parses a single type for operand `name`.
static ParseResult parseSingleType(OpAsmParser &parser, StringRef name,
                                   SmallVectorImpl<Type> &types) {
  Type type;
  if (parser.parseType(type))
    return failure();
  types.push_back(type);
  return success();
}

/// Parses a type for an optional operand: `name` `=` type.
static ParseResult parseOptionalType(OpAsmParser &parser, StringRef name,
                                     SmallVectorImpl<Type> &types) {
  if (parser.parseKeyword(name) || parser.parseEqual())
    return failure();
  return parseSingleType(parser, name, types);
}

/// Parses a list of types for a variadic operand: `name` `=` `[` types `]`.
static ParseResult parseVariadicTypes(OpAsmParser &parser, StringRef name,
                                      SmallVectorImpl<Type> &types,
                                      int32_t expectedCount) {
  if (parser.parseKeyword(name) || parser.parseEqual() || parser.parseLSquare())
    return failure();
  for (int32_t i = 0; i < expectedCount; ++i) {
    if (i > 0 && parser.parseComma())
      return failure();
    if (failed(parseSingleType(parser, name, types)))
      return failure();
  }
  return parser.parseRSquare();
}

LogicalResult mlir::aster::parseInstOperandTypes(
    OpAsmParser &parser, StringRef prefix, SmallVectorImpl<Type> &types,
    ArrayRef<StringRef> argNames, ArrayRef<ODSOperandKind> argKinds,
    ArrayRef<int32_t> segmentSizes) {
  assert(argNames.size() == argKinds.size() &&
         "names and kinds must have equal size");
  assert(argNames.size() == segmentSizes.size() &&
         "names and segment sizes must have equal size");
  if (argNames.empty())
    return success();
  if (parser.parseKeyword(prefix) || parser.parseLParen())
    return failure();

  bool needsComma = false;
  for (auto [name, kind, size] :
       llvm::zip_equal(argNames, argKinds, segmentSizes)) {
    // Skip absent operands.
    if (size == 0)
      continue;

    if (needsComma && parser.parseComma())
      return failure();

    switch (kind) {
    case ODSOperandKind::Plain:
      if (failed(parseSingleType(parser, name, types)))
        return failure();
      break;
    case ODSOperandKind::Optional:
      if (failed(parseOptionalType(parser, name, types)))
        return failure();
      break;
    case ODSOperandKind::Variadic:
      if (failed(parseVariadicTypes(parser, name, types, size)))
        return failure();
      break;
    }
    needsComma = true;
  }
  return parser.parseRParen();
}

//===----------------------------------------------------------------------===//
// Type printing helpers
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Modifier attribute parse/print helpers
//===----------------------------------------------------------------------===//

FailureOr<Attribute> mlir::aster::parseInstModAttr(OpAsmParser &parser,
                                                   StringRef name, bool isOpt) {
  if (failed(parser.parseOptionalKeyword(name))) {
    if (isOpt)
      return Attribute(nullptr);
    return parser.emitError(parser.getCurrentLocation())
           << "expected modifier keyword '" << name << "'";
  }
  Attribute attr;
  if (parser.parseLParen() || parser.parseAttribute(attr) ||
      parser.parseRParen())
    return failure();
  return attr;
}

void mlir::aster::printInstModAttr(OpAsmPrinter &printer, StringRef name,
                                   Attribute attr, Attribute defaultValue) {
  if (attr == defaultValue)
    return;
  printer << ' ' << name << '(';
  printer.printAttribute(attr);
  printer << ')';
}

//===----------------------------------------------------------------------===//
// Type printing helpers
//===----------------------------------------------------------------------===//

void mlir::aster::printInstOperandTypes(OpAsmPrinter &printer, StringRef prefix,
                                        TypeRange types,
                                        ArrayRef<StringRef> argNames,
                                        ArrayRef<ODSOperandKind> argKinds,
                                        ArrayRef<int32_t> segmentSizes) {
  assert(argNames.size() == argKinds.size() &&
         "names and kinds must have equal size");
  assert(argNames.size() == segmentSizes.size() &&
         "names and segment sizes must have equal size");
  if (argNames.empty())
    return;
  printer << ' ' << prefix << '(';

  int64_t offset = 0;
  bool needsComma = false;
  for (auto [name, kind, size] :
       llvm::zip_equal(argNames, argKinds, segmentSizes)) {
    // Skip absent operands.
    if (size == 0)
      continue;

    TypeRange segment = types.slice(offset, size);
    offset += size;

    if (needsComma)
      printer << ", ";

    switch (kind) {
    case ODSOperandKind::Plain:
      printer << segment.front();
      break;
    case ODSOperandKind::Optional:
      printer << name << " = " << segment.front();
      break;
    case ODSOperandKind::Variadic:
      printer << name << " = [";
      llvm::interleaveComma(segment, printer);
      printer << ']';
      break;
    }
    needsComma = true;
  }
  printer << ')';
}
