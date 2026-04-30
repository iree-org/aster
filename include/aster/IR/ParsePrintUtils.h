//===- ParsePrintUtils.h - Parse/Print Utilities ----------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares template utilities for parsing and printing MLIR types
// and attributes using type-switch dispatch over a variadic pack of type or
// attribute types.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_PARSEPRINTUTILS_H
#define ASTER_IR_PARSEPRINTUTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include <cstdint>
#include <type_traits>

namespace mlir::aster {
namespace detail {
/// Check if a class has a parse method.
template <typename ConcreteTy, typename = void>
struct HasParseMethod : std::false_type {};
template <typename ConcreteTy>
struct HasParseMethod<ConcreteTy,
                      std::void_t<decltype(ConcreteTy::parse(
                          std::declval<AsmParser &>(), std::declval<Type>()))>>
    : std::true_type {};

/// Check if a class has a print method.
template <typename ConcreteTy, typename = void>
struct HasPrintMethod : std::false_type {};
template <typename ConcreteTy>
struct HasPrintMethod<ConcreteTy,
                      std::void_t<decltype(std::declval<ConcreteTy>().print(
                          std::declval<AsmPrinter &>()))>> : std::true_type {};

//===----------------------------------------------------------------------===//
// printAttributesImpl
//===----------------------------------------------------------------------===//

/// Print an attribute that must be one of the specified attribute types.
template <typename... AttrTys>
LogicalResult printAttributesImpl(Attribute def, AsmPrinter &printer) {
  return llvm::TypeSwitch<Attribute, LogicalResult>(def)
      .template Case<AttrTys...>([&](auto attr) {
        // Print the mnemonic of the attribute.
        using AttrTy = std::remove_reference_t<decltype(attr)>;
        printer << AttrTy::getMnemonic();

        // Print the attribute if it has a print method.
        if constexpr (detail::HasPrintMethod<AttrTy>::value)
          attr.print(printer);
        return success();
      })
      .Default([&](Attribute) {
        // Unexpected attribute type.
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// parseAttributesImpl
//===----------------------------------------------------------------------===//

/// Parse case for an attribute.
template <typename AttrTy, typename... AttrTys>
AsmParser::KeywordSwitch<OptionalParseResult> &
parseAttrCaseImpl(AsmParser::KeywordSwitch<OptionalParseResult> &kwSwitch,
                  AsmParser &parser, StringRef *mnemonic, Type type,
                  Attribute &value) {
  // Add the case for the current attribute type.
  AsmParser::KeywordSwitch<OptionalParseResult> &kwSwitchNext =
      kwSwitch.Case(AttrTy::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
        // Check if the attribute has a parse method, if so, use it to parse the
        // attribute.
        if constexpr (detail::HasParseMethod<AttrTy>::value) {
          value = AttrTy::parse(parser, type);
        } else {
          (void)type;
          value = AttrTy::get(parser.getContext());
        }
        return success(!!value);
      });

  // If there are more attribute types, parse the next one.
  if constexpr (sizeof...(AttrTys) > 0)
    return parseAttrCaseImpl<AttrTys...>(kwSwitchNext, parser, mnemonic, type,
                                         value);

  // Return the keyword switch with the next attribute type.
  return kwSwitchNext;
}

/// Parse an attribute that must be one of the specified attribute types.
template <typename... AttrTys>
OptionalParseResult parseAttributesImpl(AsmParser &parser,
                                        llvm::StringRef *mnemonic, Type type,
                                        Attribute &value) {
  AsmParser::KeywordSwitch<OptionalParseResult> kwSwitch(parser);
  if constexpr (sizeof...(AttrTys) > 0) {
    return parseAttrCaseImpl<AttrTys...>(kwSwitch, parser, mnemonic, type,
                                         value)
        .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
          *mnemonic = keyword;
          return std::nullopt;
        });
  }

  // If there are no attributes, return the default keyword switch.
  return kwSwitch.Default([&](llvm::StringRef keyword, llvm::SMLoc) {
    *mnemonic = keyword;
    return std::nullopt;
  });
}
} // namespace detail

/// Parse a dialect attribute that must be one of the specified attribute types.
template <typename... AttrTys>
Attribute parseDialectAttributes(DialectAsmParser &parser, Type type,
                                 StringRef dialectNamespace) {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef attrTag;
  Attribute attr;
  OptionalParseResult parseResult =
      detail::parseAttributesImpl<AttrTys...>(parser, &attrTag, type, attr);
  if (parseResult.has_value())
    return attr;
  parser.emitError(typeLoc) << "unknown attribute `" << attrTag
                            << "` in dialect `" << dialectNamespace << "`";
  return {};
}

/// Print a dialect attribute that must be one of the specified attribute types.
template <typename... AttrTys>
void printDialectAttributes(Attribute attr, DialectAsmPrinter &printer) {
  if (succeeded(detail::printAttributesImpl<AttrTys...>(attr, printer)))
    return;
}

//===----------------------------------------------------------------------===//
// Instruction operand parse/print helpers
//===----------------------------------------------------------------------===//

/// Describes the kind of an SSA operand in an instruction segment.
enum class ODSOperandKind : int8_t {
  Plain = 0,
  Optional,
  Variadic,
};

/// Parses operands for one instruction segment.
/// Format: <prefix> `(` comma-separated-args `)`
///   - Plain: operand
///   - Optional: `name` `=` operand
///   - Variadic: `name` `=` `[` comma-separated-operands `]`.
///
/// All of argNames, argKinds, and segmentSizes must have the same
/// length (one entry per ODS operand in this segment).
LogicalResult
parseInstOperands(OpAsmParser &parser, StringRef prefix,
                  SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                  ArrayRef<StringRef> argNames,
                  ArrayRef<ODSOperandKind> argKinds,
                  MutableArrayRef<int32_t> segmentSizes);

/// Prints operands following the format of parseInstOperands.
void printInstOperands(OpAsmPrinter &printer, StringRef prefix,
                       OperandRange operands, ArrayRef<StringRef> argNames,
                       ArrayRef<ODSOperandKind> argKinds,
                       ArrayRef<int32_t> segmentSizes);

/// Parses types for one instruction segment, following the format of
/// parseInstOperands.
LogicalResult parseInstOperandTypes(OpAsmParser &parser, StringRef prefix,
                                    SmallVectorImpl<Type> &types,
                                    ArrayRef<StringRef> argNames,
                                    ArrayRef<ODSOperandKind> argKinds,
                                    ArrayRef<int32_t> segmentSizes);

/// Prints types for one instruction segment, following the format of
/// parseInstOperandTypes.
void printInstOperandTypes(OpAsmPrinter &printer, StringRef prefix,
                           TypeRange types, ArrayRef<StringRef> argNames,
                           ArrayRef<ODSOperandKind> argKinds,
                           ArrayRef<int32_t> segmentSizes);

} // namespace mlir::aster

#endif // ASTER_IR_PARSEPRINTUTILS_H
