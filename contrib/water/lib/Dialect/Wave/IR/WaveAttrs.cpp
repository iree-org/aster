// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#include "water/Dialect/Wave/IR/WaveEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "water/Dialect/Wave/IR/WaveAttrs.cpp.inc"

using namespace mlir;
using namespace mlir::transform;
using namespace wave;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kNullExpr = "NULL";

/// Helper function to parse an affine wave expression with the wave
/// symbol names passed in `symbolNames` and dimension names passed in
/// `dimNames`.
static ParseResult parseExprWithNames(ArrayRef<StringRef> symbolNames,
                                      ArrayRef<StringRef> dimNames,
                                      AffineExpr &outExpr, AsmParser &parser,
                                      bool allowNull = false) {
  MLIRContext *context = parser.getContext();
  if (allowNull && succeeded(parser.parseOptionalLess()) &&
      succeeded(parser.parseKeyword(kNullExpr)) &&
      succeeded(parser.parseGreater())) {
    outExpr = nullptr;
    return success();
  }

  SmallVector<std::pair<StringRef, AffineExpr>> symbolSet;
  symbolSet.reserve(symbolNames.size() + dimNames.size());
  for (auto [i, nm] : llvm::enumerate(symbolNames))
    symbolSet.emplace_back(nm, getAffineSymbolExpr(i, context));
  for (auto [i, nm] : llvm::enumerate(dimNames))
    symbolSet.emplace_back(nm, getAffineDimExpr(i, context));
  return parser.parseAffineExpr(symbolSet, outExpr);
};

// Helper function printing an affine expression with custom symbol and
// dimension names. It textually substitutes 's<i>' occurrences with the
// corresponding names from the provided `symbolNames` array and 'd<i>'
// occurrences with the corresponding names from the provided `dimNames` array.
static void printWithNames(llvm::raw_ostream &os, AffineExpr expr,
                           ArrayRef<StringRef> symbolNames,
                           ArrayRef<StringRef> dimNames = {}) {
  if (!expr) {
    os << "<" << kNullExpr.str() << ">";
    return;
  }
  std::string exprStr;
  llvm::raw_string_ostream exprOs(exprStr);
  expr.print(exprOs);
  exprOs.flush();

  size_t segmentStart = 0;
  for (size_t i = 0, e = exprStr.size(); i < e; ++i) {
    if (exprStr[i] == 's' || exprStr[i] == 'd') {
      size_t end = i + 1;
      for (; end < e && std::isdigit(exprStr[end]); ++end)
        ;
      int position;
      // Replace only when 'pattern' is a complete token (not embedded
      // inside a longer identifier or number). We approximate token
      // boundaries by checking that adjacent characters are non-alphanumeric.
      bool isWhole = (i == 0 || !std::isalnum(exprStr[i - 1])) &&
                     (end == e || !std::isalnum(exprStr[end])) &&
                     llvm::to_integer(StringRef(exprStr).slice(i + 1, end),
                                      position, /*Base=*/10);
      if (!isWhole)
        continue;

      // When matched, first push the string from the previous segment.
      os << StringRef(exprStr).slice(segmentStart, i);
      // Then push the replacement.
      if (exprStr[i] == 's') {
        assert(static_cast<size_t>(position) < symbolNames.size() &&
               "symbol index out of bounds");
        os << symbolNames[position];
      } else {
        assert(static_cast<size_t>(position) < dimNames.size() &&
               "dimension index out of bounds");
        os << dimNames[position];
      }
      // Set i to the last symbol included in the pattern so the ++i from the
      // loop moves to the next symbol.
      i = end - 1;
      segmentStart = end;
    }
  }
  os << StringRef(exprStr).slice(segmentStart, std::string::npos);
}

// Populate `names` with the names of symbols to be used in the expression
// syntax. If new names are generated on-the-fly, store them in
// `owningSymbolNames`. It is the caller's responsibility to keep those names
// alive at least as long as it needs `names`, which only stores references.
LogicalResult
getExprSymbolNames(ArrayRef<Attribute> symbols,
                   SmallVectorImpl<StringRef> &names,
                   SmallVectorImpl<llvm::SmallString<16>> &owningSymbolNames,
                   function_ref<void(StringRef)> emitError) {
  names.reserve(names.size() + symbols.size());
  for (Attribute attr : symbols) {
    if (auto sym = dyn_cast<WaveSymbolAttr>(attr)) {
      names.push_back(sym.getName());
    } else if (auto sym = dyn_cast<WaveIndexSymbolAttr>(attr)) {
      names.push_back(stringifyWaveIndexSymbol(sym.getValue()));
    } else {
      emitError("expected symbol names to be one of WaveSymbolAttr or "
                "WaveIndexSymbolAttr");
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// WaveHyperparameterAttr
//===----------------------------------------------------------------------===//

std::optional<int64_t>
WaveHyperparameterAttr::getSymbolValue(StringRef symbolName) const {
  DictionaryAttr mapping = getMapping();
  Attribute attr = mapping.get(symbolName);
  if (!attr)
    return std::nullopt;

  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();

  return std::nullopt;
}

bool WaveHyperparameterAttr::hasSymbol(StringRef symbolName) const {
  return getMapping().get(symbolName) != nullptr;
}

//===----------------------------------------------------------------------===//
// WaveSymbolAttr
//===----------------------------------------------------------------------===//

LogicalResult
WaveSymbolAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       StringRef name) {
  static const llvm::StringSet<> indexSymbols(llvm::from_range,
                                              WaveIndexSymbolAttr::AllCases);
  if (indexSymbols.contains(name)) {
    auto diag = emitError() << "illegal wave symbol name '" << name << "'";
    diag.attachNote() << "Did you mean to use WaveIndexSymbolAttr instead?";
    return diag;
  }
  if (name.starts_with("_"))
    return emitError()
           << "symbols names starting with '_' are reserved for internal use";
  return llvm::success();
}

//===----------------------------------------------------------------------===//
// WaveExprListAttr
//===----------------------------------------------------------------------===//

std::optional<llvm::SmallVector<int64_t>>
wave::WaveExprListAttr::getResolvedShape(
    wave::WaveHyperparameterAttr hyper) const {
  ArrayRef<Attribute> symbolAttrs(getSymbols().data(), getSymbols().size());
  return wave::evaluateMapWithHyperparams(getMap(), symbolAttrs, hyper);
}

llvm::LogicalResult
wave::verifyExprAttrsSameRank(llvm::ArrayRef<wave::WaveExprListAttr> exprs) {
  if (exprs.size() < 2)
    return success();

  unsigned expectedRank = exprs[0].getRank();

  for (size_t i = 1; i < exprs.size(); ++i) {
    if (exprs[i].getRank() != expectedRank)
      return failure();
  }

  return success();
}

llvm::LogicalResult
wave::verifyExprAttrsNoSymbols(llvm::ArrayRef<wave::WaveExprListAttr> exprs) {
  for (const auto &expr : exprs) {
    if (expr.getNumSymbols() != 0)
      return failure();
  }

  return success();
}

Attribute WaveExprListAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess())
    return {};

  SmallVector<Attribute> symbolNameAttrs;
  SmallVector<StringRef> symbolNames;

  // Parse '[' symbol-names ']' allowing empty or non-empty lists.
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
        Attribute attr;
        if (failed(parser.parseAttribute(attr)))
          return llvm::failure();
        symbolNameAttrs.push_back(attr);
        return llvm::success();
      }))
    return {};

  MLIRContext *context = parser.getContext();
  SmallVector<llvm::SmallString<16>> owningSymbolNames;
  if (failed(getExprSymbolNames(symbolNameAttrs, symbolNames, owningSymbolNames,
                                [&](StringRef message) {
                                  parser.emitError(parser.getCurrentLocation(),
                                                   message);
                                }))) {
    return {};
  }

  // Parse an optional block of `(` dim-names `)` allowing empty and non-empty
  // lists.
  SmallVector<StringRef> dimNames;
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseOptionalRParen())) {
      if (parser.parseCommaSeparatedList(
              AsmParser::Delimiter::None,
              [&]() {
                llvm::SMLoc loc = parser.getCurrentLocation();
                ParseResult parseResult =
                    parser.parseKeyword(&dimNames.emplace_back());
                if (failed(parseResult))
                  return parseResult;
                if (dimNames.back().size() >= 2 && dimNames.back()[0] == '_' &&
                    std::isupper(dimNames.back()[1])) {
                  parser.emitError(loc) << "dimension name '" << dimNames.back()
                                        << "' is reserved for internal use";
                  return ParseResult(failure());
                }
                if (llvm::is_contained(llvm::drop_end(dimNames),
                                       dimNames.back())) {
                  parser.emitError(loc) << "dimension name '" << dimNames.back()
                                        << "' is used more than once";
                  return ParseResult(failure());
                }
                if (!llvm::is_contained(symbolNames, dimNames.back()))
                  return parseResult;
                parser.emitError(loc) << "dimension name '" << dimNames.back()
                                      << "' is already used as a symbol name";
                return ParseResult(failure());
              }) ||
          parser.parseRParen())
        return {};
    }
  }

  // Parse: '->' '(' expr (',' expr)* ')'
  if (parser.parseArrow())
    return {};

  SmallVector<AffineExpr> results;
  auto parseOneExpr = [&]() -> ParseResult {
    AffineExpr e;
    if (failed(parseExprWithNames(symbolNames, dimNames, e, parser)))
      return failure();
    results.push_back(e);
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseOneExpr))
    return {};

  if (parser.parseGreater())
    return {};

  // Build a single map with all result expressions.
  auto shape =
      AffineMap::get(dimNames.size(), symbolNames.size(), results, context);

  return get(parser.getContext(), symbolNameAttrs, shape);
}

void WaveExprListAttr::print(AsmPrinter &printer) const {
  // Print symbol names like: [#wave.symbol<"M">, #wave.symbol<"K">] -> ( ... )
  printer << "<[";
  ArrayRef<Attribute> symbols = getSymbols();
  SmallVector<StringRef> names;
  llvm::interleaveComma(symbols, printer,
                        [&](Attribute attr) { printer.printAttribute(attr); });
  printer << "]";

  SmallVector<llvm::SmallString<4>> owningDimNames = llvm::map_to_vector(
      llvm::seq<int64_t>(getMap().getNumDims()), [&](int64_t dim) {
        llvm::SmallString<4> name;
        llvm::raw_svector_ostream os(name);
        os << "d" << dim;
        return name;
      });
  if (getMap().getNumDims() != 0) {
    printer << "(";
    llvm::interleaveComma(owningDimNames, printer, [&](StringRef name) {
      printer.printKeywordOrString(name);
    });
    printer << ")";
  }
  SmallVector<StringRef> dimNames = llvm::map_to_vector(
      owningDimNames,
      [&](const llvm::SmallString<4> &name) { return StringRef(name); });
  printer << " -> (";

  SmallVector<llvm::SmallString<16>> owningSymbolNames;
  if (failed(getExprSymbolNames(
          symbols, names, owningSymbolNames, [&](StringRef message) {
            // TODO: double-check that printer doesn't crash on malformed
            // attributes.
            llvm_unreachable("Unexpected symbol attribute type");
          }))) {
    return;
  }

  llvm::interleaveComma(
      getMap().getResults(), printer.getStream(), [&](AffineExpr expr) {
        printWithNames(printer.getStream(), expr, names, dimNames);
      });
  printer << ")>";
}

LogicalResult
WaveExprListAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         ArrayRef<Attribute> symbols, AffineMap map) {
  if (!llvm::all_of(symbols,
                    llvm::IsaPred<WaveSymbolAttr, WaveIndexSymbolAttr>))
    return emitError() << "expected all symbols to be a WaveSymbolAttr or "
                          "WaveIndexSymbolAttr";

  return success();
}

//-----------------------------------------------------------------------------
// Constraint attributes
//-----------------------------------------------------------------------------

LogicalResult HardwareConstraintAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, unsigned threadsPerWave,
    ArrayRef<unsigned> wavesPerBlock, WaveMmaKindAttr mmaType,
    DictionaryAttr vectorShapes, unsigned maxBitsPerLoad) {

  if (!(wavesPerBlock.empty() || wavesPerBlock.size() == 3))
    return emitError() << "waves_per_block (" << wavesPerBlock
                       << ") should have 3 elements";

  if (vectorShapes) {
    for (NamedAttribute attr : vectorShapes) {
      // TODO: verify that attr.getName() is a valid WaveSymbol
      Attribute value = attr.getValue();

      if (!isa<IntegerAttr>(value))
        return emitError() << attr.getName()
                           << " is not an IntegerAttr: " << attr.getValue();
    }
  }

  return success();
}

LogicalResult
WorkgroupConstraintAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                WaveSymbolAttr dim, WaveExprListAttr tile_size,
                                WaveWorkgroupDimAttr workgroup_dim,
                                bool primary) {
  if (tile_size.getSize() != 1) {
    return emitError() << "invalid ExpressionList size, expected 1";
  }
  if (!llvm::all_of(tile_size.getSymbols(),
                    llvm::IsaPred<wave::WaveSymbolAttr>)) {
    return emitError() << "tile_size must only contain WaveSymbolAttr";
  }
  return success();
}

LogicalResult
WaveConstraintAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           WaveSymbolAttr dim, WaveExprListAttr tile_size) {
  if (tile_size.getSize() != 1) {
    return emitError() << "invalid ExpressionList size, expected 1";
  }
  if (!llvm::all_of(tile_size.getSymbols(),
                    llvm::IsaPred<wave::WaveSymbolAttr>)) {
    return emitError() << "tile_size must only contain WaveSymbolAttr";
  }

  return success();
}

LogicalResult
TilingConstraintAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             WaveSymbolAttr dim, WaveExprListAttr tile_size) {
  if (tile_size.getSize() != 1) {
    return emitError() << "invalid ExpressionList size, expected 1";
  }
  if (!llvm::all_of(tile_size.getSymbols(),
                    llvm::IsaPred<wave::WaveSymbolAttr>)) {
    return emitError() << "tile_size must only contain WaveSymbolAttr";
  }
  return success();
}

LogicalResult
DeviceConstraintAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             WaveSymbolAttr dim, WaveExprListAttr tile_size,
                             unsigned int device_dim) {
  if (tile_size.getSize() != 1) {
    return emitError() << "invalid ExpressionList size, expected 1";
  }
  if (!llvm::all_of(tile_size.getSymbols(),
                    llvm::IsaPred<wave::WaveSymbolAttr>)) {
    return emitError() << "tile_size must only contain WaveSymbolAttr";
  }
  return success();
}

void wave::WaveDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "water/Dialect/Wave/IR/WaveAttrs.cpp.inc"
      >();
}

//-----------------------------------------------------------------------------
// WaveMmaKindAttr
//-----------------------------------------------------------------------------

// MMA spec mapping — we adopt IREE’s 0xABCD encoding.
// AB encodes vendor/arch, C encodes a type-class (e.g. f16/bf16, fp8),
// D is a variant. This is not sufficient to derive M/N/K or exact operand
// types uniformly across families, so we use an explicit switch to return
// exact (M,N,K) and concrete element types. For fp8/packed encodings we
// default accumulator to f32, lhs/rhs to Float8E4M3FNUZ (CDNA3) or Float8E4M3FN
// (CDNA4).
// TODO: Extend WaveMmaKind to avoid conflating fp8/fp6/fp4 variants
// (E4M3FN/E5M2,etc.).

wave::WaveMmaSpec wave::WaveMmaKindAttr::getSpec(MLIRContext *ctx,
                                                 wave::WaveMmaKind kind) {
  auto f16 = [&]() -> Type { return Float16Type::get(ctx); };
  auto bf16 = [&]() -> Type { return BFloat16Type::get(ctx); };
  auto f32 = [&]() -> Type { return Float32Type::get(ctx); };
  auto i8 = [&]() -> Type { return IntegerType::get(ctx, 8); };
  auto i32 = [&]() -> Type { return IntegerType::get(ctx, 32); };
  auto f8E4M3FNUZ = [&]() -> Type { return Float8E4M3FNUZType::get(ctx); };
  auto f8E4M3FN = [&]() -> Type { return Float8E4M3FNType::get(ctx); };

  switch (kind) {
  // CDNA1
  case wave::WaveMmaKind::F32_16x16x16_F16:
    return {16, 16, 16, f16(), f16(), f32()};
  case wave::WaveMmaKind::F32_32x32x8_F16:
    return {32, 32, 8, f16(), f16(), f32()};
  case wave::WaveMmaKind::F32_16x16x32_K8_F16:
    return {16, 16, 32, f16(), f16(), f32()};
  case wave::WaveMmaKind::F32_32x32x16_K8_F16:
    return {32, 32, 16, f16(), f16(), f32()};
  case wave::WaveMmaKind::I32_16x16x16_I8:
    return {16, 16, 16, i8(), i8(), i32()};
  case wave::WaveMmaKind::I32_32x32x8_I8:
    return {32, 32, 8, i8(), i8(), i32()};

  // CDNA3
  case wave::WaveMmaKind::F32_16x16x32_F8:
    return {16, 16, 32, f8E4M3FNUZ(), f8E4M3FNUZ(), f32()};
  case wave::WaveMmaKind::F32_32x32x16_F8:
    return {32, 32, 16, f8E4M3FNUZ(), f8E4M3FNUZ(), f32()};
  case wave::WaveMmaKind::F32_16x16x32_K4_F8:
    return {16, 16, 32, f8E4M3FNUZ(), f8E4M3FNUZ(), f32()};
  case wave::WaveMmaKind::F32_32x32x16_K4_F8:
    return {32, 32, 16, f8E4M3FNUZ(), f8E4M3FNUZ(), f32()};
  case wave::WaveMmaKind::I32_16x16x32_I8:
    return {16, 16, 32, i8(), i8(), i32()};
  case wave::WaveMmaKind::I32_32x32x16_I8:
    return {32, 32, 16, i8(), i8(), i32()};

  // CDNA4
  case wave::WaveMmaKind::F32_16x16x128_F8F6F4:
    return {16, 16, 128, f8E4M3FN(), f8E4M3FN(), f32()};
  case wave::WaveMmaKind::F32_32x32x64_F8F6F4:
    return {32, 32, 64, f8E4M3FN(), f8E4M3FN(), f32()};
  case wave::WaveMmaKind::F32_16x16x32_F16:
    return {16, 16, 32, f16(), f16(), f32()};
  case wave::WaveMmaKind::F32_32x32x16_F16:
    return {32, 32, 16, f16(), f16(), f32()};
  case wave::WaveMmaKind::F32_16x16x32_BF16:
    return {16, 16, 32, bf16(), bf16(), f32()};
  case wave::WaveMmaKind::F32_32x32x16_BF16:
    return {32, 32, 16, bf16(), bf16(), f32()};
  }

  llvm_unreachable("unhandled WaveMmaKind in WaveMmaKindAttr::getSpec");
}

std::tuple<uint32_t, uint32_t, uint32_t>
wave::WaveMmaKindAttr::getShape(MLIRContext *ctx, wave::WaveMmaKind kind) {
  wave::WaveMmaSpec spec = wave::WaveMmaKindAttr::getSpec(ctx, kind);
  return {static_cast<uint32_t>(spec.m), static_cast<uint32_t>(spec.n),
          static_cast<uint32_t>(spec.k)};
}

// Verify that wave tensor types in the given range are fully specified. Emit a
// diagnostic with the given message at the location provided, if present,
// otherwise just return failure.
static llvm::LogicalResult
verifyTypesFullySpecified(std::optional<Location> loc, TypeRange types,
                          llvm::StringRef message) {
  for (Type type : types) {
    auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(type);
    if (!tensorType || tensorType.getFullySpecified())
      continue;

    if (loc)
      emitError(*loc) << message;
    return llvm::failure();
  }
  return llvm::success();
}

// Verify that wave tensor types in the given range have memory-related address
// spaces, i.e., no unspecified and no register space.
static llvm::LogicalResult
verifyMemoryOnlyAddressSpaces(std::optional<Location> loc, TypeRange types,
                              llvm::StringRef message) {
  for (Type type : types) {
    auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(type);
    if (!tensorType || llvm::is_contained({wave::WaveAddressSpace::Global,
                                           wave::WaveAddressSpace::Shared},
                                          tensorType.getAddressSpaceValue()))
      continue;

    if (loc)
      emitError(*loc) << message;
    return llvm::failure();
  }
  return llvm::success();
}

// Type of a callback visiting and verifying types of operation operands,
// results and block argument lists. The callback is expected to emit a
// diagnostic with the given message if the location is provided, and return
// failure in any case if verification fails.
using OpTypeRangeVisitor = llvm::function_ref<llvm::LogicalResult(
    std::optional<Location>, TypeRange, llvm::StringRef)>;

// Call `visitor` to verify type ranges of operation operands, results and block
// argument lists in immediately nested regions.
static llvm::LogicalResult visitOpRelatedTypes(Operation *op,
                                               OpTypeRangeVisitor visitor,
                                               llvm::StringRef message,
                                               bool emitDiagnostics) {
  std::optional<Location> optionalLoc =
      emitDiagnostics ? std::optional(op->getLoc()) : std::nullopt;
  if (llvm::failed(visitor(optionalLoc, op->getOperandTypes(), message)))
    return llvm::failure();
  if (llvm::failed(visitor(optionalLoc, op->getResultTypes(), message)))
    return llvm::failure();
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      if (llvm::failed(visitor(optionalLoc, block.getArgumentTypes(), message)))
        return llvm::failure();
    }
  }
  return llvm::success();
}

//-----------------------------------------------------------------------------
// Normal Form attributes
//-----------------------------------------------------------------------------

// Check the wave normal form constraints on a single nested operation. Returns
// success or a definite failure with an error diagnostic emitted at the
// operation's location.
static DiagnosedSilenceableFailure
checkWaveNormalFormOnOp(WaveWaterNormalForm form, Operation *op) {
  auto emitError = [&]() { return op->emitError(); };

  if (auto func = llvm::dyn_cast<FunctionOpInterface>(op)) {
    if (wave::bitEnumContainsAll(
            form, wave::WaveWaterNormalForm::FunctionBoundarySpecified)) {
      constexpr llvm::StringLiteral kMessage =
          "normal form requires tensor types to be fully specified at "
          "function boundaries";
      if (llvm::failed(verifyTypesFullySpecified(
              /*loc*/ std::nullopt, func.getArgumentTypes(), kMessage))) {
        emitError() << kMessage;
        return DiagnosedSilenceableFailure::definiteFailure();
      }

      if (llvm::failed(verifyTypesFullySpecified(
              /*loc*/ std::nullopt, func->getResultTypes(), kMessage))) {
        emitError() << kMessage;
        return DiagnosedSilenceableFailure::definiteFailure();
      }
    }
  }

  if (wave::bitEnumContainsAll(form,
                               wave::WaveWaterNormalForm::OpTypesSpecified)) {
    constexpr llvm::StringLiteral kMessage =
        "normal form requires tensor types to be fully specified";
    if (llvm::failed(visitOpRelatedTypes(op, verifyTypesFullySpecified,
                                         kMessage,
                                         /*emitDiagnostics*/ false))) {
      emitError() << kMessage;
      return DiagnosedSilenceableFailure::definiteFailure();
    }
  }

  if (wave::bitEnumContainsAll(form,
                               wave::WaveWaterNormalForm::MemoryOnlyTypes)) {
    constexpr llvm::StringLiteral kMessage =
        "normal form requires tensor types to have only memory address spaces "
        "(elements per thread propagation missing?)";
    if (llvm::failed(visitOpRelatedTypes(op, verifyMemoryOnlyAddressSpaces,
                                         kMessage,
                                         /*emitDiagnostics*/ false))) {
      emitError() << kMessage;
      return DiagnosedSilenceableFailure::definiteFailure();
    }
  }

  if (wave::bitEnumContainsAll(
          form, wave::WaveWaterNormalForm::ResolvedAllocations)) {
    if (auto allocOp = llvm::dyn_cast<wave::AllocateOp>(op)) {
      if (!llvm::isa<MemRefType>(allocOp.getResult().getType())) {
        emitError() << "normal form requires all wave.allocate "
                       "operations to have memref result type";
        return DiagnosedSilenceableFailure::definiteFailure();
      }
    }
  }

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
WaveWaterNormalFormAttr::checkOperation(Operation *op) const {
  WaveWaterNormalForm form = getValue();

  // No normal form required.
  if (form == wave::WaveWaterNormalForm::None)
    return DiagnosedSilenceableFailure::success();

  // Walk all operations nested under `op`, including `op` itself, and check
  // each one against the wave normal form constraints. This mirrors the
  // walk done by the former water_normalform.module verifier.
  WalkResult walkResult = op->walk<WalkOrder::PreOrder>([&](Operation *nested) {
    DiagnosedSilenceableFailure result = checkWaveNormalFormOnOp(form, nested);
    if (result.isDefiniteFailure())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return DiagnosedSilenceableFailure::definiteFailure();

  return DiagnosedSilenceableFailure::success();
}
