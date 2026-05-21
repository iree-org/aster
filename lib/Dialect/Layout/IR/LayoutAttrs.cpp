//===- LayoutAttrs.cpp - Layout attribute implementation -----------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/Layout/IR/LayoutAttrs.h"
#include "aster/Dialect/Layout/IR/LayoutDialect.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::aster::layout;

//===----------------------------------------------------------------------===//
// LayoutAttr: custom assembly format
//===----------------------------------------------------------------------===//
//
// Syntax:
//   #layout.strided_layout<[4, 8] : [1, 4]>                      -- flat
//   #layout.strided_layout<[(2, 2), (2, 4)] : [(1, 4), (2, 8)]>  -- nested
//
// Grammar for an int-tuple element:
//   element ::= integer          (leaf)
//             | '(' element (',' element)* ')'  (nested tuple)
//

/// Parse a single int-tuple element: either an integer or (elem, elem, ...).
static Attribute parseIntTupleElement(AsmParser &parser) {
  // Try nested tuple: '(' element (',' element)* ')'
  if (parser.parseOptionalLParen().succeeded()) {
    SmallVector<Attribute> elements;
    do {
      Attribute elem = parseIntTupleElement(parser);
      if (!elem)
        return {};
      elements.push_back(elem);
    } while (parser.parseOptionalComma().succeeded());

    if (parser.parseRParen())
      return {};
    return ArrayAttr::get(parser.getContext(), elements);
  }

  // Otherwise, parse an integer leaf.
  int64_t value;
  if (parser.parseInteger(value))
    return {};
  return IntegerAttr::get(IntegerType::get(parser.getContext(), 64), value);
}

/// Parse a top-level int-tuple array: '[' element (',' element)* ']'
static ArrayAttr parseIntTupleArray(AsmParser &parser) {
  SmallVector<Attribute> elements;
  if (parser.parseLSquare())
    return {};
  do {
    Attribute elem = parseIntTupleElement(parser);
    if (!elem)
      return {};
    elements.push_back(elem);
  } while (parser.parseOptionalComma().succeeded());

  if (parser.parseRSquare())
    return {};
  return ArrayAttr::get(parser.getContext(), elements);
}

/// Print a single int-tuple element recursively.
static void printIntTupleElement(AsmPrinter &printer, Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    printer << intAttr.getInt();
    return;
  }

  auto arr = cast<ArrayAttr>(attr);
  printer << "(";
  llvm::interleave(
      arr, printer,
      [&](Attribute elem) { printIntTupleElement(printer, elem); }, ", ");
  printer << ")";
}

/// Print a top-level int-tuple array.
static void printIntTupleArray(AsmPrinter &printer, ArrayAttr arr) {
  printer << "[";
  llvm::interleave(
      arr, printer,
      [&](Attribute elem) { printIntTupleElement(printer, elem); }, ", ");
  printer << "]";
}

Attribute LayoutAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  ArrayAttr shape = parseIntTupleArray(parser);
  if (!shape)
    return {};

  if (parser.parseColon())
    return {};

  ArrayAttr stride = parseIntTupleArray(parser);
  if (!stride)
    return {};

  if (parser.parseGreater())
    return {};

  if (shape.size() != stride.size()) {
    parser.emitError(parser.getCurrentLocation(),
                     "shape and stride must have the same length");
    return {};
  }

  return LayoutAttr::get(parser.getContext(), shape, stride);
}

void LayoutAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printIntTupleArray(printer, getShape());
  printer << " : ";
  printIntTupleArray(printer, getStride());
  printer << ">";
}

//===----------------------------------------------------------------------===//
// LayoutAttr: query methods
//===----------------------------------------------------------------------===//
// An "int tuple" is an ArrayAttr where each element is either:
//   - IntegerAttr (leaf: a single integer)
//   - ArrayAttr   (nested: a sub-tuple)
//
// For flat (non-nested) tuples, we delegate to upstream IndexingUtils
// (computeProduct, linearize, delinearize). The recursive wrappers below
// handle nesting by flattening to leaves and recursing into sub-tuples.

/// Append leaf (shape, stride) pairs from one mode's tree to the output.
static void flattenMode(Attribute shape, Attribute stride,
                        SmallVectorImpl<int64_t> &shapeOut,
                        SmallVectorImpl<int64_t> &strideOut) {
  if (auto si = dyn_cast<IntegerAttr>(shape)) {
    shapeOut.push_back(si.getInt());
    strideOut.push_back(cast<IntegerAttr>(stride).getInt());
    return;
  }
  for (auto [s, d] : llvm::zip(cast<ArrayAttr>(shape), cast<ArrayAttr>(stride)))
    flattenMode(s, d, shapeOut, strideOut);
}

void LayoutAttr::flatten(SmallVectorImpl<int64_t> &shape,
                         SmallVectorImpl<int64_t> &stride) const {
  for (auto [s, d] : llvm::zip(getShape(), getStride()))
    flattenMode(s, d, shape, stride);
}

int64_t LayoutAttr::getSize() const {
  SmallVector<int64_t> shape, stride;
  flatten(shape, stride);
  return computeProduct(shape);
}

int64_t LayoutAttr::getFlatRank() const {
  SmallVector<int64_t> shape, stride;
  flatten(shape, stride);
  return shape.size();
}

// Extent = 1 + sum((s_i - 1) * d_i) over all leaf modes.
int64_t LayoutAttr::getExtent() const {
  SmallVector<int64_t> shape, stride;
  flatten(shape, stride);
  int64_t result = 1;
  for (auto [s, d] : llvm::zip(shape, stride))
    result += (s - 1) * d;
  return result;
}

int64_t LayoutAttr::evaluate(int64_t coord) const {
  SmallVector<int64_t> shape, stride;
  flatten(shape, stride);
  SmallVector<int64_t> basis = computeStrides(shape);
  SmallVector<int64_t> coords = delinearize(coord, basis);
  return linearize(coords, stride);
}

//===----------------------------------------------------------------------===//
// LayoutAttr: convenience builders
//===----------------------------------------------------------------------===//

LayoutAttr LayoutAttr::getFlat(MLIRContext *ctx, ArrayRef<int64_t> shape,
                               ArrayRef<int64_t> stride) {
  assert(shape.size() == stride.size());
  auto i64Ty = IntegerType::get(ctx, 64);
  SmallVector<Attribute> shapeAttrs, strideAttrs;
  for (int64_t s : shape)
    shapeAttrs.push_back(IntegerAttr::get(i64Ty, s));
  for (int64_t d : stride)
    strideAttrs.push_back(IntegerAttr::get(i64Ty, d));
  return LayoutAttr::get(ctx, ArrayAttr::get(ctx, shapeAttrs),
                         ArrayAttr::get(ctx, strideAttrs));
}

//===----------------------------------------------------------------------===//
// TableGen'd attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/Layout/IR/LayoutAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect attribute registration (must be in same TU as storage class def)
//===----------------------------------------------------------------------===//

void LayoutDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aster/Dialect/Layout/IR/LayoutAttrs.cpp.inc"
      >();
}
