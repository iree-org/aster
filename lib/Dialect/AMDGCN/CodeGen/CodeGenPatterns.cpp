//===- CodeGenPatterns.cpp ------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AMDGCN CodeGen patterns
//
//===----------------------------------------------------------------------===//

#include "aster/CodeGen/CodeGen.h"
#include "aster/Dialect/AMDGCN/CodeGen/CodeGen.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// IDDimOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename NewOpTy>
struct IDDimOpPattern : public OpCodeGenPattern<OpTy> {
  using OpCodeGenPattern<OpTy>::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// IDDimOpPattern
//===----------------------------------------------------------------------===//

template <typename OpTy, typename NewOpTy>
LogicalResult IDDimOpPattern<OpTy, NewOpTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type type = this->converter.convertType(op);
  Type regTy = std::is_same_v<OpTy, aster_utils::ThreadIdOp>
                   ? Type(amdgcn::VGPRType::get(op.getContext(), Register()))
                   : Type(amdgcn::SGPRType::get(op.getContext(), Register()));
  auto nOp = NewOpTy::create(
      rewriter, op.getLoc(), regTy,
      static_cast<amdgcn::Dim>(static_cast<int8_t>(op.getDim())));
  rewriter.replaceOpWithNewOp<lsir::RegCastOp>(op, type, nOp);
  return success();
}

//===----------------------------------------------------------------------===//
// Internal functions
//===----------------------------------------------------------------------===//

/// Untagle unrealized conversion casts to find the original value.
static Value untagleConvertValue(Value value) {
  if (isa<RegisterTypeInterface>(value.getType()))
    return value;
  auto cOp =
      dyn_cast_if_present<UnrealizedConversionCastOp>(value.getDefiningOp());
  while (cOp && cOp.getNumOperands() == 1) {
    Value value = cOp.getOperand(0);
    if (isa<RegisterTypeInterface>(value.getType()))
      return value;
    cOp =
        dyn_cast_if_present<UnrealizedConversionCastOp>(value.getDefiningOp());
  }
  return value;
}

static Type convertAttrConstraintToType(Attribute constraint,
                                        int64_t numWords) {
  auto kind = dyn_cast<amdgcn::RegisterKindAttr>(constraint);
  if (!kind)
    return nullptr;
  switch (kind.getValue()) {
  case amdgcn::RegisterKind::SGPR:
    if (numWords == 1)
      return amdgcn::SGPRType::get(kind.getContext(), Register());
    return amdgcn::SGPRRangeType::get(kind.getContext(),
                                      RegisterRange(Register(), numWords));
  case amdgcn::RegisterKind::VGPR:
    if (numWords == 1)
      return amdgcn::VGPRType::get(kind.getContext(), Register());
    return amdgcn::VGPRRangeType::get(kind.getContext(),
                                      RegisterRange(Register(), numWords));
  case amdgcn::RegisterKind::AGPR:
    if (numWords == 1)
      return amdgcn::AGPRType::get(kind.getContext(), Register());
    return amdgcn::AGPRRangeType::get(kind.getContext(),
                                      RegisterRange(Register(), numWords));
  default:
    assert(false && "nyi register kind");
  }
  return nullptr;
}

static Type convertTypeImpl(Value value, const CodeGenConverter &converter) {
  if (Operation *defOp = value.getDefiningOp();
      defOp && m_Constant().match(value.getDefiningOp()))
    return value.getType();
  value = untagleConvertValue(value);
  if (isa<RegisterTypeInterface>(value.getType()))
    return value.getType();

  int64_t typeSize = converter.getTypeSize(value.getType());
  int64_t numWords = (typeSize + 3) / 4;

  // If there is a register constraint, use it to determine the type.
  if (Attribute constraint =
          converter.getState().getRegisterConstraint(value)) {
    if (Type t = convertAttrConstraintToType(constraint, numWords))
      return t;
  }

  std::optional<bool> isUniform = converter.isThreadUniform(value);
  assert(isUniform.has_value() &&
         "Type conversion for value without known thread-uniformity");

  if (isUniform.has_value() && *isUniform) {
    if (numWords > 1)
      return amdgcn::SGPRRangeType::get(value.getContext(),
                                        RegisterRange(Register(), numWords));
    return amdgcn::SGPRType::get(value.getContext(), Register());
  }
  if (numWords > 1)
    return amdgcn::VGPRRangeType::get(value.getContext(),
                                      RegisterRange(Register(), numWords));
  return amdgcn::VGPRType::get(value.getContext(), Register());
}

static Type convertTypeImpl(Type type, const CodeGenConverter &converter) {
  if (isa<RegisterTypeInterface>(type))
    return type;
  int64_t typeSize = converter.getTypeSize(type);
  int64_t numWords = (typeSize + 3) / 4;
  if (numWords > 1)
    return amdgcn::VGPRRangeType::get(type.getContext(),
                                      RegisterRange(Register(), numWords));
  return amdgcn::VGPRType::get(type.getContext(), Register());
}

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

void mlir::aster::amdgcn::getDependentCodeGenDialects(
    DialectRegistry &registry) {
  registry.insert<amdgcn::AMDGCNDialect, lsir::LSIRDialect>();
}

void mlir::aster::amdgcn::populateCodeGenPatterns(CodeGenConverter &converter,
                                                  RewritePatternSet &patterns,
                                                  ConversionTarget &target) {

  // Add the type conversions.
  converter.addConversion(
      [&converter](Type type) { return convertTypeImpl(type, converter); });
  converter.addConversion(
      [&converter](Value value) { return convertTypeImpl(value, converter); });

  // Configure the conversion target.
  target.addLegalDialect<amdgcn::AMDGCNDialect>();

  target.addIllegalOp<aster_utils::ThreadIdOp, aster_utils::BlockIdOp,
                      aster_utils::BlockDimOp, aster_utils::GridDimOp>();

  target.addIllegalOp<aster_utils::ThreadIdOp, aster_utils::BlockIdOp,
                      aster_utils::BlockDimOp, aster_utils::GridDimOp,
                      aster_utils::AssumeRangeOp, lsir::FromRegOp,
                      lsir::ToRegOp, lsir::RegConstraintOp>();

  // Add the patterns.
  patterns.add<IDDimOpPattern<aster_utils::ThreadIdOp, amdgcn::ThreadIdOp>,
               IDDimOpPattern<aster_utils::BlockIdOp, amdgcn::BlockIdOp>,
               IDDimOpPattern<aster_utils::BlockDimOp, amdgcn::BlockDimOp>,
               IDDimOpPattern<aster_utils::GridDimOp, amdgcn::GridDimOp>>(
      converter);
}
