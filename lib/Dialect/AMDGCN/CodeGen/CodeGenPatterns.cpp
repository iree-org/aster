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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

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

//===----------------------------------------------------------------------===//
// PtrLoadOpPattern
//===----------------------------------------------------------------------===//
struct PtrLoadOpPattern : public OpCodeGenPattern<ptr::LoadOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(ptr::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// PtrStoreOpPattern
//===----------------------------------------------------------------------===//
struct PtrStoreOpPattern : public OpCodeGenPattern<ptr::StoreOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(ptr::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// PtrAddOpPattern
//===----------------------------------------------------------------------===//
struct PtrAddOpPattern : public OpCodeGenPattern<aster_utils::PtrAddOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(aster_utils::PtrAddOp op, OpAdaptor adaptor,
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
// Internal helpers
//===----------------------------------------------------------------------===//

/// Create an i32 constant value.
static Value getI32Constant(OpBuilder &builder, Location loc, int32_t value) {
  return arith::ConstantOp::create(
      builder, loc, builder.getI32Type(),
      builder.getIntegerAttr(builder.getI32Type(), value));
}

/// Get the address space kind from a ptr type. Returns std::nullopt for
/// generic/unknown memory spaces (which default to global).
static std::optional<AddressSpaceKind>
getAddressSpaceKind(ptr::PtrType ptrType) {
  auto memSpace = ptrType.getMemorySpace();
  if (!memSpace)
    return std::nullopt;
  if (auto addrSpace = dyn_cast<AddressSpaceAttr>(memSpace))
    return addrSpace.getSpace();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// PtrLoadOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
PtrLoadOpPattern::matchAndRewrite(ptr::LoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();

  // Get the result type and compute the number of 32-bit words needed.
  Type resultType = converter.convertType(op.getResult());
  int64_t typeSize = converter.getTypeSize(op.getResult().getType());
  int64_t numWords = (typeSize + 3) / 4;

  // Create the destination register allocation.
  Value dst = createAlloca(rewriter, loc, resultType);

  // Get the converted address operand.
  Value addr = adaptor.getPtr();

  // Get the memory space from the ptr type.
  auto ptrType = cast<ptr::PtrType>(op.getPtr().getType());
  std::optional<AddressSpaceKind> space = getAddressSpaceKind(ptrType);

  Value result;
  // Handle local memory (shared/LDS) with DS_READ instructions.
  if (space && *space == AddressSpaceKind::Local) {
    Value offset = getI32Constant(rewriter, loc, 0);
    switch (numWords) {
    case 1:
      result =
          DS_READ_B32::create(rewriter, loc, dst, addr, offset).getResult();
      break;
    case 2:
      result =
          DS_READ_B64::create(rewriter, loc, dst, addr, offset).getResult();
      break;
    case 3:
      result =
          DS_READ_B96::create(rewriter, loc, dst, addr, offset).getResult();
      break;
    case 4:
      result =
          DS_READ_B128::create(rewriter, loc, dst, addr, offset).getResult();
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for ptr.load from local memory");
    }
  } else {
    // Default to global memory with GLOBAL_LOAD instructions.
    switch (numWords) {
    case 1:
      result =
          GLOBAL_LOAD_DWORD::create(rewriter, loc, dst, addr, nullptr, nullptr)
              .getResult();
      break;
    case 2:
      result = GLOBAL_LOAD_DWORDX2::create(rewriter, loc, dst, addr, nullptr,
                                           nullptr)
                   .getResult();
      break;
    case 3:
      result = GLOBAL_LOAD_DWORDX3::create(rewriter, loc, dst, addr, nullptr,
                                           nullptr)
                   .getResult();
      break;
    case 4:
      result = GLOBAL_LOAD_DWORDX4::create(rewriter, loc, dst, addr, nullptr,
                                           nullptr)
                   .getResult();
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for ptr.load");
    }
  }
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// PtrStoreOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
PtrStoreOpPattern::matchAndRewrite(ptr::StoreOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();

  // Get the value to store and compute the number of 32-bit words.
  Value data = adaptor.getValue();
  int64_t typeSize = converter.getTypeSize(op.getValue().getType());
  int64_t numWords = (typeSize + 3) / 4;

  // Get the converted address operand.
  Value addr = adaptor.getPtr();

  // Get the memory space from the ptr type.
  auto ptrType = cast<ptr::PtrType>(op.getPtr().getType());
  std::optional<AddressSpaceKind> space = getAddressSpaceKind(ptrType);

  // Handle local memory (shared/LDS) with DS_WRITE instructions.
  if (space && *space == AddressSpaceKind::Local) {
    Value offset = getI32Constant(rewriter, loc, 0);
    switch (numWords) {
    case 1:
      DS_WRITE_B32::create(rewriter, loc, data, addr, offset);
      break;
    case 2:
      DS_WRITE_B64::create(rewriter, loc, data, addr, offset);
      break;
    case 3:
      DS_WRITE_B96::create(rewriter, loc, data, addr, offset);
      break;
    case 4:
      DS_WRITE_B128::create(rewriter, loc, data, addr, offset);
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for ptr.store to local memory");
    }
  } else {
    // Default to global memory with GLOBAL_STORE instructions.
    switch (numWords) {
    case 1:
      GLOBAL_STORE_DWORD::create(rewriter, loc, data, addr, nullptr, nullptr);
      break;
    case 2:
      GLOBAL_STORE_DWORDX2::create(rewriter, loc, data, addr, nullptr, nullptr);
      break;
    case 3:
      GLOBAL_STORE_DWORDX3::create(rewriter, loc, data, addr, nullptr, nullptr);
      break;
    case 4:
      GLOBAL_STORE_DWORDX4::create(rewriter, loc, data, addr, nullptr, nullptr);
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for ptr.store");
    }
  }
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// PtrAddOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
PtrAddOpPattern::matchAndRewrite(aster_utils::PtrAddOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  // Get the converted operands.
  Value ptr = adaptor.getPtr();
  Value offset = adaptor.getOffset();
  Value uniformOffset = adaptor.getUniformOffset();
  int64_t constOffset = op.getConstOffset();

  // Create the AMDGCN ptr_add operation.
  rewriter.replaceOpWithNewOp<amdgcn::PtrAddOp>(op, ptr.getType(), ptr, offset,
                                                uniformOffset, constOffset);
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
                      aster_utils::AssumeRangeOp, aster_utils::PtrAddOp,
                      lsir::FromRegOp, lsir::ToRegOp, lsir::RegConstraintOp,
                      ptr::LoadOp, ptr::StoreOp>();

  // Add the patterns.
  patterns.add<IDDimOpPattern<aster_utils::ThreadIdOp, amdgcn::ThreadIdOp>,
               IDDimOpPattern<aster_utils::BlockIdOp, amdgcn::BlockIdOp>,
               IDDimOpPattern<aster_utils::BlockDimOp, amdgcn::BlockDimOp>,
               IDDimOpPattern<aster_utils::GridDimOp, amdgcn::GridDimOp>,
               PtrLoadOpPattern, PtrStoreOpPattern, PtrAddOpPattern>(converter);
}
