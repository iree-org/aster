//===- Utils.cpp - AMDGCN utils ----------------------------------*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defined AMDGCN utility functions.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/SetOperations.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

Value mlir::aster::amdgcn::createAllocation(OpBuilder &builder, Location loc,
                                            RegisterTypeInterface rTy) {
  if (!rTy.isRegisterRange() || rTy.getAsRange().size() == 1)
    return amdgcn::AllocaOp::create(
        builder, loc, rTy.cloneRegisterType(rTy.getAsRange().begin()));
  SmallVector<Value> results;
  RegisterRange range = rTy.getAsRange();
  results.reserve(range.size());
  for (int16_t i = 0; i < range.size(); ++i) {
    Value alloc = amdgcn::AllocaOp::create(
        builder, loc, rTy.cloneRegisterType(range.begin().getWithOffset(i)));
    results.push_back(alloc);
  }
  return MakeRegisterRangeOp::create(builder, loc, rTy, results);
}

std::optional<RegisterKind>
mlir::aster::amdgcn::getRegKind(RegisterTypeInterface regTy) {
  if (auto rTy = dyn_cast<AMDGCNRegisterTypeInterface>(regTy))
    return rTy.getRegisterKind();
  return std::nullopt;
}

OperandKind mlir::aster::amdgcn::getOperandKind(Type type) {
  if (isa<SGPRType, SGPRRangeType>(type))
    return OperandKind::SGPR;
  if (isa<VGPRType, VGPRRangeType>(type))
    return OperandKind::VGPR;
  if (isa<AGPRType, AGPRRangeType>(type))
    return OperandKind::AGPR;
  if (auto iTy = dyn_cast<IntegerType>(type)) {
    if (iTy.isSignless() && (iTy.getWidth() == 32 || iTy.getWidth() == 64))
      return OperandKind::IntImm;
  }
  if (auto fTy = dyn_cast<FloatType>(type)) {
    if (fTy.getWidth() == 32 || fTy.getWidth() == 64)
      return OperandKind::FPImm;
  }
  return OperandKind::Invalid;
}

RegisterTypeInterface mlir::aster::amdgcn::getRegisterType(MLIRContext *ctx,
                                                           RegisterKind kind,
                                                           int16_t size) {
  using RTy = RegisterTypeInterface;
  switch (kind) {
  case amdgcn::RegisterKind::AGPR:
    return size == 1 ? RTy(amdgcn::AGPRType::get(ctx, Register()))
                     : RTy(amdgcn::AGPRRangeType::get(
                           ctx, RegisterRange(Register(), size)));
  case amdgcn::RegisterKind::SGPR:
    return size == 1 ? RTy(amdgcn::SGPRType::get(ctx, Register()))
                     : RTy(amdgcn::SGPRRangeType::get(
                           ctx, RegisterRange(Register(), size)));
  case amdgcn::RegisterKind::VGPR:
    return size == 1 ? RTy(amdgcn::VGPRType::get(ctx, Register()))
                     : RTy(amdgcn::VGPRRangeType::get(
                           ctx, RegisterRange(Register(), size)));
  default:
    llvm_unreachable("unknown register kind");
  }
}

bool mlir::aster::amdgcn::isAMDReg(Type regTy) {
  return isa<AGPRType, AGPRRangeType, SGPRType, SGPRRangeType, VGPRType,
             VGPRRangeType, SREGType>(regTy);
}

bool mlir::aster::amdgcn::isAMDRegOrImm(Type type) {
  if (auto iTy = dyn_cast<IntegerType>(type))
    return iTy.isSignless() && (iTy.getWidth() == 32 || iTy.getWidth() == 64);
  if (auto fTy = dyn_cast<FloatType>(type))
    return (fTy.getWidth() == 32 || fTy.getWidth() == 64);
  return isAMDReg(type);
}

bool mlir::aster::amdgcn::isAMDRegOf(Type type, RegisterKind kind,
                                     int16_t numWords) {
  if (!isAMDReg(type))
    return false;
  auto rTy = cast<AMDGCNRegisterTypeInterface>(type);
  return rTy.getRegisterKind() == kind &&
         (numWords <= 0 || rTy.getAsRange().size() == numWords);
}

ValueRange mlir::aster::amdgcn::splitRange(OpBuilder &builder, Location loc,
                                           Value value) {
  if (auto rTy = dyn_cast<RegisterTypeInterface>(value.getType());
      !rTy || !rTy.isRegisterRange())
    return ValueRange();
  return SplitRegisterRangeOp::create(builder, loc, value).getResults();
}

//===----------------------------------------------------------------------===//
// ISA/Target utilities
//===----------------------------------------------------------------------===//

ISAVersion mlir::aster::amdgcn::getIsaForTarget(Target target) {
  switch (target) {
  case Target::GFX940:
  case Target::GFX942:
    return ISAVersion::CDNA3;
  case Target::GFX1201:
    return ISAVersion::RDNA4;
  case Target::Invalid:
    return ISAVersion::Invalid;
  }
  llvm_unreachable("unknown target");
}

bool mlir::aster::amdgcn::isOpcodeValidForAllIsas(OpCode opcode,
                                                  ArrayRef<ISAVersion> isas,
                                                  MLIRContext *ctx) {
  ArrayRef<ISAVersion> instISAVersions =
      InstAttr::get(ctx, opcode).getMetadata()->getISAVersions();
  if (instISAVersions.empty()) {
    // Available on all targets.
    return true;
  }
  for (ISAVersion isa : isas) {
    if (!llvm::is_contained(instISAVersions, isa))
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// KernelArgSegmentInfo
//===----------------------------------------------------------------------===//

KernelArgSegmentInfo KernelArgSegmentInfo::get(amdgcn::KernelOp kernel) {
  ArrayRef<KernelArgAttrInterface> args = kernel.getArguments();
  if (args.empty())
    return {};

  KernelArgSegmentInfo info;
  int32_t currentOffset = 0;

  for (KernelArgAttrInterface arg : args) {
    // Align offset to the argument's alignment if specified
    std::optional<uint32_t> alignment = arg.getAlignment();
    if (alignment.has_value() && *alignment > 1) {
      currentOffset = (currentOffset + *alignment - 1) & ~(*alignment - 1);
      info.maxAlignment =
          std::max(info.maxAlignment, static_cast<int32_t>(*alignment));
    }

    // Update offsets and size
    info.offsets.push_back(currentOffset);
    currentOffset += arg.getSize();
  }

  // Final size
  info.size = currentOffset;
  return info;
}
