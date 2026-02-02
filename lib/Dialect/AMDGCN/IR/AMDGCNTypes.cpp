//===- AMDGCNTypes.cpp - AMDGCN types -------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Interfaces/ResourceInterfaces.h"
#include <optional>

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// RegisterRangeType verification helper
//===----------------------------------------------------------------------===//

namespace {
LogicalResult verifyRegisterRange(function_ref<InFlightDiagnostic()> emitError,
                                  RegisterRange range, StringRef registerKind) {
  if (range.size() <= 0)
    return emitError() << registerKind << " range size must be positive";

  if (!range.begin().isValid())
    return emitError() << "begin " << registerKind << " is invalid";

  // Check that alignment is a power of 2
  int16_t alignment = range.alignment();
  if ((alignment & (alignment - 1)) != 0)
    return emitError() << "align must be a power of 2, got " << alignment;

  // Check alignment if the range is allocated
  if (!range.begin().isRelocatable()) {
    if (alignment <= 0)
      return emitError() << "align must be positive, got " << alignment;

    int16_t begin = range.begin().getRegister();
    if (begin % alignment != 0) {
      return emitError() << "index begin (" << begin << ") must be aligned to "
                         << "align (" << alignment << ")";
    }
  }

  return success();
}
} // namespace

bool mlir::aster::amdgcn::compareLessAMDGCNRegisterTypes(
    AMDGCNRegisterTypeInterface lhs, AMDGCNRegisterTypeInterface rhs) {
  if (lhs.getRegisterKind() != rhs.getRegisterKind())
    return lhs.getRegisterKind() < rhs.getRegisterKind();
  return lhs.getAsRange() < rhs.getAsRange();
}

bool amdgcn::hasSize(Type type, ArrayRef<int32_t> size) {
  auto rangeType = dyn_cast<AMDGCNRegisterTypeInterface>(type);
  if (!rangeType || !rangeType.isRegisterRange())
    return false;

  RegisterRange range = rangeType.getAsRange();
  return llvm::any_of(size, [&](int32_t s) { return range.size() == s; });
}

//===----------------------------------------------------------------------===//
// AGPR types
//===----------------------------------------------------------------------===//

LogicalResult AGPRType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Register reg) {
  if (!reg.isValid())
    return emitError() << "AGPR must be non-negative";
  return success();
}

LogicalResult
AGPRRangeType::verify(function_ref<InFlightDiagnostic()> emitError,
                      RegisterRange range) {
  return verifyRegisterRange(emitError, range, "AGPR");
}

Resource *AGPRType::getResource() const { return AGPRResource::get(); }
Resource *AGPRRangeType::getResource() const { return AGPRResource::get(); }

//===----------------------------------------------------------------------===//
// SGPR types
//===----------------------------------------------------------------------===//

LogicalResult SGPRType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Register reg) {
  if (!reg.isValid())
    return emitError() << "SGPR must be non-negative";
  return success();
}

LogicalResult
SGPRRangeType::verify(function_ref<InFlightDiagnostic()> emitError,
                      RegisterRange range) {
  return verifyRegisterRange(emitError, range, "SGPR");
}

Resource *SGPRType::getResource() const { return SGPRResource::get(); }
Resource *SGPRRangeType::getResource() const { return SGPRResource::get(); }

//===----------------------------------------------------------------------===//
// VGPR types
//===----------------------------------------------------------------------===//

LogicalResult VGPRType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Register reg) {
  if (!reg.isValid())
    return emitError() << "VGPR must be non-negative";
  return success();
}

LogicalResult
VGPRRangeType::verify(function_ref<InFlightDiagnostic()> emitError,
                      RegisterRange range) {
  return verifyRegisterRange(emitError, range, "VGPR");
}

Resource *VGPRType::getResource() const { return VGPRResource::get(); }
Resource *VGPRRangeType::getResource() const { return VGPRResource::get(); }

//===----------------------------------------------------------------------===//
// SREG types
//===----------------------------------------------------------------------===//

LogicalResult SREGType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Register reg, SregKind kind) {
  if (!reg.isValid())
    return emitError() << "SREG must be non-negative";
  switch (kind) {
  case SregKind::Scc: {
    if (!reg.isRelocatable() && reg.getRegister() != 0) {
      return emitError() << "SCC SREG must be register 0";
    }
    break;
  }
  }
  return success();
}

Resource *SREGType::getResource() const { return SGPRResource::get(); }
