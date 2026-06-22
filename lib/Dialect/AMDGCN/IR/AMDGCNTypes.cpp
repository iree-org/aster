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
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// GPRegTrait getCompositeType / getSplitType
//===----------------------------------------------------------------------===//

template <typename T>
static RegisterTypeInterface
getCompositeGPRImpl(T type, TypeRange types, std::optional<int16_t> alignment,
                    llvm::function_ref<InFlightDiagnostic()> emitError) {
  RegisterRange range = type.getAsRange();
  if (range.size() > 1) {
    if (emitError)
      emitError() << "register is already composite";
    return nullptr;
  }

  RegisterSemantics semantics = range.getSemantics();
  RegisterKind kind = type.getRegisterKind();

  // Registers must be contiguous and in ascending order (for allocated
  // semantics). Validate kind, semantics, and size in the same pass.
  int16_t expected = semantics == RegisterSemantics::Allocated
                         ? static_cast<int16_t>(range.begin().getRegister() + 1)
                         : int16_t{-1};

  for (Type t : types) {
    auto other = dyn_cast<AMDGCNRegisterTypeInterface>(t);
    if (!other || other.getRegisterKind() != kind ||
        other.getSemantics() != semantics) {
      if (emitError)
        emitError() << "expected register of the same kind and semantics";
      return nullptr;
    }
    if (other.getAsRange().size() != 1) {
      if (emitError)
        emitError() << "expected a single register, not a range";
      return nullptr;
    }
    if (semantics == RegisterSemantics::Allocated) {
      int16_t reg = other.getAsRange().begin().getRegister();
      if (reg != expected) {
        if (emitError)
          emitError() << "expected register " << expected << " but got " << reg;
        return nullptr;
      }
      ++expected;
    }
  }

  assert(types.size() <
             static_cast<size_t>(std::numeric_limits<int16_t>::max()) &&
         "register count overflows int16_t");
  int16_t size = static_cast<int16_t>(types.size()) + 1;
  int16_t align = alignment.value_or(defaultAlignment(size));
  return T::get(type.getContext(), RegisterRange(range.begin(), size, align));
}

template <typename T>
static LogicalResult
getSplitGPRImpl(T type, SmallVectorImpl<RegisterTypeInterface> &regs,
                llvm::function_ref<InFlightDiagnostic()> emitError) {
  RegisterRange range = type.getAsRange();
  if (range.size() <= 1) {
    if (emitError)
      emitError() << "register is not composite";
    return failure();
  }
  for (int i = 0; i < range.size(); ++i)
    regs.push_back(type.cloneRegisterType(RegisterRange(
        range.begin().getWithOffset(static_cast<int16_t>(i)), 1)));
  return success();
}

//===----------------------------------------------------------------------===//
// SpecialRegTrait getCompositeType / getSplitType
//===----------------------------------------------------------------------===//

static RegisterTypeInterface
getCompositeSRegImpl(llvm::function_ref<InFlightDiagnostic()> emitError) {
  if (emitError)
    emitError() << "special registers cannot form composites";
  return nullptr;
}

static LogicalResult
getSplitSRegImpl(llvm::function_ref<InFlightDiagnostic()> emitError) {
  if (emitError)
    emitError() << "special registers cannot be split";
  return failure();
}

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
  if (range.getSemantics() == RegisterSemantics::Allocated) {
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

//===----------------------------------------------------------------------===//
// SREG type verification
//===----------------------------------------------------------------------===//

static LogicalResult
verifySRegSemantics(function_ref<InFlightDiagnostic()> emitError, Register reg,
                    RegisterProps props, StringRef name) {
  RegisterSemantics sem = reg.getSemantics();
  if (sem == RegisterSemantics::Value &&
      !bitEnumContainsAll(props, RegisterProps::AcceptsValueSemantics))
    return emitError() << name << " does not accept value semantics";
  if (sem == RegisterSemantics::Unallocated &&
      !bitEnumContainsAll(props, RegisterProps::AcceptsUnallocatedSemantics))
    return emitError() << name << " does not accept unallocated semantics";
  return success();
}

#define SREG_VERIFY(TypeName, Name)                                            \
  LogicalResult TypeName::verify(function_ref<InFlightDiagnostic()> emitError, \
                                 Register reg) {                               \
    return verifySRegSemantics(emitError, reg, kRegisterProps, Name);          \
  }

SREG_VERIFY(VCCType, "vcc")
SREG_VERIFY(VCCLoType, "vcc_lo")
SREG_VERIFY(VCCHiType, "vcc_hi")
SREG_VERIFY(VCCZType, "vccz")
SREG_VERIFY(EXECType, "exec")
SREG_VERIFY(EXECLoType, "exec_lo")
SREG_VERIFY(EXECHiType, "exec_hi")
SREG_VERIFY(EXECZType, "execz")
SREG_VERIFY(M0Type, "m0")
SREG_VERIFY(SCCType, "scc")

#undef SREG_VERIFY

//===----------------------------------------------------------------------===//

bool mlir::aster::amdgcn::compareLessAMDGCNRegisterTypes(
    AMDGCNRegisterTypeInterface lhs, AMDGCNRegisterTypeInterface rhs) {
  if (lhs.getRegisterKind() != rhs.getRegisterKind())
    return lhs.getRegisterKind() < rhs.getRegisterKind();
  return lhs.getAsRange() < rhs.getAsRange();
}

bool amdgcn::hasSize(Type type, ArrayRef<int32_t> size) {
  auto rangeType = dyn_cast<AMDGCNRegisterTypeInterface>(type);
  if (!rangeType)
    return false;

  RegisterRange range = rangeType.getAsRange();
  return llvm::any_of(size, [&](int32_t s) { return range.size() == s; });
}

//===----------------------------------------------------------------------===//
// AGPR types
//===----------------------------------------------------------------------===//

LogicalResult AGPRType::verify(function_ref<InFlightDiagnostic()> emitError,
                               RegisterRange range) {
  return verifyRegisterRange(emitError, range, "AGPR");
}

Resource *AGPRType::getResource() const { return AGPRResource::get(); }

//===----------------------------------------------------------------------===//
// SGPR types
//===----------------------------------------------------------------------===//

LogicalResult SGPRType::verify(function_ref<InFlightDiagnostic()> emitError,
                               RegisterRange range) {
  return verifyRegisterRange(emitError, range, "SGPR");
}

Resource *SGPRType::getResource() const { return SGPRResource::get(); }

//===----------------------------------------------------------------------===//
// VGPR types
//===----------------------------------------------------------------------===//

LogicalResult VGPRType::verify(function_ref<InFlightDiagnostic()> emitError,
                               RegisterRange range) {
  return verifyRegisterRange(emitError, range, "VGPR");
}

Resource *VGPRType::getResource() const { return VGPRResource::get(); }

//===----------------------------------------------------------------------===//
// TTMP types
//===----------------------------------------------------------------------===//

LogicalResult TTMPType::verify(function_ref<InFlightDiagnostic()> emitError,
                               RegisterRange range) {
  return verifyRegisterRange(emitError, range, "TTMP");
}

// TTMP is not an allocatable GP resource; report the special-register resource.
Resource *TTMPType::getResource() const { return SREGResource::get(); }

//===----------------------------------------------------------------------===//
// GPRegTrait interface method definitions
//===----------------------------------------------------------------------===//

#define GP_REG_COMPOSITE_SPLIT(TypeName)                                       \
  RegisterTypeInterface TypeName::getCompositeType(                            \
      TypeRange types, std::optional<int16_t> alignment,                       \
      llvm::function_ref<InFlightDiagnostic()> emitError) const {              \
    return getCompositeGPRImpl(*this, types, alignment, emitError);            \
  }                                                                            \
  LogicalResult TypeName::getSplitType(                                        \
      SmallVectorImpl<RegisterTypeInterface> &regs,                            \
      llvm::function_ref<InFlightDiagnostic()> emitError) const {              \
    return getSplitGPRImpl(*this, regs, emitError);                            \
  }

GP_REG_COMPOSITE_SPLIT(AGPRType)
GP_REG_COMPOSITE_SPLIT(SGPRType)
GP_REG_COMPOSITE_SPLIT(VGPRType)
GP_REG_COMPOSITE_SPLIT(TTMPType)

#undef GP_REG_COMPOSITE_SPLIT

//===----------------------------------------------------------------------===//
// SpecialRegTrait interface method definitions
//===----------------------------------------------------------------------===//

#define SPECIAL_REG_COMPOSITE_SPLIT(TypeName)                                  \
  RegisterTypeInterface TypeName::getCompositeType(                            \
      TypeRange types, std::optional<int16_t> alignment,                       \
      llvm::function_ref<InFlightDiagnostic()> emitError) const {              \
    (void)types;                                                               \
    (void)alignment;                                                           \
    return getCompositeSRegImpl(emitError);                                    \
  }                                                                            \
  LogicalResult TypeName::getSplitType(                                        \
      SmallVectorImpl<RegisterTypeInterface> &regs,                            \
      llvm::function_ref<InFlightDiagnostic()> emitError) const {              \
    (void)regs;                                                                \
    return getSplitSRegImpl(emitError);                                        \
  }

SPECIAL_REG_COMPOSITE_SPLIT(VCCHiType)
SPECIAL_REG_COMPOSITE_SPLIT(VCCZType)
SPECIAL_REG_COMPOSITE_SPLIT(EXECHiType)
SPECIAL_REG_COMPOSITE_SPLIT(EXECZType)
SPECIAL_REG_COMPOSITE_SPLIT(M0Type)
SPECIAL_REG_COMPOSITE_SPLIT(SCCType)

#undef SPECIAL_REG_COMPOSITE_SPLIT

//===----------------------------------------------------------------------===//
// VCC composite/split
//===----------------------------------------------------------------------===//

RegisterTypeInterface VCCLoType::getCompositeType(
    TypeRange types, std::optional<int16_t> alignment,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  assert(!alignment.has_value() &&
         "alignment is not supported for special register composites");
  if (types.size() != 1 || !isa<VCCHiType>(types[0])) {
    if (emitError)
      emitError() << "expected exactly one vcc_hi operand";
    return nullptr;
  }
  if (cast<VCCHiType>(types[0]).getSemantics() != getSemantics()) {
    if (emitError)
      emitError() << "expected matching semantics";
    return nullptr;
  }
  return VCCType::get(getContext(), getReg());
}

LogicalResult VCCLoType::getSplitType(
    SmallVectorImpl<RegisterTypeInterface> &regs,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  (void)regs;
  if (emitError)
    emitError() << "vcc_lo is not a composite type and cannot be split";
  return failure();
}

RegisterTypeInterface VCCType::getCompositeType(
    TypeRange types, std::optional<int16_t> alignment,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  (void)types;
  (void)alignment;
  return getCompositeSRegImpl(emitError);
}

LogicalResult VCCType::getSplitType(
    SmallVectorImpl<RegisterTypeInterface> &regs,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  regs.push_back(VCCLoType::get(getContext(), getReg()));
  regs.push_back(VCCHiType::get(getContext(), getReg()));
  return success();
}

//===----------------------------------------------------------------------===//
// EXEC composite/split
//===----------------------------------------------------------------------===//

RegisterTypeInterface EXECLoType::getCompositeType(
    TypeRange types, std::optional<int16_t> alignment,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  assert(!alignment.has_value() &&
         "alignment is not supported for special register composites");
  if (types.size() != 1 || !isa<EXECHiType>(types[0])) {
    if (emitError)
      emitError() << "expected exactly one exec_hi operand";
    return nullptr;
  }
  if (cast<EXECHiType>(types[0]).getSemantics() != getSemantics()) {
    if (emitError)
      emitError() << "expected matching semantics";
    return nullptr;
  }
  return EXECType::get(getContext(), getReg());
}

LogicalResult EXECLoType::getSplitType(
    SmallVectorImpl<RegisterTypeInterface> &regs,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  (void)regs;
  if (emitError)
    emitError() << "exec_lo is not a composite type and cannot be split";
  return failure();
}

RegisterTypeInterface EXECType::getCompositeType(
    TypeRange types, std::optional<int16_t> alignment,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  (void)types;
  (void)alignment;
  return getCompositeSRegImpl(emitError);
}

LogicalResult EXECType::getSplitType(
    SmallVectorImpl<RegisterTypeInterface> &regs,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  regs.push_back(EXECLoType::get(getContext(), getReg()));
  regs.push_back(EXECHiType::get(getContext(), getReg()));
  return success();
}
