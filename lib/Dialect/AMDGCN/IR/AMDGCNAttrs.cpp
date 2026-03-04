//===- AMDGCNInsts.cpp - AMDGCN Instructions ------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNInst.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// AMDGCN dialect
//===----------------------------------------------------------------------===//

/// These are to avoid compiler warning because MLIR unconditionally generates
/// these functions.
[[maybe_unused]] static OptionalParseResult
generatedAttributeParser(AsmParser &parser, StringRef *mnemonic, Type type,
                         Attribute &value);
[[maybe_unused]] static LogicalResult
generatedAttributePrinter(Attribute def, AsmPrinter &printer);

void AMDGCNDialect::initializeAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AddressSpaceAttr
//===----------------------------------------------------------------------===//

LogicalResult
AddressSpaceAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         AddressSpaceKind space, AccessKind kind) {
  if (space != AddressSpaceKind::Local && space != AddressSpaceKind::Global) {
    emitError() << "unsupported address space: "
                << stringifyAddressSpaceKind(space);
    return failure();
  }
  if (kind == AccessKind::Unspecified) {
    emitError() << "access kind is unspecified";
    return failure();
  }
  return success();
}

bool AddressSpaceAttr::isValidLoad(
    Type type, ptr::AtomicOrdering ordering, std::optional<int64_t> alignment,
    const DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  bool isValid = getKind() != AccessKind::WriteOnly;
  if (!isValid && emitError) {
    emitError() << "memory space '" << *this << "' is write-only";
  }
  return isValid;
}

bool AddressSpaceAttr::isValidStore(
    Type type, ptr::AtomicOrdering ordering, std::optional<int64_t> alignment,
    const DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  bool isValid = getKind() != AccessKind::ReadOnly;
  if (!isValid && emitError) {
    emitError() << "memory space '" << *this << "' is read-only";
  }
  return isValid;
}

bool AddressSpaceAttr::isValidAtomicOp(
    ptr::AtomicBinOp op, Type type, ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  bool isValid = getKind() != AccessKind::ReadWrite;
  if (!isValid && emitError) {
    emitError() << "memory space '" << *this << "' is not read-write";
  }
  return isValid;
}

bool AddressSpaceAttr::isValidAtomicXchg(
    Type type, ptr::AtomicOrdering successOrdering,
    ptr::AtomicOrdering failureOrdering, std::optional<int64_t> alignment,
    const DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  bool isValid = getKind() != AccessKind::ReadWrite;
  if (!isValid && emitError) {
    emitError() << "memory space '" << *this << "' is not read-write";
  }
  return isValid;
}

bool AddressSpaceAttr::isValidAddrSpaceCast(
    Type tgt, Type src, function_ref<InFlightDiagnostic()> emitError) const {
  // TODO: update this method once the `addrspace_cast` op is added to the
  // dialect.
  assert(false && "unimplemented, see TODO in the source.");
  return false;
}

bool AddressSpaceAttr::isValidPtrIntCast(
    Type intLikeTy, Type ptrLikeTy,
    function_ref<InFlightDiagnostic()> emitError) const {
  // TODO: update this method once the int-cast ops are added to the dialect.
  assert(false && "unimplemented, see TODO in the source.");
  return false;
}

LogicalResult
AddressSpaceAttr::getSupportedOpWidths(Type type, Value addr, Value offset,
                                       Value const_offset, bool isRead,
                                       SmallVectorImpl<int32_t> &widths) const {
  return success();
  AddressSpaceKind space = getSpace();
  OperandKind addrKind = getOperandKind(addr.getType());
  OperandKind offsetKind = getOperandKind(offset.getType());
  OperandKind constOffsetKind = getOperandKind(const_offset.getType());
  // Fail if the operand kinds are not valid.
  if (!isOperandOf(addrKind, {OperandKind::SGPR, OperandKind::VGPR}) ||
      !isOperandOf(offsetKind, {OperandKind::SGPR, OperandKind::VGPR,
                                OperandKind::IntImm}) ||
      !isOperandOf(constOffsetKind, {OperandKind::IntImm})) {
    return failure();
  }
  assert(isAddressSpaceOf(
             space, {AddressSpaceKind::Local, AddressSpaceKind::Global}) &&
         "unsupported address space");
  if (addrKind == OperandKind::SGPR && offsetKind == OperandKind::SGPR) {
    // Invalid operands.
    if (space == AddressSpaceKind::Local) {
      // TODO: Add error reporting here.
      llvm_unreachable("unhandled case in getSupportedOpWidths");
      return failure();
    }

    // These correspond to the available SMEM instructions.
    widths.push_back(32);
    widths.push_back(32 * 2);
    widths.push_back(32 * 4);
    widths.push_back(32 * 8);
    widths.push_back(32 * 16);
    return success();
  }
  if (isOperandOf(addrKind, {OperandKind::SGPR, OperandKind::VGPR}) &&
      offsetKind == OperandKind::VGPR) {
    // Invalid operands.
    if (space == AddressSpaceKind::Local) {
      // TODO: Add error reporting here.
      llvm_unreachable("unhandled case in getSupportedOpWidths");
      return failure();
    }

    // These correspond to the available FLAT instructions.
    widths.push_back(32);
    widths.push_back(32 * 2);
    widths.push_back(32 * 3);
    widths.push_back(32 * 4);
    return success();
  }
  if (isOperandOf(addrKind, {OperandKind::VGPR}) &&
      offsetKind == OperandKind::VGPR) {
    // These correspond to the available FLAT/DS instructions.
    widths.push_back(32);
    widths.push_back(32 * 2);
    widths.push_back(32 * 3);
    widths.push_back(32 * 4);
    return success();
  }
  // TODO: Add error reporting here.
  llvm_unreachable("unhandled case in getSupportedOpWidths");
  return failure();
}

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.cpp.inc"
