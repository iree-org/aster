//===- AMDGCNInsts.cpp - AMDGCN Instructions ------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNInst.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

//===----------------------------------------------------------------------===//
// GenericSchedLabelerAttr
//===----------------------------------------------------------------------===//

GenericSchedLabelerAttr GenericSchedLabelerAttr::get(MLIRContext *ctx,
                                                     StringRef path) {
  SchedLabeler labeler;
  FailureOr<SchedLabeler> result = SchedLabeler::getFromYAML(path, ctx);
  if (succeeded(result))
    labeler = std::move(*result);
  else
    labeler.setName(path);
  return Base::get(ctx, labeler);
}

LogicalResult
GenericSchedLabelerAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                SchedLabeler labeler) {
  if (labeler.isTrivial()) {
    emitError() << "scheduler labeler has no patterns; "
                   "check that the YAML file exists and is non-empty: "
                << labeler.getName();
    return failure();
  }
  return success();
}

int32_t GenericSchedLabelerAttr::getLabel(Operation *op, int32_t,
                                          const SchedGraph &) const {
  return getLabeler().getLabel(op);
}

//===----------------------------------------------------------------------===//
// NoValueSemanticRegistersAttr
//===----------------------------------------------------------------------===//

LogicalResult NoValueSemanticRegistersAttr::verifyType(
    function_ref<InFlightDiagnostic()> emitError, Type type) const {
  auto regType = dyn_cast<RegisterTypeInterface>(type);
  if (!regType)
    return success();

  if (regType.hasValueSemantics())
    return emitError() << "normal form violation: register types with value "
                          "semantics are disallowed but found: "
                       << type;

  return success();
}

//===----------------------------------------------------------------------===//
// AllRegistersAllocatedAttr
//===----------------------------------------------------------------------===//

LogicalResult AllRegistersAllocatedAttr::verifyType(
    function_ref<InFlightDiagnostic()> emitError, Type type) const {
  auto regType = dyn_cast<RegisterTypeInterface>(type);
  if (!regType)
    return success();

  if (!regType.hasAllocatedSemantics())
    return emitError() << "normal form violation: all registers must have "
                          "allocated semantics but found: "
                       << type;

  return success();
}

//===----------------------------------------------------------------------===//
// NoRegCastOpsAttr
//===----------------------------------------------------------------------===//

LogicalResult
NoRegCastOpsAttr::verifyOperation(function_ref<InFlightDiagnostic()> emitError,
                                  Operation *op) const {
  if (isa<lsir::RegCastOp>(op))
    return emitError() << "normal form violation: lsir.reg_cast should not "
                          "survive past aster-to-amdgcn; this indicates an "
                          "incorrect lsir.to_reg or lsir.from_reg surviving "
                          "from high-level (hand-authored ?) IR";
  return success();
}

//===----------------------------------------------------------------------===//
// NoLsirOpsAttr
//===----------------------------------------------------------------------===//

LogicalResult
NoLsirOpsAttr::verifyOperation(function_ref<InFlightDiagnostic()> emitError,
                               Operation *op) const {
  if (op->getDialect() && op->getDialect()->getNamespace() == "lsir")
    return emitError() << "normal form violation: LSIR dialect operations "
                          "are disallowed but found: "
                       << op->getName();

  return success();
}

//===----------------------------------------------------------------------===//
// NoLsirComputeOpsAttr
//===----------------------------------------------------------------------===//

LogicalResult NoLsirComputeOpsAttr::verifyOperation(
    function_ref<InFlightDiagnostic()> emitError, Operation *op) const {
  if (!op->getDialect() || op->getDialect()->getNamespace() != "lsir")
    return success();

  // Allow control-flow ops (lowered by LegalizeCF) and copy (regalloc
  // primitive).
  if (isa<lsir::CmpIOp, lsir::CmpFOp, lsir::SelectOp, lsir::CopyOp,
          lsir::BranchOp, lsir::CondBranchOp>(op))
    return success();

  return emitError() << "normal form violation: LSIR compute/memory "
                        "operations are disallowed but found: "
                     << op->getName();
}

//===----------------------------------------------------------------------===//
// NoLsirControlOpsAttr
//===----------------------------------------------------------------------===//

LogicalResult NoLsirControlOpsAttr::verifyOperation(
    function_ref<InFlightDiagnostic()> emitError, Operation *op) const {
  if (isa<lsir::CmpIOp, lsir::CmpFOp, lsir::SelectOp>(op))
    return emitError() << "normal form violation: LSIR control-flow "
                          "operations are disallowed but found: "
                       << op->getName();

  return success();
}

//===----------------------------------------------------------------------===//
// NoScfOpsAttr
//===----------------------------------------------------------------------===//

LogicalResult
NoScfOpsAttr::verifyOperation(function_ref<InFlightDiagnostic()> emitError,
                              Operation *op) const {
  if (op->getDialect() && op->getDialect()->getNamespace() == "scf")
    return emitError() << "normal form violation: SCF dialect operations "
                          "are disallowed but found: "
                       << op->getName();

  return success();
}

//===----------------------------------------------------------------------===//
// NoCfBranchesAttr
//===----------------------------------------------------------------------===//

LogicalResult
NoCfBranchesAttr::verifyOperation(function_ref<InFlightDiagnostic()> emitError,
                                  Operation *op) const {
  if (isa<cf::BranchOp, cf::CondBranchOp>(op))
    return emitError() << "normal form violation: cf.br/cf.cond_br operations "
                          "are disallowed but found: "
                       << op->getName();

  return success();
}

//===----------------------------------------------------------------------===//
// NoRegisterBlockArgsAttr
//===----------------------------------------------------------------------===//

LogicalResult NoRegisterBlockArgsAttr::verifyOperation(
    function_ref<InFlightDiagnostic()> emitError, Operation *op) const {
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument arg : block.getArguments()) {
        if (isa<RegisterTypeInterface>(arg.getType()))
          return emitError()
                 << "normal form violation: block arguments with register "
                    "types are disallowed but found: "
                 << arg.getType();
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// NoAffineOpsAttr
//===----------------------------------------------------------------------===//

LogicalResult
NoAffineOpsAttr::verifyOperation(function_ref<InFlightDiagnostic()> emitError,
                                 Operation *op) const {
  if (op->getDialect() && op->getDialect()->getNamespace() == "affine")
    return emitError() << "normal form violation: affine dialect operations "
                          "are disallowed but found: "
                       << op->getName();

  return success();
}

//===----------------------------------------------------------------------===//
// NoMetadataOpsAttr
//===----------------------------------------------------------------------===//

LogicalResult
NoMetadataOpsAttr::verifyOperation(function_ref<InFlightDiagnostic()> emitError,
                                   Operation *op) const {
  if (isa<LoadArgOp, ThreadIdOp, BlockDimOp, BlockIdOp, GridDimOp,
          MakeBufferRsrcOp>(op))
    return emitError() << "normal form violation: AMDGCN metadata operations "
                          "are disallowed but found: "
                       << op->getName();

  return success();
}

//===----------------------------------------------------------------------===//
// AllInlinedAttr
//===----------------------------------------------------------------------===//

LogicalResult
AllInlinedAttr::verifyOperation(function_ref<InFlightDiagnostic()> emitError,
                                Operation *op) const {
  if (isa<func::CallOp>(op))
    return emitError() << "normal form violation: func.call operations "
                          "are disallowed (all functions should be inlined) "
                          "but found call to '"
                       << cast<func::CallOp>(op).getCallee() << "'";

  return success();
}
