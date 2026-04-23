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
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
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
// Normal-form helpers
//===----------------------------------------------------------------------===//

namespace {
/// Filter callback used by `walkTypes` to decide whether to descend into a
/// given `NamedAttribute` of an operation. Returning `false` skips the
/// attribute payload entirely.
using NamedAttrFilter = llvm::function_ref<bool(Operation *, NamedAttribute)>;

/// Skips named attributes that carry ABI metadata whose register types are
/// not subject to normal-form invariants enforced on the kernel body. For
/// `amdgcn.kernel`, this excludes the `arguments` attribute (e.g.
/// `by_val_arg` parameter types).
bool skipKernelAbiMetadata(Operation *op, NamedAttribute attr) {
  if (auto kernel = dyn_cast<KernelOp>(op))
    return attr.getName() != kernel.getArgumentsAttrName();
  return true;
}

/// Aggregates `DiagnosedSilenceableFailure` results across multiple visits:
/// records the first silenceable failure (silencing later ones) and short-
/// circuits on definite failures.
struct AttrTypeAggregator {
  DiagnosedSilenceableFailure overall = DiagnosedSilenceableFailure::success();
  bool stop = false;

  void merge(DiagnosedSilenceableFailure &&result) {
    if (result.isDefiniteFailure()) {
      overall = std::move(result);
      stop = true;
      return;
    }
    if (result.isSilenceableFailure()) {
      if (overall.succeeded())
        overall = std::move(result);
      else
        (void)result.silence();
    }
  }
};

/// Walks all distinct types reachable from operations under `root`: operation
/// result types, block argument types, and types nested inside operation
/// attributes. Invokes `visitor` on each distinct type with a `Location` near
/// its discovery point. When `filter` is non-null, it is consulted for each
/// named attribute of every visited operation; returning `false` skips the
/// attribute payload (e.g. to exclude `amdgcn.kernel`'s `arguments` ABI
/// metadata).
DiagnosedSilenceableFailure walkTypes(
    Operation *root,
    llvm::function_ref<DiagnosedSilenceableFailure(Type, Location)> visitor,
    NamedAttrFilter filter = nullptr) {
  AttrTypeAggregator agg;
  llvm::SmallPtrSet<Type, 16> seenTypes;
  llvm::SmallPtrSet<Attribute, 16> seenAttrs;
  Location currentLoc = root->getLoc();
  AttrTypeWalker walker;

  walker.addWalk([&](Type type) {
    auto [it, inserted] = seenTypes.insert(type);
    if (!inserted)
      return WalkResult::skip();
    agg.merge(visitor(type, currentLoc));
    if (agg.stop)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  walker.addWalk([&](Attribute attr) {
    auto [it, inserted] = seenAttrs.insert(attr);
    if (!inserted)
      return WalkResult::skip();
    return WalkResult::advance();
  });

  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    currentLoc = op->getLoc();
    for (OpResult result : op->getResults()) {
      currentLoc = result.getLoc();
      if (walker.walk(result.getType()).wasInterrupted())
        return WalkResult::interrupt();
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          currentLoc = arg.getLoc();
          if (walker.walk(arg.getType()).wasInterrupted())
            return WalkResult::interrupt();
        }
      }
    }
    currentLoc = op->getLoc();
    for (NamedAttribute attr : op->getAttrs()) {
      if (filter && !filter(op, attr))
        continue;
      if (walker.walk(attr.getValue()).wasInterrupted())
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return std::move(agg.overall);
}

} // namespace

//===----------------------------------------------------------------------===//
// NoValueSemanticRegistersAttr
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
NoValueSemanticRegistersAttr::checkOperation(Operation *op) const {
  return walkTypes(
      op,
      [](Type type, Location loc) -> DiagnosedSilenceableFailure {
        auto regType = dyn_cast<RegisterTypeInterface>(type);
        if (!regType || !regType.hasValueSemantics())
          return DiagnosedSilenceableFailure::success();
        return emitSilenceableFailure(loc)
               << "normal form violation: register types with value "
                  "semantics are disallowed but found: "
               << type;
      },
      skipKernelAbiMetadata);
}

//===----------------------------------------------------------------------===//
// AllRegistersAllocatedAttr
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
AllRegistersAllocatedAttr::checkOperation(Operation *op) const {
  return walkTypes(
      op,
      [](Type type, Location loc) -> DiagnosedSilenceableFailure {
        auto regType = dyn_cast<RegisterTypeInterface>(type);
        if (!regType || regType.hasAllocatedSemantics())
          return DiagnosedSilenceableFailure::success();
        return emitSilenceableFailure(loc)
               << "normal form violation: all registers must have "
                  "allocated semantics but found: "
               << type;
      },
      skipKernelAbiMetadata);
}

//===----------------------------------------------------------------------===//
// NoRegCastOpsAttr
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
NoRegCastOpsAttr::checkOperation(Operation *op) const {
  AttrTypeAggregator agg;
  op->walk([&](Operation *innerOp) {
    if (!isa<lsir::RegCastOp>(innerOp))
      return WalkResult::advance();
    agg.merge(emitSilenceableFailure(innerOp)
              << "normal form violation: lsir.reg_cast should not "
                 "survive past aster-to-amdgcn; this indicates an "
                 "incorrect lsir.to_reg or lsir.from_reg surviving "
                 "from high-level (hand-authored ?) IR");
    return agg.stop ? WalkResult::interrupt() : WalkResult::advance();
  });
  return std::move(agg.overall);
}

//===----------------------------------------------------------------------===//
// NoLsirOpsAttr
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure NoLsirOpsAttr::checkOperation(Operation *op) const {
  AttrTypeAggregator agg;
  op->walk([&](Operation *innerOp) {
    if (!innerOp->getDialect() ||
        innerOp->getDialect()->getNamespace() != "lsir")
      return WalkResult::advance();
    agg.merge(emitSilenceableFailure(innerOp)
              << "normal form violation: LSIR dialect operations "
                 "are disallowed but found: "
              << innerOp->getName());
    return agg.stop ? WalkResult::interrupt() : WalkResult::advance();
  });
  return std::move(agg.overall);
}

//===----------------------------------------------------------------------===//
// NoLsirComputeOpsAttr
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
NoLsirComputeOpsAttr::checkOperation(Operation *op) const {
  AttrTypeAggregator agg;
  op->walk([&](Operation *innerOp) {
    if (!innerOp->getDialect() ||
        innerOp->getDialect()->getNamespace() != "lsir")
      return WalkResult::advance();
    // Allow control-flow ops (lowered by LegalizeCF) and copy (regalloc
    // primitive).
    if (isa<lsir::CmpIOp, lsir::CmpFOp, lsir::SelectOp, lsir::CopyOp,
            lsir::BranchOp, lsir::CondBranchOp>(innerOp))
      return WalkResult::advance();
    agg.merge(emitSilenceableFailure(innerOp)
              << "normal form violation: LSIR compute/memory "
                 "operations are disallowed but found: "
              << innerOp->getName());
    return agg.stop ? WalkResult::interrupt() : WalkResult::advance();
  });
  return std::move(agg.overall);
}

//===----------------------------------------------------------------------===//
// NoLsirControlOpsAttr
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
NoLsirControlOpsAttr::checkOperation(Operation *op) const {
  AttrTypeAggregator agg;
  op->walk([&](Operation *innerOp) {
    if (!isa<lsir::CmpIOp, lsir::CmpFOp, lsir::SelectOp>(innerOp))
      return WalkResult::advance();
    agg.merge(emitSilenceableFailure(innerOp)
              << "normal form violation: LSIR control-flow "
                 "operations are disallowed but found: "
              << innerOp->getName());
    return agg.stop ? WalkResult::interrupt() : WalkResult::advance();
  });
  return std::move(agg.overall);
}

//===----------------------------------------------------------------------===//
// NoScfOpsAttr
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure NoScfOpsAttr::checkOperation(Operation *op) const {
  AttrTypeAggregator agg;
  op->walk([&](Operation *innerOp) {
    if (!innerOp->getDialect() ||
        innerOp->getDialect()->getNamespace() != "scf")
      return WalkResult::advance();
    agg.merge(emitSilenceableFailure(innerOp)
              << "normal form violation: SCF dialect operations "
                 "are disallowed but found: "
              << innerOp->getName());
    return agg.stop ? WalkResult::interrupt() : WalkResult::advance();
  });
  return std::move(agg.overall);
}

//===----------------------------------------------------------------------===//
// NoCfBranchesAttr
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
NoCfBranchesAttr::checkOperation(Operation *op) const {
  AttrTypeAggregator agg;
  op->walk([&](Operation *innerOp) {
    if (!isa<cf::BranchOp, cf::CondBranchOp>(innerOp))
      return WalkResult::advance();
    agg.merge(emitSilenceableFailure(innerOp)
              << "normal form violation: cf.br/cf.cond_br operations "
                 "are disallowed but found: "
              << innerOp->getName());
    return agg.stop ? WalkResult::interrupt() : WalkResult::advance();
  });
  return std::move(agg.overall);
}

//===----------------------------------------------------------------------===//
// NoRegisterBlockArgsAttr
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
NoRegisterBlockArgsAttr::checkOperation(Operation *op) const {
  AttrTypeAggregator agg;
  op->walk([&](Operation *innerOp) {
    for (Region &region : innerOp->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (!isa<RegisterTypeInterface>(arg.getType()))
            continue;
          agg.merge(emitSilenceableFailure(innerOp)
                    << "normal form violation: block arguments with "
                       "register types are disallowed but found: "
                    << arg.getType());
          if (agg.stop)
            return WalkResult::interrupt();
          // Only report once per op to mirror the original behavior.
          return WalkResult::advance();
        }
      }
    }
    return WalkResult::advance();
  });
  return std::move(agg.overall);
}

//===----------------------------------------------------------------------===//
// NoAffineOpsAttr
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
NoAffineOpsAttr::checkOperation(Operation *op) const {
  AttrTypeAggregator agg;
  op->walk([&](Operation *innerOp) {
    if (!innerOp->getDialect() ||
        innerOp->getDialect()->getNamespace() != "affine")
      return WalkResult::advance();
    agg.merge(emitSilenceableFailure(innerOp)
              << "normal form violation: affine dialect operations "
                 "are disallowed but found: "
              << innerOp->getName());
    return agg.stop ? WalkResult::interrupt() : WalkResult::advance();
  });
  return std::move(agg.overall);
}

//===----------------------------------------------------------------------===//
// NoMetadataOpsAttr
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
NoMetadataOpsAttr::checkOperation(Operation *op) const {
  AttrTypeAggregator agg;
  op->walk([&](Operation *innerOp) {
    if (!isa<LoadArgOp, ThreadIdOp, BlockDimOp, BlockIdOp, GridDimOp,
             MakeBufferRsrcOp>(innerOp))
      return WalkResult::advance();
    agg.merge(emitSilenceableFailure(innerOp)
              << "normal form violation: AMDGCN metadata operations "
                 "are disallowed but found: "
              << innerOp->getName());
    return agg.stop ? WalkResult::interrupt() : WalkResult::advance();
  });
  return std::move(agg.overall);
}

//===----------------------------------------------------------------------===//
// AllInlinedAttr
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
AllInlinedAttr::checkOperation(Operation *op) const {
  AttrTypeAggregator agg;
  op->walk([&](Operation *innerOp) {
    auto callOp = dyn_cast<func::CallOp>(innerOp);
    if (!callOp)
      return WalkResult::advance();
    agg.merge(emitSilenceableFailure(innerOp)
              << "normal form violation: func.call operations "
                 "are disallowed (all functions should be inlined) "
                 "but found call to '"
              << callOp.getCallee() << "'");
    return agg.stop ? WalkResult::interrupt() : WalkResult::advance();
  });
  return std::move(agg.overall);
}
