//===- AMDGCN.cpp - AMDGCN Operations -------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNVerifiers.h"
#include "aster/Dialect/AMDGCN/IR/OpAsmUtils.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/IR/ParsePrintUtils.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// Inline literal validation
//===----------------------------------------------------------------------===//

bool amdgcn::checkFloatConst(Value value, ArrayRef<float> values) {
  APFloat apFloat(APFloat::IEEEdouble());
  if (!matchPattern(value, m_ConstantFloat(&apFloat)))
    return false;
  float fval = static_cast<float>(apFloat.convertToDouble());
  return llvm::is_contained(values, fval);
}

bool amdgcn::checkIntConst(Value value, ArrayRef<int64_t> values) {
  APInt apInt;
  if (!matchPattern(value, m_ConstantInt(&apInt)))
    return false;
  return llvm::is_contained(values, apInt.getSExtValue());
}

//===----------------------------------------------------------------------===//
// AMDGCN Inliner Interface
//===----------------------------------------------------------------------===//

namespace {
struct AMDGCNInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// Always allow inlining of AMDGCN operations.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// Always allow inlining of AMDGCN operations into regions.
  bool isLegalToInline(Operation *op, Region *region, bool wouldBeCloned,
                       IRMapping &mapping) const final {
    return true;
  }

  /// Always allow inlining of regions.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// AMDGCNDialect
//===----------------------------------------------------------------------===//

void AMDGCNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.cpp.inc"
      ,
#define GET_OP_LIST
#include "aster/Dialect/AMDGCN/IR/ControlFlow.cpp.inc"
      ,
#define GET_OP_LIST
#include "aster/Dialect/AMDGCN/IR/DS.cpp.inc"
      ,
#define GET_OP_LIST
#include "aster/Dialect/AMDGCN/IR/MMA.cpp.inc"
      ,
#define GET_OP_LIST
#include "aster/Dialect/AMDGCN/IR/WMMA.cpp.inc"
      ,
#define GET_OP_LIST
#include "aster/Dialect/AMDGCN/IR/SMem.cpp.inc"
      ,
#define GET_OP_LIST
#include "aster/Dialect/AMDGCN/IR/VMem.cpp.inc"
      ,
#define GET_OP_LIST
#include "aster/Dialect/AMDGCN/IR/SOP.cpp.inc"
      ,
#define GET_OP_LIST
#include "aster/Dialect/AMDGCN/IR/VOP.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.cpp.inc"
      >();
  initializeAttributes();
  addInterfaces<AMDGCNInlinerInterface>();
}

Attribute AMDGCNDialect::parseAttribute(DialectAsmParser &parser,
                                        Type type) const {
  return parseDialectAttributes<
#define GET_ATTRDEF_LIST
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.cpp.inc"
      >(parser, type, getDialectNamespace());
}

void AMDGCNDialect::printAttribute(Attribute attr,
                                   DialectAsmPrinter &os) const {
  return printDialectAttributes<
#define GET_ATTRDEF_LIST
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.cpp.inc"
      >(attr, os);
}

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

bool mlir::aster::amdgcn::isRegisterLike(Type type) {
  auto regType = dyn_cast<RegisterTypeInterface>(type);
  if (!regType)
    return false;

  // Check if it's a register type with size 1
  RegisterRange range = regType.getAsRange();
  return range.size() == 1;
}

bool mlir::aster::amdgcn::isAllocatableRegisterLike(Type type) {
  if (!isRegisterLike(type))
    return false;
  RegisterTypeInterface regType = cast<RegisterTypeInterface>(type);
  return !bitEnumContainsAll(regType.getProps(), RegisterProps::IsComposite);
}

bool mlir::aster::amdgcn::isCompositeRegisterLike(Type type) {
  auto regType = dyn_cast<RegisterTypeInterface>(type);
  if (!regType)
    return false;
  return bitEnumContainsAll(regType.getProps(), RegisterProps::IsComposite);
}

RegisterKind
mlir::aster::amdgcn::getRegisterKind(AMDGCNRegisterTypeInterface type) {
  if (auto rTy = dyn_cast<AMDGCNRegisterTypeInterface>(type))
    return rTy.getRegisterKind();
  return RegisterKind::Unknown;
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

LogicalResult AllocaOp::verify() {
  RegisterTypeInterface regTy = getType();
  if (bitEnumContainsAll(regTy.getProps(), RegisterProps::IsReadOnly) &&
      regTy.hasValueSemantics())
    return emitOpError("cannot allocate read-only register with value "
                       "semantics");
  return success();
}

Speculation::Speculatability AllocaOp::getSpeculatability() {
  if (getType().hasAllocatedSemantics())
    return Speculation::Speculatability::Speculatable;
  return Speculation::Speculatability::NotSpeculatable;
}

void AllocaOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getType().hasAllocatedSemantics())
    return;
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getResult(0));
}

//===----------------------------------------------------------------------===//
// AllocLDSOp
//===----------------------------------------------------------------------===//

LogicalResult AllocLDSOp::verify() {
  int64_t staticSize = getStaticSize();
  bool hasDynamicSize = getDynamicSize() != nullptr;

  // Check that we have either a static or dynamic size, not both.
  if (ShapedType::isDynamic(staticSize) && !hasDynamicSize) {
    return emitOpError("requires a dynamic size operand when static size is "
                       "dynamic");
  }
  if (!ShapedType::isDynamic(staticSize) && hasDynamicSize)
    return emitOpError("cannot have both static and dynamic size");

  // Verify static size is positive.
  if (!ShapedType::isDynamic(staticSize) && staticSize <= 0)
    return emitOpError("static size must be positive, got ") << staticSize;

  if (std::optional<uint32_t> offset = getOffset();
      offset && *offset % getAlignment() != 0) {
    return emitOpError("offset ")
           << *offset << " is not aligned to alignment " << getAlignment();
  }

  return success();
}

OpFoldResult AllocLDSOp::fold(FoldAdaptor adaptor) {
  if (!ShapedType::isDynamic(getStaticSize()))
    return nullptr;

  // Update in case the dynamic size is a constant.
  auto constValue = dyn_cast_or_null<IntegerAttr>(adaptor.getDynamicSize());
  if (!constValue)
    return nullptr;
  setStaticSize(constValue.getValue().getZExtValue());
  getDynamicSizeMutable().clear();
  return getResult();
}

//===----------------------------------------------------------------------===//
// MakeBufferRsrcOp
//===----------------------------------------------------------------------===//

LogicalResult MakeBufferRsrcOp::verify() {
  // base_addr must be a 2-SGPR range.
  auto baseAddrTy = dyn_cast<SGPRType>(getBaseAddr().getType());
  if (!baseAddrTy || baseAddrTy.getRange().size() != 2)
    return emitOpError("base_addr must be an sgpr_range of size 2, got ")
           << getBaseAddr().getType();

  // Result must be a 4-SGPR range.
  auto resultTy = dyn_cast<SGPRType>(getResult().getType());
  if (!resultTy || resultTy.getRange().size() != 4)
    return emitOpError("result must be an sgpr_range of size 4, got ")
           << getResult().getType();

  // If stride is a known constant, validate it fits in 14 bits.
  // Note: this is not standard to look at value provenance but it is a best
  // effort to avoid ValueOrAttr that upstream MLIR is flippant about.
  APInt strideVal;
  if (matchPattern(getStride(), m_ConstantInt(&strideVal))) {
    int64_t stride = strideVal.getSExtValue();
    if (stride < 0 || stride > 16383)
      return emitOpError("stride must be in [0, 16383], got ") << stride;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MakeRegisterRangeOp
//===----------------------------------------------------------------------===//

LogicalResult CrossWaveTokenBarrierOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(FenceTokenType::get(context));
  return success();
}

LogicalResult MakeRegisterRangeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Fail if there are fewer than two operands.
  if (operands.size() < 2) {
    if (location)
      mlir::emitError(*location) << "expected at least two operands";
    return failure();
  }

  RegisterTypeInterface fTy =
      cast<RegisterTypeInterface>(operands[0].getType());

  auto emitErrorLambda = [&]() { return mlir::emitError(*location); };
  llvm::function_ref<InFlightDiagnostic()> emitError = nullptr;
  if (location)
    emitError = emitErrorLambda;

  TypeRange remaining = TypeRange(operands).drop_front();
  RegisterTypeInterface result =
      fTy.getCompositeType(remaining, std::nullopt, emitError);
  if (!result)
    return failure();
  inferredReturnTypes.push_back(result);
  return success();
}

LogicalResult
MakeRegisterRangeOp::livenessTransferFunction(LivenessCallback insertCallback,
                                              LivenessCallback removeCallback,
                                              IsLiveCallback isLiveCallback) {
  Value result = getResult();
  // If the result is live, propagate the liveness to the inputs.
  if (isLiveCallback(result))
    insertCallback(getInputs());
  removeCallback(result);
  return success();
}

//===----------------------------------------------------------------------===//
// SplitRegisterRangeOp
//===----------------------------------------------------------------------===//

LogicalResult SplitRegisterRangeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // There should be exactly one operand.
  if (operands.size() != 1) {
    if (location)
      mlir::emitError(*location) << "expected exactly one operand";
    return failure();
  }

  Type inputType = operands[0].getType();
  RegisterTypeInterface rangeType = cast<RegisterTypeInterface>(inputType);

  auto emitErrorLambda = [&]() { return mlir::emitError(*location); };
  llvm::function_ref<InFlightDiagnostic()> emitError = nullptr;
  if (location)
    emitError = emitErrorLambda;

  SmallVector<RegisterTypeInterface> regs;
  if (failed(rangeType.getSplitType(regs, emitError)))
    return failure();

  for (RegisterTypeInterface reg : regs)
    inferredReturnTypes.push_back(reg);
  return success();
}

LogicalResult
SplitRegisterRangeOp::livenessTransferFunction(LivenessCallback insertCallback,
                                               LivenessCallback removeCallback,
                                               IsLiveCallback isLiveCallback) {
  ValueRange results = getResults();
  // If any of the results are live, propagate the liveness to the input.
  if (llvm::any_of(results, isLiveCallback))
    insertCallback(getInput());
  removeCallback(results);
  return success();
}

//===----------------------------------------------------------------------===//
// RegInterferenceOp
//===----------------------------------------------------------------------===//

LogicalResult
RegInterferenceOp::livenessTransferFunction(LivenessCallback insertCallback,
                                            LivenessCallback removeCallback,
                                            IsLiveCallback isLiveCallback) {
  // This operation doesn't modify liveness.
  return success();
}

//===----------------------------------------------------------------------===//
// KernelOp Verification
//===----------------------------------------------------------------------===//

LogicalResult KernelOp::verify() {
  Region &bodyRegion = getBodyRegion();

  // Check there is at least one EndKernelOp terminator.
  int32_t numEndKernel = 0;
  for (auto &block : bodyRegion) {
    if (block.empty())
      continue;
    Operation &terminator = block.back();
    if (isa<EndKernelOp>(terminator))
      numEndKernel++;
  }
  if (numEndKernel == 0)
    return emitError("kernel must have at least one EndKernelOp terminator");

  return success();
}

//===----------------------------------------------------------------------===//
// Normal form helpers (shared by ModuleOp and KernelOp)
//===----------------------------------------------------------------------===//

/// Verify all normal forms attached to an operation via its normal_forms attr.
/// Each normal form's `checkOperation` method is responsible for walking the
/// IR (and excluding op-attribute payloads that should not participate in
/// normal-form type checking; see `getNormalFormTypeWalkExcludes` in
/// AMDGCNAttrs.cpp for the kernel-arguments exclusion).
static LogicalResult verifyNormalFormsRegions(Operation *op,
                                              ArrayAttr normalFormsAttr) {
  if (!normalFormsAttr || normalFormsAttr.empty())
    return success();
  for (Attribute attr : normalFormsAttr) {
    auto nf = cast<mlir::transform::NormalFormAttrInterface>(attr);
    if (failed(nf.checkOperation(op).checkAndReport()))
      return failure();
  }
  return success();
}

/// Add normal forms to an operation's normal_forms attribute using set
/// semantics. Returns true if the attribute was changed.
static bool
addNormalFormsImpl(Operation *op, StringRef attrName, ArrayAttr currentAttr,
                   ArrayRef<mlir::transform::NormalFormAttrInterface> nfs,
                   function_ref<void(ArrayAttr)> setter) {
  if (nfs.empty())
    return false;

  SetVector<Attribute> nfSet;
  if (currentAttr)
    nfSet.insert_range(currentAttr.getValue());

  bool changed = false;
  for (mlir::transform::NormalFormAttrInterface nf : nfs)
    changed |= nfSet.insert(nf);

  if (!changed)
    return false;

  OpBuilder builder(op->getContext());
  setter(builder.getArrayAttr(nfSet.getArrayRef()));
  return true;
}

/// Remove normal forms from an operation's normal_forms attribute.
/// Returns true if the attribute was changed.
static bool
removeNormalFormsImpl(Operation *op, ArrayAttr currentAttr,
                      ArrayRef<mlir::transform::NormalFormAttrInterface> nfs,
                      function_ref<void(ArrayAttr)> setter) {
  if (nfs.empty() || !currentAttr || currentAttr.empty())
    return false;

  SetVector<Attribute> nfSet;
  nfSet.insert_range(currentAttr.getValue());

  bool changed = false;
  for (mlir::transform::NormalFormAttrInterface nf : nfs)
    changed |= nfSet.remove(nf);

  if (!changed)
    return false;

  OpBuilder builder(op->getContext());
  setter(builder.getArrayAttr(nfSet.getArrayRef()));
  return true;
}

//===----------------------------------------------------------------------===//
// ModuleOp Normal Forms
//===----------------------------------------------------------------------===//

LogicalResult amdgcn::ModuleOp::verifyRegions() {
  if (failed(verifyNormalFormsRegions(getOperation(), getNormalFormsAttr())))
    return failure();

  // Verify ISA compatibility: every nested op that declares ISA restrictions
  // must be compatible with this module's target.
  ISAVersion moduleISA = getIsaForTarget(getTarget());
  LogicalResult result = success();
  getOperation()->walk([&](ISACompatibleOpInterface op) {
    if (!op.supportsISA(moduleISA)) {
      op->emitOpError() << "is not compatible with module target "
                        << getTarget();
      result = failure();
    }
  });
  return result;
}

bool amdgcn::ModuleOp::addNormalForms(
    ArrayRef<mlir::transform::NormalFormAttrInterface> normalForms) {
  return addNormalFormsImpl(getOperation(), getNormalFormsAttrName(),
                            getNormalFormsAttr(), normalForms,
                            [&](ArrayAttr attr) { setNormalFormsAttr(attr); });
}

bool amdgcn::ModuleOp::removeNormalForms(
    ArrayRef<mlir::transform::NormalFormAttrInterface> normalForms) {
  return removeNormalFormsImpl(
      getOperation(), getNormalFormsAttr(), normalForms,
      [&](ArrayAttr attr) { setNormalFormsAttr(attr); });
}

//===----------------------------------------------------------------------===//
// KernelOp Normal Forms
//===----------------------------------------------------------------------===//

LogicalResult KernelOp::verifyRegions() {
  return verifyNormalFormsRegions(getOperation(), getNormalFormsAttr());
}

bool KernelOp::addNormalForms(
    ArrayRef<mlir::transform::NormalFormAttrInterface> normalForms) {
  return addNormalFormsImpl(getOperation(), getNormalFormsAttrName(),
                            getNormalFormsAttr(), normalForms,
                            [&](ArrayAttr attr) { setNormalFormsAttr(attr); });
}

bool KernelOp::removeNormalForms(
    ArrayRef<mlir::transform::NormalFormAttrInterface> normalForms) {
  return removeNormalFormsImpl(
      getOperation(), getNormalFormsAttr(), normalForms,
      [&](ArrayAttr attr) { setNormalFormsAttr(attr); });
}

//===----------------------------------------------------------------------===//
// LibraryOp Verification
//===----------------------------------------------------------------------===//

LogicalResult LibraryOp::verify() {
  // Libraries cannot contain amdgcn.kernel operations.
  for (Operation &op : getBodyRegion().front()) {
    if (isa<KernelOp>(op))
      return emitError(
          "amdgcn.library cannot contain amdgcn.kernel operations");
  }

  // Extract ISA versions from the isa attribute (if present).
  SmallVector<ISAVersion> isas;
  if (std::optional<ArrayAttr> isaAttr = getIsa()) {
    for (Attribute attr : *isaAttr) {
      auto isaVersionAttr = dyn_cast<ISAVersionAttr>(attr);
      if (!isaVersionAttr)
        return emitError("isa attribute must contain only ISAVersion elements");
      isas.push_back(isaVersionAttr.getValue());
    }
  }

  // Verify ISA support for all operations in the library.
  return verifyISAsSupportImpl(getBodyRegion(), isas,
                               [&]() { return emitError(); });
}

//===----------------------------------------------------------------------===//
// PtrAddOp
//===----------------------------------------------------------------------===//

LogicalResult
PtrAddOp::inferReturnTypes(MLIRContext *context,
                           std::optional<Location> location, Adaptor adaptor,
                           SmallVectorImpl<Type> &inferredReturnTypes) {
  auto ptrType =
      dyn_cast<AMDGCNRegisterTypeInterface>(adaptor.getPtr().getType());
  if (!ptrType || !llvm::is_contained({RegisterKind::SGPR, RegisterKind::VGPR},
                                      ptrType.getRegisterKind())) {
    if (location)
      mlir::emitError(*location)
          << "expected ptr to be a SGPR or VGPR register type";
    return failure();
  }
  // If the op has a dynamic offset, the result is always a VGPR type, otherwise
  // it is the same as the ptr type.
  if (adaptor.getDynamicOffset())
    ptrType = VGPRType::get(context, ptrType.getAsRange());
  inferredReturnTypes.push_back(ptrType);
  return success();
}

OpFoldResult PtrAddOp::fold(FoldAdaptor adaptor) {
  // If dynamic_offset and uniform_offset are not present and const_offset is 0,
  // fold to the ptr.
  if (!getDynamicOffset() && !getUniformOffset() && getConstOffset() == 0)
    return getPtr();
  return nullptr;
}

LogicalResult PtrAddOp::canonicalize(PtrAddOp op, PatternRewriter &rewriter) {
  auto ptrBase = op.getPtr().getDefiningOp<PtrAddOp>();
  if (!ptrBase)
    return failure();

  // Bail if the flags don't match.
  if (ptrBase.getFlags() != op.getFlags())
    return failure();

  // Bail if either op has a uniform offset.
  if (op.getUniformOffset() || ptrBase.getUniformOffset())
    return failure();

  // Bail if both ops have dynamic offsets.
  if (op.getDynamicOffset() && ptrBase.getDynamicOffset())
    return failure();

  // Get the dynamic offset.
  Value dynOff = op.getDynamicOffset();
  if (!dynOff)
    dynOff = ptrBase.getDynamicOffset();

  int64_t ptrBaseConstOff = ptrBase.getConstOffset();
  int64_t opConstOff = op.getConstOffset();
  // Bail if any of the offsets are negative
  if (opConstOff < 0 || ptrBaseConstOff < 0)
    return failure();

  // Get the constant offset.
  int64_t constOffset = opConstOff + ptrBaseConstOff;
  auto newAdd =
      PtrAddOp::create(rewriter, op.getLoc(), ptrBase.getPtr(), dynOff,
                       /*uniformOffset=*/nullptr, constOffset, op.getFlags());
  rewriter.replaceOp(op, newAdd.getResult());
  return success();
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

Value WaitOp::getOutDependency() { return Value(); }

// WaitCntOpInterface: amdgcn.wait carries the CDNA vm/lgkm counters.
SmallVector<WaitCounterKind> WaitOp::getSupportedCounters() {
  return {WaitCounterKind::Vm, WaitCounterKind::Lgkm};
}

uint16_t WaitOp::getCounterValue(WaitCounterKind kind) {
  switch (kind) {
  case WaitCounterKind::Vm:
    return getVmCnt();
  case WaitCounterKind::Lgkm:
    return getLgkmCnt();
  default:
    return kNoWaitCount;
  }
}

void WaitOp::setCounterValue(WaitCounterKind kind, uint16_t value) {
  switch (kind) {
  case WaitCounterKind::Vm:
    setVmCnt(value);
    return;
  case WaitCounterKind::Lgkm:
    setLgkmCnt(value);
    return;
  default:
    llvm_unreachable("amdgcn.wait does not carry this counter");
  }
}

ArrayRef<ISAVersion> WaitOp::getCompatibleISAVersions() {
  static ISAVersion versions[] = {ISAVersion::CDNA3, ISAVersion::CDNA4};
  return versions;
}

bool WaitOp::addDependencies(ValueRange deps) {
  bool changed = false;
  if (deps.empty())
    return changed;
  getDependenciesMutable().append(deps);
  changed = true;
  return changed;
}

bool WaitOp::removeDependencies(ValueRange deps) {
  bool changed = false;
  if (deps.empty())
    return changed;
  MutableOperandRange operands = getDependenciesMutable();
  llvm::SmallPtrSet<Value, 5> removeSet(deps.begin(), deps.end());
  SmallVector<Value> remaining;
  for (Value dep : operands.getAsOperandRange()) {
    if (removeSet.contains(dep))
      continue;
    remaining.push_back(dep);
  }
  if (remaining.size() != operands.size()) {
    operands.assign(remaining);
    changed = true;
  }
  return changed;
}

void WaitOp::setDependencies(ValueRange deps) {
  getDependenciesMutable().assign(deps);
}

/// Merge contiguous wait ops into a single wait op and canonicalize its
/// operands.
static LogicalResult canonicalizeWaitImpl(WaitOp waitOp, RewriterBase &rewriter,
                                          llvm::SetVector<Value> &deps) {
  deps.clear();
  bool changed = false;

  // Helper to remove duplicate dependency operands.
  auto removeDuplicates = [&]() {
    MutableOperandRange operands = waitOp.getDependenciesMutable();
    deps.insert_range(operands.getAsOperandRange());
    if (deps.size() != operands.size()) {
      operands.assign(deps.getArrayRef());
      changed = true;
    }
  };

  /// Early exit if the wait op is not in a block.
  Block::iterator bbEnd;
  if (Block *bb = waitOp->getBlock()) {
    bbEnd = bb->end();
  } else {
    removeDuplicates();
    if (changed)
      rewriter.modifyOpInPlace(waitOp, []() {});
    return success(changed);
  }

  Block::iterator start = waitOp->getIterator(),
                  end = ++(waitOp->getIterator());
  // Find the end of the contiguous wait ops.
  while (end != bbEnd) {
    Operation &op = *end;
    // Stop at non-wait ops.
    auto wait = dyn_cast<WaitOp>(op);
    if (!wait)
      break;
    ++end;
  }

  // Compute the new counts and arguments
  uint16_t vmCnt = waitOp.getVmCnt(), lgkmCnt = waitOp.getLgkmCnt();
  while (start != end) {
    auto wait = cast<WaitOp>(*(start++));
    deps.insert_range(wait.getDependencies());
    vmCnt = std::min(vmCnt, wait.getVmCnt());
    lgkmCnt = std::min(lgkmCnt, wait.getLgkmCnt());

    // Erase redundant wait ops.
    if (wait != waitOp) {
      wait.getFenceToken().replaceAllUsesWith(waitOp.getFenceToken());
      rewriter.eraseOp(wait);
      changed = true;
    }
  }

  // Update the original wait op.
  removeDuplicates();
  if (waitOp.getVmCnt() != vmCnt) {
    changed = true;
    waitOp.setVmCnt(vmCnt);
  }
  if (waitOp.getLgkmCnt() != lgkmCnt) {
    changed = true;
    waitOp.setLgkmCnt(lgkmCnt);
  }
  if (changed)
    rewriter.modifyOpInPlace(waitOp, []() {});
  return success(changed);
}

FailureOr<Block::iterator>
WaitOp::canonicalizeWait(WaitOp op, RewriterBase &rewriter,
                         llvm::SetVector<Value> &deps) {
  deps.clear();
  LogicalResult res = canonicalizeWaitImpl(op, rewriter, deps);

  // Get the next iterator before potentially erasing the op.
  Block::iterator nextIt;
  if (op->getBlock() != nullptr) {
    nextIt = op->getIterator();
    ++nextIt;
  }
  if (op.isNowait()) {
    rewriter.eraseOp(op);
    return nextIt;
  }
  return failed(res) ? FailureOr<Block::iterator>(res)
                     : FailureOr<Block::iterator>(nextIt);
}

LogicalResult WaitOp::canonicalize(WaitOp op, PatternRewriter &rewriter) {
  llvm::SetVector<Value> deps;
  return op.canonicalizeWait(op, rewriter, deps);
}

//===----------------------------------------------------------------------===//
// WaitGfx1250Op
//===----------------------------------------------------------------------===//

Value WaitGfx1250Op::getOutDependency() { return Value(); }

// WaitCntOpInterface: amdgcn.wait_gfx1250 carries the gfx1250 load/ds/tensor
// counters.
SmallVector<WaitCounterKind> WaitGfx1250Op::getSupportedCounters() {
  return {WaitCounterKind::Load, WaitCounterKind::Store, WaitCounterKind::Ds,
          WaitCounterKind::Km, WaitCounterKind::Tensor};
}

uint16_t WaitGfx1250Op::getCounterValue(WaitCounterKind kind) {
  switch (kind) {
  case WaitCounterKind::Load:
    return getLoadCnt();
  case WaitCounterKind::Store:
    return getStoreCnt();
  case WaitCounterKind::Ds:
    return getDsCnt();
  case WaitCounterKind::Km:
    return getKmCnt();
  case WaitCounterKind::Tensor:
    return getTensorCnt();
  default:
    return kNoWaitCount;
  }
}

void WaitGfx1250Op::setCounterValue(WaitCounterKind kind, uint16_t value) {
  switch (kind) {
  case WaitCounterKind::Load:
    setLoadCnt(value);
    return;
  case WaitCounterKind::Store:
    setStoreCnt(value);
    return;
  case WaitCounterKind::Ds:
    setDsCnt(value);
    return;
  case WaitCounterKind::Km:
    setKmCnt(value);
    return;
  case WaitCounterKind::Tensor:
    setTensorCnt(value);
    return;
  default:
    llvm_unreachable("amdgcn.wait_gfx1250 does not carry this counter");
  }
}

ArrayRef<ISAVersion> WaitGfx1250Op::getCompatibleISAVersions() {
  static ISAVersion versions[] = {ISAVersion::GFX12_50};
  return versions;
}

/// Merge contiguous wait_gfx1250 ops into one and canonicalize its operands.
static LogicalResult canonicalizeWaitGfx1250Impl(WaitGfx1250Op waitOp,
                                                 RewriterBase &rewriter,
                                                 llvm::SetVector<Value> &deps) {
  deps.clear();
  bool changed = false;

  auto removeDuplicates = [&]() {
    MutableOperandRange operands = waitOp.getDependenciesMutable();
    deps.insert_range(operands.getAsOperandRange());
    if (deps.size() != operands.size()) {
      operands.assign(deps.getArrayRef());
      changed = true;
    }
  };

  Block::iterator bbEnd;
  if (Block *bb = waitOp->getBlock()) {
    bbEnd = bb->end();
  } else {
    removeDuplicates();
    if (changed)
      rewriter.modifyOpInPlace(waitOp, []() {});
    return success(changed);
  }

  Block::iterator start = waitOp->getIterator(),
                  end = ++(waitOp->getIterator());
  while (end != bbEnd) {
    if (!isa<WaitGfx1250Op>(*end))
      break;
    ++end;
  }

  uint16_t loadCnt = waitOp.getLoadCnt(), storeCnt = waitOp.getStoreCnt(),
           dsCnt = waitOp.getDsCnt(), kmCnt = waitOp.getKmCnt(),
           tensorCnt = waitOp.getTensorCnt();
  while (start != end) {
    auto wait = cast<WaitGfx1250Op>(*(start++));
    deps.insert_range(wait.getDependencies());
    loadCnt = std::min(loadCnt, wait.getLoadCnt());
    storeCnt = std::min(storeCnt, wait.getStoreCnt());
    dsCnt = std::min(dsCnt, wait.getDsCnt());
    kmCnt = std::min(kmCnt, wait.getKmCnt());
    tensorCnt = std::min(tensorCnt, wait.getTensorCnt());
    if (wait != waitOp) {
      wait.getFenceToken().replaceAllUsesWith(waitOp.getFenceToken());
      rewriter.eraseOp(wait);
      changed = true;
    }
  }

  removeDuplicates();
  if (waitOp.getLoadCnt() != loadCnt) {
    changed = true;
    waitOp.setLoadCnt(loadCnt);
  }
  if (waitOp.getStoreCnt() != storeCnt) {
    changed = true;
    waitOp.setStoreCnt(storeCnt);
  }
  if (waitOp.getDsCnt() != dsCnt) {
    changed = true;
    waitOp.setDsCnt(dsCnt);
  }
  if (waitOp.getKmCnt() != kmCnt) {
    changed = true;
    waitOp.setKmCnt(kmCnt);
  }
  if (waitOp.getTensorCnt() != tensorCnt) {
    changed = true;
    waitOp.setTensorCnt(tensorCnt);
  }
  if (changed)
    rewriter.modifyOpInPlace(waitOp, []() {});
  return success(changed);
}

FailureOr<Block::iterator>
WaitGfx1250Op::canonicalizeWait(WaitGfx1250Op op, RewriterBase &rewriter,
                                llvm::SetVector<Value> &deps) {
  deps.clear();
  LogicalResult res = canonicalizeWaitGfx1250Impl(op, rewriter, deps);

  Block::iterator nextIt;
  if (op->getBlock() != nullptr) {
    nextIt = op->getIterator();
    ++nextIt;
  }
  if (op.isNowait()) {
    rewriter.eraseOp(op);
    return nextIt;
  }
  return failed(res) ? FailureOr<Block::iterator>(res)
                     : FailureOr<Block::iterator>(nextIt);
}

LogicalResult WaitGfx1250Op::canonicalize(WaitGfx1250Op op,
                                          PatternRewriter &rewriter) {
  llvm::SetVector<Value> deps;
  return op.canonicalizeWait(op, rewriter, deps);
}

//===----------------------------------------------------------------------===//
// AMDGCN InstOpInterface
//===----------------------------------------------------------------------===//

/// Infer types implementation for InstOp operations.
template <typename ConcreteType, typename ConcreteTypeAdaptor>
static LogicalResult
inferTypesImpl(MLIRContext *ctx, std::optional<Location> &loc,
               ConcreteTypeAdaptor &&adaptor, SmallVectorImpl<Type> &types) {
  static_assert(ConcreteType::kOutsSize > 0,
                "Output size must be greater than 0");
  for (size_t i = 0; i < ConcreteType::kOutsSize; ++i) {
    ValueRange v = adaptor.getODSOperands(i);
    for (Type ty : TypeRange(v))
      types.push_back(ty);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// IncGen
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.cpp.inc"

#include "aster/Dialect/AMDGCN/IR/Interfaces/HazardAttrInterface.cpp.inc"

#include "aster/Dialect/AMDGCN/IR/Interfaces/KernelArgInterface.cpp.inc"

#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.cpp.inc"

#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNInstOpInterface.cpp.inc"

#include "aster/Dialect/AMDGCN/IR/Interfaces/InstOpInterfaces.cpp.inc"

#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNRegisterTypeInterface.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.cpp.inc"

#define GET_OP_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.cpp.inc"
