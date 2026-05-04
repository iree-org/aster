//===- AMDGCN.cpp - AMDGCN Operations -------------------------------------===//
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
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNVerifiers.h"
#include "aster/Dialect/AMDGCN/IR/InstructionProps.h"
#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNInterfaces.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/NormalForm/IR/NormalFormInterfaces.h"
#include "aster/IR/ParsePrintUtils.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// Internal functions
//===----------------------------------------------------------------------===//

/// Pretty parser for OpCode attribute when parsed from an operation.
static ParseResult parseOpcode(OpAsmParser &parser, InstAttr &opcode) {
  StringRef opcodeStr;
  if (parser.parseKeyword(&opcodeStr))
    return failure();

  auto opcodeOpt = symbolizeOpCode(opcodeStr);
  if (!opcodeOpt)
    return parser.emitError(parser.getCurrentLocation(), "invalid opcode: ")
           << opcodeStr;

  opcode = InstAttr::get(parser.getBuilder().getContext(), *opcodeOpt);
  return success();
}

/// Pretty printer for OpCode attribute when parsed from an operation.
static void printOpcode(OpAsmPrinter &printer, Operation *, InstAttr opcode) {
  printer << stringifyOpCode(opcode.getValue());
}

/// Helper to get either the type of a Value or return the type itself.
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T>, int> = 0>
static auto getTypeOrValue(T value) {
  using Type = decltype(value.getType());
  if (value == nullptr)
    return Type();
  return value.getType();
}
/// Helper to passthrough values that are not MLIR Values.
template <typename T, std::enable_if_t<!std::is_base_of_v<Value, T>, int> = 0>
static T &&getTypeOrValue(T &&value) {
  return std::forward<T>(value);
}

//===----------------------------------------------------------------------===//
// DimAttr Parsing/Printing
//===----------------------------------------------------------------------===//

/// Parse a DimAttr from a keyword (x, y, or z).
static ParseResult parseDimAttr(OpAsmParser &parser, DimAttr &attr) {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();

  auto dimOpt = symbolizeDim(keyword);
  if (!dimOpt)
    return parser.emitError(parser.getCurrentLocation(), "invalid dimension: ")
           << keyword;

  attr = DimAttr::get(parser.getBuilder().getContext(), *dimOpt);
  return success();
}

/// Print a DimAttr as a keyword.
static void printDimAttr(OpAsmPrinter &printer, Operation *, DimAttr attr) {
  printer << stringifyDim(attr.getValue());
}

//===----------------------------------------------------------------------===//
// Offset Parsing/Printing
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LoadResults Parsing/Printing
//===----------------------------------------------------------------------===//

/// Parse the trailing results of a load instruction. The dest type has already
/// been parsed and is passed by reference. The dest_res type is inferred to be
/// the same as dest iff the register has value semantics. The token type is
/// parsed from the input.
static ParseResult parseLoadResults(OpAsmParser &parser, Type destType,
                                    Type &destResType, Type &tokenType) {
  auto regTy = dyn_cast<RegisterTypeInterface>(destType);
  if (regTy && regTy.hasValueSemantics())
    destResType = destType;
  if (parser.parseType(tokenType))
    return failure();
  return success();
}

/// Print the trailing results of a load instruction. The dest_res type is
/// inferred from dest and is not printed; only the token type is printed.
static void printLoadResults(OpAsmPrinter &printer, Operation *, Type destType,
                             Type destResType, Type tokenType) {
  printer.printType(tokenType);
}

//===----------------------------------------------------------------------===//
// AllocSize Parsing/Printing
//===----------------------------------------------------------------------===//

/// Parse a size that can be either static (integer) or dynamic (SSA value).
/// Format: `<integer>` for static, `%operand` for dynamic.
static ParseResult
parseAllocSize(OpAsmParser &parser,
               std::optional<OpAsmParser::UnresolvedOperand> &dynamicSize,
               IntegerAttr &staticSize) {
  // Try to parse an integer first (static size).
  int64_t intVal;
  auto intRes = parser.parseOptionalInteger(intVal);
  if (intRes.has_value()) {
    if (failed(*intRes))
      return failure();
    staticSize = parser.getBuilder().getI64IntegerAttr(intVal);
    dynamicSize = std::nullopt;
    return success();
  }

  // Otherwise, parse an operand (dynamic size).
  OpAsmParser::UnresolvedOperand operand;
  if (parser.parseOperand(operand))
    return failure();
  dynamicSize = operand;
  staticSize = parser.getBuilder().getI64IntegerAttr(ShapedType::kDynamic);
  return success();
}

/// Print a size that can be either static or dynamic.
static void printAllocSize(OpAsmPrinter &printer, Operation *op,
                           Value dynamicSize, IntegerAttr staticSize) {
  if (ShapedType::isDynamic(staticSize.getInt())) {
    printer.printOperand(dynamicSize);
    return;
  }
  printer << staticSize.getInt();
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
#include "aster/Dialect/AMDGCN/IR/DS.cpp.inc"
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
  initializeInstAttr();
  addInterfaces<AMDGCNInlinerInterface>();
}

Attribute AMDGCNDialect::parseAttribute(DialectAsmParser &parser,
                                        Type type) const {
  return parseDialectAttributes<
#define GET_ATTRDEF_LIST
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.cpp.inc"
      ,
#define GET_ATTRDEF_LIST
#include "aster/Dialect/AMDGCN/IR/InstAttr.cpp.inc"
      >(parser, type, getDialectNamespace());
}

void AMDGCNDialect::printAttribute(Attribute attr,
                                   DialectAsmPrinter &os) const {
  return printDialectAttributes<
#define GET_ATTRDEF_LIST
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.cpp.inc"
      ,
#define GET_ATTRDEF_LIST
#include "aster/Dialect/AMDGCN/IR/InstAttr.cpp.inc"
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

RegisterKind
mlir::aster::amdgcn::getRegisterKind(AMDGCNRegisterTypeInterface type) {
  if (auto rTy = dyn_cast<AMDGCNRegisterTypeInterface>(type))
    return rTy.getRegisterKind();
  return RegisterKind::Unknown;
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

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

LogicalResult MakeRegisterRangeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Fail if there are no operands.
  if (operands.empty()) {
    if (location)
      mlir::emitError(*location) << "expected at least one operand";
    return failure();
  }

  // Fail if any of the types is a register range.
  if (llvm::any_of(TypeRange(operands), [](Type type) {
        return cast<RegisterTypeInterface>(type).isRegisterRange();
      })) {
    if (location)
      mlir::emitError(*location) << "expected all types to be single registers";
    return failure();
  }

  // Fail if the types are not all of the same kind.
  auto fTy = cast<AMDGCNRegisterTypeInterface>(operands[0].getType());
  if (llvm::any_of(TypeRange(operands), [&](Type type) {
        auto oTy = cast<AMDGCNRegisterTypeInterface>(type);
        return fTy.getRegisterKind() != oTy.getRegisterKind() ||
               fTy.getSemantics() != oTy.getSemantics();
      })) {
    if (location) {
      mlir::emitError(*location)
          << "expected all operand types to be of the same kind";
    }
    return failure();
  }

  // Create the appropriate register range type.
  auto makeRange = [&](RegisterRange range) -> Type {
    switch (getRegisterKind(fTy)) {
    case RegisterKind::SGPR:
      return SGPRType::get(context, range);
    case RegisterKind::VGPR:
      return VGPRType::get(context, range);
    case RegisterKind::AGPR:
      return AGPRType::get(context, range);
    default:
      llvm_unreachable("nyi register kind");
    }
  };

  if (!fTy.hasAllocatedSemantics()) {
    inferredReturnTypes.push_back(
        makeRange(RegisterRange(fTy.getAsRange().begin(), operands.size())));
    return success();
  }

  // Collect unique registers and find upper bound.
  llvm::SmallDenseSet<int> uniqueRegs;
  int ub = -1;

  for (Type type : TypeRange(operands)) {
    int reg = cast<AMDGCNRegisterTypeInterface>(type)
                  .getAsRange()
                  .begin()
                  .getRegister();
    if (!uniqueRegs.insert(reg).second) {
      // Duplicate register found.
      if (location)
        mlir::emitError(*location) << "duplicate register found: " << reg;
      return failure();
    }
    ub = std::max(ub, reg);
  }

  assert(ub >= 0 && "ub should have been set");
  // Check for missing registers in the range.
  int lb = ub - uniqueRegs.size() + 1;
  for (int regNum = lb; regNum <= ub; ++regNum) {
    if (!uniqueRegs.contains(regNum)) {
      // Missing register found.
      if (location)
        mlir::emitError(*location) << "missing register in range: " << regNum;
      return failure();
    }
  }
  inferredReturnTypes.push_back(
      makeRange(RegisterRange(Register(lb), operands.size())));
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
  auto rangeType = cast<AMDGCNRegisterTypeInterface>(inputType);

  // Get the range information.
  RegisterRange range = rangeType.getAsRange();
  int size = range.size();

  // Create a function to make individual register types.
  auto makeRegister = [&](Register reg) -> Type {
    return rangeType.cloneRegisterType(reg);
  };

  // If the range doesn't have allocated semantics, create individual registers.
  if (!rangeType.hasAllocatedSemantics()) {
    for (int i = 0; i < size; ++i)
      inferredReturnTypes.push_back(makeRegister(range.begin()));
    return success();
  }

  // Otherwise, create individual registers from the range.
  int begin = range.begin().getRegister();
  for (int i = 0; i < size; ++i) {
    inferredReturnTypes.push_back(makeRegister(Register(begin + i)));
  }
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
/// `excludeAttrNames` optionally specifies named attributes whose nested
/// types should be skipped during verification.
static LogicalResult verifyNormalFormsRegions(
    Operation *op, ArrayAttr normalFormsAttr,
    const DenseSet<StringAttr> *excludeAttrNames = nullptr) {
  if (!normalFormsAttr || normalFormsAttr.empty())
    return success();
  for (Attribute attr : normalFormsAttr) {
    auto nf = cast<normalform::NormalFormAttrInterface>(attr);
    if (failed(normalform::verifyNormalForm(op, nf, /*emitDiagnostics=*/true,
                                            excludeAttrNames)))
      return failure();
  }
  return success();
}

/// Add normal forms to an operation's normal_forms attribute using set
/// semantics. Returns true if the attribute was changed.
static bool
addNormalFormsImpl(Operation *op, StringRef attrName, ArrayAttr currentAttr,
                   ArrayRef<normalform::NormalFormAttrInterface> nfs,
                   function_ref<void(ArrayAttr)> setter) {
  if (nfs.empty())
    return false;

  SetVector<Attribute> nfSet;
  if (currentAttr)
    nfSet.insert_range(currentAttr.getValue());

  bool changed = false;
  for (normalform::NormalFormAttrInterface nf : nfs)
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
                      ArrayRef<normalform::NormalFormAttrInterface> nfs,
                      function_ref<void(ArrayAttr)> setter) {
  if (nfs.empty() || !currentAttr || currentAttr.empty())
    return false;

  SetVector<Attribute> nfSet;
  nfSet.insert_range(currentAttr.getValue());

  bool changed = false;
  for (normalform::NormalFormAttrInterface nf : nfs)
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
  return verifyNormalFormsRegions(getOperation(), getNormalFormsAttr());
}

bool amdgcn::ModuleOp::addNormalForms(
    ArrayRef<normalform::NormalFormAttrInterface> normalForms) {
  return addNormalFormsImpl(getOperation(), getNormalFormsAttrName(),
                            getNormalFormsAttr(), normalForms,
                            [&](ArrayAttr attr) { setNormalFormsAttr(attr); });
}

bool amdgcn::ModuleOp::removeNormalForms(
    ArrayRef<normalform::NormalFormAttrInterface> normalForms) {
  return removeNormalFormsImpl(
      getOperation(), getNormalFormsAttr(), normalForms,
      [&](ArrayAttr attr) { setNormalFormsAttr(attr); });
}

//===----------------------------------------------------------------------===//
// KernelOp Normal Forms
//===----------------------------------------------------------------------===//

LogicalResult KernelOp::verifyRegions() {
  // Exclude 'arguments' attribute from normal form type walking: kernel
  // argument attrs (by_val_arg, buffer_arg) contain ABI metadata types,
  // not computational register types in the kernel body.
  DenseSet<StringAttr> excludeAttrs;
  excludeAttrs.insert(getArgumentsAttrName());
  return verifyNormalFormsRegions(getOperation(), getNormalFormsAttr(),
                                  &excludeAttrs);
}

bool KernelOp::addNormalForms(
    ArrayRef<normalform::NormalFormAttrInterface> normalForms) {
  return addNormalFormsImpl(getOperation(), getNormalFormsAttrName(),
                            getNormalFormsAttr(), normalForms,
                            [&](ArrayAttr attr) { setNormalFormsAttr(attr); });
}

bool KernelOp::removeNormalForms(
    ArrayRef<normalform::NormalFormAttrInterface> normalForms) {
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

#include "aster/Dialect/AMDGCN/IR/Interfaces/MemoryOpInterfaces.cpp.inc"

#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNRegisterTypeInterface.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.cpp.inc"

#define GET_OP_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.cpp.inc"

#define GET_OP_CLASSES
#include "aster/Dialect/AMDGCN/IR/DS.cpp.inc"

#define AMDGCN_GEN_INST_METHODS
#include "aster/Dialect/AMDGCN/IR/DSInst.cpp.inc"

#define GET_OP_CLASSES
#include "aster/Dialect/AMDGCN/IR/SMem.cpp.inc"

#define AMDGCN_GEN_INST_METHODS
#include "aster/Dialect/AMDGCN/IR/SMemInst.cpp.inc"

#define GET_OP_CLASSES
#include "aster/Dialect/AMDGCN/IR/SOP.cpp.inc"

#define AMDGCN_GEN_INST_METHODS
#include "aster/Dialect/AMDGCN/IR/SOPInst.cpp.inc"

#define GET_OP_CLASSES
#include "aster/Dialect/AMDGCN/IR/VMem.cpp.inc"

#define AMDGCN_GEN_INST_METHODS
#include "aster/Dialect/AMDGCN/IR/VMemInst.cpp.inc"

#define GET_OP_CLASSES
#include "aster/Dialect/AMDGCN/IR/VOP.cpp.inc"

#define AMDGCN_GEN_INST_METHODS
#include "aster/Dialect/AMDGCN/IR/VOPInst.cpp.inc"
