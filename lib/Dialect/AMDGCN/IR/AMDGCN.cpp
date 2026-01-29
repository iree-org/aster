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
#include "aster/Dialect/AMDGCN/IR/AMDGCNInterfaces.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNVerifiers.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

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

//===----------------------------------------------------------------------===//
// Offset Parsing/Printing
//===----------------------------------------------------------------------===//

static ParseResult
parseOffsets(OpAsmParser &parser,
             std::optional<OpAsmParser::UnresolvedOperand> &uOff,
             std::optional<OpAsmParser::UnresolvedOperand> &dOff,
             std::optional<OpAsmParser::UnresolvedOperand> &cOff) {
  // If there are no offsets, return success.
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseOptionalKeyword("offset")))
    return success();

  // Helper to parse an optional offset.
  auto parseOffset = [&](std::optional<OpAsmParser::UnresolvedOperand> &offset,
                         StringRef prefix) -> OptionalParseResult {
    if (failed(parser.parseOptionalKeyword(prefix)))
      return std::nullopt;
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseLParen() || parser.parseOperand(operand) ||
        parser.parseRParen())
      return failure();
    offset = operand;
    return success();
  };

  // Parse (`u` `(` operand `)` `+`?)?
  {
    OptionalParseResult result = parseOffset(uOff, "u");
    if (result.has_value() && failed(result.value()))
      return failure();
    if (result.has_value() && failed(parser.parseOptionalPlus()))
      return success();
  }

  // Parse (`d` `(` operand `)` `+`?)?
  {
    OptionalParseResult result = parseOffset(dOff, "d");
    if (result.has_value() && failed(result.value()))
      return failure();
    if (result.has_value() && failed(parser.parseOptionalPlus()))
      return success();
  }

  // Parse (`c` `(` operand `)` `+`?)?
  {
    OptionalParseResult result = parseOffset(cOff, "c");
    if (result.has_value() && failed(result.value()))
      return failure();
  }
  if (!uOff.has_value() && !dOff.has_value() && !cOff.has_value())
    return parser.emitError(loc, "expected at least one offset operand");
  return success();
}

static void printOffsets(OpAsmPrinter &printer, Operation *, Value uOff,
                         Value dOff, Value cOff) {
  if (dOff == nullptr && uOff == nullptr && cOff == nullptr)
    return;
  printer << " offset ";
  bool first = true;
  auto printOperand = [&](Value operand, StringRef prefix) {
    if (!operand)
      return;
    if (!first)
      printer << " + ";
    printer << prefix << "(";
    printer.printOperand(operand);
    printer << ")";
    first = false;
  };
  printOperand(uOff, "u");
  printOperand(dOff, "d");
  printOperand(cOff, "c");
}

static ParseResult parseOutTypes(OpAsmParser &parser, Type &dpsType,
                                 Type &resultType) {
  if (parser.parseKeyword("dps") || parser.parseLParen() ||
      parser.parseType(resultType) || parser.parseRParen())
    return failure();
  dpsType = resultType;
  return success();
}

static void printOutTypes(OpAsmPrinter &printer, Operation *, Type dpsType,
                          Type resultType) {
  printer << "dps(";
  printer << resultType;
  printer << ")";
}

static ParseResult
parseOffsetTypes(OpAsmParser &parser,
                 std::optional<OpAsmParser::UnresolvedOperand> &uOff,
                 std::optional<OpAsmParser::UnresolvedOperand> &dOff,
                 std::optional<OpAsmParser::UnresolvedOperand> &cOff,
                 Type &uOffTy, Type &dOffTy, Type &cOffTy) {
  // Helper to parse an optional type.
  auto parseType = [&](Type &type, bool present) -> ParseResult {
    if (!present)
      return success();
    if (parser.parseComma())
      return failure();
    return parser.parseType(type);
  };

  if (parseType(uOffTy, uOff.has_value()) ||
      parseType(dOffTy, dOff.has_value()) ||
      parseType(cOffTy, cOff.has_value()))
    return failure();
  return success();
}

static void printOffsetTypes(OpAsmPrinter &printer, Operation *, Value uOff,
                             Value dOff, Value cOff, Type uOffTy, Type dOffTy,
                             Type cOffTy) {
  if (uOff)
    printer << ", " << uOffTy;
  if (dOff)
    printer << ", " << dOffTy;
  if (cOff)
    printer << ", " << cOffTy;
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
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.cpp.inc"
      >();
  initializeAttributes();
  addInterfaces<AMDGCNInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

RegisterKind
mlir::aster::amdgcn::getRegisterKind(AMDGCNRegisterTypeInterface type) {
  if (auto rTy = dyn_cast<AMDGCNRegisterTypeInterface>(type))
    return rTy.getRegisterKind();
  return RegisterKind::Unknown;
}

Speculation::Speculatability
mlir::aster::amdgcn::getInstSpeculatability(InstOpInterface op) {
  if (!op.isRegAllocated())
    return Speculation::Speculatability::Speculatable;
  return Speculation::Speculatability::NotSpeculatable;
}

void mlir::aster::amdgcn::getInstEffects(
    InstOpInterface op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!op.isRegAllocated())
    return;

  // Helper to add effects for a register type with specific resources
  auto addEffectsForRegister = [&](Type type, MemoryEffects::Effect *effect) {
    auto regType = dyn_cast<AMDGCNRegisterTypeInterface>(type);
    if (!regType || regType.isRelocatable())
      return;

    RegisterRange range = regType.getAsRange();
    RegisterKind kind = regType.getRegisterKind();

    // For non-relocatable registers, get the register numbers
    int size = range.size();

    // Add effects for each register in the range
    for (int i = 0; i < size; ++i) {
      SideEffects::Resource *resource = nullptr;

      // Get the type for this specific register
      switch (kind) {
      case RegisterKind::SGPR:
        resource = SGPRResource::get();
        break;
      case RegisterKind::VGPR:
        resource = VGPRResource::get();
        break;
      case RegisterKind::AGPR:
        resource = AGPRResource::get();
        break;
      case RegisterKind::SREG:
        resource = SREGResource::get();
        break;
      default:
        llvm_unreachable("nyi register kind");
      }

      if (resource)
        effects.emplace_back(effect, resource);
    }
  };

  // Add write effects for outputs
  for (OpResult res : op.getInstResults()) {
    addEffectsForRegister(res.getType(), MemoryEffects::Write::get());
  }

  // Add read effects for inputs
  for (OpOperand &in : op.getInstInsMutable()) {
    addEffectsForRegister(in.get().getType(), MemoryEffects::Read::get());
  }
}

MemoryInstructionKind
mlir::aster::amdgcn::getMemoryInstructionKind(OpCode opCode) {
  switch (opCode) {
  case OpCode::DS_READ_B32:
  case OpCode::DS_READ_B64:
  case OpCode::DS_READ_B96:
  case OpCode::DS_READ_B128:
  case OpCode::DS_WRITE_B32:
  case OpCode::DS_WRITE_B64:
  case OpCode::DS_WRITE_B96:
  case OpCode::DS_WRITE_B128:
    return MemoryInstructionKind::Shared;
  case OpCode::S_LOAD_DWORD:
  case OpCode::S_LOAD_DWORDX2:
  case OpCode::S_LOAD_DWORDX4:
  case OpCode::S_LOAD_DWORDX8:
  case OpCode::S_LOAD_DWORDX16:
  case OpCode::S_STORE_DWORD:
  case OpCode::S_STORE_DWORDX2:
  case OpCode::S_STORE_DWORDX4:
    return MemoryInstructionKind::Constant;
  case OpCode::GLOBAL_LOAD_DWORD:
  case OpCode::GLOBAL_LOAD_DWORDX2:
  case OpCode::GLOBAL_LOAD_DWORDX3:
  case OpCode::GLOBAL_LOAD_DWORDX4:
  case OpCode::GLOBAL_STORE_DWORD:
  case OpCode::GLOBAL_STORE_DWORDX2:
  case OpCode::GLOBAL_STORE_DWORDX3:
  case OpCode::GLOBAL_STORE_DWORDX4:
    return MemoryInstructionKind::Flat;
  default:
    return MemoryInstructionKind::None;
  }
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

Speculation::Speculatability AllocaOp::getSpeculatability() {
  if (!getType().isRelocatable())
    return Speculation::Speculatability::Speculatable;
  return Speculation::Speculatability::NotSpeculatable;
}

void AllocaOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!getType().isRelocatable())
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
// LoadOp
//===----------------------------------------------------------------------===//

static Type getTokType(MLIRContext *context, OpCode code, bool isRead) {
  MemoryInstructionKind kind =
      mlir::aster::amdgcn::getMemoryInstructionKind(code);
  return isRead ? cast<Type>(ReadTokenType::get(context, kind))
                : cast<Type>(WriteTokenType::get(context, kind));
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   OpCode opcode, Value dst, Value addr, Value uniform_offset,
                   Value dynamic_offset, Value constant_offset) {
  auto &props = odsState.getOrAddProperties<Properties>();
  props.setOpcode(InstAttr::get(odsBuilder.getContext(), opcode));
  odsState.addOperands({dst, addr});
  if (uniform_offset)
    odsState.addOperands({uniform_offset});
  if (dynamic_offset)
    odsState.addOperands({dynamic_offset});
  if (constant_offset)
    odsState.addOperands({constant_offset});
  props.operandSegmentSizes =
      std::array<int32_t, 5>({1, 1, uniform_offset ? 1 : 0,
                              dynamic_offset ? 1 : 0, constant_offset ? 1 : 0});
  LogicalResult res = inferReturnTypes(
      odsBuilder.getContext(), odsState.location, odsState.operands,
      odsState.attributes.getDictionary(odsBuilder.getContext()), &props,
      odsState.regions, odsState.types);
  assert(succeeded(res) && "unexpected inferReturnTypes failure");
  (void)res;
}

MutableArrayRef<OpOperand> LoadOp::getInstInsMutable() {
  MutableArrayRef<OpOperand> operands =
      getOperation()->getOpOperands().drop_front(1);
  if (getConstantOffset())
    operands = operands.drop_back(1);
  return operands;
}

MutableOperandRange LoadOp::getDependenciesMutable() {
  return MutableOperandRange(getOperation(), 0, 0);
}

Value LoadOp::getOutDependency() { return getToken(); }

LogicalResult
LoadOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  inferredReturnTypes.push_back(getTokType(
      context, properties.as<Properties *>()->getOpcode().getValue(), true));
  return success();
}

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAddrMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestMutable());
  switch (getInstKind()) {
  case MemoryInstructionKind::Flat:
    effects.emplace_back(MemoryEffects::Read::get(),
                         GlobalMemoryResource::get());
    break;
  case MemoryInstructionKind::Shared:
    effects.emplace_back(MemoryEffects::Read::get(), LDSMemoryResource::get());
    break;
  case MemoryInstructionKind::Constant:
    effects.emplace_back(MemoryEffects::Read::get(),
                         GlobalMemoryResource::get());
    break;
  default:
    break;
  }
}

//===----------------------------------------------------------------------===//
// MakeRegisterRangeOp
//===----------------------------------------------------------------------===//

LogicalResult MakeRegisterRangeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
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
               fTy.isRelocatable() != oTy.isRelocatable();
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
      return SGPRRangeType::get(context, range);
    case RegisterKind::VGPR:
      return VGPRRangeType::get(context, range);
    case RegisterKind::AGPR:
      return AGPRRangeType::get(context, range);
    default:
      llvm_unreachable("nyi register kind");
    }
  };

  if (fTy.isRelocatable()) {
    inferredReturnTypes.push_back(
        makeRange(RegisterRange(Register(), operands.size())));
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

//===----------------------------------------------------------------------===//
// SplitRegisterRangeOp
//===----------------------------------------------------------------------===//

LogicalResult SplitRegisterRangeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // There should be exactly one operand.
  if (operands.size() != 1) {
    if (location)
      mlir::emitError(*location) << "expected exactly one operand";
    return failure();
  }

  Type inputType = operands[0].getType();
  auto rangeType = cast<AMDGCNRegisterTypeInterface>(inputType);

  // Fail if the input is not a register range.
  if (!rangeType.isRegisterRange()) {
    if (location)
      mlir::emitError(*location) << "expected register range type";
    return failure();
  }

  // Get the range information.
  RegisterRange range = rangeType.getAsRange();
  int size = range.size();

  // Create a function to make individual register types.
  auto makeRegister = [&](Register reg) -> Type {
    return rangeType.cloneRegisterType(reg);
  };

  // If the range is relocatable, create relocatable individual registers.
  if (range.begin().isRelocatable()) {
    for (int i = 0; i < size; ++i) {
      inferredReturnTypes.push_back(makeRegister(Register()));
    }
    return success();
  }

  // Otherwise, create individual registers from the range.
  int begin = range.begin().getRegister();
  for (int i = 0; i < size; ++i) {
    inferredReturnTypes.push_back(makeRegister(Register(begin + i)));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    OpCode opcode, Value data, Value addr, Value uniform_offset,
                    Value dynamic_offset, Value constant_offset) {
  auto &props = odsState.getOrAddProperties<Properties>();
  props.setOpcode(InstAttr::get(odsBuilder.getContext(), opcode));
  odsState.addOperands({data, addr});
  if (uniform_offset)
    odsState.addOperands({uniform_offset});
  if (dynamic_offset)
    odsState.addOperands({dynamic_offset});
  if (constant_offset)
    odsState.addOperands({constant_offset});
  props.operandSegmentSizes =
      std::array<int32_t, 5>({1, 1, uniform_offset ? 1 : 0,
                              dynamic_offset ? 1 : 0, constant_offset ? 1 : 0});
  LogicalResult res = inferReturnTypes(
      odsBuilder.getContext(), odsState.location, odsState.operands,
      odsState.attributes.getDictionary(odsBuilder.getContext()), &props,
      odsState.regions, odsState.types);
  assert(succeeded(res) && "unexpected inferReturnTypes failure");
  (void)res;
}

MutableArrayRef<OpOperand> StoreOp::getInstInsMutable() {
  MutableArrayRef<OpOperand> operands = getOperation()->getOpOperands();
  if (getConstantOffset())
    operands = operands.drop_back(1);
  return operands;
}

MutableOperandRange StoreOp::getDependenciesMutable() {
  return MutableOperandRange(getOperation(), 0, 0);
}

Value StoreOp::getOutDependency() { return getToken(); }

LogicalResult StoreOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(getTokType(
      context, properties.as<Properties *>()->getOpcode().getValue(), false));
  return success();
}

void StoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getAddrMutable());
  switch (getInstKind()) {
  case MemoryInstructionKind::Flat:
    effects.emplace_back(MemoryEffects::Write::get(),
                         GlobalMemoryResource::get());
    break;
  case MemoryInstructionKind::Shared:
    effects.emplace_back(MemoryEffects::Write::get(), LDSMemoryResource::get());
    break;
  case MemoryInstructionKind::Constant:
    effects.emplace_back(MemoryEffects::Write::get(),
                         GlobalMemoryResource::get());
    break;
  default:
    break;
  }
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

/// Check if a type is an unallocated register (relocatable).
static bool isUnallocatedRegister(Type type) {
  auto regType = dyn_cast<RegisterTypeInterface>(type);
  return regType && regType.isRelocatable();
}

/// Parse the output types for CmpIOp.
/// Format: `dps(type) outs(type)?` or `outs(type, type?)`
static ParseResult
parseCmpOutTypes(OpAsmParser &parser, Type &destType,
                 std::optional<OpAsmParser::UnresolvedOperand> &exec,
                 Type &execType) {
  // Parse either:
  //   dps(destType) outs(execType)?  -- unallocated dest
  //   outs(destType, execType?)      -- allocated dest

  SMLoc loc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("dps"))) {
    // dps(destType) format
    if (parser.parseLParen() || parser.parseType(destType) ||
        parser.parseRParen())
      return failure();

    if (!isUnallocatedRegister(destType)) {
      return parser.emitError(loc)
             << "expected unallocated register type for dps";
    }

    // Optionally parse outs(execType)
    if (succeeded(parser.parseOptionalKeyword("outs"))) {
      if (parser.parseLParen() || parser.parseType(execType) ||
          parser.parseRParen())
        return failure();
    }
    return success();
  }

  // outs(destType, execType?) format
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseType(destType))
    return failure();

  if (isUnallocatedRegister(destType)) {
    return parser.emitError(loc) << "expected allocated register type for outs";
  }

  // Check if exec is present
  if (exec.has_value()) {
    if (parser.parseComma() || parser.parseType(execType))
      return failure();
  }

  if (parser.parseRParen())
    return failure();
  return success();
}

/// Print the output types for CmpIOp.
static void printCmpOutTypes(OpAsmPrinter &printer, Operation *, Type destType,
                             Value exec, Type execType) {
  if (isUnallocatedRegister(destType)) {
    // dps(type) outs(type)? format
    printer << "dps(" << destType << ")";
    if (exec)
      printer << " outs(" << execType << ")";
    return;
  }
  // outs(type, type?) format
  printer << "outs(" << destType;
  if (exec)
    printer << ", " << execType;
  printer << ")";
}

void CmpIOp::build(OpBuilder &builder, OperationState &state, OpCode opcode,
                   Value dest, Value lhs, Value rhs) {
  auto &props = state.getOrAddProperties<Properties>();
  props.setOpcode(InstAttr::get(builder.getContext(), opcode));
  state.addOperands({dest, lhs, rhs});
  LogicalResult res =
      inferReturnTypes(builder.getContext(), state.location, state.operands,
                       state.attributes.getDictionary(builder.getContext()),
                       &props, state.regions, state.types);
  assert(succeeded(res) && "unexpected inferReturnTypes failure");
  (void)res;
}

void CmpIOp::build(OpBuilder &builder, OperationState &state, OpCode opcode,
                   Value dest, Value exec, Value lhs, Value rhs) {
  auto &props = state.getOrAddProperties<Properties>();
  props.setOpcode(InstAttr::get(builder.getContext(), opcode));
  state.addOperands({dest, exec, lhs, rhs});
  LogicalResult res =
      inferReturnTypes(builder.getContext(), state.location, state.operands,
                       state.attributes.getDictionary(builder.getContext()),
                       &props, state.regions, state.types);
  assert(succeeded(res) && "unexpected inferReturnTypes failure");
  (void)res;
}

LogicalResult
CmpIOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  // The dest operand is the first operand.
  Type destType = operands[0].getType();
  // If dest is an unallocated SGPR, return it as the result type.
  if (isUnallocatedRegister(destType) && isa<SGPRType>(destType))
    inferredReturnTypes.push_back(destType);
  return success();
}

void CmpIOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Always write effect on dest.
  effects.emplace_back(MemoryEffects::Write::get(), &getDestMutable());
  // Read effects on inputs.
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  // Write effect on exec if present.
  if (getExec()) {
    MutableOperandRange execRange = getExecMutable();
    effects.emplace_back(
        MemoryEffects::Write::get(),
        &getOperation()->getOpOperand(execRange[0].getOperandNumber()));
  }
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

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrInterfaces.cpp.inc"

#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.cpp.inc"

#include "aster/Dialect/AMDGCN/IR/AMDGCNInstOpInterface.cpp.inc"

#include "aster/Dialect/AMDGCN/IR/AMDGCNTypeInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.cpp.inc"

#define GET_OP_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.cpp.inc"
