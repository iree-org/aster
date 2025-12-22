//===- AMDGCN.cpp - AMDGCN Operations -------------------------------------===//
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
#include "aster/Dialect/AMDGCN/IR/AMDGCNVerifiers.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"

#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

RegisterKind
mlir::aster::amdgcn::getRegisterKind(AMDGCNRegisterTypeInterface type) {
  if (auto rTy = dyn_cast<AMDGCNRegisterTypeInterface>(type))
    return rTy.getRegisterKind();
  return RegisterKind::Unknown;
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
// OpCode Parsing/Printing
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
// CDNA3 GlobalLoadOp Builders
//===----------------------------------------------------------------------===//

void inst::GlobalLoadOp::build(::mlir::OpBuilder &builder,
                               ::mlir::OperationState &state,
                               ::mlir::aster::amdgcn::OpCode opcode,
                               ::mlir::Value vdst, ::mlir::Value addr,
                               ::mlir::Value vgpr_offset, int32_t offset) {
  state.addAttribute("opcode", ::mlir::aster::amdgcn::InstAttr::get(
                                   builder.getContext(), opcode));
  state.addOperands({vdst, addr, vgpr_offset});
  state.addAttribute("offset", builder.getI32IntegerAttr(offset));
  state.addTypes(vdst.getType());
}

void inst::GlobalLoadOp::build(::mlir::OpBuilder &builder,
                               ::mlir::OperationState &state,
                               ::mlir::aster::amdgcn::OpCode opcode,
                               ::mlir::Value vdst, ::mlir::Value addr,
                               int32_t offset) {
  state.addAttribute("opcode", ::mlir::aster::amdgcn::InstAttr::get(
                                   builder.getContext(), opcode));
  state.addOperands({vdst, addr});
  state.addAttribute("offset", builder.getI32IntegerAttr(offset));
  state.addTypes(vdst.getType());
}

//===----------------------------------------------------------------------===//
// CDNA3 GlobalStoreOp Builders
//===----------------------------------------------------------------------===//

void inst::GlobalStoreOp::build(::mlir::OpBuilder &builder,
                                ::mlir::OperationState &state,
                                ::mlir::aster::amdgcn::OpCode opcode,
                                ::mlir::Value data, ::mlir::Value addr,
                                ::mlir::Value vgpr_offset, int32_t offset) {
  state.addAttribute("opcode", ::mlir::aster::amdgcn::InstAttr::get(
                                   builder.getContext(), opcode));
  state.addOperands({data, addr, vgpr_offset});
  state.addAttribute("offset", builder.getI32IntegerAttr(offset));
}

void inst::GlobalStoreOp::build(::mlir::OpBuilder &builder,
                                ::mlir::OperationState &state,
                                ::mlir::aster::amdgcn::OpCode opcode,
                                ::mlir::Value data, ::mlir::Value addr,
                                int32_t offset) {
  state.addAttribute("opcode", ::mlir::aster::amdgcn::InstAttr::get(
                                   builder.getContext(), opcode));
  state.addOperands({data, addr});
  state.addAttribute("offset", builder.getI32IntegerAttr(offset));
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
