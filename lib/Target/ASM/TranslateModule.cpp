//===- TranslateModule.cpp - Export ASM -------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements translation utilities for the export ASM target.
//
//===----------------------------------------------------------------------===//

#include "aster/Target/ASM/TranslateModule.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Target/ASM/AsmPrinter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::amdgcn::target;

#include "AMDGCNAsmPrinter.cpp.inc"

/// Prints the given instruction using the AsmPrinter.
static llvm::LogicalResult printInstruction(amdgcn::AsmPrinter &printer,
                                            AMDGCNInstOpInterface op) {
  OpCode opcode = op.getOpcodeAttr().getValue();
  assert(opcode != OpCode::Invalid && "invalid opcode in instruction");
  if (_instPrinters.size() <= static_cast<size_t>(opcode) ||
      !_instPrinters[static_cast<size_t>(opcode)]) {
    return op.emitError() << "no printer defined for opcode: "
                          << stringifyOpCode(opcode);
  }
  return _instPrinters[static_cast<size_t>(opcode)](opcode, printer, op);
}

namespace {
/// Structure to hold register usage.
struct RegisterUsage {
  int32_t maxVGPR = 0;
  int32_t maxSGPR = 0;
  int32_t maxAGPR = 0;
  int32_t countVGPR = 0;
  int32_t countSGPR = 0;
  int32_t countAGPR = 0;
  /// Compute register usage for the given kernel.
  static FailureOr<RegisterUsage> countKernelRegisters(KernelOp kernel);

private:
  /// Update the maximum register number for the given kind.
  void updateRegisters(AMDGCNRegisterTypeInterface kind, int32_t regNum,
                       DenseSet<int16_t> &usedVGPRs,
                       DenseSet<int16_t> &usedSGPRs,
                       DenseSet<int16_t> &usedAGPRs);
};

/// Implementation class for translating an AMDGCN module to assembly.
struct TranslateModuleImpl {
  TranslateModuleImpl(amdgcn::ModuleOp module, llvm::raw_ostream &os,
                      bool debugPrint)
      : module(module), os(os), debugPrint(debugPrint) {}

  /// Translate the module to assembly.
  LogicalResult translate();

private:
  /// Emit the given kernel.
  LogicalResult emitKernel(KernelOp kernel, size_t kernelIndex);
  /// Emit the epilogue for the given kernel.
  LogicalResult emitKernelEpilogue(KernelOp kernel, size_t kernelIndex);
  /// Emit the prologue for the given kernel.
  LogicalResult emitKernelPrologue(KernelOp kernel, size_t kernelIndex);
  /// Emit the given block.
  LogicalResult emitBlock(amdgcn::AsmPrinter &printer, Block *block);
  /// Emit the given operation.
  LogicalResult emitOperation(amdgcn::AsmPrinter &printer, Operation *op);
  /// Emit a kernel argument. Returns the argument offset after this argument.
  int32_t emitKernelArgument(KernelArgAttrInterface arg, int32_t offset);
  /// Emit metadata for all kernels.
  LogicalResult emitMetadata();
  /// Emit the given kernel metadata.
  LogicalResult emitKernelMetadata(KernelOp kernel, RegisterUsage &regInfo,
                                   size_t kernelIndex);
  /// Print a field with a key and value, skipping if the value is the default.
  void printField(StringRef key, int32_t value, int32_t defaultValue);
  void printField(StringRef key, StringRef value, StringRef defaultValue);
  /// Print a field with a key and value.
  void printField(StringRef key, int32_t value);
  void printField(StringRef key, StringRef value);
  /// The module being translated.
  amdgcn::ModuleOp module;
  /// The output stream.
  raw_indented_ostream os;
  /// Enable debug printing.
  bool debugPrint;
  /// The register usage information for each kernel.
  SmallVector<std::pair<KernelOp, RegisterUsage>> kernels;
};

/// Helper class to emit YAML lists with proper indentation.
struct YAMLList {
  YAMLList(raw_indented_ostream &s) : os(s) { os.indent(); }
  ~YAMLList() {
    os.unindent();
    if (numElems > 0)
      os.unindent();
  }

  raw_ostream &emit() {
    if (numElems++ == 0) {
      os << "- ";
      os.indent();
    }
    return os;
  }
  void printArrayField(StringRef key, ArrayRef<int32_t> value,
                       ArrayRef<int32_t> defaultValue) {
    if (value == defaultValue)
      return;
    emit() << key << ": ";
    llvm::interleaveComma(value, os);
    os << "\n";
  }
  void printField(StringRef key, int32_t value, int32_t defaultValue) {
    if (value == defaultValue)
      return;
    emit() << key << ": " << value << "\n";
  }
  void printField(StringRef key, StringRef value, StringRef defaultValue) {
    if (value == defaultValue)
      return;
    emit() << key << ": " << value << "\n";
  }
  void printField(StringRef key, int32_t value) {
    emit() << key << ": " << value << "\n";
  }
  void printField(StringRef key, StringRef value) {
    emit() << key << ": " << value << "\n";
  }
  void printField(StringRef key) { emit() << key << "\n"; }

private:
  raw_indented_ostream &os;
  int32_t numElems = 0;
};
} // namespace

//===----------------------------------------------------------------------===//
// RegisterUsage
//===----------------------------------------------------------------------===//

void RegisterUsage::updateRegisters(AMDGCNRegisterTypeInterface ty,
                                    int32_t regNum,
                                    DenseSet<int16_t> &usedVGPRs,
                                    DenseSet<int16_t> &usedSGPRs,
                                    DenseSet<int16_t> &usedAGPRs) {
  RegisterKind kind = ty.getRegisterKind();
  switch (kind) {
  case RegisterKind::VGPR:
    maxVGPR = std::max(maxVGPR, regNum + 1);
    usedVGPRs.insert(regNum);
    return;
  case RegisterKind::SGPR:
    maxSGPR = std::max(maxSGPR, regNum + 1);
    usedSGPRs.insert(regNum);
    return;
  case RegisterKind::AGPR:
    maxAGPR = std::max(maxAGPR, regNum + 1);
    usedAGPRs.insert(regNum);
    return;
  case RegisterKind::SREG: {
    SregKind sregKind = cast<SREGType>(ty).getKind();
    switch (sregKind) {
    case SregKind::Scc:
      return;
    }
    llvm_unreachable("nyi sreg kind");
    return;
  }
  default:
    llvm_unreachable("nyi register kind");
  }
}

FailureOr<RegisterUsage> RegisterUsage::countKernelRegisters(KernelOp kernel) {
  RegisterUsage counts;
  DenseSet<int16_t> usedVGPRs;
  DenseSet<int16_t> usedSGPRs;
  DenseSet<int16_t> usedAGPRs;
  auto result = kernel.walk([&](AllocaOp op) -> WalkResult {
    AMDGCNRegisterTypeInterface type = op.getType();
    if (type.isRelocatable()) {
      op->emitError() << "expected non-relocatable registers";
      return WalkResult::interrupt();
    }
    RegisterRange range = type.getAsRange();
    assert(range.size() == 1 && "expected single register");
    counts.updateRegisters(type, range.begin().getRegister(), usedVGPRs,
                           usedSGPRs, usedAGPRs);
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  counts.countSGPR = usedSGPRs.size();
  counts.countVGPR = usedVGPRs.size();
  counts.countAGPR = usedAGPRs.size();
  return counts;
}

//===----------------------------------------------------------------------===//
// TranslateModuleImpl
//===----------------------------------------------------------------------===//

void TranslateModuleImpl::printField(StringRef key, int32_t value,
                                     int32_t defaultValue) {
  if (!debugPrint && value == defaultValue)
    return;
  os << key << " " << value << "\n";
}

void TranslateModuleImpl::printField(StringRef key, StringRef value,
                                     StringRef defaultValue) {
  if (!debugPrint && value == defaultValue)
    return;
  os << key << " " << value << "\n";
}

void TranslateModuleImpl::printField(StringRef key, int32_t value) {
  os << key << " " << value << "\n";
}

void TranslateModuleImpl::printField(StringRef key, StringRef value) {
  os << key << " " << value << "\n";
}

int32_t TranslateModuleImpl::emitKernelArgument(KernelArgAttrInterface arg,
                                                int32_t offset) {
  YAMLList yOs(os);
  if (AccessKind access = arg.getAccess(); access != AccessKind::Unspecified)
    yOs.printField(".access", stringifyAccessKind(access), "");
  if (AccessKind actualAccess = arg.getActualAccess();
      actualAccess != AccessKind::Unspecified)
    yOs.printField(".actual_access", stringifyAccessKind(actualAccess), "");
  if (AddressSpaceKind addrSpace = arg.getAddressSpace();
      addrSpace != AddressSpaceKind::Invalid)
    yOs.printField(".address_space", stringifyAddressSpaceKind(addrSpace), "");
  // Flags
  KernelArgumentFlags flags = arg.getFlags();
  if ((flags & KernelArgumentFlags::Const) != KernelArgumentFlags::None)
    yOs.printField(".is_const", "true", "false");
  if ((flags & KernelArgumentFlags::Pipe) != KernelArgumentFlags::None)
    yOs.printField(".is_pipe", "true", "false");
  if ((flags & KernelArgumentFlags::Restrict) != KernelArgumentFlags::None)
    yOs.printField(".is_restrict", "true", "false");
  if ((flags & KernelArgumentFlags::Volatile) != KernelArgumentFlags::None)
    yOs.printField(".is_volatile", "true", "false");
  yOs.printField(".name", arg.getName(), "");
  yOs.printField(".offset", offset);
  if (std::optional<uint32_t> pointeeAlign = arg.getPointeeAlign())
    yOs.printField(".pointee_align", *pointeeAlign, 0);
  yOs.printField(".size", arg.getSize());
  yOs.printField(".value_kind", stringifyArgumentValueKind(arg.getValueKind()));

  // Compute the new offset
  int32_t size = arg.getSize();
  // Align offset to the argument's alignment if specified
  std::optional<uint32_t> alignment = arg.getAlignment();
  if (alignment.has_value())
    return offset + (size + *alignment - 1) & ~(*alignment - 1);
  return offset + size;
}

LogicalResult TranslateModuleImpl::emitBlock(amdgcn::AsmPrinter &printer,
                                             Block *block) {
  printer.getStream() << printer.getBranchLabel(block) << ":\n";
  os.indent();
  for (Operation &op : *block) {
    if (failed(emitOperation(printer, &op)))
      return failure();
  }
  os.unindent();
  return success();
}

LogicalResult TranslateModuleImpl::emitOperation(amdgcn::AsmPrinter &printer,
                                                 Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<amdgcn::BranchOp>([&](amdgcn::BranchOp branchOp) {
        // If the branch label is the next block, don't emit the branch and let
        // it fallthrough.
        if (branchOp.getDest() == branchOp->getBlock()->getNextNode()) {
          printer.getStream()
              << "; fallthrough: " << printer.getBranchLabel(branchOp.getDest())
              << "\n";
          return success();
        }
        return printInstruction(printer, branchOp);
      })
      .Case<amdgcn::CBranchOp>([&](amdgcn::CBranchOp cbranchOp)
                                   -> LogicalResult {
        // The fallthrough branch has to be the next block, otherwise the asm
        // emission is invalid.
        if (cbranchOp.getFallthrough() !=
            cbranchOp->getBlock()->getNextNode()) {
          return cbranchOp->emitError()
                 << "conditional branch fallthrough target must be the next "
                    "block";
        }
        if (failed(printInstruction(printer, cbranchOp)))
          return failure();
        printer.getStream()
            << "; fallthrough: "
            << printer.getBranchLabel(cbranchOp.getFallthrough()) << "\n";
        return success();
      })
      .Case<AMDGCNInstOpInterface>([&](AMDGCNInstOpInterface op) {
        return printInstruction(printer, op);
      })
      .Case<AllocaOp, MakeRegisterRangeOp, SplitRegisterRangeOp,
            arith::ConstantIntOp>([&](auto op) {
        // Alloca, MakeRegisterRange and arith::ConstantInt operations are not
        // printed, they are implied by the register or attribute usage.
        if (debugPrint)
          printer.getStream() << "; " << *op << "\n";
        return success();
      })
      .Case<KernelOp>([&](KernelOp kernelOp) { return success(); })
      .Default([&](Operation *defaultOp) {
        return defaultOp->emitError()
               << "cannot be translated to AMDGPU assembly";
      });
}

LogicalResult TranslateModuleImpl::emitKernelPrologue(KernelOp kernel,
                                                      size_t kernelIndex) {
  os.indent();
  os << ".text\n";
  os << ".globl " << kernel.getName() << "\n";
  os << ".p2align 8\n";
  os << ".type " << kernel.getName() << ",@function\n";
  os.unindent();
  return success();
}

LogicalResult TranslateModuleImpl::emitKernel(KernelOp kernel,
                                              size_t kernelIndex) {
  amdgcn::AsmPrinter printer(os);

  // Emit prologue
  if (failed(emitKernelPrologue(kernel, kernelIndex)))
    return failure();

  // TODO: Reorder blocks to get better fallthrough paths instead of having
  // explicit branching.

  // Emit instructions
  WalkResult result = kernel.walk<WalkOrder::PreOrder>([&](Block *block) {
    if (failed(emitBlock(printer, block)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  // Emit epilogue
  if (failed(emitKernelEpilogue(kernel, kernelIndex)))
    return failure();

  os << "\n";
  return success();
}

LogicalResult TranslateModuleImpl::emitKernelEpilogue(KernelOp kernel,
                                                      size_t kernelIndex) {
  RegisterUsage regUsage = kernels[kernelIndex].second;

  Target target = module.getTarget();
  bool hasAGPR = (target == Target::GFX940 || target == Target::GFX942);

  // Calculate kernarg segment information
  KernelArgSegmentInfo argInfo = KernelArgSegmentInfo::get(kernel);

  // Calculate the number of user SGPRs needed
  // TODO: Support argument preloading to SGPRs
  int32_t userSGPRCount = 0;
  // Values taken from:
  // https://llvm.org/docs/AMDGPUUsage.html#initial-kernel-execution-state
  // TODO: Support the other options.
  userSGPRCount += kernel.getEnablePrivateSegmentBuffer() ? 4 : 0;
  userSGPRCount += kernel.getEnableDispatchPtr() ? 2 : 0;
  userSGPRCount += kernel.getEnableKernargSegmentPtr() ? 2 : 0;

  os.indent();

  os << ".section .rodata,\"a\",@progbits\n";
  os << ".p2align 6, 0x0\n";

  // Emit .amdhsa metadata section
  os << ".amdhsa_kernel " << kernel.getName() << "\n";
  os.indent();

  printField(".amdhsa_group_segment_fixed_size", kernel.getSharedMemorySize(),
             0);
  printField(".amdhsa_private_segment_fixed_size",
             kernel.getPrivateMemorySize(), 0);
  printField(".amdhsa_kernarg_size", argInfo.size, 0);
  printField(".amdhsa_user_sgpr_count", userSGPRCount);
  printField(".amdhsa_user_sgpr_private_segment_buffer",
             kernel.getEnablePrivateSegmentBuffer(), false);
  printField(".amdhsa_user_sgpr_dispatch_ptr", kernel.getEnableDispatchPtr(),
             false);
  printField(".amdhsa_user_sgpr_kernarg_segment_ptr",
             kernel.getEnableKernargSegmentPtr(), false);
  // TODO: FIXME @fabianmcg
  // printField(".amdhsa_wavefront_size32", kernel.getWavefrontSize32(), false);
  printField(".amdhsa_system_sgpr_workgroup_id_x",
             kernel.getEnableWorkgroupIdX(), true);
  printField(".amdhsa_system_sgpr_workgroup_id_y",
             kernel.getEnableWorkgroupIdY(), false);
  printField(".amdhsa_system_sgpr_workgroup_id_z",
             kernel.getEnableWorkgroupIdZ(), false);
  printField(".amdhsa_system_vgpr_workitem_id",
             static_cast<int32_t>(kernel.getWorkitemIdMode()),
             static_cast<int32_t>(WorkitemIDMode::X));

  os << ".amdhsa_next_free_vgpr " << regUsage.maxVGPR << "\n";
  os << ".amdhsa_next_free_sgpr " << regUsage.maxSGPR << "\n";
  if (hasAGPR) {
    unsigned accumOffset =
        regUsage.maxVGPR > 0 ? ((regUsage.maxVGPR + 3) & ~3) : 4;
    os << ".amdhsa_accum_offset " << accumOffset << "\n";
  }

  printField(".amdhsa_float_round_mode_32",
             static_cast<int32_t>(kernel.getF32RoundMode()),
             static_cast<int32_t>(FloatRoundMode::NearEven));
  printField(".amdhsa_float_round_mode_16_64",
             static_cast<int32_t>(kernel.getF16F64RoundMode()),
             static_cast<int32_t>(FloatRoundMode::NearEven));
  printField(".amdhsa_float_denorm_mode_32",
             static_cast<int32_t>(kernel.getF32DenormMode()),
             static_cast<int32_t>(FloatDenormMode::SrcDst));
  printField(".amdhsa_float_denorm_mode_16_64",
             static_cast<int32_t>(kernel.getF16F64DenormMode()),
             static_cast<int32_t>(FloatDenormMode::None));
  printField(".amdhsa_ieee_mode", kernel.getIeeeMode(), true);
  printField(".amdhsa_exception_fp_ieee_invalid_op",
             kernel.getExceptionFpIeeeInvalidOp(), false);
  printField(".amdhsa_exception_fp_denorm_src",
             kernel.getExceptionFpDenormSrc(), false);
  printField(".amdhsa_exception_fp_ieee_div_zero",
             kernel.getExceptionFpIeeeDivZero(), false);
  printField(".amdhsa_exception_fp_ieee_overflow",
             kernel.getExceptionFpIeeeOverflow(), false);
  printField(".amdhsa_exception_fp_ieee_underflow",
             kernel.getExceptionFpIeeeUnderflow(), false);
  printField(".amdhsa_exception_fp_ieee_inexact",
             kernel.getExceptionFpIeeeInexact(), false);
  printField(".amdhsa_exception_int_div_zero", kernel.getExceptionIntDivZero(),
             false);
  os.unindent();
  os << ".end_amdhsa_kernel\n";
  os << ".text\n";
  os.unindent();
  os << ".Lfunc_end" << kernelIndex << ":\n";
  os.indent();
  os << ".size " << kernel.getName() << ", " << ".Lfunc_end" << kernelIndex
     << "-" << kernel.getName() << "\n";
  os.unindent();
  return success();
}

LogicalResult TranslateModuleImpl::emitKernelMetadata(KernelOp kernel,
                                                      RegisterUsage &regInfo,
                                                      size_t kernelInde) {
  YAMLList yOs(os);
  Target target = module.getTarget();
  bool hasAGPR = (target == Target::GFX940 || target == Target::GFX942);
  if (hasAGPR)
    yOs.printField(".agpr_count", regInfo.countAGPR);

  // Calculate kernarg segment information
  KernelArgSegmentInfo argInfo = KernelArgSegmentInfo::get(kernel);

  // Emit kernel arguments
  if (kernel.getArguments().size() > 0) {
    yOs.printField(".args:");
    for (auto [i, arg] : llvm::enumerate(kernel.getArguments()))
      emitKernelArgument(arg, argInfo.offsets[i]);
  }

  yOs.printField(".group_segment_fixed_size", kernel.getSharedMemorySize());
  yOs.printField(".kernarg_segment_align", argInfo.maxAlignment);
  yOs.printField(".kernarg_segment_size", argInfo.size);
  yOs.printField(".language", "Assembler");
  yOs.printField(".max_flat_workgroup_size", kernel.getMaxFlatWorkgroupSize());
  yOs.printField(".name", kernel.getName());
  yOs.printField(".private_segment_fixed_size", kernel.getPrivateMemorySize());
  yOs.printArrayField(".reqd_workgroup_size", kernel.getReqdWorkgroupSize(),
                      {0, 0, 0});
  yOs.printField(".sgpr_count", regInfo.countSGPR);
  yOs.printField(".sgpr_spill_count", 0);
  yOs.printField(".symbol", (kernel.getName() + ".kd").str());
  yOs.printField(".vgpr_count", regInfo.countVGPR);
  yOs.printField(".vgpr_spill_count", 0);
  yOs.printField(".wavefront_size", kernel.getWavefrontSize32() ? 32 : 64);
  return success();
}

LogicalResult TranslateModuleImpl::emitMetadata() {
  if (kernels.empty())
    return success();
  os.indent();
  os << ".amdgpu_metadata\n";
  os.unindent();
  os << "---\n";
  os << "amdhsa.kernels:\n";
  for (auto [kernelIndex, kernel] : llvm::enumerate(kernels)) {
    if (failed(emitKernelMetadata(kernel.first, kernel.second, kernelIndex)))
      return failure();
  }
  os << "amdgcn_target: amdgcn-amd-amdhsa--"
     << stringifyTarget(module.getTarget()) << "\n";
  os << "amdhsa.version:\n";
  os.indent();
  os << "- 1\n";
  os << "- 2\n";
  os.unindent();
  os << "---\n\n";
  os.indent();
  os << ".end_amdgpu_metadata\n";
  return success();
}

LogicalResult TranslateModuleImpl::translate() {
  StringRef targetName = stringifyTarget(module.getTarget());
  os << "  ; Module: " << module.getName() << "\n";
  os << "  .amdgcn_target \"amdgcn-amd-amdhsa--" << targetName << "\"\n";
  for (auto [kernelIndex, kernel] :
       llvm::enumerate(module.getOps<KernelOp>())) {
    assert(kernelIndex == kernels.size() && "Kernel index mismatch");
    FailureOr<RegisterUsage> regUsage =
        RegisterUsage::countKernelRegisters(kernel);
    if (failed(regUsage))
      return failure();
    kernels.emplace_back(kernel, *regUsage);
    if (failed(emitKernel(kernel, kernelIndex)))
      return failure();
  }
  return emitMetadata();
}

LogicalResult mlir::aster::amdgcn::target::translateModule(
    amdgcn::ModuleOp module, llvm::raw_ostream &os, bool debugPrint) {
  if (failed(runVerifiersOnOp<IsTranslatableOpAttr>(module)))
    return failure();
  return TranslateModuleImpl(module, os, debugPrint).translate();
}
