//===- LowerToAMDGCN.cpp --------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower a simple copy kernel to AMDGCN ops with relocatable registers
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/ConvertFuncToAMDGCN.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_LOWERTOAMDGCNPASS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::amdgcn::inst;
using namespace mlir::ptr;

/// Convert a func::FuncOp to an amdgcn::KernelOp.
/// Only supports ptr and POD (integer/float) arguments that are translated
/// one-to-one. Index type is not supported. The function must be within an
/// amdgcn::ModuleOp.
FailureOr<amdgcn::KernelOp>
mlir::convertFuncOpToAMDGCNKernel(func::FuncOp funcOp, RewriterBase &rewriter) {
  auto funcTy = dyn_cast<FunctionType>(funcOp.getFunctionType());
  if (funcOp.getBody().empty()) {
    return rewriter.notifyMatchFailure(
        funcOp, "Function body is empty - cannot convert to kernel");
  }

  // Create the kernel op with the same name.
  OpBuilder builder = rewriter;
  auto kernelName = builder.getStringAttr(funcOp.getName());

  // Collect argument types for the kernel region (from function signature)
  SmallVector<Type> kernelArgTypes;
  for (int i = 0, e = static_cast<int>(funcTy.getNumInputs()); i < e; ++i) {
    Type argTy = funcTy.getInput(i);
    kernelArgTypes.push_back(argTy);
  }

  auto kernel = builder.create<amdgcn::KernelOp>(funcOp.getLoc(), kernelName);

  // Inline the function body into the kernel.
  // The function's entry block arguments will become the kernel's first block
  // arguments.
  rewriter.inlineRegionBefore(funcOp.getBody(), kernel.getBodyRegion(),
                              kernel.getBodyRegion().end());

  return kernel;
}

/// Calculate kernarg offset for a kernel argument
static int calculateKernargOffset(Block *entryBlock, int argIndex) {
  int offset = 0;
  for (int i = 0; i < argIndex; ++i) {
    Type argType = entryBlock->getArgument(i).getType();
    int size = 0;
    if (isa<ptr::PtrType>(argType)) {
      size = 8; // 64-bit pointer
    } else if (auto intType = dyn_cast<IntegerType>(argType)) {
      int bitwidth = intType.getWidth();
      size = (bitwidth + 31) / 32 * 4; // Round to 4-byte alignment
    } else if (auto floatType = dyn_cast<FloatType>(argType)) {
      size = floatType.getWidth() / 8; // bytes
    }
    // Kernarg segment is 8-byte aligned
    offset = (offset + size + 7) & ~7;
  }
  return offset;
}

/// Create or get kernarg pointer (uses relocatable registers)
static Value getKernargPointer(OpBuilder &builder, Location loc) {
  auto s0Type = SGPRType::get(builder.getContext(), Register());
  auto s1Type = SGPRType::get(builder.getContext(), Register());
  auto s0Alloca = builder.create<amdgcn::AllocaOp>(loc, s0Type);
  auto s1Alloca = builder.create<amdgcn::AllocaOp>(loc, s1Type);
  auto kernargRangeType =
      SGPRRangeType::get(builder.getContext(), RegisterRange(Register(), 2));
  return builder
      .create<amdgcn::MakeRegisterRangeOp>(loc, kernargRangeType,
                                           ValueRange{s0Alloca, s1Alloca})
      .getResult();
}

/// Load pointer from kernarg segment (without moving to VGPRs yet)
/// Returns info needed to do the move later
/// Uses relocatable registers
static TypedValue<SGPRRangeType> loadPointerToSGPRs(OpBuilder &builder,
                                                    Location loc,
                                                    Value kernargPtr,
                                                    int kernargOffset) {
  // Allocate SGPRs for pointer (relocatable)
  auto sType1 = SGPRType::get(builder.getContext(), Register());
  auto sType2 = SGPRType::get(builder.getContext(), Register());
  auto s1Alloca = builder.create<amdgcn::AllocaOp>(loc, sType1);
  auto s2Alloca = builder.create<amdgcn::AllocaOp>(loc, sType2);
  auto ptrRangeType =
      SGPRRangeType::get(builder.getContext(), RegisterRange(Register(), 2));
  auto ptrRange = builder.create<amdgcn::MakeRegisterRangeOp>(
      loc, ptrRangeType, ValueRange{s1Alloca, s2Alloca});

  // Load pointer from kernarg segment
  auto smemLoadOpcode = OpCode::S_LOAD_DWORDX2;
  auto smemLoad = builder.create<inst::SMEMLoadOp>(
      loc, smemLoadOpcode, ptrRange.getResult(), kernargPtr, kernargOffset);
  return smemLoad.getResult();
}

/// Convert an SGPR range to a VGPR range by splitting and moving registers.
/// If the input is already a VGPR range, returns it unchanged.
/// Returns nullptr if the input is not an SGPR range.
/// Uses relocatable registers.
static Value convertSGPRRangeToVGPRRange(OpBuilder &builder, Location loc,
                                         Value addrRange) {
  // If already a VGPR range, return as-is
  if (isa<VGPRRangeType>(addrRange.getType()))
    return addrRange;

  // If not an SGPR range, can't convert
  auto sgprRangeType = dyn_cast<SGPRRangeType>(addrRange.getType());
  if (!sgprRangeType)
    return nullptr;

  // Split the SGPR range into individual SGPRs
  auto splitOp = builder.create<amdgcn::SplitRegisterRangeOp>(loc, addrRange);

  // Get the number of registers in the range
  int numRegs = sgprRangeType.size();

  // Allocate VGPRs for the pointer (relocatable)
  SmallVector<Value> vgprAllocas;
  SmallVector<Value> vgprResults;
  for (int i = 0; i < numRegs; ++i) {
    auto vType = VGPRType::get(builder.getContext(), Register());
    auto vAlloca = builder.create<amdgcn::AllocaOp>(loc, vType);
    vgprAllocas.push_back(vAlloca.getResult());

    // Move pointer component from SGPR to VGPR
    Value sgpr = splitOp.getResult(i);
    auto vResult =
        V_MOV_B32_E32::create(builder, loc, vAlloca.getResult(), sgpr);
    vgprResults.push_back(vResult.getResult());
  }

  // Create VGPR range
  auto vgprRangeType = VGPRRangeType::get(builder.getContext(),
                                          RegisterRange(Register(), numRegs));
  return builder
      .create<amdgcn::MakeRegisterRangeOp>(loc, vgprRangeType, vgprResults)
      .getResult();
}

namespace {
/// FuncOp conversion pattern that converts func::FuncOp to amdgcn::KernelOp.
/// Only supports ptr and POD (integer/float) arguments that are translated
/// one-to-one. Index type is not supported.
class FuncToKernelConversionPattern : public OpConversionPattern<func::FuncOp> {
public:
  explicit FuncToKernelConversionPattern(MLIRContext *ctx, IRMapping &mapping)
      : OpConversionPattern<func::FuncOp>(ctx, /*benefit=*/1),
        mapping(mapping) {}

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<amdgcn::KernelOp> newKernelOp =
        convertFuncOpToAMDGCNKernel(funcOp, rewriter);
    if (failed(newKernelOp))
      return rewriter.notifyMatchFailure(funcOp, "Could not convert funcop");

    amdgcn::KernelOp kernelOp = *newKernelOp;
    Block *entryBlock = &kernelOp.getBodyRegion().front();

    // Load all pointer kernel arguments to registers at the beginning
    Location loc = kernelOp.getLoc();
    OpBuilder builder(entryBlock, entryBlock->begin());

    // Get kernarg pointer (relocatable registers)
    Value kernargPtr = getKernargPointer(builder, loc);

    // Load all pointer arguments to SGPRs first
    SmallVector<TypedValue<SGPRRangeType>> argLoadInfos;
    for (BlockArgument arg : entryBlock->getArguments()) {
      if (!isa<ptr::PtrType>(arg.getType()))
        continue;

      int kernargOffset =
          calculateKernargOffset(entryBlock, arg.getArgNumber());
      argLoadInfos.push_back(
          loadPointerToSGPRs(builder, loc, kernargPtr, kernargOffset));
      mapping.map(arg, argLoadInfos.back());
    }

    // Add a single s_waitcnt after all SMEM loads
    if (!argLoadInfos.empty()) {
      S_WAITCNT::create(builder, loc, /*vmcnt=*/0);
    }

    rewriter.eraseOp(funcOp);
    return success();
  }

  IRMapping &mapping;
};

/// ReturnOp conversion pattern that converts func::ReturnOp to
/// amdgcn::EndKernelOp.
class ReturnToEndKernelConversionPattern
    : public OpConversionPattern<func::ReturnOp> {
public:
  explicit ReturnToEndKernelConversionPattern(MLIRContext *ctx)
      : OpConversionPattern<func::ReturnOp>(ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // amdgcn.end_kernel doesn't take any operands
    rewriter.replaceOpWithNewOp<amdgcn::EndKernelOp>(returnOp);
    return success();
  }
};

/// Helper struct to hold size and opcode information
struct SizeAndOpcode {
  int size;
  OpCode loadOpcode;
  OpCode storeOpcode;
};

/// Helper function to get size and opcodes from a type
static FailureOr<SizeAndOpcode> getSizeAndOpcodeFromType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    int bitwidth = intType.getWidth();
    if (bitwidth == 32) {
      return SizeAndOpcode{1, OpCode::GLOBAL_LOAD_DWORD,
                           OpCode::GLOBAL_STORE_DWORD};
    } else if (bitwidth == 64) {
      return SizeAndOpcode{2, OpCode::GLOBAL_LOAD_DWORDX2,
                           OpCode::GLOBAL_STORE_DWORDX2};
    }
  }
  return failure();
}

/// PtrLoadOp conversion pattern that lowers ptr.load to AMDGCN operations
class PtrLoadToAMDGCNConversionPattern
    : public OpConversionPattern<ptr::LoadOp> {
public:
  explicit PtrLoadToAMDGCNConversionPattern(MLIRContext *ctx,
                                            IRMapping &mapping)
      : OpConversionPattern<ptr::LoadOp>(ctx, /*benefit=*/1), mapping(mapping) {
  }

  LogicalResult
  matchAndRewrite(ptr::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = adaptor.getPtr();

    auto blockArg = dyn_cast<BlockArgument>(ptr);
    if (!blockArg) {
      return rewriter.notifyMatchFailure(loadOp,
                                         "only supports kernel arguments");
    }

    // Look up the loaded pointer in the mapping
    Value addrRange = mapping.lookupOrNull(blockArg);
    if (addrRange == nullptr) {
      return rewriter.notifyMatchFailure(
          loadOp, "kernel argument not found in mapping");
    }

    // If addr range comes from a smem_load (SGPRRangeType), convert it to
    // VGPR range and update the mapping.
    Location loc = loadOp.getLoc();
    OpBuilder builder = rewriter;
    Value vgprRange = convertSGPRRangeToVGPRRange(builder, loc, addrRange);

    if (vgprRange && vgprRange != addrRange) {
      // Update the mapping to use the VGPR range instead
      mapping.map(blockArg, vgprRange);
      addrRange = vgprRange;
    }

    Type resultType = loadOp.getResult().getType();
    auto sizeAndOpcodeOr = getSizeAndOpcodeFromType(resultType);
    if (failed(sizeAndOpcodeOr))
      return rewriter.notifyMatchFailure(loadOp, "unsupported result type");

    // Allocate VGPRs for result (relocatable)
    SmallVector<Value> resultAllocas;
    for (int i = 0; i < sizeAndOpcodeOr->size; ++i) {
      auto vType = VGPRType::get(builder.getContext(), Register());
      auto alloca = builder.create<amdgcn::AllocaOp>(loc, vType);
      resultAllocas.push_back(alloca.getResult());
    }

    auto vdstRangeType = VGPRRangeType::get(
        builder.getContext(), RegisterRange(Register(), sizeAndOpcodeOr->size));
    auto vdstRange = builder.create<amdgcn::MakeRegisterRangeOp>(
        loc, vdstRangeType, resultAllocas);

    // Load data from global memory
    auto globalLoad = builder.create<inst::GlobalLoadOp>(
        loc, sizeAndOpcodeOr->loadOpcode, vdstRange.getResult(), addrRange, 0);

    rewriter.replaceOp(loadOp, globalLoad.getResult());
    return success();
  }

private:
  IRMapping &mapping;
};

/// PtrStoreOp conversion pattern that lowers ptr.store to AMDGCN operations
class PtrStoreToAMDGCNConversionPattern
    : public OpConversionPattern<ptr::StoreOp> {
public:
  explicit PtrStoreToAMDGCNConversionPattern(MLIRContext *ctx,
                                             IRMapping &mapping)
      : OpConversionPattern<ptr::StoreOp>(ctx, /*benefit=*/1),
        mapping(mapping) {}

  LogicalResult
  matchAndRewrite(ptr::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = adaptor.getPtr();
    Value value = adaptor.getValue();

    auto blockArg = dyn_cast<BlockArgument>(ptr);
    if (!blockArg) {
      return rewriter.notifyMatchFailure(storeOp,
                                         "only supports kernel arguments");
    }

    // Look up the loaded pointer in the mapping
    Value addrRange = mapping.lookupOrNull(blockArg);
    if (addrRange == nullptr) {
      return rewriter.notifyMatchFailure(
          storeOp, "kernel argument not found in mapping");
    }

    // If addr range comes from a smem_load (SGPRRangeType), convert it to
    // VGPR range and update the mapping.
    Location loc = storeOp.getLoc();
    OpBuilder builder = rewriter;
    Value vgprRange = convertSGPRRangeToVGPRRange(builder, loc, addrRange);

    if (vgprRange && vgprRange != addrRange) {
      // Update the mapping to use the VGPR range instead
      mapping.map(blockArg, vgprRange);
      addrRange = vgprRange;
    }

    // Get the value type - check both the converted type and original type
    Type valueType = value.getType();
    Type originalValueType = storeOp.getValue().getType();

    // If value is already a VGPRRangeType (from converted load), use it
    // directly
    Value dataRange = value;
    OpCode storeOpcode;

    if (auto vgprRangeType = dyn_cast<VGPRRangeType>(valueType)) {
      // Value is already a VGPR range from converted load
      int storeSize = vgprRangeType.size();
      // Determine opcode from size
      if (storeSize == 1) {
        storeOpcode = OpCode::GLOBAL_STORE_DWORD;
      } else if (storeSize == 2) {
        storeOpcode = OpCode::GLOBAL_STORE_DWORDX2;
      } else {
        return rewriter.notifyMatchFailure(storeOp,
                                           "unsupported VGPR range size");
      }
    } else {
      // Value is still in original type, need to get size from original type
      auto sizeAndOpcodeOr = getSizeAndOpcodeFromType(originalValueType);
      if (failed(sizeAndOpcodeOr))
        return rewriter.notifyMatchFailure(storeOp, "unsupported value type");

      storeOpcode = sizeAndOpcodeOr->storeOpcode;
      int storeSize = sizeAndOpcodeOr->size;

      // Convert value to VGPR range if needed
      // For now, assume value needs to be moved to VGPRs
      // This handles cases where value comes from a non-load source
      // Allocate relocatable VGPRs
      SmallVector<Value> dataAllocas;
      for (int i = 0; i < storeSize; ++i) {
        auto vType = VGPRType::get(builder.getContext(), Register());
        auto alloca = builder.create<amdgcn::AllocaOp>(loc, vType);
        dataAllocas.push_back(alloca.getResult());
        // TODO: Move value to VGPR - for now, assume value is already in a
        // register
      }

      auto dataRangeType = VGPRRangeType::get(
          builder.getContext(), RegisterRange(Register(), storeSize));
      dataRange = builder
                      .create<amdgcn::MakeRegisterRangeOp>(loc, dataRangeType,
                                                           dataAllocas)
                      .getResult();
    }

    // Store data to global memory
    builder.create<inst::GlobalStoreOp>(loc, storeOpcode, dataRange, addrRange,
                                        0);

    rewriter.eraseOp(storeOp);
    return success();
  }

private:
  IRMapping &mapping;
};

} // namespace

struct LowerToAMDGCNPass
    : public amdgcn::impl::LowerToAMDGCNPassBase<LowerToAMDGCNPass> {
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    // Set up conversion target: mark func::FuncOp, func::ReturnOp, ptr.load,
    // and ptr.store as illegal
    ConversionTarget target(*ctx);
    target.addIllegalOp<func::FuncOp, func::ReturnOp, ptr::LoadOp,
                        ptr::StoreOp>();
    target.addLegalDialect<amdgcn::AMDGCNDialect>();

    // Create shared mapping for kernel arguments
    IRMapping mapping;

    // Add the conversion patterns
    RewritePatternSet patterns(ctx);
    patterns.add<ReturnToEndKernelConversionPattern>(ctx);
    patterns
        .add<FuncToKernelConversionPattern, PtrLoadToAMDGCNConversionPattern,
             PtrStoreToAMDGCNConversionPattern>(ctx, mapping);

    // Apply the conversion
    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
