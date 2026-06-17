// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- ConvertToAMDGCNLibraryCalls.cpp - ops -> AMDGCN library calls ------===//

#include "air/Dialect/AIR/AIRDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Interfaces/ModuleOpInterface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

static std::string buildFuncName(StringRef prefix, MemRefType ty) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << prefix;
  Type elt = ty.getElementType();
  if (elt.isF16())
    os << "_f16";
  else if (elt.isF32())
    os << "_f32";
  else if (elt.isBF16())
    os << "_bf16";
  else
    os << "_unknown";
  auto shape = ty.getShape();
  for (size_t i = 0; i < shape.size(); ++i)
    os << (i == 0 ? "_" : "x") << shape[i];
  return name;
}

static void ensureDecl(OpBuilder &builder, Block &block, Location loc,
                       StringRef name, FunctionType funcTy) {
  for (auto &op : block)
    if (auto fn = dyn_cast<func::FuncOp>(&op))
      if (fn.getName() == name)
        return;
  auto savedIP = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&block);
  auto decl = func::FuncOp::create(builder, loc, name, funcTy);
  decl.setPrivate();
  builder.restoreInsertionPoint(savedIP);
}

static bool isPromotedBuffer(Value v) {
  if (auto viewOp = v.getDefiningOp<memref::ViewOp>()) {
    if (auto allocaOp = viewOp.getSource().getDefiningOp<memref::AllocaOp>())
      return allocaOp.getMemref().getType().getMemorySpace() != nullptr;
  }
  if (auto allocOp = v.getDefiningOp<memref::AllocOp>()) {
    auto memSpace = allocOp.getMemref().getType().getMemorySpace();
    if (!memSpace)
      return false;
    if (auto intAttr = dyn_cast<IntegerAttr>(memSpace))
      return intAttr.getInt() == 2;
    if (auto addrSpace = dyn_cast<amdgcn::AddressSpaceAttr>(memSpace))
      return addrSpace.getSpace() == amdgcn::AddressSpaceKind::Local;
    return false;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Shared context for patterns (populated by analysis, read by patterns).
// ---------------------------------------------------------------------------

struct ConversionContext {
  Block *declBlock = nullptr;
  int64_t numWavefronts = 1;
  DenseMap<Value, Value> ldsCache;
};

/// Emit amdgcn.alloc_lds + get_lds_offset for a promoted buffer at function
/// entry.  When numWavefronts > 1, allocates nWf * tileSizeBytes and adds a
/// per-wavefront offset (wavefrontId * tileSizeBytes).
/// Uses ldsCache so the same buffer gets the same LDS region regardless of
/// which pattern fires first.
static Value emitLDSOffset(OpBuilder &builder, Location loc, Value memrefVal,
                           ConversionContext &convCtx) {
  auto it = convCtx.ldsCache.find(memrefVal);
  if (it != convCtx.ldsCache.end())
    return it->second;

  int64_t sizeBytes = 0;
  Value byteShift;
  if (auto viewOp = memrefVal.getDefiningOp<memref::ViewOp>()) {
    auto allocaOp = viewOp.getSource().getDefiningOp<memref::AllocaOp>();
    sizeBytes = allocaOp.getMemref().getType().getNumElements();
    byteShift = viewOp.getByteShift();
  } else if (auto allocOp = memrefVal.getDefiningOp<memref::AllocOp>()) {
    auto mrTy = allocOp.getMemref().getType();
    unsigned eltBits = mrTy.getElementType().getIntOrFloatBitWidth();
    sizeBytes = mrTy.getNumElements() * eltBits / 8;
  }

  // Insert at function entry so LDS allocation dominates all uses.
  auto funcOp = memrefVal.getParentRegion()->getParentOfType<func::FuncOp>();
  auto savedIP = builder.saveInsertionPoint();
  if (funcOp)
    builder.setInsertionPointToStart(&funcOp.front());

  int64_t nWf = convCtx.numWavefronts;
  auto ldsAlloc = AllocLDSOp::create(builder, loc, /*dynamic_size=*/Value(),
                                     nWf * sizeBytes, /*alignment=*/16,
                                     /*offset=*/IntegerAttr{});
  auto ldsOffset =
      GetLDSOffsetOp::create(builder, loc, builder.getIndexType(), ldsAlloc);
  Value result = ldsOffset.getResult();

  if (nWf > 1) {
    Value wavefrontSize = arith::ConstantIndexOp::create(builder, loc, 64);
    Value threadIdX =
        gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
    Value wavefrontId =
        arith::DivUIOp::create(builder, loc, threadIdX, wavefrontSize);
    Value tileSizeVal =
        arith::ConstantIndexOp::create(builder, loc, sizeBytes);
    Value wavefrontOffset =
        arith::MulIOp::create(builder, loc, wavefrontId, tileSizeVal);
    result = arith::AddIOp::create(builder, loc, result, wavefrontOffset);
  }

  if (byteShift)
    result = builder.create<arith::AddIOp>(loc, result, byteShift);

  builder.restoreInsertionPoint(savedIP);
  convCtx.ldsCache[memrefVal] = result;
  return result;
}

static std::pair<Value, Value>
decomposeGlobalMemref(OpBuilder &builder, Location loc, Value memref) {
  auto mrTy = cast<MemRefType>(memref.getType());
  unsigned eltBytes = mrTy.getElementType().getIntOrFloatBitWidth() / 8;
  auto metadata =
      memref::ExtractStridedMetadataOp::create(builder, loc, memref);
  Value baseBuffer = metadata.getBaseBuffer();
  Value offset = metadata.getOffset();
  Value leadingStride = metadata.getStrides()[0];
  Value eltSize = arith::ConstantIndexOp::create(builder, loc, eltBytes);
  Value byteStride =
      arith::MulIOp::create(builder, loc, leadingStride, eltSize);
  Value byteOffset = arith::MulIOp::create(builder, loc, offset, eltSize);
  auto addrSpace = cast<ptr::MemorySpaceAttrInterface>(mrTy.getMemorySpace());
  auto ptrTy = ptr::PtrType::get(builder.getContext(), addrSpace);
  Value ptrVal = ptr::ToPtrOp::create(builder, loc, ptrTy, baseBuffer);
  auto sx2Ty = amdgcn::SGPRType::get(builder.getContext(), Register(),
                                     /*size=*/2, /*alignment=*/2);
  Value rawPtr = lsir::ToRegOp::create(builder, loc, sx2Ty, ptrVal);
  Value ptrFromReg = lsir::FromRegOp::create(builder, loc, ptrTy, rawPtr);
  Value adjusted =
      ptr::PtrAddOp::create(builder, loc, ptrTy, ptrFromReg, byteOffset);
  Value result = lsir::ToRegOp::create(builder, loc, sx2Ty, adjusted);
  return {result, byteStride};
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

/// Convert air.dma_memcpy_nd to a library call.
/// Handles both directions: Global→LDS and LDS→Global.
struct DmaToLibraryCall
    : public OpRewritePattern<xilinx::air::DmaMemcpyNdOp> {
  ConversionContext &convCtx;

  DmaToLibraryCall(MLIRContext *ctx, ConversionContext &convCtx)
      : OpRewritePattern(ctx), convCtx(convCtx) {}

  LogicalResult matchAndRewrite(xilinx::air::DmaMemcpyNdOp dma,
                                PatternRewriter &rewriter) const override {
    Value dst = dma.getDstMemref();
    Value src = dma.getSrcMemref();
    bool dstIsLDS = isPromotedBuffer(dst);
    bool srcIsLDS = isPromotedBuffer(src);
    if (!dstIsLDS && !srcIsLDS)
      return failure();

    Location loc = dma.getLoc();
    auto indexTy = rewriter.getIndexType();
    auto sx2Ty = amdgcn::SGPRType::get(rewriter.getContext(), Register(),
                                       /*size=*/2, /*alignment=*/2);
    SmallVector<Value> callArgs;
    SmallVector<Type> argTypes;

    // Use the LDS memref type for function naming.
    // Append direction suffix to distinguish global→LDS from LDS→global.
    auto ldsTy = cast<MemRefType>(dstIsLDS ? dst.getType() : src.getType());
    std::string name = buildFuncName(
        srcIsLDS ? "store_global" : "copy", ldsTy);

    if (dstIsLDS && !srcIsLDS) {
      // Detect boundary tile padding from either:
      // (a) pad_after attribute (set by air-split-launch-for-padding), or
      // (b) DMA src_sizes[0] < dst memref dim 0 (from transform tensor.pad).
      auto padAfterAttr = dma->getAttrOfType<DenseI32ArrayAttr>("pad_after");
      bool hasPadding = false;
      int32_t rowPad = 0; // pad rows appended after the valid region
      if (padAfterAttr) {
        auto padArr = padAfterAttr.asArrayRef();
        if (!padArr.empty() && padArr[0] > 0) {
          hasPadding = true;
          rowPad = padArr[0];
        }
      }
      // Also detect from DMA sizes: if src_sizes[0] is a constant < dst dim 0,
      // this is a partial copy into a padded LDS buffer (from tensor.pad path).
      if (!hasPadding) {
        auto srcSizes = dma.getSrcSizes();
        if (!srcSizes.empty()) {
          auto srcRowOpt = getConstantIntValue(srcSizes[0]);
          int64_t dstDim0 = ldsTy.getDimSize(0);
          if (srcRowOpt && dstDim0 > 0 && *srcRowOpt < dstDim0) {
            hasPadding = true;
            rowPad = dstDim0 - *srcRowOpt;
          }
        }
      }

      Value ldsOffset = emitLDSOffset(rewriter, loc, dst, convCtx);

      if (hasPadding) {
        // When padding is detected from pad_after attribute (air-split-launch),
        // emit fill explicitly. When detected from DMA sizes (tensor.pad path),
        // the linalg.fill from transform already handles the zero-fill and is
        // converted separately by LinalgToLibraryCall<FillOp>.
        if (padAfterAttr) {
          std::string fillName = buildFuncName("fill", ldsTy);
          Type f16Ty = rewriter.getF16Type();
          Value zeroF16 = arith::ConstantOp::create(
              rewriter, loc, f16Ty,
              rewriter.getF16FloatAttr(0.0f));
          auto fillTy = rewriter.getFunctionType({f16Ty, indexTy}, {});
          ensureDecl(rewriter, *convCtx.declBlock, loc, fillName, fillTy);
          func::CallOp::create(rewriter, loc, fillName, TypeRange{},
                               ValueRange{zeroF16, ldsOffset});
        }

        // Partial copy — copy only the valid rows.
        // copy_f16_16x64_padded(global_ptr, stride, row, col, actual_rows, lds_dst)
        // The library function reads actual_rows rows from global (not all 16).
        auto [ptrVal, byteStride] = decomposeGlobalMemref(rewriter, loc, src);
        auto srcSizes = dma.getSrcSizes();
        // actual_rows = src_sizes[0] (set by split-launch pass to actualLast).
        Value actualRows;
        if (!srcSizes.empty())
          actualRows = srcSizes[0];
        else
          actualRows = arith::ConstantIndexOp::create(
              rewriter, loc, ldsTy.getDimSize(0) - rowPad);

        std::string copyName = buildFuncName("copy", ldsTy) + "_padded";
        SmallVector<Value> copyArgs = {ptrVal, byteStride};
        SmallVector<Type> copyArgTypes = {sx2Ty, indexTy};
        auto srcOffsets = dma.getSrcOffsets();
        if (srcOffsets.size() >= 2) {
          copyArgs.push_back(srcOffsets[0]);
          copyArgs.push_back(srcOffsets[1]);
        } else {
          Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
          copyArgs.push_back(zero);
          copyArgs.push_back(zero);
        }
        copyArgTypes.push_back(indexTy);
        copyArgTypes.push_back(indexTy);
        copyArgs.push_back(actualRows);
        copyArgTypes.push_back(indexTy);
        copyArgs.push_back(ldsOffset);
        copyArgTypes.push_back(indexTy);
        auto copyTy = rewriter.getFunctionType(copyArgTypes, {});
        ensureDecl(rewriter, *convCtx.declBlock, loc, copyName, copyTy);
        func::CallOp::create(rewriter, loc, copyName, TypeRange{}, copyArgs);
        rewriter.eraseOp(dma);
        return success();
      }

      // Non-padded: Global→LDS: copy(global_ptr, stride, row, col, lds_dst)
      auto [ptrVal, byteStride] = decomposeGlobalMemref(rewriter, loc, src);
      callArgs.push_back(ptrVal);
      argTypes.push_back(sx2Ty);
      callArgs.push_back(byteStride);
      argTypes.push_back(indexTy);
      auto srcOffsets = dma.getSrcOffsets();
      if (srcOffsets.size() >= 2) {
        callArgs.push_back(srcOffsets[0]);
        callArgs.push_back(srcOffsets[1]);
      } else {
        Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
        callArgs.push_back(zero);
        callArgs.push_back(zero);
      }
      argTypes.push_back(indexTy);
      argTypes.push_back(indexTy);
      callArgs.push_back(ldsOffset);
      argTypes.push_back(indexTy);
    } else if (srcIsLDS && !dstIsLDS) {
      // LDS→Global: copy(lds_src, global_ptr, stride, row, col)
      callArgs.push_back(emitLDSOffset(rewriter, loc, src, convCtx));
      argTypes.push_back(indexTy);
      auto [ptrVal, byteStride] = decomposeGlobalMemref(rewriter, loc, dst);
      callArgs.push_back(ptrVal);
      argTypes.push_back(sx2Ty);
      callArgs.push_back(byteStride);
      argTypes.push_back(indexTy);
      auto dstOffsets = dma.getDstOffsets();
      if (dstOffsets.size() >= 2) {
        callArgs.push_back(dstOffsets[0]);
        callArgs.push_back(dstOffsets[1]);
      } else {
        Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
        callArgs.push_back(zero);
        callArgs.push_back(zero);
      }
      argTypes.push_back(indexTy);
      argTypes.push_back(indexTy);
    } else {
      // LDS→LDS: not supported.
      return failure();
    }

    auto funcTy = rewriter.getFunctionType(argTypes, {});
    ensureDecl(rewriter, *convCtx.declBlock, loc, name, funcTy);
    func::CallOp::create(rewriter, loc, name, TypeRange{}, callArgs);
    rewriter.eraseOp(dma);
    return success();
  }
};

/// Convert a linalg op (or memref.copy) with at least one promoted (LDS)
/// operand to a library call.
template <typename OpTy>
struct LinalgToLibraryCall : public OpRewritePattern<OpTy> {
  ConversionContext &convCtx;
  StringRef namePrefix;

  LinalgToLibraryCall(MLIRContext *ctx, ConversionContext &convCtx,
                      StringRef namePrefix)
      : OpRewritePattern<OpTy>(ctx), convCtx(convCtx),
        namePrefix(namePrefix) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    bool hasPromoted = false;
    for (Value operand : op->getOperands())
      if (isPromotedBuffer(operand))
        hasPromoted = true;
    if (!hasPromoted)
      return failure();

    MemRefType namingType;
    for (Value operand : op->getOperands())
      if (auto mrTy = dyn_cast<MemRefType>(operand.getType()))
        if (!namingType)
          namingType = mrTy;
    if (!namingType)
      return failure();

    std::string name = buildFuncName(namePrefix, namingType);
    Location loc = op->getLoc();
    auto indexTy = rewriter.getIndexType();
    auto sx2Ty = amdgcn::SGPRType::get(rewriter.getContext(), Register(),
                                       /*size=*/2, /*alignment=*/2);
    SmallVector<Value> callArgs;
    SmallVector<Type> argTypes;

    for (Value operand : op->getOperands()) {
      if (auto mrTy = dyn_cast<MemRefType>(operand.getType())) {
        if (isPromotedBuffer(operand)) {
          callArgs.push_back(
              emitLDSOffset(rewriter, loc, operand, convCtx));
          argTypes.push_back(indexTy);
        } else {
          Value baseMemref = operand;
          SmallVector<Value> tileOffsets;
          if (auto svOp = operand.getDefiningOp<memref::SubViewOp>()) {
            baseMemref = svOp.getSource();
            for (auto off : svOp.getMixedOffsets()) {
              if (auto val = dyn_cast<Value>(off))
                tileOffsets.push_back(val);
              else
                tileOffsets.push_back(arith::ConstantIndexOp::create(
                    rewriter, loc,
                    cast<IntegerAttr>(off.get<Attribute>()).getInt()));
            }
          }
          auto [ptrVal, byteStride] =
              decomposeGlobalMemref(rewriter, loc, baseMemref);
          callArgs.push_back(ptrVal);
          argTypes.push_back(sx2Ty);
          callArgs.push_back(byteStride);
          argTypes.push_back(indexTy);
          if (tileOffsets.empty()) {
            for (int64_t i = 0; i < mrTy.getRank(); ++i) {
              callArgs.push_back(
                  arith::ConstantIndexOp::create(rewriter, loc, 0));
              argTypes.push_back(indexTy);
            }
          } else {
            for (auto off : tileOffsets) {
              callArgs.push_back(off);
              argTypes.push_back(indexTy);
            }
          }
        }
      } else {
        callArgs.push_back(operand);
        argTypes.push_back(operand.getType());
      }
    }

    auto funcTy = rewriter.getFunctionType(argTypes, {});
    ensureDecl(rewriter, *convCtx.declBlock, loc, name, funcTy);
    func::CallOp::create(rewriter, loc, name, TypeRange{}, callArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Match linalg.generic with matmul semantics (2 inputs, 1 output, 1 reduction).
struct GenericMatmulToLibraryCall : public OpRewritePattern<linalg::GenericOp> {
  ConversionContext &convCtx;

  GenericMatmulToLibraryCall(MLIRContext *ctx, ConversionContext &convCtx)
      : OpRewritePattern(ctx), convCtx(convCtx) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
        op.getNumReductionLoops() != 1)
      return failure();
    // Use _lds_c suffix when the output operand is in LDS.
    StringRef prefix = "mfma_matmul";
    bool outputIsLDS = false;
    for (Value out : op.getDpsInits())
      if (isPromotedBuffer(out))
        outputIsLDS = true;
    std::string prefixStr =
        outputIsLDS ? "mfma_matmul_lds_c" : "mfma_matmul";
    LinalgToLibraryCall<linalg::GenericOp> inner(
        rewriter.getContext(), convCtx, prefixStr);
    return inner.matchAndRewrite(op, rewriter);
  }
};

/// Erase linalg.fill on global (non-LDS) buffers.
/// The library's zero_C handles accumulator init.
struct EraseGlobalFill : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fill,
                                PatternRewriter &rewriter) const override {
    for (Value out : fill.getDpsInits())
      if (isa<MemRefType>(out.getType()) && !isPromotedBuffer(out)) {
        rewriter.eraseOp(fill);
        return success();
      }
    return failure();
  }
};

// ---------------------------------------------------------------------------
// Analysis: detect numWavefronts from IR and pre-allocate LDS.
// ---------------------------------------------------------------------------

static int64_t detectNumWavefronts(Operation *moduleOp) {
  int64_t result = 1;
  moduleOp->walk([&](arith::DivUIOp divOp) {
    if (result > 1)
      return;
    auto threadId = divOp.getLhs().getDefiningOp<gpu::ThreadIdOp>();
    if (!threadId || threadId.getDimension() != gpu::Dimension::x)
      return;
    auto cst = divOp.getRhs().getDefiningOp<arith::ConstantIndexOp>();
    if (!cst || cst.value() != 64)
      return;
    for (Operation *user : divOp.getResult().getUsers()) {
      auto applyOp = dyn_cast<affine::AffineApplyOp>(user);
      if (!applyOp || applyOp.getAffineMap().getNumResults() != 1)
        continue;
      unsigned wfPos = 0;
      bool found = false;
      for (auto [idx, operand] : llvm::enumerate(applyOp.getMapOperands())) {
        if (operand == divOp.getResult()) {
          wfPos = idx;
          found = true;
          break;
        }
      }
      if (!found)
        continue;
      int64_t stride = 0;
      AffineExpr expr = applyOp.getAffineMap().getResult(0);
      expr.walk([&](AffineExpr e) {
        auto mul = dyn_cast<AffineBinaryOpExpr>(e);
        if (!mul || mul.getKind() != AffineExprKind::Mul)
          return;
        auto sym = dyn_cast<AffineSymbolExpr>(mul.getLHS());
        auto con = dyn_cast<AffineConstantExpr>(mul.getRHS());
        if (!sym || !con)
          return;
        if (sym.getPosition() == wfPos)
          stride = con.getValue();
      });
      if (stride <= 0)
        continue;
      moduleOp->walk([&](xilinx::air::DmaMemcpyNdOp dma2) {
        if (result > 1)
          return;
        Value src = dma2.getSrcMemref();
        auto srcTy = dyn_cast<MemRefType>(src.getType());
        if (!srcTy || srcTy.getRank() < 1 || srcTy.getDimSize(0) <= 0)
          return;
        result = srcTy.getDimSize(0) / stride;
      });
    }
  });
  return result;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct ConvertToAMDGCNLibraryCalls
    : public PassWrapper<ConvertToAMDGCNLibraryCalls,
                         InterfacePass<aster::ModuleOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToAMDGCNLibraryCalls)
  StringRef getArgument() const override {
    return "convert-to-amdgcn-library-calls";
  }
  StringRef getDescription() const override {
    return "Convert linalg/AIR ops to AMDGCN library calls";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ptr::PtrDialect>();
    registry.insert<lsir::LSIRDialect>();
    registry.insert<amdgcn::AMDGCNDialect>();
  }

  void runOnOperation() override {
    Operation *moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Find the declaration block (inside amdgcn.module if present).
    Operation *declParent = moduleOp;
    if (isa<mlir::ModuleOp>(moduleOp))
      moduleOp->walk([&](amdgcn::ModuleOp m) { declParent = m; });

    ConversionContext convCtx;
    convCtx.declBlock = &declParent->getRegion(0).front();
    convCtx.numWavefronts = detectNumWavefronts(moduleOp);

    // Apply conversion patterns.
    RewritePatternSet patterns(ctx);
    patterns.add<DmaToLibraryCall>(ctx, convCtx);
    patterns.add<LinalgToLibraryCall<linalg::FillOp>>(ctx, convCtx, "fill");
    patterns.add<LinalgToLibraryCall<linalg::CopyOp>>(ctx, convCtx, "copy");
    patterns.add<LinalgToLibraryCall<memref::CopyOp>>(ctx, convCtx, "copy");
    patterns.add<LinalgToLibraryCall<linalg::MatmulOp>>(ctx, convCtx,
                                                        "mfma_matmul");
    patterns.add<GenericMatmulToLibraryCall>(ctx, convCtx);
    patterns.add<EraseGlobalFill>(ctx);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::aster::mlir_air {
std::unique_ptr<Pass> createConvertToAMDGCNLibraryCalls() {
  return std::make_unique<ConvertToAMDGCNLibraryCalls>();
}
} // namespace mlir::aster::mlir_air
