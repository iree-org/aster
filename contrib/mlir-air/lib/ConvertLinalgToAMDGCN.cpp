// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- ConvertLinalgToAMDGCN.cpp - linalg ops -> AMDGCN library calls -----===//

#include "air/Dialect/AIR/AIRDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Interfaces/ModuleOpInterface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {

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

/// Check if a memref value comes from promote to shared memory.
/// Matches two patterns:
///   1. memref.view(memref.alloca) with non-default memory space (from promote)
///   2. memref.alloc with non-default memory space (from bufferize_to_allocation)
static bool isPromotedBuffer(Value v) {
  // Pattern 1: memref.view(memref.alloca) — from transform.structured.promote.
  if (auto viewOp = v.getDefiningOp<memref::ViewOp>()) {
    if (auto allocaOp = viewOp.getSource().getDefiningOp<memref::AllocaOp>()) {
      return allocaOp.getMemref().getType().getMemorySpace() != nullptr;
    }
  }
  // Pattern 2: memref.alloc with L1/local memory space —
  // from bufferize_to_allocation.
  if (auto allocOp = v.getDefiningOp<memref::AllocOp>()) {
    auto memSpace = allocOp.getMemref().getType().getMemorySpace();
    if (!memSpace)
      return false;
    // Integer memory space 2 = L1 (AIR convention).
    if (auto intAttr = dyn_cast<IntegerAttr>(memSpace))
      return intAttr.getInt() == 2;
    // #amdgcn.addr_space<local> = LDS.
    if (auto addrSpace = dyn_cast<amdgcn::AddressSpaceAttr>(memSpace))
      return addrSpace.getSpace() == amdgcn::AddressSpaceKind::Local;
    return false;
  }
  return false;
}

/// Emit amdgcn.alloc_lds + get_lds_offset for a promoted buffer.
/// Uses a cache so the same promoted buffer gets the same LDS region
/// for both write (copy) and read (matmul).
static Value emitLDSOffset(OpBuilder &builder, Location loc, Value memrefVal,
                           DenseMap<Value, Value> &ldsCache) {
  auto it = ldsCache.find(memrefVal);
  if (it != ldsCache.end())
    return it->second;

  int64_t sizeBytes = 0;
  Value byteShift;

  // Pattern 1: memref.view(memref.alloca) — from promote.
  if (auto viewOp = memrefVal.getDefiningOp<memref::ViewOp>()) {
    auto allocaOp = viewOp.getSource().getDefiningOp<memref::AllocaOp>();
    sizeBytes = allocaOp.getMemref().getType().getNumElements();
    byteShift = viewOp.getByteShift();
  }
  // Pattern 2: memref.alloc — from bufferize_to_allocation.
  else if (auto allocOp = memrefVal.getDefiningOp<memref::AllocOp>()) {
    auto mrTy = allocOp.getMemref().getType();
    unsigned eltBits = mrTy.getElementType().getIntOrFloatBitWidth();
    sizeBytes = mrTy.getNumElements() * eltBits / 8;
  }

  auto ldsAlloc = AllocLDSOp::create(builder, loc, /*dynamic_size=*/Value(),
                                     sizeBytes, /*alignment=*/16,
                                     /*offset=*/IntegerAttr{});
  auto ldsOffset =
      GetLDSOffsetOp::create(builder, loc, builder.getIndexType(), ldsAlloc);

  Value result = ldsOffset.getResult();
  if (byteShift)
    result = builder.create<arith::AddIOp>(loc, result, byteShift);

  ldsCache[memrefVal] = result;
  return result;
}

/// Decompose a global memref into (!sx2, byte_stride: index).
/// Emits: extract_strided_metadata -> ptr.to_ptr -> lsir.to_reg -> ptr_add.
static std::pair<Value, Value>
decomposeGlobalMemref(OpBuilder &builder, Location loc, Value memref) {
  auto mrTy = cast<MemRefType>(memref.getType());
  unsigned eltBytes = mrTy.getElementType().getIntOrFloatBitWidth() / 8;

  // extract_strided_metadata -> (base_memref, offset, sizes..., strides...)
  auto metadata =
      memref::ExtractStridedMetadataOp::create(builder, loc, memref);
  Value baseBuffer = metadata.getBaseBuffer();
  Value offset = metadata.getOffset();
  // Leading stride is strides[0] (row stride in elements).
  Value leadingStride = metadata.getStrides()[0];

  // byte_stride = leading_stride * elt_bytes
  Value eltSize = arith::ConstantIndexOp::create(builder, loc, eltBytes);
  Value byteStride =
      arith::MulIOp::create(builder, loc, leadingStride, eltSize);

  // byte_offset = offset * elt_bytes
  Value byteOffset = arith::MulIOp::create(builder, loc, offset, eltSize);

  // ptr.to_ptr base_memref -> !ptr.ptr<addr_space>
  auto addrSpace = cast<ptr::MemorySpaceAttrInterface>(mrTy.getMemorySpace());
  auto ptrTy = ptr::PtrType::get(builder.getContext(), addrSpace);
  Value ptrVal = ptr::ToPtrOp::create(builder, loc, ptrTy, baseBuffer);

  // lsir.to_reg ptr -> !sx2
  auto sx2Ty = amdgcn::SGPRType::get(builder.getContext(), Register(),
                                     /*size=*/2, /*alignment=*/2);
  Value rawPtr = lsir::ToRegOp::create(builder, loc, sx2Ty, ptrVal);

  // Add byte offset: from_reg -> ptr_add -> to_reg
  Value ptrFromReg = lsir::FromRegOp::create(builder, loc, ptrTy, rawPtr);
  Value adjusted =
      ptr::PtrAddOp::create(builder, loc, ptrTy, ptrFromReg, byteOffset);
  Value result = lsir::ToRegOp::create(builder, loc, sx2Ty, adjusted);

  return {result, byteStride};
}

/// Replace a linalg op with a library call.
/// Global memrefs -> decomposed (!sx2, byte_stride) args.
/// Promoted buffers -> index (LDS offset).
static void replaceWithCall(OpBuilder &builder, Block &declBlock, Operation *op,
                            StringRef namePrefix,
                            SmallVector<Operation *> &toErase,
                            DenseMap<Value, Value> &ldsCache) {
  // Only convert ops that involve at least one promoted (LDS) buffer.
  bool hasPromotedOperand = false;
  for (Value operand : op->getOperands())
    if (isPromotedBuffer(operand))
      hasPromotedOperand = true;
  if (!hasPromotedOperand)
    return;

  auto indexTy = builder.getIndexType();
  SmallVector<Value> callArgs;
  SmallVector<Type> argTypes;

  MemRefType namingType;
  for (Value operand : op->getOperands())
    if (auto mrTy = dyn_cast<MemRefType>(operand.getType()))
      if (!namingType)
        namingType = mrTy;
  if (!namingType)
    return;
  std::string name = buildFuncName(namePrefix, namingType);

  builder.setInsertionPoint(op);
  Location loc = op->getLoc();

  auto sx2Ty = amdgcn::SGPRType::get(builder.getContext(), Register(),
                                     /*size=*/2, /*alignment=*/2);

  for (Value operand : op->getOperands()) {
    if (auto mrTy = dyn_cast<MemRefType>(operand.getType())) {
      if (isPromotedBuffer(operand)) {
        callArgs.push_back(emitLDSOffset(builder, loc, operand, ldsCache));
        argTypes.push_back(indexTy);
      } else {
        // Global memref: if this is a subview, decompose the BASE memref
        // (clean sgpr) and pass tile offsets separately. This avoids
        // baking wavefront-varying offsets into the pointer.
        Value baseMemref = operand;
        SmallVector<Value> tileOffsets;
        if (auto svOp = operand.getDefiningOp<memref::SubViewOp>()) {
          baseMemref = svOp.getSource();
          for (auto off : svOp.getMixedOffsets()) {
            if (auto val = dyn_cast<Value>(off))
              tileOffsets.push_back(val);
            else
              tileOffsets.push_back(arith::ConstantIndexOp::create(
                  builder, loc,
                  cast<IntegerAttr>(off.get<Attribute>()).getInt()));
          }
        }
        auto [ptrVal, byteStride] =
            decomposeGlobalMemref(builder, loc, baseMemref);
        callArgs.push_back(ptrVal);
        argTypes.push_back(sx2Ty);
        callArgs.push_back(byteStride);
        argTypes.push_back(indexTy);
        // Pass tile offsets (or zeros if no subview).
        if (tileOffsets.empty()) {
          auto rank = mrTy.getRank();
          for (int64_t i = 0; i < rank; ++i) {
            callArgs.push_back(
                arith::ConstantIndexOp::create(builder, loc, 0));
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

  auto funcTy = builder.getFunctionType(argTypes, {});
  ensureDecl(builder, declBlock, loc, name, funcTy);
  func::CallOp::create(builder, loc, name, TypeRange{}, callArgs);
  toErase.push_back(op);
}

struct ConvertLinalgToAMDGCN
    : public PassWrapper<ConvertLinalgToAMDGCN,
                         InterfacePass<aster::ModuleOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToAMDGCN)
  StringRef getArgument() const override { return "convert-linalg-to-amdgcn"; }
  StringRef getDescription() const override {
    return "Convert tiled linalg ops to AMDGCN library calls";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ptr::PtrDialect>();
    registry.insert<lsir::LSIRDialect>();
    registry.insert<amdgcn::AMDGCNDialect>();
  }


  void runOnOperation() override {
    Operation *moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    Operation *declParent = moduleOp;
    if (isa<mlir::ModuleOp>(moduleOp))
      moduleOp->walk([&](amdgcn::ModuleOp m) { declParent = m; });
    auto &declBlock = declParent->getRegion(0).front();
    OpBuilder builder(ctx);
    SmallVector<Operation *> toErase;
    DenseMap<Value, Value> ldsCache;

    // Pre-allocate LDS at function entry for channel get destinations FIRST,
    // before processing linalg ops. This ensures the matmul (which shares the
    // same memref.alloc as the channel.get) hits the cache and uses the
    // function-entry LDS offset instead of creating one inside the loop.
    DenseMap<StringRef, SmallVector<xilinx::air::ChannelPutOp>> putsByChannel;
    moduleOp->walk([&](xilinx::air::ChannelPutOp put) {
      putsByChannel[put.getChanName()].push_back(put);
    });
    // Determine number of wavefronts from channel array dimensions.
    int64_t numWavefronts = 1;
    moduleOp->walk([&](xilinx::air::ChannelOp chan) {
      auto sizes = chan.getSize();
      int64_t total = 1;
      for (auto s : sizes)
        total *= cast<IntegerAttr>(s).getInt();
      if (total > numWavefronts)
        numWavefronts = total;
    });

    // Pre-allocate LDS for channel get destinations (Global→LDS direction).
    // Each wavefront gets its own LDS region: alloc numWavefronts * size,
    // offset by wavefront_id * size.
    moduleOp->walk([&](xilinx::air::ChannelGetOp get) {
      Value dst = get.getDst();
      if (!isPromotedBuffer(dst))
        return;
      auto funcOp = get->getParentOfType<func::FuncOp>();
      if (!funcOp || funcOp.empty())
        return;
      auto savedIP = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(&funcOp.front());
      Location loc = funcOp.getLoc();

      // Get per-wavefront tile size.
      int64_t tileSizeBytes = 0;
      if (auto allocOp = dst.getDefiningOp<memref::AllocOp>()) {
        auto mrTy = allocOp.getMemref().getType();
        unsigned eltBits = mrTy.getElementType().getIntOrFloatBitWidth();
        tileSizeBytes = mrTy.getNumElements() * eltBits / 8;
      }

      // Allocate numWavefronts * tileSizeBytes.
      auto ldsAlloc = AllocLDSOp::create(builder, loc, /*dynamic_size=*/Value(),
                                         numWavefronts * tileSizeBytes,
                                         /*alignment=*/16,
                                         /*offset=*/IntegerAttr{});
      auto ldsBaseOffset =
          GetLDSOffsetOp::create(builder, loc, builder.getIndexType(), ldsAlloc);

      // Per-wavefront offset: base + wavefront_id * tileSizeBytes.
      Value wavefrontSize =
          arith::ConstantIndexOp::create(builder, loc, 64);
      Value threadIdX =
          gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
      Value wavefrontId =
          arith::DivUIOp::create(builder, loc, threadIdX, wavefrontSize);
      Value tileSizeVal =
          arith::ConstantIndexOp::create(builder, loc, tileSizeBytes);
      Value wavefrontOffset =
          arith::MulIOp::create(builder, loc, wavefrontId, tileSizeVal);
      Value adjustedOffset = builder.create<arith::AddIOp>(
          loc, ldsBaseOffset.getResult(), wavefrontOffset);

      ldsCache[dst] = adjustedOffset;
      builder.restoreInsertionPoint(savedIP);
    });
    // Pre-allocate LDS for channel put sources (LDS→Global direction).
    moduleOp->walk([&](xilinx::air::ChannelPutOp put) {
      Value src = put.getSrc();
      if (!isPromotedBuffer(src))
        return;
      if (ldsCache.count(src))
        return; // Already allocated (shared with channel get).
      auto funcOp = put->getParentOfType<func::FuncOp>();
      if (!funcOp || funcOp.empty())
        return;
      auto savedIP = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(&funcOp.front());
      Location loc = funcOp.getLoc();

      int64_t tileSizeBytes = 0;
      if (auto allocOp = src.getDefiningOp<memref::AllocOp>()) {
        auto mrTy = allocOp.getMemref().getType();
        unsigned eltBits = mrTy.getElementType().getIntOrFloatBitWidth();
        tileSizeBytes = mrTy.getNumElements() * eltBits / 8;
      }

      auto ldsAlloc = AllocLDSOp::create(builder, loc, /*dynamic_size=*/Value(),
                                         numWavefronts * tileSizeBytes,
                                         /*alignment=*/16,
                                         /*offset=*/IntegerAttr{});
      auto ldsBaseOffset =
          GetLDSOffsetOp::create(builder, loc, builder.getIndexType(), ldsAlloc);

      Value wavefrontSize =
          arith::ConstantIndexOp::create(builder, loc, 64);
      Value threadIdX =
          gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
      Value wavefrontId =
          arith::DivUIOp::create(builder, loc, threadIdX, wavefrontSize);
      Value tileSizeVal =
          arith::ConstantIndexOp::create(builder, loc, tileSizeBytes);
      Value wavefrontOffset =
          arith::MulIOp::create(builder, loc, wavefrontId, tileSizeVal);
      Value adjustedOffset = builder.create<arith::AddIOp>(
          loc, ldsBaseOffset.getResult(), wavefrontOffset);

      ldsCache[src] = adjustedOffset;
      builder.restoreInsertionPoint(savedIP);
    });

    // Detect numWavefronts from the IR.
    // air-to-amdgcn emits: wavefrontId = gpu.thread_id x / 64
    // The affine.apply for M row uses: #map()[%loop_var, %wavefrontId]
    //   = loop_var + wavefrontId * herdTileM
    // numWavefronts = totalM / herdTileM.
    // We find herdTileM from the coefficient of the wavefrontId symbol in the
    // affine map by scanning AffineApplyOp users of the divui result.
    int64_t detectedNumWavefronts = 1;
    moduleOp->walk([&](arith::DivUIOp divOp) {
      if (detectedNumWavefronts > 1)
        return;
      auto threadId = divOp.getLhs().getDefiningOp<gpu::ThreadIdOp>();
      if (!threadId || threadId.getDimension() != gpu::Dimension::x)
        return;
      auto cst = divOp.getRhs().getDefiningOp<arith::ConstantIndexOp>();
      if (!cst || cst.value() != 64)
        return;
      // wavefrontId = divOp.getResult(). Find its use in affine.apply.
      for (Operation *user : divOp.getResult().getUsers()) {
        auto applyOp = dyn_cast<affine::AffineApplyOp>(user);
        if (!applyOp || applyOp.getAffineMap().getNumResults() != 1)
          continue;
        // Find the position of wavefrontId in the operand list.
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
        // Extract coefficient of symbol/dim at wfPos in the affine map.
        AffineMap map = applyOp.getAffineMap();
        // The map operands are pure symbols in this context.
        int64_t stride = 0;
        AffineExpr expr = map.getResult(0);
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
        // Get totalM from DMA src memref.
        moduleOp->walk([&](xilinx::air::DmaMemcpyNdOp dma2) {
          if (detectedNumWavefronts > 1)
            return;
          Value src = dma2.getSrcMemref();
          auto srcTy = dyn_cast<MemRefType>(src.getType());
          if (!srcTy || srcTy.getRank() < 1 || srcTy.getDimSize(0) <= 0)
            return;
          detectedNumWavefronts = srcTy.getDimSize(0) / stride;
        });
      }
    });

    // Pre-allocate LDS for air.dma_memcpy_nd destinations (no-channel path).
    // Must run before linalg op processing so matmul hits the same ldsCache.
    // Allocate numWavefronts * tileSizeBytes and stripe by wavefront_id.
    moduleOp->walk([&](xilinx::air::DmaMemcpyNdOp dma) {
      Value dst = dma.getDstMemref();
      if (!isPromotedBuffer(dst) || ldsCache.count(dst))
        return;
      auto funcOp = dma->getParentOfType<func::FuncOp>();
      if (!funcOp || funcOp.empty())
        return;
      auto savedIP = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(&funcOp.front());
      Location loc = funcOp.getLoc();

      int64_t tileSizeBytes = 0;
      if (auto allocOp = dst.getDefiningOp<memref::AllocOp>()) {
        auto mrTy = allocOp.getMemref().getType();
        unsigned eltBits = mrTy.getElementType().getIntOrFloatBitWidth();
        tileSizeBytes = mrTy.getNumElements() * eltBits / 8;
      }
      int64_t nWf = detectedNumWavefronts;
      auto ldsAlloc = AllocLDSOp::create(builder, loc, /*dynamic_size=*/Value(),
                                         nWf * tileSizeBytes, /*alignment=*/16,
                                         /*offset=*/IntegerAttr{});
      auto ldsBaseOffset =
          GetLDSOffsetOp::create(builder, loc, builder.getIndexType(), ldsAlloc);
      Value result = ldsBaseOffset.getResult();
      if (nWf > 1) {
        Value wavefrontSize = arith::ConstantIndexOp::create(builder, loc, 64);
        Value threadIdX =
            gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
        Value wavefrontId =
            arith::DivUIOp::create(builder, loc, threadIdX, wavefrontSize);
        Value tileSizeVal =
            arith::ConstantIndexOp::create(builder, loc, tileSizeBytes);
        Value wavefrontOffset =
            arith::MulIOp::create(builder, loc, wavefrontId, tileSizeVal);
        result = arith::AddIOp::create(builder, loc, result, wavefrontOffset);
      }
      ldsCache[dst] = result;
      builder.restoreInsertionPoint(savedIP);
    });

    // Convert air.dma_memcpy_nd directly (Global→LDS only; no channels).
    // Emit: copy_<dtype>_<MxN>(base_sgpr, stride, row_off, col_off, lds_dst)
    moduleOp->walk([&](xilinx::air::DmaMemcpyNdOp dma) {
      Value dst = dma.getDstMemref();
      Value src = dma.getSrcMemref();
      bool dstIsLDS = isPromotedBuffer(dst);
      if (!dstIsLDS)
        return; // Only handle Global→LDS DMAs here.

      auto dstTy = cast<MemRefType>(dst.getType());
      builder.setInsertionPoint(dma);
      Location loc = dma.getLoc();

      std::string name = buildFuncName("copy", dstTy);

      auto indexTy = builder.getIndexType();
      auto sx2Ty = amdgcn::SGPRType::get(ctx, Register(), /*size=*/2,
                                          /*alignment=*/2);
      SmallVector<Value> callArgs;
      SmallVector<Type> argTypes;

      // Decompose BASE src memref → (sgpr_base, byte_stride).
      auto [ptrVal, byteStride] = decomposeGlobalMemref(builder, loc, src);
      callArgs.push_back(ptrVal);
      argTypes.push_back(sx2Ty);
      callArgs.push_back(byteStride);
      argTypes.push_back(indexTy);

      // Src tile offsets (row, col) from DMA operands.
      auto srcOffsets = dma.getSrcOffsets();
      if (srcOffsets.size() >= 2) {
        callArgs.push_back(srcOffsets[0]);
        argTypes.push_back(indexTy);
        callArgs.push_back(srcOffsets[1]);
        argTypes.push_back(indexTy);
      } else {
        Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
        callArgs.push_back(zero);
        argTypes.push_back(indexTy);
        callArgs.push_back(zero);
        argTypes.push_back(indexTy);
      }

      // LDS dst offset from cache.
      assert(ldsCache.count(dst) && "DMA dst LDS not pre-allocated");
      callArgs.push_back(ldsCache[dst]);
      argTypes.push_back(indexTy);

      auto funcTy = builder.getFunctionType(argTypes, {});
      ensureDecl(builder, declBlock, loc, name, funcTy);
      func::CallOp::create(builder, loc, name, TypeRange{}, callArgs);
      toErase.push_back(dma);
    });

    // Now process linalg ops — they'll hit the ldsCache for shared allocs.
    moduleOp->walk([&](linalg::FillOp op) {
      replaceWithCall(builder, declBlock, op, "fill", toErase, ldsCache);
    });
    moduleOp->walk([&](linalg::CopyOp op) {
      replaceWithCall(builder, declBlock, op, "copy", toErase, ldsCache);
    });
    moduleOp->walk([&](memref::CopyOp op) {
      replaceWithCall(builder, declBlock, op, "copy", toErase, ldsCache);
    });
    moduleOp->walk([&](linalg::MatmulOp op) {
      replaceWithCall(builder, declBlock, op, "mfma_matmul", toErase, ldsCache);
    });
    moduleOp->walk([&](linalg::GenericOp op) {
      if (op.getNumDpsInputs() == 2 && op.getNumDpsInits() == 1 &&
          op.getNumReductionLoops() == 1)
        replaceWithCall(builder, declBlock, op, "mfma_matmul", toErase,
                        ldsCache);
    });

    // Emit copy call at each put site (where global src operands dominate).
    moduleOp->walk([&](xilinx::air::ChannelGetOp get) {
      StringRef chanName = get.getChanName();
      auto it = putsByChannel.find(chanName);
      if (it == putsByChannel.end() || it->second.empty())
        return;
      xilinx::air::ChannelPutOp put = it->second.front();

      Value dst = get.getDst();
      auto dstTy = dyn_cast<MemRefType>(dst.getType());
      if (!dstTy)
        return;

      Value src = put.getSrc();
      bool srcIsLDS = isPromotedBuffer(src);
      bool dstIsLDS = isPromotedBuffer(dst);

      // Determine direction and emit at the appropriate site.
      // Global→LDS: emit at put site (global src operands dominate).
      // LDS→Global: emit at get site (global dst operands dominate).
      if (srcIsLDS && !dstIsLDS) {
        // LDS→Global (C write-back): emit at get site.
        builder.setInsertionPoint(get);
      } else {
        // Global→LDS (A/B copy): emit at put site.
        builder.setInsertionPoint(put);
      }
      Location loc = builder.getInsertionPoint()->getLoc();

      // Use the L1 (smaller) memref type for the function name.
      auto namingTy = srcIsLDS ? cast<MemRefType>(src.getType()) : dstTy;
      std::string name = buildFuncName("copy", namingTy);

      auto indexTy = builder.getIndexType();
      auto sx2Ty = amdgcn::SGPRType::get(ctx, Register(),
                                         /*size=*/2, /*alignment=*/2);
      SmallVector<Value> callArgs;
      SmallVector<Type> argTypes;

      // Decompose src side.
      if (srcIsLDS) {
        // LDS src.
        assert(ldsCache.count(src) && "LDS offset not pre-allocated for channel put src");
        callArgs.push_back(ldsCache[src]);
        argTypes.push_back(indexTy);
      } else {
        // Global src: decompose BASE memref (not subview) to get a clean
        // sgpr pointer. Pass the channel's tile offsets separately so the
        // library function handles them (kittens pattern).
        auto [ptrVal, byteStride] =
            decomposeGlobalMemref(builder, loc, src);
        callArgs.push_back(ptrVal);
        argTypes.push_back(sx2Ty);
        callArgs.push_back(byteStride);
        argTypes.push_back(indexTy);
        // Tile offsets from the channel put (element-level indices).
        auto putOffsets = put.getSrcOffsets();
        if (putOffsets.size() >= 2) {
          callArgs.push_back(putOffsets[0]); // row offset
          argTypes.push_back(indexTy);
          callArgs.push_back(putOffsets[1]); // col offset
          argTypes.push_back(indexTy);
        } else {
          Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
          callArgs.push_back(c0);
          argTypes.push_back(indexTy);
          callArgs.push_back(c0);
          argTypes.push_back(indexTy);
        }
      }

      // Decompose dst side.
      if (dstIsLDS) {
        // LDS dst.
        assert(ldsCache.count(dst) && "LDS offset not pre-allocated for channel get dst");
        callArgs.push_back(ldsCache[dst]);
        argTypes.push_back(indexTy);
      } else {
        // Global dst: decompose BASE memref, pass offsets separately.
        auto [ptrVal, byteStride] =
            decomposeGlobalMemref(builder, loc, dst);
        callArgs.push_back(ptrVal);
        argTypes.push_back(sx2Ty);
        callArgs.push_back(byteStride);
        argTypes.push_back(indexTy);
        auto getOffsets = get.getDstOffsets();
        if (getOffsets.size() >= 2) {
          callArgs.push_back(getOffsets[0]);
          argTypes.push_back(indexTy);
          callArgs.push_back(getOffsets[1]);
          argTypes.push_back(indexTy);
        } else {
          Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
          callArgs.push_back(c0);
          argTypes.push_back(indexTy);
          callArgs.push_back(c0);
          argTypes.push_back(indexTy);
        }
      }

      auto funcTy = builder.getFunctionType(argTypes, {});
      ensureDecl(builder, declBlock, loc, name, funcTy);
      func::CallOp::create(builder, loc, name, TypeRange{}, callArgs);

      toErase.push_back(put);
      toErase.push_back(get);
    });

    // Clean up channel declarations.
    moduleOp->walk([&](xilinx::air::ChannelOp chan) {
      toErase.push_back(chan);
    });

    for (auto *op : toErase)
      op->erase();

    // Erase linalg.fill on global (non-LDS) buffers.
    // The library's zero_C handles accumulator init, so the fill is redundant.
    // It must be erased because the aster backend cannot lower linalg.fill
    // on global memrefs.
    SmallVector<linalg::FillOp> globalFills;
    moduleOp->walk([&](linalg::FillOp fill) {
      for (Value out : fill.getDpsInits())
        if (isa<MemRefType>(out.getType()) && !isPromotedBuffer(out))
          globalFills.push_back(fill);
    });
    for (auto fill : globalFills)
      fill->erase();

  }
};

} // namespace

namespace mlir::aster::mlir_air {
std::unique_ptr<Pass> createConvertLinalgToAMDGCN() {
  return std::make_unique<ConvertLinalgToAMDGCN>();
}
} // namespace mlir::aster::mlir_air
