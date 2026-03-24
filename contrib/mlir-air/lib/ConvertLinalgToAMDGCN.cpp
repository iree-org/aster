// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- ConvertLinalgToAMDGCN.cpp - linalg ops -> AMDGCN library calls -----===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Interfaces/ModuleOpInterface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

/// Check if a memref value comes from promote to shared memory
/// (memref.view(memref.alloca) with workgroup address space).
static bool isPromotedBuffer(Value v) {
  auto viewOp = v.getDefiningOp<memref::ViewOp>();
  if (!viewOp)
    return false;
  auto allocaOp = viewOp.getSource().getDefiningOp<memref::AllocaOp>();
  if (!allocaOp)
    return false;
  auto memSpace = allocaOp.getMemref().getType().getMemorySpace();
  return memSpace != nullptr;
}

/// Emit amdgcn.alloc_lds + get_lds_offset for a promoted buffer.
/// Uses a cache so the same promoted buffer (same memref.view(memref.alloca))
/// gets the same LDS region for both write (copy) and read (matmul).
static Value emitLDSOffset(OpBuilder &builder, Location loc, Value memrefVal,
                           DenseMap<Value, Value> &ldsCache) {
  auto it = ldsCache.find(memrefVal);
  if (it != ldsCache.end())
    return it->second;

  auto viewOp = memrefVal.getDefiningOp<memref::ViewOp>();
  auto allocaOp = viewOp.getSource().getDefiningOp<memref::AllocaOp>();
  int64_t sizeBytes = allocaOp.getMemref().getType().getNumElements();
  auto ldsAlloc = AllocLDSOp::create(builder, loc, /*dynamic_size=*/Value(),
                                     sizeBytes, /*alignment=*/16,
                                     /*offset=*/IntegerAttr{});
  auto ldsOffset =
      GetLDSOffsetOp::create(builder, loc, builder.getIndexType(), ldsAlloc);
  Value result = builder.create<arith::AddIOp>(loc, ldsOffset.getResult(),
                                               viewOp.getByteShift());
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
        // Decompose global memref into (!sx2, byte_stride)
        auto [ptrVal, byteStride] =
            decomposeGlobalMemref(builder, loc, operand);
        callArgs.push_back(ptrVal);
        argTypes.push_back(sx2Ty);
        callArgs.push_back(byteStride);
        argTypes.push_back(indexTy);
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

    moduleOp->walk([&](linalg::FillOp op) {
      replaceWithCall(builder, declBlock, op, "fill", toErase, ldsCache);
    });
    moduleOp->walk([&](linalg::CopyOp op) {
      replaceWithCall(builder, declBlock, op, "copy", toErase, ldsCache);
    });
    moduleOp->walk([&](linalg::MatmulOp op) {
      replaceWithCall(builder, declBlock, op, "mfma_matmul", toErase, ldsCache);
    });
    // Also handle linalg.generic with matmul-like semantics (e.g.,
    // matmul_transpose_b expressed as generic with (m,n,k)->(m,k),(n,k),(m,n)).
    moduleOp->walk([&](linalg::GenericOp op) {
      if (op.getNumDpsInputs() == 2 && op.getNumDpsInits() == 1 &&
          op.getNumReductionLoops() == 1)
        replaceWithCall(builder, declBlock, op, "mfma_matmul", toErase,
                        ldsCache);
    });

    for (auto *op : toErase)
      op->erase();

    // DCE unused alloca/view/dealloc.
    SmallVector<Operation *> deadOps;
    moduleOp->walk([&](Operation *op) {
      if (isa<memref::DeallocOp>(op) ||
          (isa<memref::ViewOp, memref::AllocaOp>(op) && op->use_empty()))
        deadOps.push_back(op);
    });
    for (auto *op : deadOps)
      op->erase();
    deadOps.clear();
    moduleOp->walk([&](memref::AllocaOp op) {
      if (op->use_empty())
        deadOps.push_back(op);
    });
    for (auto *op : deadOps)
      op->erase();

    // Erase transform dialect ops.
    if (auto builtinMod = dyn_cast<mlir::ModuleOp>(moduleOp)) {
      SmallVector<Operation *> transformOps;
      for (auto &op : builtinMod.getBody()->getOperations())
        if (op.getDialect() && op.getDialect()->getNamespace() == "transform")
          transformOps.push_back(&op);
      for (auto *op : transformOps)
        op->erase();
      builtinMod->removeAttr("transform.with_named_sequence");
    }
  }
};

} // namespace

namespace mlir::aster::mlir_air {
std::unique_ptr<Pass> createConvertLinalgToAMDGCN() {
  return std::make_unique<ConvertLinalgToAMDGCN>();
}
} // namespace mlir::aster::mlir_air
