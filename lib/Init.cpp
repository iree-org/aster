//===- Init.cpp -----------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Init.h"
#include "aster/CodeGen/Passes.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Interfaces/UpstreamExternalModels.h"
#include "aster/Transforms/Passes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

///
/// Upstream MLIR C++ stuff
///
void mlir::aster::initUpstreamMLIRDialects(DialectRegistry &registry) {
  registry.insert<affine::AffineDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<gpu::GPUDialect>();
  registry.insert<index::IndexDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<ptr::PtrDialect>();
  registry.insert<scf::SCFDialect>();
}

//===----------------------------------------------------------------------===//
// Dialect Inliner Interfaces
//===----------------------------------------------------------------------===//
namespace {
/// Inliner interface for Affine dialect operations.
struct AffineInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All Affine operations can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  /// All Affine regions can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

/// Inliner interface for Index dialect operations.
struct IndexInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All Index operations can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  /// All Index regions can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
//  External model for MemRef to support SROA.
//===----------------------------------------------------------------------===//
namespace {

/// Walks over the indices of the elements of a tensor of a given
/// `shape` by updating `index` in place to the next index. This returns failure
/// updating `index` in place to the next index. This returns failure if the
/// provided index was the last index.
static LogicalResult nextIndex(ArrayRef<int64_t> shape,
                               MutableArrayRef<int64_t> index) {
  for (size_t i = 0; i < shape.size(); ++i) {
    index[i]++;
    if (index[i] < shape[i])
      return success();
    index[i] = 0;
  }
  return failure();
}

/// Calls `walker` for each index within a tensor of a given `shape`, providing
/// the index as an array attribute of the coordinates.
template <typename CallableT>
static void walkIndicesAsAttr(MLIRContext *ctx, ArrayRef<int64_t> shape,
                              CallableT &&walker) {
  Type indexType = IndexType::get(ctx);
  SmallVector<int64_t> shapeIter(shape.size(), 0);
  do {
    SmallVector<Attribute> indexAsAttr;
    for (int64_t dim : shapeIter)
      indexAsAttr.push_back(IntegerAttr::get(indexType, dim));
    walker(ArrayAttr::get(ctx, indexAsAttr));
  } while (succeeded(nextIndex(shape, shapeIter)));
}

struct MemRefDestructurableTypeExternalModel
    : public DestructurableTypeInterface::ExternalModel<
          MemRefDestructurableTypeExternalModel, MemRefType> {
  std::optional<DenseMap<Attribute, Type>>
  getSubelementIndexMap(Type type) const {
    auto memrefType = llvm::cast<MemRefType>(type);
    constexpr int64_t maxMemrefSizeForDestructuring = 1024;
    if (!memrefType.hasStaticShape() ||
        memrefType.getNumElements() > maxMemrefSizeForDestructuring ||
        memrefType.getNumElements() == 1)
      return {};

    DenseMap<Attribute, Type> destructured;
    walkIndicesAsAttr(
        memrefType.getContext(), memrefType.getShape(), [&](Attribute index) {
          destructured.insert({index, memrefType.getElementType()});
        });

    return destructured;
  }

  Type getTypeAtIndex(Type type, Attribute index) const {
    auto memrefType = llvm::cast<MemRefType>(type);
    auto coordArrAttr = llvm::dyn_cast<ArrayAttr>(index);
    if (!coordArrAttr || coordArrAttr.size() != memrefType.getShape().size())
      return {};

    Type indexType = IndexType::get(memrefType.getContext());
    for (const auto &[coordAttr, dimSize] :
         llvm::zip(coordArrAttr, memrefType.getShape())) {
      auto coord = llvm::dyn_cast<IntegerAttr>(coordAttr);
      if (!coord || coord.getType() != indexType || coord.getInt() < 0 ||
          coord.getInt() >= dimSize)
        return {};
    }

    return memrefType.getElementType();
  }
};

void registerMemorySlotExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    MemRefType::attachInterface<MemRefDestructurableTypeExternalModel>(*ctx);
  });
}

} // namespace

/// See mlir/lib/RegisterAllExtensions.cpp
void mlir::aster::registerUpstreamMLIRInterfaces(DialectRegistry &registry) {
  mlir::func::registerInlinerExtension(registry);

  // Register Affine and Index dialect inliner interface
  registry.addExtension(+[](MLIRContext *ctx, affine::AffineDialect *dialect) {
    dialect->addInterfaces<AffineInlinerInterface>();
  });
  registry.addExtension(+[](MLIRContext *ctx, index::IndexDialect *dialect) {
    dialect->addInterfaces<IndexInlinerInterface>();
  });

  // Register all conversions to LLVM extensions.
  // registerConvertArithToEmitCInterface(registry);
  // arith::registerConvertArithToLLVMInterface(registry);
  // registerConvertComplexToLLVMInterface(registry);
  // cf::registerConvertControlFlowToLLVMInterface(registry);
  // func::registerAllExtensions(registry);
  // tensor::registerAllExtensions(registry);
  // registerConvertFuncToEmitCInterface(registry);
  // registerConvertFuncToLLVMInterface(registry);
  // index::registerConvertIndexToLLVMInterface(registry);
  // registerConvertMathToLLVMInterface(registry);
  // mpi::registerConvertMPIToLLVMInterface(registry);
  // registerConvertMemRefToEmitCInterface(registry);
  // registerConvertMemRefToLLVMInterface(registry);
  // registerConvertNVVMToLLVMInterface(registry);
  // ptr::registerConvertPtrToLLVMInterface(registry);
  // registerConvertOpenMPToLLVMInterface(registry);
  // registerConvertSCFToEmitCInterface(registry);
  // ub::registerConvertUBToLLVMInterface(registry);
  // registerConvertAMXToLLVMInterface(registry);
  // gpu::registerConvertGpuToLLVMInterface(registry);
  // NVVM::registerConvertGpuToNVVMInterface(registry);
  // vector::registerConvertVectorToLLVMInterface(registry);
  // registerConvertXeVMToLLVMInterface(registry);

  // // Register all transform dialect extensions.
  // affine::registerTransformDialectExtension(registry);
  // bufferization::registerTransformDialectExtension(registry);
  // dlti::registerTransformDialectExtension(registry);
  // func::registerTransformDialectExtension(registry);
  // gpu::registerTransformDialectExtension(registry);
  // linalg::registerTransformDialectExtension(registry);
  // memref::registerTransformDialectExtension(registry);
  // nvgpu::registerTransformDialectExtension(registry);
  // scf::registerTransformDialectExtension(registry);
  // sparse_tensor::registerTransformDialectExtension(registry);
  // tensor::registerTransformDialectExtension(registry);
  // transform::registerDebugExtension(registry);
  // transform::registerIRDLExtension(registry);
  // transform::registerLoopExtension(registry);
  // transform::registerPDLExtension(registry);
  // transform::registerSMTExtension(registry);
  // transform::registerTuneExtension(registry);
  // vector::registerTransformDialectExtension(registry);
  // xegpu::registerTransformDialectExtension(registry);
  // arm_neon::registerTransformDialectExtension(registry);
  // arm_sve::registerTransformDialectExtension(registry);
}

/// See mlir/lib/RegisterAllDialects.cpp
void mlir::aster::registerUpstreamMLIRExternalModels(
    DialectRegistry &registry) {
  // Register all external models.
  affine::registerValueBoundsOpInterfaceExternalModels(registry);
  // arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  // arith::registerBufferizableOpInterfaceExternalModels(registry);
  // arith::registerBufferViewFlowOpInterfaceExternalModels(registry);
  // arith::registerShardingInterfaceExternalModels(registry);
  arith::registerValueBoundsOpInterfaceExternalModels(registry);
  // bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
  //     registry);
  builtin::registerCastOpInterfaceExternalModels(registry);
  // cf::registerBufferizableOpInterfaceExternalModels(registry);
  // cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  // gpu::registerBufferDeallocationOpInterfaceExternalModels(registry);
  // gpu::registerValueBoundsOpInterfaceExternalModels(registry);
  // LLVM::registerInlinerInterface(registry);
  // NVVM::registerInlinerInterface(registry);
  // linalg::registerAllDialectInterfaceImplementations(registry);
  // linalg::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  memref::registerAllocationOpInterfaceExternalModels(registry);
  // memref::registerBufferViewFlowOpInterfaceExternalModels(registry);
  // memref::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  memref::registerValueBoundsOpInterfaceExternalModels(registry);

  //
  // Technically we need this but upstream hardcodes a small constant...
  // memref::registerMemorySlotExternalModels(registry);
  // so we copy our own implementation here.
  registerMemorySlotExternalModels(registry);

  // ml_program::registerBufferizableOpInterfaceExternalModels(registry);
  // scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  // scf::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerValueBoundsOpInterfaceExternalModels(registry);
  // shape::registerBufferizableOpInterfaceExternalModels(registry);
  // sparse_tensor::registerBufferizableOpInterfaceExternalModels(registry);
  // tensor::registerBufferizableOpInterfaceExternalModels(registry);
  // tensor::registerFindPayloadReplacementOpInterfaceExternalModels(registry);
  // tensor::registerInferTypeOpInterfaceExternalModels(registry);
  // tensor::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  // tensor::registerSubsetOpInterfaceExternalModels(registry);
  // tensor::registerTilingInterfaceExternalModels(registry);
  // tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  // tosa::registerShardingInterfaceExternalModels(registry);
  // vector::registerBufferizableOpInterfaceExternalModels(registry);
  // vector::registerSubsetOpInterfaceExternalModels(registry);
  // vector::registerValueBoundsOpInterfaceExternalModels(registry);
  // NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  // ROCDL::registerROCDLTargetInterfaceExternalModels(registry);
  // spirv::registerSLSIRVTargetInterfaceExternalModels(registry);
  // xevm::registerXeVMTargetInterfaceExternalModels(registry);
}

/// See mlir/lib/RegisterAllPasses.cpp
void mlir::aster::registerUpstreamMLIRPasses() {
  // General passes

  registerTransformsPasses();
  // mlir::registerCanonicalizerPass();
  // mlir::registerCSEPass();
  // mlir::registerInlinerPass();
  // mlir::registerSymbolDCEPass();
  // mlir::registerSROAPass();
  // mlir::registerSCCPPass();
  // mlir::registerMem2RegPass();

  // // Conversion passes
  // registerConversionPasses();

  // // Dialect passes
  // acc::registerOpenACCPasses();
  affine::registerAffinePasses();
  // amdgpu::registerAMDGPUPasses();
  // registerAsyncPasses();
  // arith::registerArithPasses();
  // bufferization::registerBufferizationPasses();
  // func::registerFuncPasses();
  // registerGPUPasses();
  // registerLinalgPasses();
  // registerNVGPUPasses();
  // registerSparseTensorPasses();
  // LLVM::registerLLVMPasses();
  // LLVM::registerTargetLLVMIRTransformsPasses();
  // math::registerMathPasses();
  // memref::registerMemRefPasses();
  // shard::registerShardPasses();
  // ml_program::registerMLProgramPasses();
  // omp::registerOpenMPPasses();
  // quant::registerQuantPasses();
  // registerSCFPasses();
  // registerShapePasses();
  // spirv::registerSLSIRVPasses();
  // tensor::registerTensorPasses();
  // tosa::registerTosaOptPasses();
  // transform::registerTransformPasses();
  // vector::registerVectorPasses();
  // arm_sme::registerArmSMEPasses();
  // arm_sve::registerArmSVEPasses();
  // emitc::registerEmitCPasses();
  // xegpu::registerXeGPUPasses();

  // // Dialect pipelines
  // bufferization::registerBufferizationPipelines();
  // sparse_tensor::registerSparseTensorPipelines();
  // tosa::registerTosaToLinalgPipelines();
  // gpu::registerGPUToNVVMPipeline();
  // gpu::registerGPUToXeVMPipeline();
}

///
/// Upstream MLIR CAPI stuff
///
void mlir::aster::asterRegisterUpstreamMLIRDialects(
    MlirDialectRegistry registry) {
  initUpstreamMLIRDialects(*unwrap(registry));
}

void mlir::aster::asterRegisterUpstreamMLIRInterfaces(
    MlirDialectRegistry registry) {
  registerUpstreamMLIRInterfaces(*unwrap(registry));
}

void mlir::aster::asterRegisterUpstreamMLIRExternalModels(
    MlirDialectRegistry registry) {
  registerUpstreamMLIRExternalModels(*unwrap(registry));
}

///
/// Aster C++ stuff
///
void mlir::aster::initDialects(DialectRegistry &registry) {
  registry.insert<amdgcn::AMDGCNDialect>();
  registry.insert<lsir::LSIRDialect>();
  registry.insert<aster_utils::AsterUtilsDialect>();
  registerUpstreamExternalModels(registry);
}

void mlir::aster::registerPasses() {
  amdgcn::registerAMDGCNPasses();
  aster_utils::registerAsterUtilsPasses();
  aster::registerAsterPasses();
  aster::registerCodeGenPasses();
}

///
/// Aster CAPI stuff
///
void mlir::aster::asterInitDialects(MlirDialectRegistry registry) {
  initDialects(*unwrap(registry));
}
