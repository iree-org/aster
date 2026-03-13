// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/IR/Interfaces/KernelArgInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "water/Dialect/Wave/Transforms/Passes.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "water/Dialect/Wave/Transforms/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include <functional>

using namespace mlir;
using namespace wave;

namespace wave {
#define GEN_PASS_DEF_WATERWAVEASTERIZEPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace wave

// FIXME: copied from WaveInterfaces.cpp
/// Parse and validate wave constraints from an attribute array.
/// Returns the hardware constraint or nullptr on failure.
static wave::HardwareConstraintAttr parseWaveConstraints(
    Location loc, Attribute constraints,
    llvm::DenseMap<wave::WaveSymbolAttr, llvm::SmallVector<Attribute>>
        &symbolConstraints) {
  wave::HardwareConstraintAttr hardwareConstraint;
  for (Attribute constraint : llvm::cast<ArrayAttr>(constraints)) {
    if (auto workgroup =
            llvm::dyn_cast<wave::WorkgroupConstraintAttr>(constraint)) {
      symbolConstraints[workgroup.getDim()].push_back(workgroup);
    } else if (auto tiling =
                   llvm::dyn_cast<wave::TilingConstraintAttr>(constraint)) {
      symbolConstraints[tiling.getDim()].push_back(tiling);
    } else if (auto waveConstraint =
                   llvm::dyn_cast<wave::WaveConstraintAttr>(constraint)) {
      symbolConstraints[waveConstraint.getDim()].push_back(waveConstraint);
    } else if (auto hardware =
                   llvm::dyn_cast<wave::HardwareConstraintAttr>(constraint)) {
      assert(hardwareConstraint == nullptr &&
             "multiple hardware constraints are not supported");
      hardwareConstraint = hardware;
    } else {
      emitError(loc) << "unsupported constraint type: " << constraint;
      return nullptr;
    }
  }

  if (!hardwareConstraint) {
    emitError(loc) << "expected a hardware constraint";
    return nullptr;
  }

  return hardwareConstraint;
}

namespace {

struct WaterWaveAsterizePass
    : public wave::impl::WaterWaveAsterizePassBase<WaterWaveAsterizePass> {
  void runOnOperation() override {
    getOperation()->walk([&](func::FuncOp funcOp) {
      OpBuilder builder(funcOp);
      SmallVector<aster::amdgcn::KernelArgAttrInterface> argAttributes;
      argAttributes.reserve(funcOp.getNumArguments());
      for (BlockArgument arg : funcOp.getArguments()) {
        bool isReadOnly = true;
        bool isWriteOnly = true;
        for (Operation *user : arg.getUsers()) {
          if (!isa<wave::ReadOp>(user))
            isReadOnly = false;
          if (!isa<wave::WriteOp>(user))
            isWriteOnly = false;
        }
        argAttributes.push_back(aster::amdgcn::BufferArgAttr::get(
            builder.getContext(), aster::amdgcn::AddressSpaceKind::Generic,
            isReadOnly ? aster::amdgcn::AccessKind::ReadOnly
                       : (isWriteOnly ? aster::amdgcn::AccessKind::WriteOnly
                                      : aster::amdgcn::AccessKind::ReadWrite),
            aster::amdgcn::KernelArgumentFlags::None,
            /*name=*/"", builder.getType<mlir::ptr::PtrType>()));
      }

      wave::WaveHyperparameterAttr hyper = wave::getHyperparameters(funcOp);
      // TODO: thread this through as graceful failure
      assert(hyper && "expected hyperparameters on functions");
      uint64_t totalSharedSize = 0;
      funcOp->walk([&](wave::AllocateOp allocate) {
        WaveExprListAttr distributedShape = allocate.getDistributedShape();
        std::optional<SmallVector<int64_t>> numericShape =
            distributedShape.getResolvedShape(hyper);
        // TODO: ditto
        assert(numericShape && "expected numeric shape");
        uint64_t sizeInBytes = 1;
        for (int64_t dim : *numericShape) {
          // TODO: ditto
          assert(dim > 0 && "expected positive, static size");
          sizeInBytes *= dim;
        }
        totalSharedSize += sizeInBytes;
      });
      auto kernel = aster::amdgcn::KernelOp::create(
          builder, funcOp.getLoc(), funcOp.getSymName(), argAttributes,
          static_cast<uint32_t>(totalSharedSize));

      SmallVector<int32_t, 3> blockDims(/*Size=*/3, /*Value=*/1);
      SmallVector<int32_t, 3> waveDims(/*Size=*/3, /*Value=*/1);
      SmallVector<WaveSymbolAttr, 3> blockSymbols(/*Size=*/3);
      auto constraints = funcOp->getAttrOfType<ArrayAttr>(
          wave::WaveDialect::kWaveConstraintsAttrName);
      // TODO: ditto
      assert(constraints &&
             "expected constraints to be present on the function");

      DenseMap<wave::WaveSymbolAttr, SmallVector<Attribute>> symbolConstraints;
      HardwareConstraintAttr hwConstraint = parseWaveConstraints(
          funcOp->getLoc(), constraints, symbolConstraints);
      // TODO: this should just return, we will have complained already at
      // `loc`.
      assert(hwConstraint && "");

      for (auto &&[symbol, constraints] : symbolConstraints) {
        // TODO: at this point, which symbols are MNK for matmul, only where
        // they are mapped. That will only come when we hit an mma op.
        auto wgConstraintIt =
            llvm::find_if(constraints, llvm::IsaPred<WorkgroupConstraintAttr>);
        if (wgConstraintIt == constraints.end())
          continue;

        auto workgroupConstraint =
            cast<WorkgroupConstraintAttr>(*wgConstraintIt);
        std::optional<SmallVector<int64_t>> numericWgTileSize =
            wave::evaluateMapWithHyperparams(
                workgroupConstraint.getTileSize().getMap(),
                workgroupConstraint.getTileSize().getSymbols(), hyper);
        // TODO: convert into an error and return gracefully
        assert(numericWgTileSize && numericWgTileSize->size() == 1 &&
               "expected block size to be computable from hyperparameters");

        auto waveConstraintIt =
            llvm::find_if(constraints, llvm::IsaPred<WaveConstraintAttr>);
        // TODO: ditto
        assert(waveConstraintIt != constraints.end() &&
               "expected a wave constraint to be present along a workgroup "
               "constraint");
        auto waveConstraint = cast<WaveConstraintAttr>(*waveConstraintIt);
        std::optional<SmallVector<int64_t>> numericWaveTileSize =
            wave::evaluateMapWithHyperparams(
                waveConstraint.getTileSize().getMap(),
                waveConstraint.getTileSize().getSymbols(), hyper);
        assert(numericWgTileSize.has_value() &&
               numericWaveTileSize->size() == 1 &&
               "expected wave size to be computable from hyperparameters");

        assert(constraints.size() <= 2 && "unexepected constraint kind");

        switch (workgroupConstraint.getWorkgroupDim().getValue()) {
        case wave::WaveWorkgroupDim::X:
          blockDims[0] = (*numericWgTileSize)[0];
          blockSymbols[0] = workgroupConstraint.getDim();
          waveDims[0] = (*numericWaveTileSize)[0];
          break;
        case wave::WaveWorkgroupDim::Y:
          blockDims[1] = (*numericWgTileSize)[0];
          blockSymbols[1] = workgroupConstraint.getDim();
          waveDims[1] = (*numericWaveTileSize)[0];
          break;
        case wave::WaveWorkgroupDim::Z:
          blockDims[2] = (*numericWgTileSize)[0];
          blockSymbols[2] = workgroupConstraint.getDim();
          waveDims[2] = (*numericWaveTileSize)[0];
          break;
        }
      }

      for (unsigned i = 0; i < 3; ++i) {
        if (!blockSymbols.back()) {
          blockSymbols.pop_back();
          blockDims.pop_back();
        }
      }
      // TODO: later, we can redefine this to be all ones and proceeed
      assert(!blockDims.empty() && "kernel not mapped to blocks and threads");
      SmallVector<int32_t, 3> gridDims;
      SmallVector<int32_t, 3> numWaves;
      gridDims.reserve(blockDims.size());
      numWaves.reserve(blockDims.size());
      for (auto [symbol, blockDim, waveDim] :
           llvm::zip(blockSymbols, blockDims, waveDims)) {
        std::optional<int64_t> totalSize =
            hyper.getSymbolValue(symbol.getName());
        // this must remain as an actual assertion
        assert(totalSize.has_value() &&
               "expected hyperparams to provide values for shapes");
        // TODO: turn into an error
        assert((*totalSize % blockDim) == 0 &&
               "shape not divisible by block size not currently supported");
        gridDims.push_back(*totalSize / blockDim);
        assert((blockDim % waveDim) == 0 &&
               "block size should be divisible by wave size");
        numWaves.push_back(blockDim / waveDim);
      }

      // compute m_tiles, n_tiles, k_tiles
      // m_tiles = m_tiles_wg // m_waves
      // m_dim = m_wg * m_tiles_wg * 16
      // m_wg = gridDims[indexed dimension mapped to M]
      // => m_tiles_wg = m_dim / gridDims[indexed dimension mapped to M] / 16
      // m_waves = waves per WG, can compute from wave constraints, I expect
      // => m_tiles by formula

      // FIXME: XXX: at this point, we don't know which symbols correspond to
      // MNK, we need some sort of propagation from mma, or some better
      // reasoning for tile sizes, for now just go ahead and assume M is mapped
      // to TX and N is mapped to TY.
      assert(gridDims.size() == 2 && "overfit for matmul...");
      int32_t mWgTiles = *hyper.getSymbolValue(blockSymbols[0].getName());
      int32_t nWgTiles = *hyper.getSymbolValue(blockSymbols[1].getName());
      assert((mWgTiles % numWaves[0]) == 0 && "expected divisibility");
      assert((nWgTiles % numWaves[1]) == 0 && "expected divisibility");
      int32_t mTiles = mWgTiles / numWaves[0];
      int32_t nTiles = nWgTiles / numWaves[0];

      // TODO: k_tiles from tiling constraint

      // Linearize grids and blocks into one dim, they will be delinearlized
      // later.
      int32_t numBlocks =
          llvm::accumulate(gridDims, 1, std::multiplies<int32_t>());
      int32_t numThreads =
          llvm::accumulate(blockDims, 1, std::multiplies<int32_t>());
      kernel.setGridDims(ArrayRef<int32_t>{numBlocks, 1, 1});
      kernel.setBlockDims(ArrayRef<int32_t>{numThreads, 1, 1});

      // Arguments.
      builder.setInsertionPointToStart(&kernel.getBodyRegion().front());
      auto sgpr2x = aster::amdgcn::SGPRType::get(builder.getContext(),
                                                 aster::Register(), /*size=*/2);
      SmallVector<Value> kernelArgs;
      kernelArgs.reserve(funcOp.getNumArguments());
      for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
        kernelArgs.push_back(aster::amdgcn::LoadArgOp::create(
            builder, funcOp.getArgument(i).getLoc(), sgpr2x,
            /*index=*/i));
      }
      aster::amdgcn::S_WAITCNT::create(builder, kernel.getLoc());

      // Delinearize block id into workgroup coordinates.
      Value linearBlockId =
          gpu::BlockIdOp::create(builder, funcOp.getLoc(), gpu::Dimension::x);
      SmallVector<Value, 3> gridSizeBasis =
          llvm::map_to_vector<3>(gridDims, [&](int32_t d) -> Value {
            return arith::ConstantIndexOp::create(builder, funcOp.getLoc(), d);
          });
      affine::AffineDelinearizeIndexOp::create(builder, funcOp.getLoc(),
                                               linearBlockId, gridSizeBasis);

      Value linearThreadId =
          gpu::ThreadIdOp::create(builder, funcOp.getLoc(), gpu::Dimension::x);
      Value threadsPerWave = arith::ConstantIndexOp::create(
          builder, funcOp.getLoc(), hwConstraint.getThreadsPerWave());
      Value linearWaveId = arith::FloorDivSIOp::create(
          builder, funcOp.getLoc(), linearThreadId, threadsPerWave);
      SmallVector<Value, 3> wavesBasis =
          llvm::map_to_vector<3>(numWaves, [&](int32_t d) -> Value {
            return arith::ConstantIndexOp::create(builder, funcOp.getLoc(), d);
          });
      affine::AffineDelinearizeIndexOp::create(builder, funcOp.getLoc(),
                                               linearWaveId, wavesBasis);

      // TODO: how do we know it's a matmul to start with? do we care?

      funcOp->erase();
    });
  }
};

} // namespace
