// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/IR/Interfaces/KernelArgInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Transforms/SchedUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "water/Dialect/Wave/Transforms/Passes.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.h"

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

struct PipelineStages {
  int32_t globalLoad;
  int32_t dsWrite;
  int32_t dsRead;
  int32_t compute;
};

// Returns the pipeline stage indices for the given number of stages.
static FailureOr<PipelineStages> getPipelineStages(int32_t stages) {
  switch (stages) {
  case 1:
    return PipelineStages{0, 0, 0, 0};
  case 2:
    return PipelineStages{0, 1, 1, 1};
  case 3:
    return PipelineStages{0, 1, 2, 2};
  case 4:
    return PipelineStages{0, 1, 2, 3};
  case 5:
    return PipelineStages{0, 2, 3, 4};
  case 6:
    return PipelineStages{0, 3, 4, 5};
  }
  return failure();
}

struct WaterWaveAsterizePass
    : public wave::impl::WaterWaveAsterizePassBase<WaterWaveAsterizePass> {
  using WaterWaveAsterizePassBase::WaterWaveAsterizePassBase;

  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());
    // Wraps a freshly-created func::CallOp: if the callee has no declaration
    // in the parent module, inserts a private func.func for it. Returns the
    // same CallOp so the wrapper can be used inline.
    auto ensureCalledFuncExists = [&symbolTable, parentOp = getOperation()](
                                      OpBuilder &builder,
                                      func::CallOp callOp) -> func::CallOp {
      OpBuilder::InsertionGuard guard(builder);
      StringRef funcName = callOp.getCallee();
      if (!symbolTable.lookup(funcName)) {
        FunctionType funcType =
            FunctionType::get(callOp.getContext(), callOp.getOperandTypes(),
                              callOp.getResultTypes());
        Block &moduleBody = parentOp->getRegion(0).front();
        builder.setInsertionPointToEnd(&moduleBody);
        func::FuncOp funcDecl =
            func::FuncOp::create(builder, callOp.getLoc(), funcName, funcType);
        funcDecl.setPrivate();
        symbolTable.insert(funcDecl);
      }
      return callOp;
    };

    FailureOr<PipelineStages> maybeStages = getPipelineStages(stages);
    if (failed(maybeStages)) {
      getOperation()->emitError()
          << "invalid number of pipeline stages: " << stages;
      return signalPassFailure();
    }

    WalkResult walkResult = getOperation()->walk([&](func::FuncOp funcOp) {
      wave::MmaOp singleMmaOp;
      WalkResult walkResult = funcOp->walk([&](wave::MmaOp mmaOp) {
        if (singleMmaOp) {
          mmaOp.emitError() << "NYI: multiple mmas in the function";
          return WalkResult::interrupt();
        }
        singleMmaOp = mmaOp;
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted())
        return WalkResult::interrupt();
      if (!singleMmaOp) {
        funcOp.emitError() << "NYI: expected a single mma op in the function";
        return WalkResult::interrupt();
      }
      auto iterateOp = dyn_cast<wave::IterateOp>(singleMmaOp->getParentOp());
      if (!iterateOp) {
        singleMmaOp.emitError()
            << "NYI: expected a mma op to be nested in an iterate op";
        return WalkResult::interrupt();
      }
      wave::WaveSymbolAttr kSymbol = iterateOp.getIterator();
      auto accumulatorType = dyn_cast<wave::WaveTensorType>(
          singleMmaOp.getAccumulator().getType());
      if (!accumulatorType || !accumulatorType.getFullySpecified()) {
        // TODO: we have a normal form precondition for this.
        singleMmaOp.emitError() << "expected fully-specified tensor types";
        return WalkResult::interrupt();
      }
      wave::WaveSymbolAttr mSymbol =
          accumulatorType.getShape().drop_back().back();
      wave::WaveSymbolAttr nSymbol = accumulatorType.getShape().back();
      IRMapping valueMap;

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
            /*name=*/"", Type()));
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
      // TODO: later, we can redefine this to be all ones and proceed
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

      auto it = llvm::find(blockSymbols, mSymbol);
      if (it == blockSymbols.end()) {
        funcOp.emitError() << "expected MMA m dimension to be mapped";
        return WalkResult::interrupt();
      }
      int32_t mMappedDim = std::distance(blockSymbols.begin(), it);
      it = llvm::find(blockSymbols, nSymbol);
      if (it == blockSymbols.end()) {
        funcOp.emitError() << "expected MMA n dimension to be mapped";
        return WalkResult::interrupt();
      }
      int32_t nMappedDim = std::distance(blockSymbols.begin(), it);

      // compute m_tiles, n_tiles, k_tiles
      // m_tiles = m_tiles_wg // m_waves
      // m_dim = m_wg * m_tiles_wg * 16
      // m_wg = gridDims[indexed dimension mapped to M]
      // => m_tiles_wg = m_dim / gridDims[indexed dimension mapped to M] / 16
      // m_waves = waves per WG, can compute from wave constraints, I expect
      // => m_tiles by formula

      // Starting, we have MxNxK matmul divided into m_wg, n_wg workgroups
      // and m_waves, n_waves waves. The size of the workgroup is given by
      // workgroup constraints.
      // Each workgroup then has a certain number of 16x16x32 tiles processed by
      // a single wave, and expects the number of waves. This gives a larger
      // tile processed by all waves. If tile size is smaller than size of the
      // workgroup, this creates workgroup tiles.
      // In the input, we may have wave constraints. We relate those to
      // workgroup constraints to obtain the number of waves per workgroup. Then
      // compute the rest.

      assert(gridDims.size() == 2 && "overfit for matmul...");
      int32_t mSize = *hyper.getSymbolValue(mSymbol.getName());
      int32_t nSize = *hyper.getSymbolValue(nSymbol.getName());
      int32_t kSize = *hyper.getSymbolValue(kSymbol.getName());
      if (mSize % 16 != 0) {
        singleMmaOp.emitError() << "m dimension not divisible by 16";
        return WalkResult::interrupt();
      }
      if (nSize % 16 != 0) {
        singleMmaOp.emitError() << "n dimension not divisible by 16";
        return WalkResult::interrupt();
      }
      if (kSize % 32 != 0) {
        singleMmaOp.emitError() << "k dimension not divisible by 32";
        return WalkResult::interrupt();
      }

      if (mSize % gridDims[mMappedDim] != 0) {
        singleMmaOp.emitError()
            << "m dimension " << mSize << " not divisible by grid dimension "
            << gridDims[mMappedDim];
        return WalkResult::interrupt();
      }
      if (nSize % gridDims[nMappedDim] != 0) {
        singleMmaOp.emitError()
            << "n dimension " << nSize << "not divisible by grid dimension "
            << gridDims[nMappedDim];
        return WalkResult::interrupt();
      }
      int32_t mWgTileSize = mSize / gridDims[mMappedDim];
      int32_t nWgTileSize = nSize / gridDims[nMappedDim];
      int32_t mAllWaves = numWaves[mMappedDim] * 16;
      int32_t nAllWaves = numWaves[nMappedDim] * 16;
      if (mWgTileSize % mAllWaves != 0) {
        singleMmaOp.emitError()
            << "m workgroup tile size " << mWgTileSize
            << " not divisible by the number of elements " << mAllWaves
            << " processed by all " << numWaves[mMappedDim] << " waves\n";
        return WalkResult::interrupt();
      }
      if (nWgTileSize % nAllWaves != 0) {
        singleMmaOp.emitError()
            << "n workgroup tile size " << nWgTileSize
            << " not divisible by the number of elements " << nAllWaves
            << " processed by all " << numWaves[nMappedDim] << " waves\n";
        return WalkResult::interrupt();
      }
      int32_t mWgTiles = mWgTileSize / mAllWaves;
      int32_t nWgTiles = nWgTileSize / nAllWaves;
      assert(16 * numWaves[mMappedDim] * mWgTiles * gridDims[mMappedDim] ==
             mSize);
      assert(16 * numWaves[nMappedDim] * nWgTiles * gridDims[nMappedDim] ==
             nSize);

      int32_t kTiles = -1;
      for (Attribute attr : symbolConstraints.lookup(kSymbol)) {
        if (auto tilingConstraint =
                dyn_cast<wave::TilingConstraintAttr>(attr)) {
          if (kTiles != -1) {
            singleMmaOp.emitError()
                << "multiple tiling constraints for k dimension";
            return WalkResult::interrupt();
          }
          std::optional<SmallVector<int64_t>> numericTileSize =
              wave::evaluateMapWithHyperparams(
                  tilingConstraint.getTileSize().getMap(),
                  tilingConstraint.getTileSize().getSymbols(), hyper);
          assert(numericTileSize.has_value() && numericTileSize->size() == 1 &&
                 "expected tile size to be computable from hyperparameters");
          kTiles = kSize / (*numericTileSize)[0];
        }
      }
      if (kTiles == -1) {
        singleMmaOp.emitError() << "no tiling constraint for k dimension";
        return WalkResult::interrupt();
      }

      // Linearize grids and blocks into one dim, they will be delinearlized
      // later.
      int32_t numBlocks =
          llvm::accumulate(gridDims, 1, std::multiplies<int32_t>());
      int32_t numThreads =
          hwConstraint.getThreadsPerWave() *
          llvm::accumulate(numWaves, 1, std::multiplies<int32_t>());
      kernel.setGridDims(ArrayRef<int32_t>{numBlocks, 1, 1});
      kernel.setBlockDims(ArrayRef<int32_t>{numThreads, 1, 1});

      // Arguments.

      kernel.getBodyRegion().takeBody(funcOp.getBody());
      Block *body = &kernel.getBodyRegion().front();
      builder.setInsertionPointToStart(body);
      auto sgpr2x = aster::amdgcn::SGPRType::get(builder.getContext(),
                                                 aster::Register(), /*size=*/2);
      SmallVector<Value> kernelArgs;
      kernelArgs.reserve(body->getNumArguments());
      for (unsigned i = 0, e = body->getNumArguments(); i < e; ++i) {
        kernelArgs.push_back(aster::amdgcn::LoadArgOp::create(
            builder, body->getArgument(i).getLoc(), sgpr2x,
            /*index=*/i));
      }
      // XXX: need to provide explicit "default" counts for vmcnt and expcnt
      // because the default builder provides 0, which are *NOT* default values.
      aster::amdgcn::S_WAITCNT::create(builder, kernel.getLoc(), /*vmcnt=*/64,
                                       /*expcnt=*/8, /*lgkmcnt=*/0);
      for (unsigned i = 0, e = body->getNumArguments(); i < e; ++i) {
        auto anyType =
            aster::aster_utils::AnyTypeType::get(builder.getContext());
        kernelArgs[i] =
            ensureCalledFuncExists(
                builder,
                func::CallOp::create(builder, body->getArgument(i).getLoc(),
                                     "prepare_ptr", anyType, kernelArgs[i]))
                ->getResult(0);
      }

      // Delinearize block id into workgroup coordinates.
      Value linearBlockId =
          gpu::BlockIdOp::create(builder, funcOp.getLoc(), gpu::Dimension::x);
      SmallVector<Value, 3> gridSizeBasis =
          llvm::map_to_vector<3>(gridDims, [&](int32_t d) -> Value {
            return arith::ConstantIndexOp::create(builder, funcOp.getLoc(), d);
          });
      auto wgCoordinates = affine::AffineDelinearizeIndexOp::create(
          builder, funcOp.getLoc(), linearBlockId, gridSizeBasis);

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
      auto waveCoordinates = affine::AffineDelinearizeIndexOp::create(
          builder, funcOp.getLoc(), linearWaveId, wavesBasis);

      // Prepare "base" values for each dimension.
      AffineExpr d0, d1;
      AffineExpr s0, s1;
      bindDims(builder.getContext(), d0, d1);
      bindSymbols(builder.getContext(), s0, s1);

      // TODO: let's not restrict this to hardcoded m,n and do it for all
      // dimensions?
      struct DimensionIndexing {
        Value base;
        Value waveBase;
        Value wgTiles;
        Value waveTiles;
      };
      DenseMap<int32_t, DimensionIndexing> dimensionIndexing;
      Location indexingLoc = funcOp.getLoc();
      for (auto mappedDim : {nMappedDim, mMappedDim}) {
        int32_t wgTiles = mappedDim == mMappedDim ? mWgTiles : nWgTiles;
        dimensionIndexing[mappedDim].wgTiles =
            arith::ConstantIndexOp::create(builder, indexingLoc, wgTiles);
        dimensionIndexing[mappedDim].waveTiles = arith::ConstantIndexOp::create(
            builder, indexingLoc, wgTiles / numWaves[mappedDim]);
        dimensionIndexing[mappedDim].base = affine::AffineApplyOp::create(
            builder, indexingLoc, ArrayRef<AffineExpr>({d0 * s0 + d1 * s1}),
            {wgCoordinates->getResult(mappedDim),
             waveCoordinates->getResult(mappedDim),
             dimensionIndexing[mappedDim].wgTiles,
             dimensionIndexing[mappedDim].waveTiles});
        dimensionIndexing[mappedDim].waveBase = affine::AffineApplyOp::create(
            builder, indexingLoc, ArrayRef<AffineExpr>({d0 * s0}),
            {waveCoordinates->getResult(mappedDim),
             dimensionIndexing[mappedDim].waveTiles});
      }

      // Initialize accumulators.
      DenseMap<wave::RegisterOp, Value> registers;
      walkResult = kernel->walk([&](wave::RegisterOp registerOp) {
        FloatAttr value;
        if (!matchPattern(registerOp.getInit(), m_Constant(&value))) {
          registerOp.emitError() << "NYI: non-constant register initializer";
          return WalkResult::interrupt();
        }
        if (!value.getValue().isZero()) {
          registerOp.emitError() << "NYI: non-zero register initializer";
          return WalkResult::interrupt();
        }
        if (!isa<Float32Type>(wave::getElementType(registerOp.getType()))) {
          registerOp.emitError() << "NYI: non-f32 register type";
          return WalkResult::interrupt();
        }
        auto registerType =
            dyn_cast<wave::WaveTensorType>(registerOp.getType());
        // TODO: we have a normal form for this.
        if (!registerType || !registerType.getFullySpecified()) {
          registerOp.emitError() << "expected fully-specified register type";
          return WalkResult::interrupt();
        }
        for (WaveSymbolAttr symbol : registerType.getShape()) {
          if (symbol != mSymbol && symbol != nSymbol) {
            // We likely want some generic mapping "symbol -> number of tiles"
            // for this purpose.
            registerOp.emitError() << "NYI: non-m/n dimension in register type";
            return WalkResult::interrupt();
          }
        }

        builder.setInsertionPoint(registerOp);
        Value allocationSize =
            arith::MulIOp::create(builder, registerOp.getLoc(),
                                  dimensionIndexing[mMappedDim].waveTiles,
                                  dimensionIndexing[nMappedDim].waveTiles);

        // TODO: not sure if this is always 4 agprs.
        auto agpr4x = aster::amdgcn::AGPRType::get(
            builder.getContext(), aster::Register(), /*size=*/4);
        auto memrefType = MemRefType::get({ShapedType::kDynamic}, agpr4x);
        Value alloca = memref::AllocaOp::create(builder, registerOp.getLoc(),
                                                memrefType, allocationSize);
        valueMap.map(registerOp.getResult(), alloca);
        auto zero =
            arith::ConstantIndexOp::create(builder, registerOp.getLoc(), 0);
        auto one =
            arith::ConstantIndexOp::create(builder, registerOp.getLoc(), 1);

        scf::ForOp forOp = scf::ForOp::create(
            builder, registerOp.getLoc(),
            /*lowerBound=*/zero,
            /*upperBound=*/allocationSize,
            /*step=*/one,
            /*initArgs=*/ValueRange{},
            /*bodyBuilder=*/
            [alloca, agpr4x, &ensureCalledFuncExists](
                OpBuilder &builder, Location loc, Value iv, ValueRange) {
              Value call = ensureCalledFuncExists(
                               builder, func::CallOp::create(builder, loc,
                                                             "zero_C", agpr4x))
                               .getResult(0);
              memref::StoreOp::create(builder, loc, call, alloca, iv);
              scf::YieldOp::create(builder, loc, ValueRange());
            });
        forOp->setAttr("aster.constexpr", builder.getUnitAttr());

        registers.try_emplace(registerOp, alloca);
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted())
        return WalkResult::interrupt();

      // K-loop
      walkResult = kernel->walk([&](wave::IterateOp iterateOp) {
        builder.setInsertionPoint(iterateOp);

        if (iterateOp.getIterator() != kSymbol) {
          iterateOp.emitError() << "NYI: non-k dimension in iterate op";
          return WalkResult::interrupt();
        }

        Value zero =
            arith::ConstantIndexOp::create(builder, iterateOp.getLoc(), 0);
        Value cKT = arith::ConstantIndexOp::create(
            builder, iterateOp.getLoc(),
            kTiles); // TODO: IMPORTANT this must be c_K_T equivalent
        if (*hyper.getSymbolValue(kSymbol.getName()) % 32 != 0) {
          iterateOp.emitError() << "NYI: k dimension not divisible by 32";
          return WalkResult::interrupt();
        }
        // This is confusingly called K_TILES which is different from K_T that
        // is set from cfg.k_tiles.
        Value kTilesForLoop = arith::ConstantIndexOp::create(
            builder, iterateOp.getLoc(),
            *hyper.getSymbolValue(kSymbol.getName()) / 32);
        auto loop = scf::ForOp::create(builder, iterateOp.getLoc(),
                                       /*lowerBound=*/zero,
                                       /*upperBound=*/kTilesForLoop,
                                       /*step=*/cKT);

        loop.getBodyRegion().takeBody(iterateOp.getBody());
        // FIXME: this should go through a rewriter once there is one. We will
        // also need to be much smarter about replacing iter args with
        // essentially a bufferized value.
        {
          for (auto [arg, init] : llvm::zip(loop.getBody()->getArguments(),
                                            iterateOp.getIterArgs())) {
            arg.replaceAllUsesWith(init);
          }
          loop.getBody()->eraseArguments(0, loop.getBody()->getNumArguments());
          loop.getBody()->addArgument(builder.getIndexType(),
                                      iterateOp.getLoc());

          Operation *yield = loop.getBody()->getTerminator();
          builder.setInsertionPoint(yield);
          scf::YieldOp::create(builder, yield->getLoc(), ValueRange());
          for (auto [yielded, result] :
               llvm::zip(yield->getOperands(), iterateOp.getResults())) {
            auto mmaOp = yielded.getDefiningOp<wave::MmaOp>();
            if (!mmaOp) {
              // TODO: this will need some sort of DestinationStyle interface on
              // ops so we can figure out what the buffer is.
              yield->emitError() << "NYI: non-mma defining operand of yield op";
              return WalkResult::interrupt();
            }
            result.replaceAllUsesWith(mmaOp.getAccumulator());
          }
          yield->erase();
        }

        iterateOp.erase();
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted())
        return WalkResult::interrupt();

      auto isReadingMMAOperand = [](wave::ReadOp readOp, wave::MmaOp mmaOp,
                                    bool &isA, bool &isB) -> bool {
        if (!llvm::hasSingleElement(readOp->getUses()))
          return false;
        if (cast<WaveTensorType>(readOp.getMemory().getType())
                .getAddressSpaceValue() != wave::WaveAddressSpace::Shared)
          return false;
        for (OpOperand &use : readOp->getUses()) {
          if (use.getOwner() != mmaOp)
            return false;
          if (use.getOperandNumber() ==
              mmaOp.getLhsMutable().getOperandNumber()) {
            isA = true;
            return true;
          } else if (use.getOperandNumber() ==
                     mmaOp.getRhsMutable().getOperandNumber()) {
            isB = true;
            return true;
          }
        }
        return false;
      };

      // Collect all reads that need to be marshalled through shared memory.
      SmallVector<wave::ReadOp> readsThroughSharedMemory;
      kernel->walk([&](wave::ReadOp readOp) {
        if (cast<WaveTensorType>(readOp.getMemory().getType())
                .getAddressSpaceValue() != wave::WaveAddressSpace::Shared)
          return WalkResult::advance();
        readsThroughSharedMemory.push_back(readOp);
        return WalkResult::advance();
      });

      // Work by groups of reads adjacent in the block, they may share a
      // barrier.
      for (auto firstReadIt = readsThroughSharedMemory.begin(),
                lastReadIt = std::next(firstReadIt),
                finalReadIt = readsThroughSharedMemory.end();
           firstReadIt != finalReadIt; firstReadIt = lastReadIt) {
        while (true) {
          if (lastReadIt == finalReadIt)
            break;
          if ((*lastReadIt) != (*std::prev(lastReadIt))->getNextNode())
            break;
          std::advance(lastReadIt, 1);
        }

        builder.setInsertionPointAfter(*std::prev(lastReadIt));
        auto barrier = aster::amdgcn::inst::SOPPOp::create(
            builder, (*firstReadIt)->getLoc(), aster::amdgcn::OpCode::S_BARRIER,
            0);
        barrier->setAttr(
            aster::kSchedStageAttr,
            IntegerAttr::get(builder.getI32Type(), maybeStages->dsRead));
        // TODO: this can be done by some sort of scheduling / interleaving
        // instead.
        OperationState dummyOp(builder.getUnknownLoc(),
                               "waster.insertion_placeholder");
        builder.setInsertionPoint(barrier);
        Operation *allocIP = builder.create(dummyOp);
        Operation *glReadIP = builder.create(dummyOp);
        Operation *ldsComputeWriteAddrIP = builder.create(dummyOp);
        Operation *ldsComputeReadAddrIP = builder.create(dummyOp);
        Operation *ldsStoreIP = builder.create(dummyOp);
        builder.setInsertionPointAfter(barrier);
        Operation *waitIP = builder.create(dummyOp);
        Operation *ldsReadIP = builder.create(dummyOp);

        wave::ReadOp readOp = *firstReadIt;
        // Index computations are hardcoded specifically for mma in a loop...
        // Can be used either by a store to LDS or by the MMA.
        if (!llvm::hasSingleElement(readOp->getUses())) {
          readOp.emitError() << "NYI: multiple uses of read op";
          return WalkResult::interrupt();
        }
        auto parentLoop = dyn_cast<scf::ForOp>(readOp->getParentOp());
        if (!parentLoop) {
          readOp.emitError() << "NYI: read op not nested in a loop";
          return WalkResult::interrupt();
        }

        for (auto it = firstReadIt; it != lastReadIt; ++it) {
          readOp = *it;
          auto readType = cast<WaveTensorType>(readOp.getMemory().getType());
          // TODO: we have a normal form for fully-specified types.
          if (!readType || !readType.getFullySpecified()) {
            readOp.emitError()
                << "expected fully-specified tensor type in shared memory";
            return WalkResult::interrupt();
          }

          bool isA = false;
          bool isB = false;
          bool okay = isReadingMMAOperand(readOp, singleMmaOp, isA, isB);
          if (!okay || (!isA && !isB)) {
            InFlightDiagnostic diag =
                readOp.emitError()
                << "NYI: read op not reading A or B of the mma";
            diag.attachNote(singleMmaOp.getLoc()) << "mma op";
            return WalkResult::interrupt();
          }

          int32_t mappedDim = isA ? mMappedDim : nMappedDim;
          if (readType.getShape()[0] != (isA ? mSymbol : nSymbol) ||
              readType.getShape()[1] != kSymbol) {
            readOp.emitError()
                << "NYI: only MK, NK shapes are currently supported";
            return WalkResult::interrupt();
          }

          builder.setInsertionPoint(allocIP);
          // TODO: later, we can define this generically for any dimension by
          // computing per-dimension tile sizes.
          int32_t ldsBytes = 1;
          bool seenK = false;
          for (WaveSymbolAttr symbol : readType.getShape()) {
            if (symbol == mSymbol) {
              ldsBytes *= mWgTiles;
            } else if (symbol == nSymbol) {
              ldsBytes *= nWgTiles;
            } else if (symbol == kSymbol) {
              seenK = true;
              ldsBytes *=
                  kTiles; // TODO: IMPORTANT this must be c_K_T equivalent
            }
          }
          if (!seenK) {
            // This is likely related to the hardcoded value below.
            readOp.emitError() << "NYI: k dimension not present in tensor type";
            return WalkResult::interrupt();
          }
          // TODO: this is copied from test_perf_001_gemm_fp16_weak_scaled.py
          // that doesn't really explain the magic number.
          ldsBytes *= 1024;

          Value alloc = aster::amdgcn::AllocLDSOp::create(
              builder, readOp.getLoc(),
              /*dynamic_size=*/Value(), ldsBytes,
              /*alignment=*/16,
              /*offset=*/IntegerAttr());
          alloc.getDefiningOp()->setAttr(
              aster::kSchedStageAttr,
              IntegerAttr::get(builder.getI32Type(), maybeStages->globalLoad));
          Value sharedBase = aster::amdgcn::GetLDSOffsetOp::create(
              builder, readOp.getLoc(), builder.getIndexType(), alloc);
          sharedBase.getDefiningOp()->setAttr(
              aster::kSchedStageAttr,
              IntegerAttr::get(builder.getI32Type(), maybeStages->globalLoad));

          builder.setInsertionPoint(glReadIP);
          Value wgTilesValue = dimensionIndexing[mappedDim].wgTiles;
          Value waveTilesValue = dimensionIndexing[mappedDim].waveTiles;
          Value kTilesValue =
              arith::ConstantIndexOp::create(builder, readOp.getLoc(), kTiles);

          int32_t kSize = *hyper.getSymbolValue(kSymbol.getName());
          // TODO: hardcoded for the 16x32 tiles (2 times 16x16x16 mfma).
          int32_t strideAB = kSize * 2;
          Value strideABValue = arith::ConstantIndexOp::create(
              builder, readOp.getLoc(), strideAB);

          Value base = dimensionIndexing[mappedDim].base;
          Value waveBase = dimensionIndexing[mappedDim].waveBase;

          auto asterAny =
              aster::aster_utils::AnyTypeType::get(builder.getContext());
          auto readToken = aster::amdgcn::ReadTokenType::get(
              builder.getContext(), aster::amdgcn::MemoryInstructionKind::Flat);
          auto futureGlobalRead = aster::aster_utils::StructType::get(
              builder.getContext(),
              {builder.getStringAttr("value"), builder.getStringAttr("token")},
              {asterAny, readToken});
          MemRefType resultType =
              MemRefType::get({ShapedType::kDynamic}, futureGlobalRead);

          auto argIt = llvm::find(body->getArguments(), readOp.getMemory());
          if (argIt == body->getArguments().end()) {
            readOp.emitError()
                << "NYI: global reads from something other than func arguments";
            return WalkResult::interrupt();
          }
          Value rsrc =
              kernelArgs[std::distance(body->getArguments().begin(), argIt)];

          // Read from global asynchronously.
          // TODO: hardcoded functions specific to MMA, sad...
          StringRef funcName =
              isA ? "k_load_a_16x32_from_global" : "k_load_b_16x32_from_global";
          Value glReadToken =
              ensureCalledFuncExists(
                  builder, func::CallOp::create(
                               builder, readOp.getLoc(), funcName, resultType,
                               ValueRange({waveTilesValue, kTilesValue, rsrc,
                                           parentLoop.getInductionVar(),
                                           strideABValue, base})))
                  ->getResult(0);

          // Compute LDS write addresses immediately.
          builder.setInsertionPoint(ldsComputeWriteAddrIP);
          StringRef ldsWriteFuncName = isA ? "k_compute_lds_write_addrs_a"
                                           : "k_compute_lds_write_addrs_b";
          // XXX: A/B_TILES_PER_SLICE is same as M/N_TILES_WG in
          // substitutions...
          Value tilesPerSlice = wgTilesValue;
          auto memrefOfIndex =
              MemRefType::get({ShapedType::kDynamic}, builder.getIndexType());
          Value ldsWriteAddr =
              ensureCalledFuncExists(
                  builder,
                  func::CallOp::create(
                      builder, readOp.getLoc(), ldsWriteFuncName, memrefOfIndex,
                      ValueRange({waveTilesValue, kTilesValue, sharedBase,
                                  waveBase, tilesPerSlice})))
                  ->getResult(0);

          // Compute LDS read addresses immediately.
          builder.setInsertionPoint(ldsComputeReadAddrIP);
          StringRef ldsComputeReadAddrFuncName =
              isA ? "k_compute_lds_read_addrs_a" : "k_compute_lds_read_addrs_b";
          Value ldsReadAddr =
              ensureCalledFuncExists(
                  builder,
                  func::CallOp::create(
                      builder, readOp.getLoc(), ldsComputeReadAddrFuncName,
                      memrefOfIndex,
                      ValueRange({waveTilesValue, kTilesValue, sharedBase,
                                  waveBase, tilesPerSlice})))
                  ->getResult(0);

          // Wait flor global loads + store to LDS at pre-computed addresses.
          builder.setInsertionPoint(ldsStoreIP);
          StringRef ldsStoreToLdsAtAddrsFuncName =
              isA ? "k_store_to_lds_at_addrs_a" : "k_store_to_lds_at_addrs_b";
          auto ldsWriteToken = aster::amdgcn::WriteTokenType::get(
              builder.getContext(),
              aster::amdgcn::MemoryInstructionKind::Shared);
          auto ldsWriteTokenBuffer =
              MemRefType::get({ShapedType::kDynamic}, ldsWriteToken);
          Value ldsStoreToken =
              ensureCalledFuncExists(
                  builder,
                  func::CallOp::create(builder, readOp.getLoc(),
                                       ldsStoreToLdsAtAddrsFuncName,
                                       ldsWriteTokenBuffer,
                                       ValueRange({ldsWriteAddr, waveTilesValue,
                                                   kTilesValue, glReadToken})))
                  ->getResult(0);

          // Wait for LDS write tokens.
          builder.setInsertionPoint(waitIP);
          StringRef waitLDSWritesFuncName = "k_wait_lds_writes";
          ensureCalledFuncExists(
              builder, func::CallOp::create(builder, readOp.getLoc(),
                                            waitLDSWritesFuncName, TypeRange(),
                                            ValueRange({ldsStoreToken})));

          builder.setInsertionPoint(ldsReadIP);
          auto ldsReadTokenType = aster::amdgcn::ReadTokenType::get(
              builder.getContext(),
              aster::amdgcn::MemoryInstructionKind::Shared);
          auto ldsReadStruct = aster::aster_utils::StructType::get(
              builder.getContext(),
              {builder.getStringAttr("value"), builder.getStringAttr("token")},
              {asterAny, ldsReadTokenType});
          auto ldsReadFutureType =
              MemRefType::get({ShapedType::kDynamic}, ldsReadStruct);
          StringRef ldsReadFuncName =
              isA ? "k_read_lds_at_addrs_a" : "k_read_lds_at_addrs_b";
          Value ldsReadFuture =
              ensureCalledFuncExists(
                  builder, func::CallOp::create(
                               builder, readOp.getLoc(), ldsReadFuncName,
                               ldsReadFutureType, ValueRange({ldsReadAddr})))
                  ->getResult(0);
          valueMap.map(readOp.getResult(), ldsReadFuture);

          builder.setInsertionPoint(parentLoop.getBody()->getTerminator());
          auto deallocOp = aster::amdgcn::DeallocLDSOp::create(
              builder, readOp.getLoc(), alloc);
          deallocOp->setAttr(
              aster::kSchedStageAttr,
              IntegerAttr::get(builder.getI32Type(), maybeStages->compute));
        }

        allocIP->erase();
        ldsReadIP->erase();
        ldsStoreIP->erase();
        ldsComputeReadAddrIP->erase();
        ldsComputeWriteAddrIP->erase();
        glReadIP->erase();
        waitIP->erase();
      }

      builder.setInsertionPoint(singleMmaOp);

      auto parentLoop = dyn_cast<scf::ForOp>(singleMmaOp->getParentOp());
      assert(parentLoop && "checked above that mma was in a loop, but it "
                           "wasn't converted/inlined");

      // XXX: the factor of 2 is hardcoded from specific sizes.
      Value cKT = arith::ConstantIndexOp::create(
          builder, singleMmaOp.getLoc(),
          kTiles); // TODO: IMPORTANT this must be c_K_T equivalent
      Value kMfma = affine::AffineApplyOp::create(
          builder, singleMmaOp.getLoc(), ArrayRef<AffineExpr>({2 * s0}), {cKT});

      Value aFuture = valueMap.lookupOrNull(singleMmaOp.getLhs());
      if (!aFuture) {
        singleMmaOp.emitError() << "NYI: couldn't convert MMA source value";
        return WalkResult::interrupt();
      }
      Value bFuture = valueMap.lookupOrNull(singleMmaOp.getRhs());
      if (!bFuture) {
        singleMmaOp.emitError() << "NYI: couldn't convert MMA source value";
        return WalkResult::interrupt();
      }
      Value cBuffer = valueMap.lookupOrNull(singleMmaOp.getAccumulator());
      if (!cBuffer) {
        singleMmaOp.emitError() << "NYI: couldn't convert MMA result buffer";
        return WalkResult::interrupt();
      }
      ensureCalledFuncExists(
          builder, func::CallOp::create(
                       builder, singleMmaOp.getLoc(),
                       "k_wait_and_compute_mfmas", TypeRange(),
                       ValueRange({dimensionIndexing[mMappedDim].waveTiles,
                                   dimensionIndexing[nMappedDim].waveTiles,
                                   kMfma, aFuture, bFuture, cBuffer})));

      walkResult = kernel->walk([&](wave::WriteOp writeOp) {
        if (writeOp.getValueToStore() != singleMmaOp.getAccumulator()) {
          writeOp.emitError()
              << "NYI: don't know how to subscript a write that is something "
                 "else than the result/accumulator of the MMA";
          return WalkResult::interrupt();
        }
        // TODO: go through the mapping...
        auto argIt = std::find(body->getArguments().begin(),
                               body->getArguments().end(), writeOp.getMemory());
        if (argIt == body->getArguments().end()) {
          writeOp.emitError() << "NYI: don't know how to get a buffer that is "
                                 "not a function argument";
          return WalkResult::interrupt();
        }
        Value cRsrc =
            kernelArgs[std::distance(body->getArguments().begin(), argIt)];
        // XXX: the logic in
        // test_perf_001_gemm_fp16_weak_scaled.py:_make_substitutions overrides
        // STRIDE_C set in constexpr_substitutions_16x32 previously. The code
        // ifdef'ed below corresponds to overridden value.
        // XXX: hardcoded for 16x32 tiles (2 times 16x16x16 mfma).
#if 0
        llvm::APInt nTiles;
        [[maybe_unused]] bool matched =
        matchPattern(dimensionIndexing[nMappedDim].waveTiles,
                     m_ConstantInt(&nTiles));
        assert(matched && "expected constant number of wave tiles");
        int64_t strideC = nTiles.getSExtValue() * 16 * 4;
#endif
        // TODO: take element bitwidth instead of hardcoding it.
        int64_t strideC = nSize * 4; // f32 = 4 bytes
        builder.setInsertionPoint(writeOp);
        Value strideCValue =
            arith::ConstantIndexOp::create(builder, writeOp.getLoc(), strideC);

        // TODO: use shape instead of hardcoded mMappedDim, nMappedDim (assumes
        // M, N shape).
        Value cBuffer = valueMap.lookupOrNull(singleMmaOp.getAccumulator());
        assert(cBuffer && "checked above that cBuffer was in the value map");
        ensureCalledFuncExists(
            builder,
            func::CallOp::create(
                builder, writeOp.getLoc(), "store_c_tiles", TypeRange(),
                ValueRange({dimensionIndexing[mMappedDim].waveTiles,
                            dimensionIndexing[nMappedDim].waveTiles, cBuffer,
                            cRsrc, strideCValue,
                            dimensionIndexing[mMappedDim].base,
                            dimensionIndexing[nMappedDim].base})));

        writeOp->erase();
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted())
        return WalkResult::interrupt();

      singleMmaOp->erase();
      for (auto readOp : readsThroughSharedMemory) {
        readOp->erase();
      }
      for (auto registerOp : llvm::make_first_range(registers)) {
        registerOp->erase();
      }

      // TODO: erase ops here, we shouldn't immediately erase because the
      // mapping will go stale... If this turns to patterns, this becomes less
      // of a problem, but we can't really because of the need to group ops
      // together per kind.

      // Generalization: each tensor has a shape such as MK, NK, MN, etc. These
      // shapes have certain number of workgroup tiles (?), wave tiles. These
      // are the values passed into the corresponding
      // functions-that-should-become-ops. We can compute these values at the
      // start and carry them around in some sort of context or emit every time
      // and run CSE. When emitting specific function calls, we look up/generate
      // the corresponding values based on the tensor shape, as long as we know
      // the structure of function arguments.
      //
      // For strides, we can similarly compute based on shapes.

      if (!body->getTerminator()->getOperands().empty()) {
        body->getTerminator()->emitError() << "NYI: non-empty yield op";
        return WalkResult::interrupt();
      }
      Operation *yield = body->getTerminator();
      builder.setInsertionPoint(yield);
      aster::amdgcn::EndKernelOp::create(builder, kernel.getLoc());
      yield->erase();

      funcOp->erase();
      for (unsigned i = 0, e = body->getNumArguments(); i < e; ++i) {
        body->getArgument(i).replaceAllUsesWith(kernelArgs[i]);
      }
      body->eraseArguments(0, body->getNumArguments());

      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace
