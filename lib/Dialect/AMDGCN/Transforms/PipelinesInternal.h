//===- PipelinesInternal.h - Shared AMDGCN pipeline helpers ---------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_PIPELINES_INTERNAL_H
#define AMDGCN_PIPELINES_INTERNAL_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace mlir::aster::amdgcn {

/// Options for the RegAlloc pass pipeline.
struct RegAllocPipelineOptions
    : public PassPipelineOptions<RegAllocPipelineOptions> {
  mlir::detail::PassOptions::Option<std::string> buildMode{
      *this, "mode",
      llvm::cl::desc("Graph build mode: \"minimal\" (default) or \"full\""),
      llvm::cl::init("minimal")};
  mlir::detail::PassOptions::Option<int32_t> numVGPRs{
      *this, "num-vgprs",
      llvm::cl::desc("Maximum VGPRs for allocation (default 256)"),
      llvm::cl::init(256)};
  mlir::detail::PassOptions::Option<int32_t> numAGPRs{
      *this, "num-agprs",
      llvm::cl::desc("Maximum AGPRs for allocation (default 256)"),
      llvm::cl::init(256)};
  mlir::detail::PassOptions::Option<int32_t> numSGPRs{
      *this, "num-sgprs",
      llvm::cl::desc("Maximum SGPRs for allocation (default 102)"),
      llvm::cl::init(102)};
  mlir::detail::PassOptions::Option<bool> hoistIterArgWaits{
      *this, "hoist-iter-arg-waits",
      llvm::cl::desc("Hoist iter_arg-dependent waits to loop head"),
      llvm::cl::init(false)};
};

struct AMDGCNBackendPipelineOptions
    : public PassPipelineOptions<AMDGCNBackendPipelineOptions> {
  mlir::detail::PassOptions::Option<int32_t> numVGPRs{
      *this, "num-vgprs",
      llvm::cl::desc("Maximum VGPRs for allocation (default 256)"),
      llvm::cl::init(256)};
  mlir::detail::PassOptions::Option<int32_t> numAGPRs{
      *this, "num-agprs",
      llvm::cl::desc("Maximum AGPRs for allocation (default 256)"),
      llvm::cl::init(256)};
  mlir::detail::PassOptions::Option<bool> hoistIterArgWaits{
      *this, "hoist-iter-arg-waits",
      llvm::cl::desc("Hoist iter_arg-dependent waits to loop head"),
      llvm::cl::init(false)};
  mlir::detail::PassOptions::Option<int32_t> llSched{
      *this, "ll-sched",
      llvm::cl::desc("Low-level scheduler preset (0 = off, 1+ = run with "
                     "preset N; see SchedAttrs.cpp)"),
      llvm::cl::init(0)};
  mlir::detail::PassOptions::Option<bool> setMfmaPriority{
      *this, "set-mfma-priority",
      llvm::cl::desc("Insert s_setprio around MFMA groups"),
      llvm::cl::init(false)};
};

void buildRegAllocPassPipeline(OpPassManager &pm,
                               const RegAllocPipelineOptions &options);

/// Build the LDS allocation sub-pipeline (assign byte offsets, fold
/// get_lds_offset to constants, CSE/canonicalize). Runs after scheduling and
/// before register allocation so the materialized LDS-offset alloca is colored.
void buildLdsAllocPassPipeline(OpPassManager &pm);

} // namespace mlir::aster::amdgcn

#endif // AMDGCN_PIPELINES_INTERNAL_H
