// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_TRANSFORMS_PASSES
#define WATER_DIALECT_WAVE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def WaterWaveInferTypesPass : Pass<"water-wave-infer-types", "::mlir::ModuleOp"> {
  let summary = "Infer fully-specified Wave dialect types";
  let description = [{
    Performs forward and backward sparse dataflow analysis propagating shape
    information for Wave tensor types. All operations using these types are
    expected to implement WaveInferTypeOpInterface to indicate rules for shape
    propagation across them. Traits provide implementations for common cases.

    Reports errors if a type conflict is detected, for example, if a value with
    `any` shape would need two different concrete shapes based on propagation
    from different control flow paths or from different directions.
  }];

  let options = [
    Option</*varName=*/"force",
           /*arg=*/"force",
           /*valueType=*/"bool",
           /*default=*/"false",
           "Keep processing after error emission without updating the type. "
           "May construct invalid IR useful for debugging">,
    Option</*varName=*/"partial",
           /*arg=*/"partial",
           /*valueType=*/"bool",
           /*default=*/"false",
           "Allow some types to remain underspecified.">
  ];
}

def WaterWaveAsterizePass : Pass<"water-wave-asterize"> {
  let summary = "Asterize operations in the Wave dialect";
  let description = [{
    This pass asterizes operations within the Wave dialect.
  }];
  let dependentDialects = [
    "::mlir::aster::amdgcn::AMDGCNDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::aster::amdgcn::AMDGCNDialect",
    "::mlir::aster::aster_utils::AsterUtilsDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::arith::ArithDialect",
  ];
  let options = [
    Option</*varName=*/"pipelineStrategy",
           /*arg=*/"pipeline-strategy",
           /*valueType=*/"int",
           /*default=*/"0",
           "Pipeline strategy index (0-10), same as kittens_helpers.PIPELINE_STRATEGIES">,
  ];
}

def WaterWaveAsterLoweringPass : Pass<"water-wave-aster-lowering"> {
  let summary = "Lower Aster to assembly";
  let description = [{
    Runs the default Aster-to-assembly lowering pipeline. The pipeline matches
    `make_default_pass_pipeline` from `aster/pass_pipelines.py` and includes
    pre-scheduling cleanup, constexpr expansion, SCF pipelining, SROA,
    lowering to AMDGCN, register allocation and hazard insertion.
  }];
  let dependentDialects = [
    // Aster dialects
    "::mlir::aster::amdgcn::AMDGCNDialect",
    "::mlir::aster::aster_utils::AsterUtilsDialect",
    "::mlir::aster::lsir::LSIRDialect",
    // Upstream MLIR dialects
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::DLTIDialect",
    "::mlir::func::FuncDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect",
    "::mlir::ptr::PtrDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::ub::UBDialect",
    "::mlir::vector::VectorDialect",
  ];
  let options = [
    Option</*varName=*/"lcmUnroll",
           /*arg=*/"lcm-unroll",
           /*valueType=*/"bool",
           /*default=*/"false",
           "Enable LCM unrolling in SCF pipelining">,
    Option</*varName=*/"numVGPRs",
           /*arg=*/"num-vgprs",
           /*valueType=*/"int",
           /*default=*/"256",
           "Maximum number of VGPRs for register allocation">,
    Option</*varName=*/"numAGPRs",
           /*arg=*/"num-agprs",
           /*valueType=*/"int",
           /*default=*/"256",
           "Maximum number of AGPRs for register allocation">,
    Option</*varName=*/"unrollFactorMultiplier",
           /*arg=*/"unroll-factor-multiplier",
           /*valueType=*/"int",
           /*default=*/"1",
           "Multiplier for the unroll factor in SCF pipelining">,
    Option</*varName=*/"epiloguePeeling",
           /*arg=*/"epilogue-peeling",
           /*valueType=*/"bool",
           /*default=*/"true",
           "Enable epilogue peeling in SCF pipelining">,
    Option</*varName=*/"libraryFile",
           /*arg=*/"library-file",
           /*valueType=*/"std::string",
           /*default=*/"",
           "Path to an MLIR library file to load and merge before lowering">,
    Option</*varName=*/"pipelineStrategy",
           /*arg=*/"pipeline-strategy",
           /*valueType=*/"int",
           /*default=*/"0",
           "Pipeline strategy index (0-10) for helper template substitutions; same keys as kittens_helpers.pipeline_strategy_substitutions">,
    ListOption</*varName=*/"libraryPaths",
               /*arg=*/"library-paths",
               /*valueType=*/"std::string",
               "Paths to MLIR library files for amdgcn-preload-library">,
    Option</*varName=*/"target",
           /*arg=*/"target",
           /*valueType=*/"std::string",
           /*default=*/"\"gfx942\"",
           "AMDGCN target for the amdgcn.module wrapper">,
    Option</*varName=*/"isa",
           /*arg=*/"isa",
           /*valueType=*/"std::string",
           /*default=*/"\"cdna3\"",
           "AMDGCN ISA version for the amdgcn.module wrapper">,
  ];
}

def WaterWaveTranslateToASMPass : Pass<"water-wave-translate-to-asm"> {
  let summary = "Translate lowered AMDGCN module to assembly";
  let description = [{
    Finds the amdgcn.module inside the root operation, translates it to AMDGPU
    assembly using the ASTER backend, attaches the resulting assembly string as
    a `asm` StringAttr on the root operation, and erases the root's body.
  }];
  let dependentDialects = [
    "::mlir::aster::amdgcn::AMDGCNDialect",
  ];
}

#endif // WATER_DIALECT_WAVE_TRANSFORMS_PASSES
