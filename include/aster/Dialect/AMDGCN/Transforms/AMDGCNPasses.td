//===- AMDGCNPasses.td - AMDGCN Pass Definitions ----------------*- tblgen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AMDGCN pass definitions using tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_PASSES
#define AMDGCN_PASSES

include "mlir/Pass/PassBase.td"

def AMDGCNConvertWaits : InterfacePass<"amdgcn-convert-waits", "mlir::aster::ModuleOpInterface"> {
  let summary = "Convert wait operations to hardware wait instructions";
  let description = [{
    This pass converts high-level amdgcn.wait operations into the appropriate
    hardware wait instructions (s_waitcnt) based on the types of dependencies
    being waited on.
  }];
}

def LegalizeOperands : Pass<"amdgcn-legalize-operands"> {
  let summary = "Legalize operand constraints for AMDGCN instructions";
  let description = [{
    This pass legalizes operand constraints that cannot be satisfied by the
    hardware encoding. For example, SOP2 instructions allow at most one literal
    (non-inline) constant operand. If an lsir.select has two non-inline literal
    operands, this pass materializes one into a register via alloca + s_mov_b32.

    This pass runs before bufferization and register allocation so that the
    newly created allocas participate in the normal register allocation flow.
  }];
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect",
    "mlir::arith::ArithDialect"
  ];
}

def Bufferization : Pass<"amdgcn-bufferization"> {
  let summary = "Insert phi-breaking copies to prepare for register allocation";
  let description = [{
    This pass runs value provenance analysis and inserts copy instructions before
    branches that pass alloca-derived values to block arguments where multiple
    allocas merge (phi-coalescing scenarios).

    When multiple allocas flow to the same block argument, they are "phi-equivalent"
    and traditionally share a register. However, if those allocas interfere (both
    live at some point), they need separate registers.

    This can be viewed as a "bufferization" pass that prepares the IR to go out
    of DPS semantics towards side-effects and before register allocation.

    This pass should run just before register allocation.
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
}

def RegisterAlloc : Pass<"amdgcn-register-allocation"> {
  let summary = "Perform register allocation for AMDGCN kernels";
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect", "mlir::aster::lsir::LSIRDialect"];
}

def RegisterDealloc : Pass<"amdgcn-register-dealloc"> {
  let summary = "Convert a register allocated kernel to an unallocated one";
}

def InstructionSchedulingAutoschedule : Pass<"amdgcn-instruction-scheduling-autoschedule"> {
  let summary = "Apply autoschedule rules to operations without explicit schedules";
  let description = [{
    This pass applies autoschedule rules to operations without explicit schedule
    attributes in scf.for loops marked with sched.dims. It should run before the
    amdgcn-instruction-scheduling pass, if one want schedule automation.

    Operations without explicit schedules will inherit schedules based on:
    1. Their single consumer's schedule (if they have exactly one user)
    2. The smallest schedule from any consumer (if they have multiple users)
    3. Default schedule (delay=0, rate=1) otherwise

    This simple heuristic will at least keep SSA def-use chains valid for the
    autoscheduled operations but has no performance guarantees.

    In the future, better heuristics can be added.
  }];
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

def Mem2Reg : Pass<"amdgcn-mem2reg"> {
  let summary = "Promote memory operations to register operations";
  let description = [{
    This pass promotes memory allocations to register usage where possible,
    converting load and store operations to direct register operations.
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
}

def SetABI : Pass<"aster-amdgcn-set-abi"> {
  let summary = "Set AMDGCN ABI information";
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
}

def ToAMDGCN : InterfacePass<"aster-to-amdgcn", "mlir::aster::ModuleOpInterface"> {
  let summary = "Convert operations to AMDGCN dialect";
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect",
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
    "mlir::arith::ArithDialect"
  ];
  let options = [
    Option<"inputFile",
           "input-file",
           "std::string",
           "\"\"",
           "Input file to parse for ISel">,
  ];
}

def AMDGCNOptimizeStraightLineWaits : Pass<"amdgcn-optimize-straight-line-waits"> {
  let summary = "Optimize straight-line waits between first load and last store";
  let description = [{
    This pass identifies top-level blocks (not in loops) and finds the first load
    operation and last store operation in each block. All operations within the
    first-load, last-write range must be operations without regions and only
    AMDGCN operations.
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
  let options = [
    Option<"flushAllMemoryOnExit",
           "flush-all-memory-on-exit",
           "bool",
           "true",
           "If true, flush all pending memory operations before end_kernel">
  ];
}

def AMDGCNNopInsertion : Pass<"amdgcn-nop-insertion"> {
  let summary = "Insert NOP instructions in AMDGCN kernels";
  let description = [{
    This pass inserts NOP instructions in AMDGCN kernels.

    Section "4.5. Manually Inserted Wait States (NOPs)" of the "AMD Instinct MI300"
    Instruction Set Architecture manual lists the following dependencies that must
    be resolved by inserting NOPs or independent instructions.

    This pass inserts such dependencies.
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
  let options = [
    Option<"conservativeExtraDelays",
           "conservative-extra-delays",
           "unsigned",
           "0",
           "If non-zero, insert this many NOPs after each MFMA, global_load, ds_read, global_store, ds_write operation">
  ];
}

def HoistOps : Pass<"aster-hoist-ops"> {
  let summary = "Hoist special AMDGCN operations to the entry block";
}

def ExpandMetadataOps : Pass<"aster-amdgcn-expand-md-ops", "::mlir::aster::amdgcn::KernelOp"> {
  let summary = "Expand special AMDGCN metadata operations";
  let dependentDialects = [
    "mlir::arith::ArithDialect"
  ];
}

def PreloadLibrary : Pass<"amdgcn-preload-library", "mlir::ModuleOp"> {
  let summary = "Preload AMDGCN library functions into modules";
  let description = [{
    This pass preloads functions from amdgcn.library operations into amdgcn.module
    operations. For each amdgcn.module in the input, it:

    1. Parses library files from the specified paths
    2. For each func.call in the module that references an undefined function,
       looks up the function in the loaded libraries
    3. Clones the library function into the module

    This enables modular development of reusable kernel components.
  }];
  let dependentDialects = [
    // Aster dialects
    "mlir::aster::amdgcn::AMDGCNDialect",
    "mlir::aster::aster_utils::AsterUtilsDialect",
    "mlir::aster::lsir::LSIRDialect",
    // MLIR dialects
    "mlir::affine::AffineDialect",
    "mlir::func::FuncDialect",
    "mlir::gpu::GPUDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
  let options = [
    ListOption<"libraryPaths", "library-paths", "std::string",
               "Paths to MLIR files containing amdgcn.library operations">
  ];
}

def RemoveTestInst : Pass<"amdgcn-remove-test-inst"> {
  let summary = "Remove all amdgcn.test_inst operations";
  let description = [{
    This pass removes all amdgcn.test_inst operations from the IR.
    It is useful for cleaning up test instructions used to prevent DCE
    in nanobenchmarks before final code generation.
  }];
}

def LDSAlloc : Pass<"amdgcn-lds-alloc"> {
  let summary = "Perform LDS buffer allocation analysis";
  let description = [{
    This pass analyzes LDS buffer allocations using the BufferAnalysis dataflow
    framework. It builds an interference graph where nodes are AllocLDSOp
    operations and edges connect buffers that are simultaneously live.

    The pass will emit errors for:
    - Any buffer with a `top` liveness state (conflicting information)
    - Any `get_lds_offset` operation acting on a dead buffer
    - Any allocation without a constant size

    This analysis is a prerequisite for LDS memory layout optimization.
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
}

def ConvertLDSBuffers : Pass<"amdgcn-convert-lds-buffers"> {
  let summary = "Convert LDS buffer operations to constants";
  let description = [{
    This pass converts allocated LDS buffer operations to their final form:
    - Replaces `get_lds_offset` operations with `arith.constant` using the
      buffer's assigned offset
    - Erases `dealloc_lds` operations
    - Removes `alloc_lds` operations if all uses are removed

    This pass should run after `amdgcn-lds-alloc` has assigned offsets to all
    LDS allocations.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
  ];
}

def ConvertSCFControlFlow : Pass<"amdgcn-convert-scf-control-flow"> {
  let summary = "Convert SCF control flow to AMDGCN control flow instructions";
  let description = [{
    This pass converts SCF structured control flow operations (such as scf.for,
    scf.if, scf.while) to AMDGCN control flow instructions.

    The pass first runs thread uniform analysis to determine whether loop
    induction variables and conditions are uniform across all threads. Based on
    this analysis:

    - For thread-uniform conditions: emit scalar compare instructions (s_cmp_*)
      and branch on SCC
    - For thread-divergent conditions: emit vector compare instructions (v_cmp_*)
      and branch on VCC/VCCZ

    This pass should run after the ABI has been set and before register
    allocation.
  }];
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect",
    "mlir::aster::lsir::LSIRDialect"
  ];
}

def LegalizeCF : Pass<"amdgcn-legalize-cf"> {
  let summary = "Legalize CF dialect ops to AMDGCN scalar branch instructions";
  let description = [{
    This pass legalizes CF dialect operations (cf.cond_br, cf.br) and lsir.cmpi
    to AMDGCN scalar branch and compare instructions. It runs after register
    allocation when operands are in physical registers and values from through
    side effects.

    Transformations:
    - lsir.cmpi (returns i1) -> s_cmp_* (sets SCC flag)
    - cf.cond_br -> s_cbranch_scc1 / scc0 + s_branch
    - cf.br -> s_branch

    Prerequisites:
    - Register allocation must have run (all operands must be in allocatedSGPRs)
    - Block arguments must have been eliminated

    TODO: AMDGCN supports only a very limited form of branches. At this time,
    the pass errors when we reach more advanced control flow patterns. Still, the
    current behavior is enough to model `scf.for` loops.
  }];
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect"
  ];
}

#endif // AMDGCN_PASSES
