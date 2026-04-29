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
  let dependentDialects = ["mlir::ub::UBDialect"];
  let options = [
    Option<"removeTokenArgs", "remove-cf-args", "bool", "true",
           "If true, remove token arguments from control-flow arguments">,
  ];
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

def LowerSRegBlockArgs : Pass<"amdgcn-lower-sreg-block-args"> {
  let summary = "Lower special-register block arguments through SGPRs";
  let description = [{
    Transforms block arguments typed as AMDGCN special registers (SCC, VCC, etc.)
    into SGPRs. Concretely, this pass inserts copies at branch points from the
    SREG to a freshly allocated SGPR, and in the successor block, it inserts
    copies back from the SGPR to the actual register type.
  }];
  let dependentDialects = [
    "mlir::aster::lsir::LSIRDialect"
  ];
}

def AMDGCNBufferization : Pass<"aster-amdgcn-bufferization"> {
  let summary = "Insert phi-breaking copies to prepare for register allocation";
  let description = [{
    This pass inserts copies to remove potentially clobbered values, and removes
    phi-nodes arguments with register value semantics. Phi-nodes are handled by
    inserting allocations at dominator points and copying the values to the new
    allocations.

    This can be viewed as a "bufferization" pass that prepares the IR to go out
    of DPS semantics towards side-effects and before register allocation.

    Precondition: all structured control flow (scf.for, scf.if, etc.) must be
    lowered to the cf dialect before this pass runs. The pass uses
    BranchOpInterface to insert phi-breaking copies at branch points.

    Post-condition: #amdgcn.no_register_block_args

    This pass should run just before register coloring.
  }];
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect",
    "mlir::aster::lsir::LSIRDialect",
    "mlir::cf::ControlFlowDialect"
  ];
}

def RegisterColoring : Pass<"amdgcn-register-coloring"> {
  let summary = "Perform register coloring for functions";
  let description = [{
    This pass performs register coloring on functions. It uses liveness analysis
    to determine live ranges, create the interference graph, and assign registers
    accordingly. This pass should run after bufferization and to-register-semantics.

    Pre-condition: #amdgcn.no_value_semantic_registers
    Post-condition: #amdgcn.all_registers_allocated
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
  let options = [
    Option<"buildMode", "mode", "std::string", "\"minimal\"",
           "Graph build mode: \"minimal\" (default) or \"full\"">,
    Option<"optimize", "optimize", "bool", "true",
           "If true, run optimizeGraph to coalesce non-interfering nodes">,
    Option<"numVGPRs", "num-vgprs", "int32_t", "256",
           "Maximum number of VGPRs available for allocation (default 256)">,
    Option<"numAGPRs", "num-agprs", "int32_t", "256",
           "Maximum number of AGPRs available for allocation (default 256)">,
  ];
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
  let summary = "Replace ub.poison of register types with amdgcn.alloca";
  let description = [{
    Post-processing pass for upstream --mem2reg. Upstream mem2reg promotes
    memref allocations to SSA values but leaves uninitialized slots as
    ub.poison. This pass replaces ub.poison of ASTER register types with
    amdgcn.alloca (and make_register_range for register ranges).
    Token and struct types are left as poison.

    Must run after upstream --mem2reg.
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
}

def SetABI : Pass<"aster-amdgcn-set-abi"> {
  let summary = "Set AMDGCN ABI information";
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
}

def ToAMDGCN : InterfacePass<"aster-to-amdgcn", "mlir::aster::ModuleOpInterface"> {
  let summary = "Convert operations to AMDGCN dialect";
  let description = [{
    This pass converts LSIR dialect operations to their AMDGCN equivalents
    using pattern-based rewriting.

    Post-condition: #amdgcn.no_reg_cast_ops (set on amdgcn.module)

    Note: #amdgcn.no_lsir_ops is NOT set because some lowering patterns
    (ExtSI, ExtUI, PtrAdd) create new LSIR ops as intermediaries that
    may not fully resolve in pipelined loop bodies.
  }];
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect",
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
    "mlir::arith::ArithDialect",
    "mlir::aster::lsir::LSIRDialect"
  ];
  let options = [
    Option<"inputFile",
           "input-file",
           "std::string",
           "\"\"",
           "Input file to parse for ISel">,
  ];
}

def HoistOps : Pass<"aster-hoist-ops"> {
  let summary = "Hoist special AMDGCN operations to the entry block";
}

def ExpandMetadataOps : Pass<"aster-amdgcn-expand-md-ops", "::mlir::aster::amdgcn::KernelOp"> {
  let summary = "Expand special AMDGCN metadata operations";
  let description = [{
    This pass expands high-level AMDGCN metadata operations (load_arg,
    thread_id, block_dim, block_id, grid_dim, make_buffer_rsrc) into
    concrete hardware instructions.

    Related normal form: #amdgcn.no_metadata_ops (not auto-set because the
    pipeline may run this pass multiple times with intervening codegen that
    re-introduces metadata ops)
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect"
  ];
  let options = [
    Option<"forceUnpackedTID",
           "force-unpacked-tid",
           "bool",
           "false",
           "Force unpacked workitem ID mode (X=VGPR0, Y=VGPR1, Z=VGPR2) "
           "regardless of the ISA version. Used for testing the legacy path.">,
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
    "mlir::ptr::PtrDialect",
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

def RegisterDCE : Pass<"amdgcn-register-dce"> {
  let summary = "Dead code elimination for register copy/mov operations";
  let description = [{
    This pass performs dead code elimination on copy and mov operations.
    It runs the LivenessAnalysis and removes lsir.copy operations
    and AMDGCN mov instructions (s_mov_b32, v_mov_b32_e32) whose target
    register is not live after the operation.
  }];
}

def WaitInsertion : Pass<"amdgcn-wait-insertion"> {
  let summary = "Insert wait operations for reaching loads";
  let description = [{
    This pass inserts wait operations for loads consumed by instructions.
    It should run after register allocation, and before convert waits.
    Further, mem2reg pass must run before convert waits.
  }];
  let dependentDialects = ["mlir::memref::MemRefDialect"];
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

    Post-condition: #amdgcn.no_scf_ops
  }];
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect",
    "mlir::aster::lsir::LSIRDialect",
    "mlir::cf::ControlFlowDialect"
  ];
}

def LegalizeCF : Pass<"amdgcn-legalize-cf"> {
  let summary = "Legalize CF dialect ops to AMDGCN scalar branch instructions";
  let description = [{
    This pass legalizes CF dialect operations (cf.cond_br, cf.br) and lsir.cmpi
    to AMDGCN scalar branch and compare instructions. It runs after register
    allocation when operands are in physical registers and values flow through
    side effects.

    Transformations:
    - lsir.cmpi (returns i1) -> s_cmp_* (sets SCC flag)
    - cf.cond_br -> s_cbranch_scc1 / scc0 + s_branch
    - cf.br -> s_branch

    Pre-condition: #amdgcn.all_registers_allocated
    Post-condition: #amdgcn.no_cf_branches
    Additional prerequisites:
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

def ToRegisterSemantics : Pass<"amdgcn-to-register-semantics"> {
  let summary = "Propagate non-value register semantics through the IR";
  let description = [{
    This pass propagates non-value register semantics through the IR. It converts
    value allocas to unallocated allocas, and propagates them through the IR.
    After this pass, no instruction results should be used by other operations.
    This pass should run after bufferization, and before register coloring.

    Post-condition: #amdgcn.no_value_semantic_registers
  }];
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect"
  ];
}

def OptimizeAMDGCN : Pass<"amdgcn-optimize"> {
  let summary = "Optimize AMDGCN dialect operations";
  let description = [{
    This pass applies optimization patterns to AMDGCN dialect operations using
    the greedy pattern rewriter. It includes canonicalization patterns for
    LSIR and AMDGCN dialects and operations.

    Thhis pass should run before register allocation. After this pass,
    it might be necessary to run to-amdgcn pass.
  }];
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect",
    "mlir::aster::lsir::LSIRDialect"
  ];
}

def SetNormalForms : Pass<"amdgcn-set-normal-forms"> {
  let summary = "Assert and set normal form attributes on container ops";
  let description = [{
    This pass sets normal form attributes on amdgcn.module and/or amdgcn.kernel
    container ops. It is used as a pipeline checkpoint to assert that certain
    invariants hold at a specific point in the compilation pipeline.

    Normal form attributes are specified by their mnemonic name (without the
    `#amdgcn.` prefix). Module-level forms are set on amdgcn.module ops,
    kernel-level forms are set on amdgcn.kernel ops.

    Once set, MLIR's built-in verifyRegions() will enforce the constraints
    on all subsequent passes.

    Example usage:
      amdgcn-set-normal-forms{module-forms=no_lsir_ops kernel-forms=no_scf_ops}
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
  let options = [
    ListOption<"moduleForms", "module-forms", "std::string",
               "Normal form mnemonics to set on amdgcn.module ops">,
    ListOption<"kernelForms", "kernel-forms", "std::string",
               "Normal form mnemonics to set on amdgcn.kernel ops">
  ];
}

def LowLevelScheduler : Pass<"amdgcn-low-level-scheduler",
                              "::mlir::aster::amdgcn::KernelOp"> {
  let summary = "Pre-register-allocation instruction scheduler";
  let description = [{
    Reorders instructions within basic blocks to hide issue latency by
    modeling AMD GPU hardware execution queues (VALU, XDL, SALU, VMEM, LGKM).
    Uses a greedy algorithm that fills issue slots across queue types to
    maximize instruction-level parallelism.

    Runs after aster-to-amdgcn and before the register allocation pipeline.
    Operates on flat basic blocks (post-SCF lowering).
  }];
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect",
    "mlir::aster::aster_utils::AsterUtilsDialect"
  ];
  let options = [
    Option<"debugStalls", "debug-stalls", "bool", "false",
           "Annotate each op with sched.stall_cycles and sched.stall_reason">,
  ];
}

def AMDGCNHazards : Pass<"amdgcn-hazards"> {
  let summary = "Resolve hazards for AMDGCN instructions";
  let description = [{
    This pass resolves hazards for AMDGCN instructions. It runs the hazard analysis
    on AMDGCN operations and inserts NOP instructions to resolve the hazards.
    It should run after register allocation and convert-waits.

    Pre-condition: #amdgcn.all_registers_allocated
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
  let options = [
    Option<"extraVNops", "v_nops", "unsigned", "0",
           "Extra vector NOPs to insert before every instruction">,
    Option<"extraSNops", "s_nops", "unsigned", "0",
           "Extra scalar NOPs to insert before every instruction">
  ];
}

def SetMFMAPriority : Pass<"aster-set-mfma-priority"> {
  let summary = "Insert s_setprio around MFMA instruction groups";
  let description = [{
    Inserts s_setprio(1) before the first MFMA in each contiguous group and
    s_setprio(0) after the last MFMA. An MFMA group is a maximal run of
    operations where no memory instruction (VMEM, LGKM, SMEM) separates two
    MFMAs. VALU/SALU ops between MFMAs do not break a group.

    The pass is idempotent: blocks that already contain s_setprio instructions
    are skipped entirely.

    Designed to run after the low-level scheduler (amdgcn-low-level-scheduler)
    when the instruction order is final.
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
}

def HoistIterArgWaits : Pass<"amdgcn-hoist-iter-arg-waits"> {
  let summary = "Hoist iter_arg-dependent waits to loop head";
  let description = [{
    After SCF pipelining, loops carry token dependencies (read_token, write_token)
    as iter_args from previous iterations. The corresponding amdgcn.wait ops are
    placed near their consumers, deep in the loop body, fragmenting the scheduling
    window.

    This pass hoists waits whose dependencies are all iter_arg block arguments to
    the top of the loop body. For waits with mixed dependencies (some iter_arg,
    some intra-iteration), it clones the wait: one copy at the loop head with only
    iter_arg deps, the original retains only intra-iteration deps.

    After hoisting, a subsequent canonicalize pass merges adjacent waits at the
    loop head into a single aggregated wait.

    Runs after aster-scf-pipeline and before amdgcn-low-level-scheduler.
  }];
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect",
    "mlir::scf::SCFDialect"
  ];
}

#endif // AMDGCN_PASSES
