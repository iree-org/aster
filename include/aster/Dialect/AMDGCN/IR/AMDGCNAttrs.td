//===- AMDGCNAttrs.td - AMDGCN attrs ------------------------*- tablegen-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_ATTRS_TD
#define ASTER_AMDGCN_ATTRS_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/KernelArgInterface.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/HazardAttrInterface.td"
include "aster/Dialect/AMDGCN/IR/Hazards.td"
include "aster/Dialect/NormalForm/IR/NormalFormInterfaces.td"
include "aster/Interfaces/MemorySpaceConstraints.td"
include "aster/Interfaces/SchedInterfaces.td"
include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td"

//===----------------------------------------------------------------------===//
// AddressSpaceAttr
//===----------------------------------------------------------------------===//

def AddressSpaceAttr : AMDGCN_Attr<"AddressSpace", "addr_space", [
    DeclareAttrInterfaceMethods<MemorySpaceAttrInterface>,
    DeclareAttrInterfaceMethods<MemorySpaceConstraintsAttrInterface>
  ]> {
  let summary = "Address space attribute";
  let parameters = (ins "AddressSpaceKind":$space, "AccessKind":$kind);
  let assemblyFormat = "`<` $space `,` $kind `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Inst attribute
//===----------------------------------------------------------------------===//

/// Attribute constraint for AMDGCN's InstAttr attribute.
def AMDGCN_InstAttr :
  Attr<CPred<"isa<::mlir::aster::amdgcn::InstAttr>($_self)">,
    "AMDGCN instruction attribute"> {
  let storageType = "::mlir::aster::amdgcn::InstAttr";
  let constBuilderCall =
      "::mlir::aster::amdgcn::InstAttr::get($_builder.getContext(), $0)";
  let returnType = "::mlir::aster::amdgcn::OpCode";
  let convertFromStorage = "$_self.getValue()";
}

//===----------------------------------------------------------------------===//
// Scheduling attributes
//===----------------------------------------------------------------------===//

def ValueSchedulerAttr : AMDGCN_Attr<"ValueScheduler", "value_scheduler", [
    DeclareAttrInterfaceMethods<SchedGraphAttrInterface>
  ]> {
  let summary = "Value-based scheduling graph builder for AMDGCN instructions";
  let description = [{
    Builds a scheduling graph by iterating over ops in the block and adding
    edges from each op to all ops that produce its input operands. This builder
    is also aware of wait and barrier operations. Unknown non-pure ops
    are treated as possible sync points.

    After SSA and sync edges are built, adds serialization edges between
    i1-producing operations in the block.
  }];
}

def InstPropLabelerAttr : AMDGCN_Attr<"InstPropLabeler", "inst_prop_labeler", [
    DeclareAttrInterfaceMethods<SchedLabelerAttrInterface>
  ]> {
  let summary = "Label nodes by instruction properties";
  let description = [{
    Labels nodes with the given stage if the operation has any of the specified
    instruction properties. Returns -1 if the operation does not match.

    If the list is empty, it matches all instructions.
  }];
  let parameters = (ins
    ArrayRefParameter<"::mlir::aster::amdgcn::InstProp", "instruction properties">:$instMatcher,
    "uint16_t":$stage
  );
  let assemblyFormat = "`<` `[` $instMatcher `]` `,` $stage `>`";
}

def OpCodeLabelerAttr : AMDGCN_Attr<"OpCodeLabeler", "opcode_labeler", [
    DeclareAttrInterfaceMethods<SchedLabelerAttrInterface>
  ]> {
  let summary = "Label nodes by opcode";
  let description = [{
    Labels nodes with the given stage if the operation's opcode is in the
    specified list. Returns -1 if the operation does not match.

    If the list is empty, it matches all instructions.
  }];
  let parameters = (ins
    ArrayRefParameter<"::mlir::aster::amdgcn::OpCode", "opcodes">:$opCodeMatcher,
    "uint16_t":$stage
  );
  let assemblyFormat = "`<` `[` $opCodeMatcher `]` `,` $stage `>`";
}

def LatencyPipelinerSchedAttr : AMDGCN_Attr<"LatencyPipelinerSched",
    "latency_pipeliner_sched", [
    DeclareAttrInterfaceMethods<SchedBuilderAttrInterface>
  ]> {
  let summary = "Latency-aware pipelining scheduler";
  let description = [{
    Schedules operations in two phases:

    1. If any ready operation has label 0, it is scheduled immediately (these
       are considered zero-latency or pinned operations).

    2. When no label-0 operations are ready, the scheduler picks the
       highest-latency (highest-label) operation, preferring a different label
       than the previously scheduled one in order to interleave distinct
       instruction types and maximize latency hiding.

    The `schedLimit` parameter controls how many ready operations are scheduled
    per invocation of the scheduling function. If positive, at most
    `schedLimit` operations are scheduled per invocation. If non-positive, all
    ready operations are scheduled.
  }];
  let parameters = (ins
    DefaultValuedParameter<"int32_t", "1">:$schedLimit
  );
  let assemblyFormat = "`<` $schedLimit `>`";
}

def LowLevelSchedulerAttr : AMDGCN_Attr<"LowLevelScheduler", "low_level_scheduler", [
    DeclareAttrInterfaceMethods<SchedBuilderAttrInterface>
  ]> {
  let summary = "Greedy queue-aware scheduler for AMDGCN";
  let description = [{
    Schedules using a hardware queue simulator (VALU, XDL, SALU, VMEM, LGKM)
    with stall avoidance and interleaving heuristics.

    When `annotateStalls` is true, records per-op stall metadata on the IR
    during scheduling.
  }];
  let parameters = (ins
    DefaultValuedParameter<"bool", "false">:$annotateStalls
  );
  let assemblyFormat = "`<` $annotateStalls `>`";
}

//===----------------------------------------------------------------------===//
// Kernel argument attributes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ByValue argument attribute
//===----------------------------------------------------------------------===//

/// AMDGCN by-value argument attribute.
def ByValueArgAttr : AMDGCN_Attr<"ByValueArg", "by_val_arg", [
    KernelArgInterface
  ]> {
  let summary = "AMDGCN kernel by-value argument attribute";
  let description = [{
    Attribute representing an AMDGCN kernel by-value argument.

    Example:

    ```mlir
    #amdgcn.by_val_arg<size = 4, name = "int_arg", type = i32>,
    ```
  }];
  let parameters = (ins "uint32_t":$size,
                        OptionalParameter<"uint32_t">:$alignment,
                        StringRefParameter<"name", "\"\"">:$name,
                        OptionalParameter<"Type">:$type);
  let assemblyFormat = "`<` struct(params) `>`";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // KernelArgInterface methods
    //===------------------------------------------------------------------===//
    /// Get the value kind of the kernel argument.
    ArgumentValueKind getValueKind() const {
      return ArgumentValueKind::ByValue;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Buffer argument attribute
//===----------------------------------------------------------------------===//

/// AMDGCN buffer argument attribute.
def BufferArgAttr : AMDGCN_Attr<"BufferArg", "buffer_arg", [
    KernelArgInterface
  ]> {
  let summary = "AMDGCN kernel buffer argument attribute";
  let description = [{
    Attribute representing an AMDGCN kernel buffer argument.

    Example:

    ```mlir
    #amdgcn.buffer_arg<address_space = generic, access = read_write, type = !ptr.ptr<#ptr.generic_space>>
    ```
  }];
  let parameters = (ins
    DefaultValuedParameter<"AddressSpaceKind", "AddressSpaceKind::Global">:$address_space,
    DefaultValuedParameter<"AccessKind", "AccessKind::ReadWrite">:$access,
    DefaultValuedParameter<"KernelArgumentFlags", "KernelArgumentFlags::None">:$flags,
    StringRefParameter<"name", "\"\"">:$name,
    OptionalParameter<"Type">:$type
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // KernelArgInterface methods
    //===------------------------------------------------------------------===//
    /// Get the value kind of the kernel argument.
    ArgumentValueKind getValueKind() const {
      return ArgumentValueKind::GlobalBuffer;
    }
    // Get the actual memory access pattern.
    AccessKind getActualAccess() const {
      return getAccess();
    }
    // Get the alignment of the argument in bytes.
    std::optional<uint32_t> getAlignment() const {
      return 8u;
    }
    // Get the size of the argument in bytes.
    uint32_t getSize() const {
      return 8u;
    }
  }];
}

//===----------------------------------------------------------------------===//
// BlockDim argument attribute
//===----------------------------------------------------------------------===//

/// AMDGCN block dimension argument attribute.
def BlockDimArgAttr : AMDGCN_Attr<"BlockDimArg", "block_dim_arg", [
    KernelArgInterface
  ]> {
  let summary = "AMDGCN kernel block dimension argument attribute";
  let description = [{
    Attribute representing an AMDGCN kernel block dimension argument.

    Example:

    ```mlir
    #amdgcn.block_dim_arg<x>
    ```
  }];
  let parameters = (ins "Dim":$dimension);
  let assemblyFormat = "`<` $dimension `>`";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // KernelArgInterface methods
    //===------------------------------------------------------------------===//
    /// Get the value kind of the kernel argument.
    ArgumentValueKind getValueKind() const {
      switch(getDimension()) {
        case Dim::X: return ArgumentValueKind::HiddenGroupSizeX;
        case Dim::Y: return ArgumentValueKind::HiddenGroupSizeY;
        case Dim::Z: return ArgumentValueKind::HiddenGroupSizeZ;
      }
    }
    // Get the alignment of the argument in bytes.
    std::optional<uint32_t> getAlignment() const {
      return 4u;
    }
    // Get the size of the argument in bytes.
    uint32_t getSize() const {
      return 4u;
    }
  }];
}

//===----------------------------------------------------------------------===//
// GridDim argument attribute
//===----------------------------------------------------------------------===//

/// AMDGCN grid dimension argument attribute.
def GridDimArgAttr : AMDGCN_Attr<"GridDimArg", "grid_dim_arg", [
    KernelArgInterface
  ]> {
  let summary = "AMDGCN kernel grid dimension argument attribute";
  let description = [{
    Attribute representing an AMDGCN kernel grid dimension argument.

    Example:

    ```mlir
    #amdgcn.grid_dim_arg<x>
    ```
  }];
  let parameters = (ins "Dim":$dimension);
  let assemblyFormat = "`<` $dimension `>`";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // KernelArgInterface methods
    //===------------------------------------------------------------------===//
    /// Get the value kind of the kernel argument.
    ArgumentValueKind getValueKind() const {
      switch(getDimension()) {
        case Dim::X: return ArgumentValueKind::HiddenBlockCountX;
        case Dim::Y: return ArgumentValueKind::HiddenBlockCountY;
        case Dim::Z: return ArgumentValueKind::HiddenBlockCountZ;
      }
    }
    // Get the alignment of the argument in bytes.
    std::optional<uint32_t> getAlignment() const {
      return 4u;
    }
    // Get the size of the argument in bytes.
    uint32_t getSize() const {
      return 4u;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Kernel arguments attribute
//===----------------------------------------------------------------------===//

def KernelArgumentsAttr
  : ArrayOfAttr<AMDGCN_Dialect, "KernelArguments", "kernel_arguments",
                "::mlir::aster::amdgcn::KernelArgAttrInterface"> {
  let summary = "AMDGCN kernel arguments attribute";
  let constBuilderCall = [{
    $_builder.getAttr<KernelArgumentsAttr>(ArrayRef<KernelArgAttrInterface>($0))
  }];
  let defaultValue = "{}";
  let assemblyFormat = [{
    `<`(`[` $value^ `]`)? `>`
  }];
}

//===----------------------------------------------------------------------===//
// Normal form attributes
//===----------------------------------------------------------------------===//

def NoValueSemanticRegistersAttr : AMDGCN_Attr<
    "NoValueSemanticRegisters", "no_value_semantic_registers",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyType"]>]> {
  let summary = "Normal form: no register types with value semantics";
  let description = [{
    Verifies that no register types in the IR have value semantics.
    This is the post-condition of the `amdgcn-to-register-semantics` pass,
    which converts all value-semantic allocas to unallocated register semantics.

    After this normal form holds, every `RegisterTypeInterface` type in the
    module has either `Unallocated` or `Allocated` semantics -- never `Value`.
  }];
}

def AllRegistersAllocatedAttr : AMDGCN_Attr<
    "AllRegistersAllocated", "all_registers_allocated",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyType"]>]> {
  let summary = "Normal form: all register types have allocated semantics";
  let description = [{
    Verifies that every register type in the IR has allocated semantics
    (i.e., has been assigned a concrete physical register).
    This is the post-condition of the `amdgcn-register-coloring` pass.

    After this normal form holds, no `RegisterTypeInterface` type has
    `Value` or `Unallocated` semantics -- all are `Allocated`.
  }];
}

def NoRegCastOpsAttr : AMDGCN_Attr<
    "NoRegCastOps", "no_reg_cast_ops",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyOperation"]>]> {
  let summary = "Normal form: no lsir.reg_cast operations";
  let description = [{
    Verifies that no `lsir.reg_cast` operations exist in the IR.
    This is the post-condition of the `aster-to-amdgcn` pass.

    A surviving `lsir.reg_cast` indicates an incorrect `lsir.to_reg` or
    `lsir.from_reg` remaining from above, potentially from high-level authored
    IR.
  }];
}

def NoCfBranchesAttr : AMDGCN_Attr<
    "NoCfBranches", "no_cf_branches",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyOperation"]>]> {
  let summary = "Normal form: no CF dialect branch operations";
  let description = [{
    Verifies that no `cf.br` or `cf.cond_br` operations exist in the IR.
    This is the post-condition of the `amdgcn-legalize-cf` pass, which
    lowers CF dialect branches to AMDGCN scalar branch instructions
    (`s_branch`, `s_cbranch_scc0/1`).
  }];
}

def NoScfOpsAttr : AMDGCN_Attr<
    "NoScfOps", "no_scf_ops",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyOperation"]>]> {
  let summary = "Normal form: no SCF dialect operations";
  let description = [{
    Verifies that no SCF dialect operations (`scf.for`, `scf.if`, `scf.while`,
    etc.) exist in the IR. This is the post-condition of the
    `amdgcn-convert-scf-control-flow` pass, which lowers SCF structured
    control flow to CF dialect branches and AMDGCN compare instructions.
  }];
}

def NoLsirOpsAttr : AMDGCN_Attr<
    "NoLsirOps", "no_lsir_ops",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyOperation"]>]> {
  let summary = "Normal form: no LSIR dialect operations";
  let description = [{
    Verifies that no LSIR dialect operations exist in the IR.
    This is the combined form of `no_lsir_compute_ops` and
    `no_lsir_control_ops`. Set at the end of the `amdgcn-backend` pipeline
    after `LegalizeCF` has lowered the last surviving LSIR ops.
  }];
}

def NoLsirComputeOpsAttr : AMDGCN_Attr<
    "NoLsirComputeOps", "no_lsir_compute_ops",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyOperation"]>]> {
  let summary = "Normal form: no LSIR compute/memory operations";
  let description = [{
    Verifies that no LSIR arithmetic, memory, or utility operations exist in
    the IR. This includes all LSIR ops except the control-flow-related ops
    (`lsir.cmpi`, `lsir.cmpf`, `lsir.select`) and the regalloc primitive
    (`lsir.copy`).

    Set at the entry of the `amdgcn-backend` pipeline, after `aster-to-amdgcn`
    and canonicalize/CSE have eliminated all compute LSIR ops.

    Allowed ops (not checked by this form):
    - `lsir.cmpi`, `lsir.cmpf`, `lsir.select` -- lowered by `LegalizeCF`
    - `lsir.copy` -- created by bufferization, consumed by register coloring
  }];
}

def NoLsirControlOpsAttr : AMDGCN_Attr<
    "NoLsirControlOps", "no_lsir_control_ops",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyOperation"]>]> {
  let summary = "Normal form: no LSIR control-flow operations";
  let description = [{
    Verifies that no LSIR control-flow-related operations (`lsir.cmpi`,
    `lsir.cmpf`, `lsir.select`) exist in the IR.

    These ops survive through `aster-to-amdgcn` because they operate on i1
    conditions and SCC flags, which are only lowered by `amdgcn-legalize-cf`
    after register allocation.

    Set after `LegalizeCF` in the `amdgcn-backend` pipeline.
  }];
}

def NoRegisterBlockArgsAttr : AMDGCN_Attr<
    "NoRegisterBlockArgs", "no_register_block_args",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyOperation"]>]> {
  let summary = "Normal form: no block arguments with register types";
  let description = [{
    Verifies that no block arguments have register types
    (`RegisterTypeInterface`). This is the post-condition of the
    `aster-amdgcn-bufferization` pass, which replaces register-typed
    phi-nodes with alloca + copy pairs.
  }];
}

def NoAffineOpsAttr : AMDGCN_Attr<
    "NoAffineOps", "no_affine_ops",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyOperation"]>]> {
  let summary = "Normal form: no affine dialect operations";
  let description = [{
    Verifies that no affine dialect operations (`affine.apply`, `affine.for`,
    etc.) exist in the IR. This is the post-condition of the
    `aster-to-int-arith` pass, which lowers affine operations to standard
    arith operations and converts index types to integer types.
  }];
}

def NoMetadataOpsAttr : AMDGCN_Attr<
    "NoMetadataOps", "no_metadata_ops",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyOperation"]>]> {
  let summary = "Normal form: no AMDGCN metadata operations";
  let description = [{
    Verifies that no AMDGCN metadata operations remain in the IR.
    This includes `amdgcn.load_arg`, `amdgcn.thread_id`, `amdgcn.block_dim`,
    `amdgcn.block_id`, `amdgcn.grid_dim`, and `amdgcn.make_buffer_rsrc`.
    This is the post-condition of the `aster-amdgcn-expand-md-ops` pass,
    which expands these high-level metadata operations into concrete
    hardware instructions.
  }];
}

def AllInlinedAttr : AMDGCN_Attr<
    "AllInlined", "all_inlined",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyOperation"]>]> {
  let summary = "Normal form: all function calls inlined";
  let description = [{
    Verifies that no `func.call` operations exist in the IR.
    This is the post-condition of `aster-selective-inlining`, which
    inlines all function calls within kernels.

    After this point, all library functions have been expanded inline
    and passes like `amdgcn-low-level-scheduler` can see the full
    instruction stream without func.call barriers.
  }];
}

#endif // ASTER_AMDGCN_ATTRS_TD
