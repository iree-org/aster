//===- AsterUtilsAttrs.td - AsterUtils attributes ----------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSATTRS_TD
#define ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSATTRS_TD

include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.td"
include "aster/Dialect/AsterUtils/IR/AsterUtilsEnums.td"
include "aster/Interfaces/SchedInterfaces.td"

//===----------------------------------------------------------------------===//
// GPU dimension identifier.
//===----------------------------------------------------------------------===//

def DimAttr : AsterUtils_EnumAttr<"dim", DimEnum, []> {
  let description = [{
    AsterUtils gpu dimension identifier attribute.
  }];
}

//===----------------------------------------------------------------------===//
// Scheduling attributes
//===----------------------------------------------------------------------===//

def SSASchedulerAttr : AsterUtils_Attr<"SSAScheduler", "ssa_scheduler", [
    DeclareAttrInterfaceMethods<SchedGraphAttrInterface>
  ]> {
  let summary = "SSA-based scheduling graph builder";
  let description = [{
    Builds a scheduling graph by iterating over ops in the block and adding
    edges from each op to all ops that produce its operands (SSA dependencies).

    NOTE: This scheduler doesn't take into account side-effects.
    TODO: Add support for side-effects.
  }];
}

def SchedStageLabelerAttr : AsterUtils_Attr<"SchedStageLabeler", "sched_stage_labeler", [
    DeclareAttrInterfaceMethods<SchedLabelerAttrInterface>
  ]> {
  let summary = "Label nodes by `sched.stage`";
  let description = [{
    Labels nodes using the `sched.stage` IntegerAttr from the op's discardable
    attributes. Returns int32_t::max if not valid/present.
  }];
}

def SchedListLabelerAttr : AsterUtils_Attr<"SchedListLabeler", "sched_list_labeler", [
    DeclareAttrInterfaceMethods<SchedLabelerAttrInterface>
  ]> {
  let summary = "Label nodes by trying labelers in order";
  let description = [{
    Tries each labeler in the array in order and returns the first label that
    is not negative. If all labelers return negative, returns a negative value.
  }];
  let parameters = (ins
    ArrayRefParameter<"::mlir::aster::SchedLabelerAttrInterface", "sched labelers">:$labelers
  );
  let assemblyFormat = "`<` `[` $labelers `]` `>`";
}

def OpNameLabelerAttr : AsterUtils_Attr<"OpNameLabeler", "op_name_labeler", [
    DeclareAttrInterfaceMethods<SchedLabelerAttrInterface>
  ]> {
  let summary = "Label nodes by operation name";
  let description = [{
    Labels nodes with the given stage if the operation's name is in the
    specified list of op names. Returns -1 if the operation does not match.

    If the list is empty, it matches all operations.
  }];
  let parameters = (ins
    ArrayRefParameter<"::mlir::StringAttr", "op names">:$opNameMatcher,
    "uint16_t":$stage
  );
  let assemblyFormat = "`<` `[` $opNameMatcher `]` `,` $stage `>`";
}

def StageTopoSortSchedAttr : AsterUtils_Attr<"StageTopoSortSched", "stage_topo_sort_sched", [
    DeclareAttrInterfaceMethods<SchedBuilderAttrInterface>
  ]> {
  let summary = "Stage-aware topological sort scheduler";
  let description = [{
    Sorts the scheduling graph in topological order and emits ops in that order,
    with the constraint that ops with smaller stage labels are scheduled first.
  }];
}

def GenericSchedulerAttr : AsterUtils_Attr<"GenericScheduler", "generic_scheduler", [
    DeclareAttrInterfaceMethods<SchedAttrInterface>
  ]> {
  let summary = "Generic scheduler matching all blocks";
  let parameters = (ins
    "SchedGraphAttrInterface":$graph,
    "SchedLabelerAttrInterface":$labeler,
    "SchedBuilderAttrInterface":$scheduler
  );
  let assemblyFormat = "`<` $graph `,` $labeler `,` $scheduler `>`";
}

#endif // ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSATTRS_TD
