//===- Passes.td - Test Pass Definitions ------------------------*- tblgen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test pass definitions using tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TEST_PASSES
#define ASTER_TEST_PASSES

include "mlir/Pass/PassBase.td"

def TestAMDGCNInterferenceAnalysis : Pass<"test-amdgcn-interference-analysis"> {
  let summary = "Test pass for AMDGCN register interference analysis";
  let description = [{
    This pass runs AMDGCN register interference analysis and prints the
    interference graph in DOT format. Use this to verify that interference
    edges are correctly computed between live values.
  }];
  let options = [
    Option<"buildMode", "mode", "std::string", "\"minimal\"",
           "Graph build mode: \"minimal\" (default) or \"full\"">,
  ];
}

def TestAMDGCNRangeAnalysis : Pass<"test-amdgcn-range-analysis"> {
  let summary = "Test pass for AMDGCN register range constraint analysis";
  let description = [{
    This pass runs AMDGCN register range constraint analysis and prints the
    range constraints including allocations and alignment. Use this to verify
    that range constraints are correctly computed and satisfiable.
  }];
}

def TestCFGWalker : Pass<"test-cfg-walker"> {
  let summary = "Test pass that walks the CFG and prints IR elements";
  let description = [{
    This pass uses the CFG walker to traverse the control flow graph
    (blocks and region branches) and prints each operation and
    control flow edge encountered during the walk.
  }];
}

def TestLivenessAnalysis : Pass<"test-liveness-analysis"> {
  let summary = "Test pass for liveness analysis";
  let description = [{
    This pass runs liveness analysis and prints the live values
    before and after each operation in kernel functions.
  }];
}

def TestDPSAnalysis : Pass<"test-dps-analysis"> {
  let summary = "Test pass for DPS analysis";
  let description = [{
    This pass runs DPS analysis and prints the DPS alias equivalence
    classes and value provenance for block arguments.
  }];
}

def TestDPSClobberingAnalysis : Pass<"test-dps-clobbering"> {
  let summary = "Test pass for DPS clobbering analysis";
  let description = [{
    This pass runs DPS clobbering analysis and for each instruction prints the
    results that require de-clobbering.
  }];
}

def TestBufferAnalysis : Pass<"test-buffer-analysis"> {
  let summary = "Test pass for buffer analysis";
  let description = [{
    This pass runs buffer analysis and prints the analysis results,
    including live buffers before and after each operation.
  }];
}

def TestReachingDefinitions : Pass<"test-reaching-definitions"> {
  let summary = "Test pass for reaching definitions analysis";
  let description = [{
    This pass runs reaching definitions analysis and prints the analysis
    results, including reaching definitions before and after each operation.
  }];
  let options = [
    Option<"onlyLoads", "only-loads", "bool", "false",
           "Only track load operations">,
  ];
}

def TestMemoryDependenceAnalysis : Pass<"test-memory-dependence-analysis"> {
  let summary = "Test pass for memory dependence analysis";
  let description = [{
    This pass runs memory dependence analysis and prints pending
    memory operations and flush requirements for each operation.
  }];
}

def TestWaitAnalysis : Pass<"test-wait-analysis"> {
  let summary = "Test pass for wait analysis";
  let description = [{
    This pass runs wait analysis and prints the wait state
    before and after each operation.
  }];
}

def TestLDSInterferenceGraph
    : InterfacePass<"test-lds-interference-graph", "::mlir::FunctionOpInterface"> {
  let summary = "Test pass that prints the LDS interference graph";
  let description = [{
    This pass builds and prints the LDS interference graph for testing purposes.
    The graph is printed in DOT format to stderr.
  }];
}

def TestLDSMultibufferPrep : Pass<"test-lds-multibuffer-prep"> {
  let summary = "Test pass for LDS multi-buffer prep transformation";
  let description = [{
    This pass runs the LDS multi-buffer prep transformation for testing.
    In production, this transformation is called directly from the
    SCF pipelining pass.
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::aster::amdgcn::AMDGCNDialect",
  ];
}

#endif // ASTER_TEST_PASSES
