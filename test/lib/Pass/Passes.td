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
}

def TestAMDGCNRangeAnalysis : Pass<"test-amdgcn-range-analysis"> {
  let summary = "Test pass for AMDGCN register range constraint analysis";
  let description = [{
    This pass runs AMDGCN register range constraint analysis and prints the
    range constraints including allocations and alignment. Use this to verify
    that range constraints are correctly computed and satisfiable.
  }];
}

def TestLivenessAnalysis : Pass<"test-liveness-analysis"> {
  let summary = "Test pass for liveness analysis";
  let description = [{
    This pass runs liveness analysis and prints the live values
    before and after each operation in kernel functions.
  }];
}

def TestBufferAnalysis : Pass<"test-buffer-analysis"> {
  let summary = "Test pass for buffer analysis";
  let description = [{
    This pass runs buffer analysis and prints the analysis results,
    including live buffers before and after each operation.
  }];
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
