//===- VerifierAttr.cpp - Verifier attribute interface ----------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Interfaces/VerifierAttr.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include <cstdint>

#include "aster/Interfaces/VerifierAttr.cpp.inc"

#define DEBUG_TYPE "aster-verifiers"

using namespace mlir;
using namespace mlir::aster;

llvm::cl::opt<bool>
    clDisableVerifiers("aster-disable-verifiers",
                       llvm::cl::desc("Disable ASTER verifiers"),
                       llvm::cl::init(false));

llvm::cl::opt<bool> clSuppressDisabledVerifierWarning(
    "aster-suppress-disabled-verifier-warning",
    llvm::cl::desc("Suppress warnings for disabled ASTER verifiers"),
    llvm::cl::init(false));

LogicalResult VerifierRunner::run() {
  if (clDisableVerifiers) {
    if (!clSuppressDisabledVerifierWarning) {
      llvm::errs()
          << "Warning: ASTER verifiers are disabled via command line option\n";
    }
    return success();
  }
  LDBG() << "Running verifiers on operation: " << state.root->getName()
         << ", loc = " << state.root->getLoc();
  LDBG() << "Verifiers to run: " << llvm::interleaved_array(verifiers);
  for (auto [index, verifier] : llvm::enumerate(verifiers)) {
    LDBG() << "Initializing verifier (" << index << "): " << verifier;
    verifier.initializeVerifierState(state);
  }
  SmallVector<VerifierAttrInterface> scratch;
  int64_t vIt = 0;
  int64_t total = static_cast<int64_t>(verifiers.size());
  while (vIt < total) {
    int64_t start = vIt;
    scratch.clear();
    // Collect all verifiers applicable at this scope
    VerifierScope scope = verifiers[vIt].verifierScope();
    do {
      scratch.push_back(verifiers[vIt]);
      vIt++;
    } while (vIt < total && verifiers[vIt].verifierScope() == scope);

    // Run verifiers at the root scope.
    if (scope == VerifierScope::RunOnRoot) {
      for (auto [index, verifier] : llvm::enumerate(scratch)) {
        LDBG() << "Running verifier (" << (index + start) << "): " << verifier
               << " on root operation";
        if (failed(verifier.verifyOp(state.root, state))) {
          LDBG() << "Verifier (" << (index + start) << ") failed";
          return failure();
        }
        LDBG() << "Verifier (" << (index + start) << ") succeeded";
      }
      continue;
    }

    // Run verifiers at nested operations.
    LDBG_OS([&](llvm::raw_ostream &os) {
      os << "Running verifiers on nested operations:\n";
      llvm::interleave(
          llvm::enumerate(scratch), os,
          [&](auto pair) {
            auto [index, verifier] = pair;
            os << "  Verifier (" << (index + start) << "): " << verifier
               << "\n";
          },
          "\n");
    });
    WalkResult walkResult = state.root->walk([&](Operation *op) -> WalkResult {
      for (auto [index, verifier] : llvm::enumerate(scratch)) {
        if (failed(verifier.verifyOp(op, state))) {
          LDBG() << "Nested verifier (" << (index + start)
                 << ") failed on op: " << *op;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();
  }
  LDBG() << "All verifiers succeeded on operation: " << state.root->getName()
         << ", loc = " << state.root->getLoc();
  return success();
}
