//===-- AMDGCNVerifiers.td - AMDGCN verifier attributes -------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_VERIFIERS_TD
#define ASTER_AMDGCN_VERIFIERS_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/Interfaces/VerifierAttr.td"

//===----------------------------------------------------------------------===//
// IsInKernel
//===----------------------------------------------------------------------===//

/// AMDGCN verifier to check if an operation is inside an AMDGCN kernel.
def IsInKernelAttr :
    AMDGCN_Attr<"IsInKernel", "is_in_kernel", [VerifierAttr]> {
  let summary = "AMDGCN verifier to check if an operation is inside a kernel.";
  let extraClassDeclaration = [{
    /// Return the scope in which this verifier attribute operates.
    VerifierScope verifierScope() const {
      return VerifierScope::RunOnRoot;
    }
    /// Verify that the operation is inside a kernel.
    LogicalResult verifyOp(Operation *op,
                           const ::mlir::aster::VerifierState &state) const {
      return verifyIsInKernelImpl(op, state);
    }
  }];
}

//===----------------------------------------------------------------------===//
// IsInModule
//===----------------------------------------------------------------------===//

/// AMDGCN verifier to check if an operation is inside an AMDGCN module.
def IsInModuleAttr :
    AMDGCN_Attr<"IsInModule", "is_in_module", [VerifierAttr]> {
  let summary = "AMDGCN verifier to check if an operation is inside an AMDGCN module.";
  let extraClassDeclaration = [{
    /// Return the scope in which this verifier attribute operates.
    VerifierScope verifierScope() const {
      return VerifierScope::RunOnRoot;
    }
    /// Verify that the operation is inside an AMDGCN module.
    LogicalResult verifyOp(Operation *op,
                           const ::mlir::aster::VerifierState &state) const {
      return verifyIsInModuleImpl(op, state);
    }
  }];
}

///===----------------------------------------------------------------------===//
// IsAllocatableOp
//===----------------------------------------------------------------------===//

/// AMDGCN verifier to check if an operation is allocatable.
def IsAllocatableOpAttr :
    AMDGCN_Attr<"IsAllocatableOp", "is_allocatable", [VerifierAttr]> {
  let summary = "AMDGCN verifier to check if an operation is allocatable.";
  let extraClassDeclaration = [{
    /// Return the scope in which this verifier attribute operates.
    VerifierScope verifierScope() const {
      return VerifierScope::RunOnNested;
    }
    /// Verify that the operation is translatable.
    LogicalResult verifyOp(Operation *op,
                                const ::mlir::aster::VerifierState &state) const {
      return verifyIsAllocatableOpImpl(op, state);
    }
  }];
}

///===----------------------------------------------------------------------===//
// IsAllocatedOpAttr
//===----------------------------------------------------------------------===//

/// AMDGCN verifier to check if an operation is allocated.
def IsAllocatedOpAttr :
    AMDGCN_Attr<"IsAllocatedOp", "is_allocated", [VerifierAttr]> {
  let summary = "AMDGCN verifier to check if an operation is allocated.";
  let extraClassDeclaration = [{
    /// Return the scope in which this verifier attribute operates.
    VerifierScope verifierScope() const {
      return VerifierScope::RunOnNested;
    }
    /// Verify that the operation is allocated.
    LogicalResult verifyOp(Operation *op,
                                const ::mlir::aster::VerifierState &state) const {
      return verifyIsAllocatedOpImpl(op, state);
    }
  }];
}

///===----------------------------------------------------------------------===//
// IsTranslatableOpAttr
//===----------------------------------------------------------------------===//

/// AMDGCN verifier to check if a module is translatable.
def IsTranslatableOpAttr :
    AMDGCN_Attr<"IsTranslatableOp", "is_translatable", [VerifierAttr]> {
  let summary = "AMDGCN verifier to check if an operation is translatable.";
  let extraClassDeclaration = [{
    /// Return the scope in which this verifier attribute operates.
    VerifierScope verifierScope() const {
      return VerifierScope::RunOnRoot;
    }
    /// Use the dependent verifiers to verify that the operation is translatable.
    LogicalResult verifyOp(Operation *op,
                                const ::mlir::aster::VerifierState &state) const {
      return success();
    }
    /// Get the dependent verifiers that should be run before this verifier.
    void getDependentVerifiers(
        SmallVectorImpl<VerifierAttrInterface> &dependentVerifiers) const;
  }];
}

///===----------------------------------------------------------------------===//
// VerifyConstantsAttr
//===----------------------------------------------------------------------===//

/// AMDGCN verifier to check constant ranges.
def VerifyConstantsAttr :
    AMDGCN_Attr<"VerifyConstants", "verify_constants", [VerifierAttr]> {
  let summary = "AMDGCN verifier to check constant ranges.";
  let extraClassDeclaration = [{
    /// Return the scope in which this verifier attribute operates.
    VerifierScope verifierScope() const {
      return VerifierScope::RunOnNested;
    }
    /// Use the dependent verifiers to verify that the operation is translatable.
    LogicalResult verifyOp(Operation *op,
                                const ::mlir::aster::VerifierState &state) const {
      return verifyConstantsImpl(op, state);
    }
  }];
}

#endif // ASTER_AMDGCN_VERIFIERS_TD
