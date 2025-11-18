//===- VerifierAttr.td - Verifier attribute interface ------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the VerifierAttr interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_VERIFIERATTR_TD
#define ASTER_INTERFACES_VERIFIERATTR_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// VerifierAttr interface
//===----------------------------------------------------------------------===//

def VerifierAttr : AttrInterface<"VerifierAttrInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    This interface provides a mechanism for attributes to perform custom
    verification logic. It is illegal to modify the IR during verification.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Initialize the verifier state before verification begins.
      }],
      /*retTy=*/"void",
      /*methodName=*/"initializeVerifierState",
      /*args=*/(ins "::mlir::aster::VerifierState &":$state), [{}], [{
        (void)state;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the dependent verifiers that should be run before this verifier.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getDependentVerifiers", (ins
        "llvm::SmallVectorImpl<::mlir::aster::VerifierAttrInterface> &":$dependentVerifiers
      ), [{}], [{
        (void)dependentVerifiers;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        The scope in which this verifier attribute operates.
      }],
      /*retTy=*/"::mlir::aster::VerifierScope",
      /*methodName=*/"verifierScope"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Verify the operation using this attribute.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"verifyOp",
      /*args=*/(ins "::mlir::Operation *":$op,
        "const ::mlir::aster::VerifierState &":$state)
    >
  ];
}

#endif // ASTER_INTERFACES_VERIFIERATTR_TD
