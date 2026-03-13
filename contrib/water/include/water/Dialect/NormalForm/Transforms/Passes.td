// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_NORMALFORM_TRANSFORMS_PASSES
#define WATER_DIALECT_NORMALFORM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LowerWaterNormalFormModulePass : Pass<"lower-water_normalform-module"> {
  let summary = "Lower water_normalform.module to builtin.module";
  let description = [{
    This pass converts `water_normalform.module` operations to `builtin.module`
    operations. The normal form attributes are not retained on the resulting
    module.
  }];
  let dependentDialects = [];
}

#endif // WATER_DIALECT_NORMALFORM_TRANSFORMS_PASSES
