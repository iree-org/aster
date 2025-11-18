//===- site_init.cpp ---------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===////
//
// This file implements the site initialization for the AMDGCN Python module.
//
//===----------------------------------------------------------------------===//

#include "aster/Init.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(_site_initialize_0, m) {
  m.doc() = "amdgcn registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    /// Upstream MLIR CAPI stuff
    mlir::aster::asterRegisterUpstreamMLIRDialects(registry);
    mlir::aster::asterRegisterUpstreamMLIRInterfaces(registry);
    mlir::aster::asterRegisterUpstreamMLIRExternalModels(registry);
    /// Aster CAPI stuff
    mlir::aster::asterInitDialects(registry);
  });

  // Register passes when module is imported
  /// Upstream MLIR CAPI stuff
  mlir::aster::registerUpstreamMLIRPasses();
  /// Aster CAPI stuff
  mlir::aster::registerPasses();
}
