//===- pdl.cpp ------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace nb = nanobind;

using namespace nb::literals;

using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

NB_MODULE(_pdl, m) {
  nb::set_leak_warnings(false);
  llvm::sys::PrintStackTraceOnErrorSignal("pdl");
}
