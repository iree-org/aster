//===- amdgcn.cpp ------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Python bindings for the AMDGCN dialect.
//
//===----------------------------------------------------------------------===//

#include "aster/API/API.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Init.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace nb = nanobind;

using namespace nb::literals;

using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

template <RegisterKind kind>
static void bindRegisterType(nb::module_ &m, std::string_view name) {
  mlir_type_subclass(
      m, name.data(),
      +[](MlirType type) { return isRegisterType(type, kind, false); })
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context,
             std::optional<int16_t> regNum) {
            return cls(getRegisterType(
                context, kind, regNum ? Register(*regNum) : Register()));
          },
          "cls"_a, "context"_a = nb::none(), "reg"_a = nb::none())
      .def("is_relocatable", [](MlirType self) {
        std::optional<RegisterRange> range = getRegisterRange(self);
        assert(range && "invalid type");
        return range->begin().isRelocatable();
      });
}

template <RegisterKind kind>
static void bindRegisterRangeType(nb::module_ &m, std::string_view name) {
  mlir_type_subclass(
      m, name.data(),
      +[](MlirType type) { return isRegisterType(type, kind, true); })
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context, int16_t size,
             std::optional<int16_t> regNum, std::optional<int16_t> alignment) {
            auto reg = regNum ? Register(*regNum) : Register();
            return cls(getRegisterRangeType(
                context, kind, RegisterRange(reg, size, alignment)));
          },
          "cls"_a, "context"_a = nb::none(), "size"_a, "reg"_a = nb::none(),
          "alignment"_a = nb::none())
      .def("is_relocatable", [](MlirType self) {
        std::optional<RegisterRange> range = getRegisterRange(self);
        assert(range && "invalid type");
        return range->begin().isRelocatable();
      });
}

NB_MODULE(_amdgcn, m) {
  nb::set_leak_warnings(false);
  llvm::sys::PrintStackTraceOnErrorSignal("amdgcn");
  //===--------------------------------------------------------------------===//
  // API
  //===--------------------------------------------------------------------===//
  m.def("register_all",
        [](MlirDialectRegistry &registry) { asterInitDialects(registry); });

  //===--------------------------------------------------------------------===//
  // Translate Module to Assembly
  //===--------------------------------------------------------------------===//
  m.def(
      "translate_module",
      [](MlirOperation moduleOp, bool debugPrint = false) -> std::string {
        std::optional<std::string> asmOpt =
            translateMlirModule(moduleOp, debugPrint);
        if (!asmOpt.has_value())
          throw std::runtime_error("Failed to translate module to assembly");
        return *asmOpt;
      },
      "module_op"_a, "debug_print"_a = false);

  //===--------------------------------------------------------------------===//
  // AMDGPU Target Check
  //===--------------------------------------------------------------------===//
  m.def("has_amdgpu_target", hasAMDGPUTarget,
        "Check if the AMDGPU target is available in the current LLVM build.");

  //===--------------------------------------------------------------------===//
  // Compile Assembly to Binary
  //===--------------------------------------------------------------------===//
  m.def(
      "compile_asm",
      [](MlirLocation loc, const std::string &asmCode, std::string_view chip,
         std::string_view features,
         std::string_view triple = "amdgcn-amd-amdhsa",
         std::optional<std::string_view> path = std::nullopt,
         bool isLLDPath = false) -> std::optional<nb::bytes> {
        std::vector<char> binary;

        bool success = compileAsm(loc, asmCode, binary, chip, features, triple,
                                  path, isLLDPath);
        if (!success)
          return std::nullopt;

        return nb::bytes(binary.data(), binary.size());
      },
      "loc"_a, "asm"_a, "chip"_a, "features"_a,
      "triple"_a = "amdgcn-amd-amdhsa", "path"_a = nb::none(),
      "is_lld_path"_a = false,
      "Compile AMDGPU assembly code to binary and link it into an HSA code "
      "object.");

  //===--------------------------------------------------------------------===//
  // Register Types
  //===--------------------------------------------------------------------===//
  bindRegisterType<RegisterKind::AGPR>(m, "AGPRType");
  bindRegisterType<RegisterKind::SGPR>(m, "SGPRType");
  bindRegisterType<RegisterKind::VGPR>(m, "VGPRType");

  bindRegisterRangeType<RegisterKind::AGPR>(m, "AGPRRangeType");
  bindRegisterRangeType<RegisterKind::SGPR>(m, "SGPRRangeType");
  bindRegisterRangeType<RegisterKind::VGPR>(m, "VGPRRangeType");
}
