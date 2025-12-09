//===- InstCommon.cpp ---------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the utility functions and classes for the AMDGCN
// tblgen tool.
//
//===----------------------------------------------------------------------===//

#include "InstCommon.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::amdgcn::tblgen;

std::string mlir::aster::amdgcn::tblgen::getQualName(StringRef cppNamespace,
                                                     StringRef className) {
  if (cppNamespace.empty())
    return className.str();
  return (cppNamespace.trim("::") + "::" + className).str();
}

std::string
mlir::aster::amdgcn::tblgen::getInstOpName(const mlir::tblgen::Operator &instOp,
                                           bool addNamespace) {
  if (!addNamespace)
    return instOp.getCppClassName().str();
  return instOp.getQualCppClassName();
}

std::string mlir::aster::amdgcn::tblgen::getOpCode(const AMDInst &inst,
                                                   bool isPython) {
  if (isPython)
    return ("OpCode." + inst.getAsEnumCase().getIdentifier()).str();
  return ("::mlir::aster::amdgcn::OpCode::" +
          inst.getAsEnumCase().getIdentifier())
      .str();
}

std::string mlir::aster::amdgcn::tblgen::genParamList(
    const Builder &b, mlir::tblgen::FmtContext &ctx, bool isCpp, bool isDecl,
    bool prefixComma, bool postfixComma) {
  StrStream str;
  raw_ostream &os = str.os;
  if (b.getParameters().size() == 0)
    return "";
  if (prefixComma)
    os << ", ";
  auto printer = [&](const Builder::Parameter &p) {
    if (!p.getName().has_value())
      llvm::PrintFatalError("Builder parameter must have a name.");
    if (!isCpp) {
      os << *(p.getName()) << ": " << p.getCppType();
      if (std::optional<StringRef> defaultValue = p.getDefaultValue())
        os << " = " << *defaultValue;
      return;
    }
    os << p.getCppType() << " " << *(p.getName());
    if (std::optional<StringRef> defaultValue = p.getDefaultValue();
        defaultValue && isDecl)
      os << " = " << *defaultValue;
  };
  llvm::interleaveComma(b.getParameters(), os, printer);
  if (postfixComma)
    os << ", ";
  return mlir::tblgen::tgfmt(str.str.data(), &ctx);
}

std::string
mlir::aster::amdgcn::tblgen::genArgList(const Builder &b,
                                        mlir::tblgen::FmtContext &ctx,
                                        bool isCpp, bool useKwArgs) {
  StrStream str;
  raw_ostream &os = str.os;
  if (b.getParameters().size() == 0)
    return "";
  auto printer = [&](const Builder::Parameter &p) {
    if (!p.getName().has_value())
      llvm::PrintFatalError("Builder parameter must have a name.");
    if (!isCpp) {
      os << *(p.getName());
      if (useKwArgs)
        os << "=" << *(p.getName());
      return;
    }
    os << *(p.getName());
  };
  llvm::interleaveComma(b.getParameters(), os, printer);
  return mlir::tblgen::tgfmt(str.str.data(), &ctx);
}

/// Get the list of target names for the given instruction.
std::string
mlir::aster::amdgcn::tblgen::getISAVersionList(const AMDInst &inst) {
  SmallVector<ISAVersion> isa = inst.getISAVersions();
  StrStream tgts;
  llvm::interleaveComma(isa, tgts.os, [&](const ISAVersion &tgt) {
    tgts.os << "::mlir::aster::amdgcn::ISAVersion::" +
                   tgt.getAsEnumCase().getIdentifier();
  });
  return tgts.str;
}

/// Populate the format context with common substitutions.
void mlir::aster::amdgcn::tblgen::populateFmtContext(
    const AMDInst &inst, mlir::tblgen::FmtContext &ctx) {
  ctx.addSubst("_name", inst.getName());
  ctx.addSubst("_cppNamespace", inst.getCppNamespace());
  ctx.addSubst("_cppClass",
               getQualName(inst.getCppNamespace(), inst.getName()));
  ctx.addSubst("_instOp", getInstOpName(inst.getInstOp()));
  ctx.addSubst("_opcode", getOpCode(inst));
  ctx.addSubst("_mnemonic", inst.getMnemonic());
  ctx.addSubst("_isa", getISAVersionList(inst));
  ctx.addSubst("_numISAVersions", std::to_string(inst.getISAVersions().size()));
}
