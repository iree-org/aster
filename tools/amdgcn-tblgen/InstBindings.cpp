//===- InstBindings.cpp -------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file generates the AMDGCN instructions bindings for Python.
//
//===----------------------------------------------------------------------===//

#include "InstCommon.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::amdgcn::tblgen;

static llvm::cl::OptionCategory instPyGenCat("Inst Generator Options");

static llvm::cl::opt<std::string> pyImportFile(
    "inst-py-import",
    llvm::cl::desc("File with the py bindings definitions to include"),
    llvm::cl::cat(instPyGenCat));

//===----------------------------------------------------------------------===//
// Generate the python bindings for all instructions in the given record keeper.
//===----------------------------------------------------------------------===//

/// Get the builder function for the given instruction.
static std::string getPythonBuilder(const AMDInst &inst,
                                    mlir::tblgen::FmtContext &ctx) {
  std::optional<Builder> b = inst.getPythonBuilder();
  if (!b.has_value())
    return "";
  if (b->getBody().has_value() == false) {
    llvm::PrintFatalError(&inst.getDef(),
                          "python builder for instruction must have a body.");
  }
  ctx.addSubst("_args", genArgList(*b, ctx, /*isCpp=*/false));
  ctx.addSubst("_loc", "loc");
  ctx.addSubst("_ip", "ip");
  ctx.addSubst("_lastArgs", "loc=loc, ip=ip");
  // Print the body and description with indentation.
  StrStream bodyStream, descriptionStream;
  {
    mlir::raw_indented_ostream os(bodyStream.os);
    os.indent();
    os.indent();
    os.printReindented(*b->getBody());
    bodyStream.str = mlir::tblgen::tgfmt(bodyStream.str, &ctx).str();
  }
  {
    mlir::raw_indented_ostream os(descriptionStream.os);
    os.indent();
    os.indent();
    os.printReindented(inst.getDescription());
    descriptionStream.str =
        mlir::tblgen::tgfmt(descriptionStream.str, &ctx).str();
  }
  std::string_view body = R"(
# Instruction: $_mnemonic
def $_name($0$_loc=None, ip=None):
    """
$1
    """
    InstOp = _inst_base.$_pyClass
$2
  )";
  std::string params =
      genParamList(*b, ctx, /*isCpp=*/false, false, /*prefixComma=*/false,
                   /*postfixComma=*/true);
  return mlir::tblgen::tgfmt(body.data(), &ctx, params, descriptionStream.str,
                             bodyStream.str, &ctx)
      .str();
}

/// Generate the binding for the given instruction.
static void genInstBinding(const AMDInst &inst, raw_ostream &os) {
  mlir::tblgen::FmtContext ctx;
  ctx.addSubst("_pyMod", "_inst_base");
  ctx.addSubst("_pyClass", getInstOpName(inst.getInstOp(), false));
  ctx.addSubst("_name", inst.getAsEnumCase().getSymbol());
  ctx.addSubst("_opcode", "_inst_base." + getOpCode(inst, true));
  ctx.addSubst("_mnemonic", inst.getMnemonic());
  ctx.addSubst("_typing", "_typing");
  ctx.addSubst("_ir", "_ods_ir");
  ctx.addSubst("_instOp", "InstOp");
  ctx.addSubst("_create", "InstOp");
  ctx.addSubst("_loc", "loc");
  ctx.addSubst("_ip", "ip");
  os << getPythonBuilder(inst, ctx);
}

// Generate the binding for the given attribute.
static void genPyAttrBinding(const Record &record, raw_ostream &os) {
  std::optional<Builder> b =
      record.getOptionalDefAs<Builder>("builder", ArrayRef<llvm::SMLoc>());
  if (!b.has_value())
    return;
  if (b->getBody().has_value() == false) {
    llvm::PrintFatalError(&record.getDef(),
                          "python builder for attr must have a body.");
  }
  mlir::tblgen::FmtContext ctx;
  ctx.addSubst("_pyMod", "_inst_base");
  ctx.addSubst("_name", record.getName());
  ctx.addSubst("_typing", "_typing");
  ctx.addSubst("_ir", "_ods_ir");
  ctx.addSubst("_ctx", "context");
  std::string_view body = R"(
@_register_attribute_builder("$_name")
def _$0($1$_ctx):
$2
)";
  ctx.addSubst("_args", genArgList(*b, ctx, /*isCpp=*/false, false));
  // Print the body and description with indentation.
  StrStream bodyStream;
  {
    mlir::raw_indented_ostream os(bodyStream.os);
    os.indent();
    os.indent();
    os.printReindented(*b->getBody());
    bodyStream.str = mlir::tblgen::tgfmt(bodyStream.str, &ctx).str();
  }
  std::string params =
      genParamList(*b, ctx, /*isCpp=*/false, false, /*prefixComma=*/false,
                   /*postfixComma=*/true);
  os << mlir::tblgen::tgfmt(body.data(), &ctx,
                            llvm::convertToSnakeFromCamelCase(record.getName()),
                            params, bodyStream.str);
}

/// Generate the python bindings for all instructions in the given record
/// keeper.
static bool generateInstBindings(const llvm::RecordKeeper &records,
                                 raw_ostream &os) {
  llvm::SmallVector<const llvm::Record *> instRecs =
      llvm::to_vector(records.getAllDerivedDefinitions(AMDInst::ClassType));
  // Sort by ID to have a deterministic order.
  llvm::sort(instRecs, llvm::LessRecordByID());
  os << R"(
# Autogenerated by amdgcn-tblgen; don't manually edit.

from ._ods_common import _cext as _ods_cext
from ._ods_common import (
    get_op_result_or_value as _get_op_result_or_value,
)
from ..ir import register_attribute_builder as _register_attribute_builder
_ods_ir = _ods_cext.ir

import builtins
import typing as _typing
)";
  if (llvm::StringRef(pyImportFile).trim().empty())
    llvm::PrintFatalError("inst-py-import option is required");
  os << "import " << pyImportFile << " as _inst_base\n\n";
  for (const llvm::Record *instRec : instRecs)
    genInstBinding(AMDInst(instRec), os);
  {
    llvm::SmallVector<const llvm::Record *> attrRecs =
        llvm::to_vector(records.getAllDerivedDefinitions("PyAttr"));
    // Sort by ID to have a deterministic order.
    llvm::sort(attrRecs, llvm::LessRecordByID());
    for (const llvm::Record *attrRec : attrRecs)
      genPyAttrBinding(Record(attrRec, "PyAttr"), os);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// TableGen Registration
//===----------------------------------------------------------------------===//

// Generator that generates AMDGCN instructions bindings.
static GenRegistration generateInstBindingsReg("gen-inst-bindings",
                                               "Generate inst bindings",
                                               generateInstBindings);
