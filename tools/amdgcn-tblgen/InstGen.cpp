//===- InstGen.cpp -------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file generates the AMDGCN instructions declarations and definitions.
//
//===----------------------------------------------------------------------===//

#include "InstCommon.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/CodeGenHelpers.h"

using namespace mlir;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::amdgcn::tblgen;

//===----------------------------------------------------------------------===//
// Instruction Declaration Generation
//===----------------------------------------------------------------------===//

/// Get the builder function for the given instruction.
static std::string getBuilderDecl(const AMDInst &inst,
                                  mlir::tblgen::FmtContext &ctx) {
  std::optional<Builder> b = inst.getCppBuilder();
  if (!b.has_value())
    return "";
  StringRef body = "\n  static InstOp create(mlir::OpBuilder &builder, "
                   "mlir::Location loc$0);";
  std::string params =
      genParamList(*b, ctx, /*isCpp=*/true, true, /*prefixComma=*/true);
  return mlir::tblgen::tgfmt(body.data(), &ctx,
                             /*0=*/params);
}

/// Generate declaration for the given instruction.
static void genInstDecl(const AMDInst &inst, raw_ostream &os) {
  llvm::NamespaceEmitter ns(os, inst.getCppNamespace());
  std::string_view body =
      R"(struct $0 : public ::mlir::aster::amdgcn::InstMD<$0> {
  using Base::Base;
  using InstOp = $_instOp;
  static constexpr mlir::aster::amdgcn::OpCode kOpCode = $_opcode;
  static constexpr std::string_view mnemonic = "$_mnemonic";
  static constexpr std::array<mlir::aster::amdgcn::Target, $_numTargets> targets = {$_targets};$1
  mlir::LogicalResult verifyImpl(InstOp op) const;$2$3
};
)";
  llvm::StringRef initDecl =
      inst.hasInit()
          ? "\nprivate:\n  void initialize(mlir::MLIRContext *ctx) override;"
          : "";
  mlir::tblgen::FmtContext ctx;
  populateFmtContext(inst, ctx);
  os << mlir::tblgen::tgfmt(body.data(), &ctx,
                            /*0=*/inst.getName(),
                            /*1=*/getBuilderDecl(inst, ctx),
                            /*2=*/inst.getExtraClassDeclaration(),
                            /*3=*/initDecl);
}

/// Generate declarations for all instructions in the given record keeper.
static bool generateInstDecls(const llvm::RecordKeeper &records,
                              raw_ostream &os) {
  llvm::IfDefEmitter ifdefEmitter(os, "AMDGCN_GEN_INST_DECLS");
  llvm::SmallVector<const llvm::Record *, 8> instRecs =
      llvm::to_vector(records.getAllDerivedDefinitions(AMDInst::ClassType));
  // Sort by ID to have a deterministic order.
  llvm::sort(instRecs, llvm::LessRecordByID());

  // Generate each declaration.
  for (const llvm::Record *instRec : instRecs) {
    AMDInst inst(instRec);
    genInstDecl(inst, os);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Instruction Definitions Generation
//===----------------------------------------------------------------------===//

/// Get the 'self' expression for the given operand name.
static std::string getSelf(StringRef name) {
  if (name.empty())
    return "op";
  return "getTypeOrValue(op.get" +
         llvm::convertToCamelFromSnakeCase(name, true) + "())";
}

/// Generate code for the given constraint.
static void genConstraint(raw_ostream &os, mlir::tblgen::FmtContext &ctx,
                          StringRef self,
                          const mlir::tblgen::Constraint &constraint) {
  bool isOptional = false;
  if (auto c = dyn_cast<mlir::tblgen::TypeConstraint>(&constraint)) {
    isOptional = c->isOptional();
  } else if (auto c = dyn_cast<mlir::tblgen::Attribute>(&constraint)) {
    isOptional = c->isOptional();
  }
  const std::string_view body = R"(  {
    auto &&_self = $_selfExpr;
    if ($0!($1)) {
      return $_op.emitError() << R"delim($2)delim"$3$4;
    }
  })";
  ctx.withSelf("_self");
  ctx.addSubst("_selfExpr", self);
  StringRef optionalStr = isOptional ? "_self && " : "";
  StringRef errExtra = constraint.getDescription().trim();
  StringRef errPrefix = errExtra.empty() ? "" : " << ";
  os << mlir::tblgen::tgfmt(body.data(), &ctx,
                            /*0=*/optionalStr,
                            /*1=*/constraint.getConditionTemplate(),
                            /*2=*/constraint.getSummary(),
                            /*3=*/errPrefix,
                            /*4=*/errExtra);
}

/// Generate code for the given instruction argument constraints.
static void genConstraints(DagArg arg, raw_ostream &os,
                           mlir::tblgen::FmtContext &ctx) {
  InstConstraint instArg(arg.getAsRecord());
  llvm::interleave(
      instArg.getConstraints(), os,
      [&](const mlir::tblgen::Constraint &c) {
        ctx.addSubst("_argName", arg.getName());
        StrStream stream;
        genConstraint(stream.os, ctx, getSelf(arg.getName()), c);
        os << mlir::tblgen::tgfmt(stream.str.data(), &ctx);
      },
      "\n");
}

/// Get the verify body for the given instruction.
static std::string getVerifyBody(const AMDInst &inst,
                                 mlir::tblgen::FmtContext &ctx) {
  StrStream stream;
  ctx.addSubst("_op", "op");
  llvm::interleave(
      inst.getConstraints().getAsRange(), stream.os,
      [&](DagArg arg) { genConstraints(arg, stream.os, ctx); }, "\n");
  return stream.str;
}

/// Get the builder function for the given instruction.
static std::string getBuilderDef(const AMDInst &inst,
                                 mlir::tblgen::FmtContext &ctx) {
  std::optional<Builder> b = inst.getCppBuilder();
  if (!b.has_value())
    return "";
  std::optional<StringRef> bodyOpt = b->getBody();
  if (!bodyOpt.has_value())
    return "";
  const std::string_view body = R"(
$_cppClass::InstOp $_cppClass::create(mlir::OpBuilder &builder, mlir::Location loc$0) {
$1
}
)";
  ctx.withBuilder("builder");
  ctx.addSubst("_loc", "loc");
  ctx.addSubst("_create", "InstOp::create");
  ctx.addSubst("_args", genArgList(*b, ctx, /*isCpp=*/true));
  return mlir::tblgen::tgfmt(
      body.data(), &ctx,
      /*0=*/genParamList(*b, ctx, /*isCpp=*/true, false),
      /*1=*/mlir::tblgen::tgfmt(bodyOpt.value().data(), &ctx));
}

/// Generate declaration for the given instruction.
static void genInstDef(const AMDInst &inst, raw_ostream &os) {
  mlir::tblgen::FmtContext ctx;
  populateFmtContext(inst, ctx);
  os << "// Instruction: " << inst.getMnemonic() << "\n";
  if (std::string builderDef = getBuilderDef(inst, ctx); !builderDef.empty())
    os << builderDef;
  {
    std::string_view body =
        R"(
mlir::LogicalResult $_cppClass::verifyImpl($_instOp op) const {
$0
  return mlir::success();
}
)";
    os << mlir::tblgen::tgfmt(body.data(), &ctx,
                              /*0=*/getVerifyBody(inst, ctx));
  }
  if (StringRef extraDef = inst.getExtraClassDefinition();
      !extraDef.trim().empty()) {
    os << mlir::tblgen::tgfmt(extraDef.data(), &ctx);
  }
}

/// Generate the instruction metadata tables.
static void genTables(ArrayRef<const llvm::Record *> insts, raw_ostream &os) {
  os << R"(
static ::mlir::aster::amdgcn::InstMetadata *
getMetadataForOpCode(::mlir::AttributeStorageAllocator &allocator,
                     ::mlir::aster::amdgcn::OpCode opCode) {
  switch (opCode) {
)";
  auto getTable = [&](const AMDInst &inst) {
    std::string_view body =
        "  case {0}:\n    return allocateInstMetadata<{1}>(allocator);";
    os << llvm::formatv(body.data(), getOpCode(inst),
                        getQualName(inst.getCppNamespace(), inst.getName()));
  };

  // Generate the instruction metadata table.
  auto amdInst = llvm::map_range(
      insts, [](const llvm::Record *rec) { return AMDInst(rec); });
  llvm::interleave(amdInst, os, getTable, "\n");
  os << "\n  default:\n    return nullptr;\n  }\n}\n\n";
}

/// Generate declarations for all instructions in the given record keeper.
static bool generateInstDefs(const llvm::RecordKeeper &records,
                             raw_ostream &os) {
  llvm::IfDefEmitter ifdefEmitter(os, "AMDGCN_GEN_INST_DEFS");
  llvm::SmallVector<const llvm::Record *> instRecs =
      llvm::to_vector(records.getAllDerivedDefinitions(AMDInst::ClassType));
  // Sort by ID to have a deterministic order.
  llvm::sort(instRecs, llvm::LessRecordByID());

  // Generate each instruction definition.
  for (const llvm::Record *instRec : instRecs)
    genInstDef(AMDInst(instRec), os);

  // Generate the instruction metadata table.
  genTables(instRecs, os);

  // Generate verify methods for instructions that need them.
  {
    llvm::SmallVector<const llvm::Record *> instOps =
        llvm::to_vector(llvm::make_filter_range(
            records.getAllDerivedDefinitions(InstOpClassType),
            [](const llvm::Record *rec) {
              return Record(rec, InstOpClassType).getBit("genInstVerifier");
            }));
    // Sort by ID to have a deterministic order.
    llvm::sort(instOps, llvm::LessRecordByID());
    for (const llvm::Record *instOpRec : instOps) {
      Operator instOp(instOpRec);
      std::string name = instOp.getQualCppClassName();
      os << "mlir::LogicalResult " << StringRef(name).trim("::")
         << "::verify() {\n";
      os << "  return getOpcodeAttr().getMetadata()->verify(getOperation());\n";
      os << "}\n";
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// TableGen Registration
//===----------------------------------------------------------------------===//

// Generator that generates AMDGCN instructions declarations.
static GenRegistration generateInstDeclsReg(
    "gen-inst-decls", "Generate inst declarations",
    [](const llvm::RecordKeeper &records, raw_ostream &os) {
      return generateInstDecls(records, os);
    });

// Generator that generates AMDGCN instructions definitions.
static GenRegistration
    generateInstDefsReg("gen-inst-defs", "Generate inst definitions",
                        [](const llvm::RecordKeeper &records, raw_ostream &os) {
                          return generateInstDefs(records, os);
                        });
