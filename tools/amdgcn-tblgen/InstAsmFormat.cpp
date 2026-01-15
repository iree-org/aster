//===- InstAsmFormat.cpp ------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file generates the AMDGCN instructions assembly format.
//
//===----------------------------------------------------------------------===//

#include "InstCommon.h"
#include "aster/Support/Lexer.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::amdgcn::tblgen;

//===----------------------------------------------------------------------===//
// Generate the asm printers for all instructions in the given record keeper.
//===----------------------------------------------------------------------===//

namespace {
/// Handler to generate the asm printer for an instruction.
struct ASMFormatHandler {
  ASMFormatHandler(const AMDInst &inst);
  void genPrinter(raw_ostream &os);

private:
  using ArgTy = std::optional<std::pair<DagArg, AsmArgFormat>>;
  /// Emit the code to print the given variant.
  void emitVariant(AsmVariant variant, mlir::raw_indented_ostream &os);
  /// Emit the code to print the given argument.
  void emitArg(DagArg dagArg, AsmArgFormat arg, mlir::raw_indented_ostream &os);
  /// Emit the custom syntax at the current lexer position.
  LogicalResult emitCustomSyntax(Lexer &lexer, mlir::raw_indented_ostream &os);
  /// Emit an error message.
  void emitError(Twine msg) { llvm::PrintFatalError(&inst.getDef(), msg); }
  AMDInst inst;
  mlir::tblgen::FmtContext ctx;
  llvm::StringMap<ArgTy> arguments;
  DenseSet<StringRef> opArgs;
};
} // namespace

ASMFormatHandler::ASMFormatHandler(const AMDInst &inst) : inst(inst) {
  Dag args = Record(&inst.getInstOp().getDef(), "Op").getDag("arguments");
  Dag successors =
      Record(&inst.getInstOp().getDef(), "Op").getDag("successors");
  // Collect the arguments.
  for (auto [i, arg] : llvm::enumerate(args.getAsRange())) {
    opArgs.insert(arg.getName());
    if (!AsmArgFormat::isa(arg.getAsRecord()))
      continue;
    arguments[arg.getName()] = {arg, AsmArgFormat(arg.getAsRecord())};
  }
  // Collect the successors.
  for (auto [i, succ] : llvm::enumerate(successors.getAsRange())) {
    opArgs.insert(succ.getName());
    if (!AsmArgFormat::isa(succ.getAsRecord()))
      continue;
    arguments[succ.getName()] = {succ, AsmArgFormat(succ.getAsRecord())};
  }
  // Populate the format context.
  populateFmtContext(inst, ctx);
  ctx.addSubst("_inst", "_inst");
  ctx.addSubst("_parser", "parser");
  ctx.addSubst("_printer", "printer");
}

void ASMFormatHandler::emitArg(DagArg dagArg, AsmArgFormat arg,
                               mlir::raw_indented_ostream &os) {
  ctx.withSelf("_inst.get" +
               llvm::convertToCamelFromSnakeCase(dagArg.getName(), true) +
               "()");
  os.printReindented(mlir::tblgen::tgfmt(arg.getPrinter(), &ctx).str());
  os << "\n";
  // Restore context.
  ctx.withSelf("_inst");
}

LogicalResult
ASMFormatHandler::emitCustomSyntax(Lexer &lexer,
                                   mlir::raw_indented_ostream &os) {
  // Expect '@'.
  if (lexer.currentChar() != '@')
    return failure();
  lexer.consumeChar();

  // Parse identifier.
  FailureOr<StringRef> id = lexer.lexIdentifier();

  // Couldn't lex identifier.
  if (failed(id)) {
    emitError("failed to lex identifier in asm format: " +
              lexer.getCurrentPos());
    return failure();
  }

  // Lex the argument list.
  if (lexer.currentChar() != '(') {
    emitError("expected '(' after custom syntax: " + lexer.getCurrentPos());
    return failure();
  }
  lexer.consumeChar(); // consume '('
  StringRef argStr =
      lexer.getCurrentPos().take_until([](char c) { return c == ')'; });
  if (lexer.getCurrentPos() == argStr) {
    emitError("expected ')' to terminate custom syntax: " +
              lexer.getCurrentPos());
    return failure();
  }
  (void)lexer.consume(argStr);
  lexer.consumeChar(); // consume ')'

  // Parse the arguments.
  SmallVector<StringRef, 4> args;
  argStr.split(args, ',', -1, true);
  for (StringRef &arg : args) {
    arg = arg.trim(" \t\n\v\f\r$");
    if (!arguments.contains(arg)) {
      emitError("unknown argument in custom syntax: " + arg);
      return failure();
    }
    if (arg.empty()) {
      emitError("empty argument in custom syntax");
      return failure();
    }
  }
  os << id << "(" << mlir::tblgen::tgfmt("$_printer, $_self", &ctx);
  if (!args.empty())
    os << ", ";
  llvm::interleaveComma(args, os, [&](StringRef arg) {
    os << "_inst.get" + llvm::convertToCamelFromSnakeCase(arg, true) + "()";
  });
  os << ");\n";
  return success();
}

void ASMFormatHandler::emitVariant(AsmVariant variant,
                                   mlir::raw_indented_ostream &os) {
  Lexer lexer(variant.getAsmFormat());
  os << "if ((";
  os << mlir::tblgen::tgfmt(variant.getPredicate().getConditionTemplate(), &ctx,
                            "_inst");
  os << ")) {\n";
  os.indent();
  // Parse the asm format string and emit code.
  while (lexer.currentChar() != '\0') {
    lexer.consumeWhiteSpace();
    if (lexer.currentChar() == '\0')
      break;

    // Handle argument.
    if (lexer.currentChar() == '$') {
      lexer.consumeChar();
      FailureOr<StringRef> id = lexer.lexIdentifier();

      // Couldn't lex identifier.
      if (failed(id)) {
        llvm::PrintFatalError(&inst.getDef(),
                              "failed to lex identifier in asm format: " +
                                  lexer.getCurrentPos());
        return;
      }

      // Lookup the argument.
      ArgTy argumentsIt = arguments.lookup(*id);
      // Check if the argument derives from `AsmArgFormat`.
      if (!argumentsIt.has_value()) {
        if (opArgs.contains(*id)) {
          llvm::PrintFatalError(
              &inst.getDef(),
              "argument in asm format doesn't derive from `AsmArgFormat`: " +
                  *id);
          return;
        }

        // Unknown argument.
        llvm::PrintFatalError(&inst.getDef(),
                              "unknown argument in asm format: " + *id);
        return;
      }

      // Emit the argument.
      emitArg(argumentsIt->first, argumentsIt->second, os);
      continue;
    }

    // Handle comma.
    if (lexer.currentChar() == ',') {
      lexer.consumeChar();
      os << "$_printer.printComma();\n";
      continue;
    }

    // Handle keywords.
    if (lexer.currentChar() == '_' || std::isalpha(lexer.currentChar())) {
      FailureOr<StringRef> id = lexer.lexIdentifier();
      if (failed(id)) {
        llvm::PrintFatalError(&inst.getDef(),
                              "failed to lex keyword in asm format: " +
                                  lexer.getCurrentPos());
        return;
      }
      os << llvm::formatv("$_printer.printKeyword(\"{0}\");\n", *id);
      continue;
    }

    // Handle custom syntax.
    if (lexer.currentChar() == '@') {
      if (failed(emitCustomSyntax(lexer, os)))
        return;
      continue;
    }

    // Unexpected character.
    llvm::PrintFatalError(&inst.getDef(),
                          "unexpected character in asm format: " +
                              lexer.getCurrentPos());
  }
  os << "return success();\n";
  os.unindent();
  os << "}\n";
}

void ASMFormatHandler::genPrinter(raw_ostream &osOut) {
  StrStream strStream;
  mlir::raw_indented_ostream os(strStream.os);

  // Populate the format context.
  populateFmtContext(inst, ctx);
  ctx.withSelf("_inst");

  // Generate the printer function.
  os << "static mlir::LogicalResult print$0(mlir::aster::amdgcn::OpCode "
        "opcode, "
        "mlir::aster::amdgcn::AsmPrinter &printer, mlir::Operation *op) {\n";
  os.indent();
  os << "assert(opcode == $_opcode && \"unexpected opcode\");\n";
  os << "auto _inst = llvm::cast<$_instOp>(op);\n";
  os << "(void)_inst;\n";
  os << "auto _grd = $_printer.printMnemonic(\"$_mnemonic\");\n";
  os << "(void)_grd;\n";
  for (const AsmVariant &variant : inst.getAsmFormat())
    emitVariant(variant, os);
  // Finish the function.
  os << "llvm_unreachable(\"failed to print instruction\");\n";
  os << "return failure();\n";
  os.unindent();
  os << "}\n";
  osOut << mlir::tblgen::tgfmt(strStream.str, &ctx, inst.getName());
}

static bool generateInstPrinters(const llvm::RecordKeeper &records,
                                 raw_ostream &os) {
  llvm::SmallVector<const llvm::Record *> instRecs =
      llvm::to_vector(records.getAllDerivedDefinitions(AMDInst::ClassType));
  StrStream stream;
  // Sort by ID to have a deterministic order.
  llvm::sort(instRecs, llvm::LessRecord());

  // Generate each printer function.
  llvm::interleave(
      instRecs, os,
      [&](const llvm::Record *instRec) {
        AMDInst inst(instRec);
        ASMFormatHandler handler(inst);
        handler.genPrinter(os);
      },
      "\n");

  // Generate the opcode to printer function table.
  os << "\nstatic const llvm::SmallVector<"
        "llvm::function_ref<mlir::LogicalResult(mlir::aster::amdgcn::OpCode, "
        "mlir::aster::amdgcn::AsmPrinter &, mlir::Operation *)>> "
        "_instPrinters = {\n";
  os << "  nullptr, // OpCode::Invalid\n";

  // Generate each table entry.
  llvm::interleave(
      instRecs, os,
      [&](const llvm::Record *instRec) {
        AMDInst inst(instRec);
        os << "  print" << inst.getName() << ",";
      },
      "\n");
  os << "\n};\n";
  return false;
}

//===----------------------------------------------------------------------===//
// TableGen Registration
//===----------------------------------------------------------------------===//

// Generator that generates AMDGCN printer definitions.
static GenRegistration generateInstPrintersReg("gen-inst-printers",
                                               "Generate inst printers",
                                               generateInstPrinters);
