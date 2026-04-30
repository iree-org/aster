//===- InstAsmPrinterGen.cpp ----------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file generates encoding-aware assembly printers for AMDGCN instructions.
//
//===----------------------------------------------------------------------===//

#include "InstCommon.h"
#include "aster/Support/Lexer.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/TableGen/Error.h"

using namespace mlir;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::amdgcn::tblgen;

//===----------------------------------------------------------------------===//
// Generate encoding-aware asm printers for AMDISAInstruction records.
//===----------------------------------------------------------------------===//

/// A flattened (arch, encoding, syntax, predicate) entry for dispatch.
struct ArchEncodingSyntax {
  StringRef archId;
  StringRef encodingId;
  StringRef syntax;
  mlir::tblgen::Pred pred;
};

/// An (arch, syntax, predicate) tuple within a single encoding.
struct ArchSyntaxPred {
  StringRef archId;
  StringRef syntax;
  mlir::tblgen::Pred pred;
};

namespace {
/// Handler to generate the encoding-aware asm printer for an instruction.
struct ASMPrinterHandler {
  ASMPrinterHandler(const llvm::Record *rec);
  void genPrinter(raw_ostream &os);

private:
  using ArgTy = std::optional<std::pair<DagArg, ASMArgFormat>>;

  //===--------------------------------------------------------------------===//
  // Syntax token emitters
  //===--------------------------------------------------------------------===//

  /// Emit a `${mnemonic}` interpolation token.
  void emitMnemonicInterpolation(Lexer &lexer, mlir::raw_indented_ostream &os);
  /// Emit a `$identifier` operand reference token.
  void emitOperandRef(Lexer &lexer, mlir::raw_indented_ostream &os);
  /// Emit a `[identifier]` modifier reference token.
  void emitModifierRef(Lexer &lexer, mlir::raw_indented_ostream &os);
  /// Emit a keyword token.
  void emitKeyword(Lexer &lexer, mlir::raw_indented_ostream &os);

  //===--------------------------------------------------------------------===//
  // Higher-level emitters
  //===--------------------------------------------------------------------===//

  /// Emit the printer code for a complete ASMString syntax.
  void emitSyntax(StringRef syntax, mlir::raw_indented_ostream &os);
  /// Emit the printer code for a single argument.
  void emitArg(DagArg dagArg, ASMArgFormat arg, mlir::raw_indented_ostream &os);
  /// Emit encoding dispatch logic for all entries.
  void emitEncodingDispatch(llvm::ArrayRef<ArchEncodingSyntax> entries,
                            mlir::raw_indented_ostream &os);
  /// Emit syntax guarded by a predicate (handles true-predicate elision).
  void emitGuardedSyntax(StringRef syntax, const mlir::tblgen::Pred &pred,
                         mlir::raw_indented_ostream &os);
  /// Emit the body when all entries for an encoding share the same syntax.
  void emitUniformEncoding(llvm::ArrayRef<ArchSyntaxPred> archEntries,
                           mlir::raw_indented_ostream &os);
  /// Emit the body when entries differ by arch within a single encoding.
  void emitArchGroupedEncoding(llvm::ArrayRef<ArchSyntaxPred> archEntries,
                               mlir::raw_indented_ostream &os);

  /// Emit an error message.
  void emitError(Twine msg) { llvm::PrintFatalError(&instOp.getDef(), msg); }

  InstOp instOp;
  mlir::tblgen::FmtContext ctx;
  llvm::StringMap<ArgTy> arguments;
};
} // namespace

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

ASMPrinterHandler::ASMPrinterHandler(const llvm::Record *rec) : instOp(rec) {
  // Collect arguments from outputs, inputs, and trailingArgs.
  for (const Dag &dag :
       {instOp.getOutputs(), instOp.getInputs(), instOp.getTrailingArgs()}) {
    for (auto [i, arg] : llvm::enumerate(dag.getAsRange())) {
      if (!ASMArgFormat::isa(arg.getAsRecord()))
        continue;
      arguments[arg.getName()] = {arg, ASMArgFormat(arg.getAsRecord())};
    }
  }
  // Set up the format context.
  ctx.addSubst("_inst", "_inst");
  ctx.addSubst("_printer", "printer");
}

//===----------------------------------------------------------------------===//
// Syntax token emitters
//===----------------------------------------------------------------------===//

void ASMPrinterHandler::emitMnemonicInterpolation(
    Lexer &lexer, mlir::raw_indented_ostream &os) {
  lexer.consumeChar(); // consume '$'
  lexer.consumeChar(); // consume '{'
  FailureOr<StringRef> id = lexer.lexIdentifier();
  if (failed(id))
    emitError("failed to lex interpolation in asm_syntax");
  if (*id != "mnemonic")
    emitError("unknown interpolation ${" + *id + "} in asm_syntax");
  if (lexer.currentChar() != '}')
    emitError("expected '}' in asm_syntax interpolation");
  lexer.consumeChar(); // consume '}'

  // Collect any trailing suffix (e.g. `_e64` in `${mnemonic}_e64`).
  std::string suffix;
  while (lexer.currentChar() == '_' || std::isalnum(lexer.currentChar())) {
    suffix += lexer.currentChar();
    lexer.consumeChar();
  }
  os << "auto _grd = $_printer.printMnemonic(\"" << instOp.getOpName() << suffix
     << "\");\n";
}

void ASMPrinterHandler::emitOperandRef(Lexer &lexer,
                                       mlir::raw_indented_ostream &os) {
  lexer.consumeChar(); // consume '$'
  FailureOr<StringRef> id = lexer.lexIdentifier();
  if (failed(id))
    emitError("failed to lex identifier in asm_syntax: " +
              lexer.getCurrentPos());

  ArgTy arg = arguments.lookup(*id);
  if (!arg.has_value())
    emitError("unknown operand $" + *id + " in asm_syntax");

  emitArg(arg->first, arg->second, os);
}

void ASMPrinterHandler::emitModifierRef(Lexer &lexer,
                                        mlir::raw_indented_ostream &os) {
  lexer.consumeChar(); // consume '['
  lexer.consumeWhiteSpace();
  FailureOr<StringRef> id = lexer.lexIdentifier();
  if (failed(id))
    emitError("failed to lex modifier name in asm_syntax: " +
              lexer.getCurrentPos());
  lexer.consumeWhiteSpace();
  if (lexer.currentChar() != ']')
    emitError("expected ']' after modifier name in asm_syntax");
  lexer.consumeChar(); // consume ']'

  ArgTy arg = arguments.lookup(*id);
  if (!arg.has_value())
    emitError("unknown modifier [" + *id + "] in asm_syntax");

  emitArg(arg->first, arg->second, os);
}

void ASMPrinterHandler::emitKeyword(Lexer &lexer,
                                    mlir::raw_indented_ostream &os) {
  FailureOr<StringRef> id = lexer.lexIdentifier();
  if (failed(id))
    emitError("failed to lex keyword in asm_syntax: " + lexer.getCurrentPos());
  os << llvm::formatv("$_printer.printKeyword(\"{0}\");\n", *id);
}

//===----------------------------------------------------------------------===//
// Higher-level emitters
//===----------------------------------------------------------------------===//

void ASMPrinterHandler::emitArg(DagArg dagArg, ASMArgFormat arg,
                                mlir::raw_indented_ostream &os) {
  ctx.withSelf("_inst.get" +
               llvm::convertToCamelFromSnakeCase(dagArg.getName(), true) +
               "()");
  os.printReindented(mlir::tblgen::tgfmt(arg.getPrinter(), &ctx).str());
  os << "\n";
  ctx.withSelf("_inst");
}

void ASMPrinterHandler::emitSyntax(StringRef syntax,
                                   mlir::raw_indented_ostream &os) {
  Lexer lexer(syntax);
  while (lexer.currentChar() != '\0') {
    lexer.consumeWhiteSpace();
    if (lexer.currentChar() == '\0')
      break;

    // `${mnemonic}` interpolation.
    if (lexer.currentChar() == '$' && lexer.getCurrentPos().size() > 1 &&
        lexer.getCurrentPos()[1] == '{') {
      emitMnemonicInterpolation(lexer, os);
      continue;
    }

    // `$identifier` operand reference.
    if (lexer.currentChar() == '$') {
      emitOperandRef(lexer, os);
      continue;
    }

    // `[identifier]` modifier reference.
    if (lexer.currentChar() == '[') {
      emitModifierRef(lexer, os);
      continue;
    }

    // Comma.
    if (lexer.currentChar() == ',') {
      lexer.consumeChar();
      os << "$_printer.printComma();\n";
      continue;
    }

    // Keyword.
    if (lexer.currentChar() == '_' || std::isalpha(lexer.currentChar())) {
      emitKeyword(lexer, os);
      continue;
    }

    emitError("unexpected character in asm_syntax: " + lexer.getCurrentPos());
  }
  os << "return success();\n";
}

//===----------------------------------------------------------------------===//
// Encoding dispatch
//===----------------------------------------------------------------------===//

void ASMPrinterHandler::emitGuardedSyntax(StringRef syntax,
                                          const mlir::tblgen::Pred &pred,
                                          mlir::raw_indented_ostream &os) {
  std::string predStr = mlir::tblgen::tgfmt(pred.getCondition(), &ctx, "_inst");
  bool isTruePred = (predStr == "true");
  if (!isTruePred) {
    os << "if ((" << predStr << ")) {\n";
    os.indent();
  }
  emitSyntax(syntax, os);
  if (!isTruePred) {
    os.unindent();
    os << "}\n";
  }
}

void ASMPrinterHandler::emitUniformEncoding(
    llvm::ArrayRef<ArchSyntaxPred> archEntries,
    mlir::raw_indented_ostream &os) {
  emitGuardedSyntax(archEntries.front().syntax, archEntries.front().pred, os);
}

void ASMPrinterHandler::emitArchGroupedEncoding(
    llvm::ArrayRef<ArchSyntaxPred> archEntries,
    mlir::raw_indented_ostream &os) {
  // Collect arch order preserving encounter order.
  llvm::SmallVector<StringRef> archOrder;
  llvm::StringSet<> seenArchs;
  for (const ArchSyntaxPred &asp : archEntries) {
    if (seenArchs.insert(asp.archId).second)
      archOrder.push_back(asp.archId);
  }

  for (StringRef archId : archOrder) {
    llvm::SmallVector<const ArchSyntaxPred *> archGroup;
    for (const ArchSyntaxPred &asp : archEntries) {
      if (asp.archId == archId)
        archGroup.push_back(&asp);
    }

    os << "if (tgt.getTargetFamily() == "
          "::mlir::aster::amdgcn::ISAVersion::"
       << archId << ") {\n";
    os.indent();
    for (const ArchSyntaxPred *asp : archGroup)
      emitGuardedSyntax(asp->syntax, asp->pred, os);
    os.unindent();
    os << "}\n";
  }
}

void ASMPrinterHandler::emitEncodingDispatch(
    llvm::ArrayRef<ArchEncodingSyntax> entries,
    mlir::raw_indented_ostream &os) {
  // Collect encoding order.
  llvm::SmallVector<StringRef> encOrder;
  llvm::StringSet<> seenEncs;
  for (const ArchEncodingSyntax &e : entries) {
    if (seenEncs.insert(e.encodingId).second)
      encOrder.push_back(e.encodingId);
  }

  for (StringRef encId : encOrder) {
    // Collect all (arch, syntax, pred) tuples for this encoding.
    llvm::SmallVector<ArchSyntaxPred> archEntries;
    for (const ArchEncodingSyntax &e : entries) {
      if (e.encodingId != encId)
        continue;
      archEntries.push_back({e.archId, e.syntax, e.pred});
    }

    os << "if (_enc == ::mlir::aster::amdgcn::AMDGCNEncoding::" << encId
       << ") {\n";
    os.indent();

    // Check if all entries share the same syntax and predicate.
    bool allSame = llvm::all_of(archEntries, [&](const ArchSyntaxPred &asp) {
      return asp.syntax == archEntries.front().syntax &&
             asp.pred.getCondition() == archEntries.front().pred.getCondition();
    });

    if (allSame)
      emitUniformEncoding(archEntries, os);
    else
      emitArchGroupedEncoding(archEntries, os);

    os.unindent();
    os << "}\n";
  }
}

//===----------------------------------------------------------------------===//
// Printer function generation
//===----------------------------------------------------------------------===//

void ASMPrinterHandler::genPrinter(raw_ostream &osOut) {
  StrStream strStream;
  mlir::raw_indented_ostream os(strStream.os);
  std::string qualClass = instOp.getQualCppClassName();

  // Read the asm_syntax list.
  llvm::SmallVector<ASMStringRecord> asmStrings = instOp.getAsmSyntax();
  if (asmStrings.empty())
    return;

  // Build a flat list of (arch, encoding, syntax, predicate) entries.
  llvm::SmallVector<ArchEncodingSyntax> entries;
  for (const ASMStringRecord &asmStr : asmStrings) {
    mlir::tblgen::Pred pred = asmStr.getPred();
    for (const EncodedArchRecord &ea : asmStr.getArchs())
      entries.push_back({ea.getArch().getIdentifier(),
                         ea.getEncoding().getIdentifier(), asmStr.getSyntax(),
                         pred});
  }

  // Generate the printer function.
  ctx.withSelf("_inst");
  os << "static ::mlir::LogicalResult print" << instOp.getCppClassName()
     << "(\n";
  os << "    ::mlir::aster::amdgcn::AsmPrinter &printer,\n";
  os << "    ::mlir::aster::TargetAttrInterface tgt,\n";
  os << "    ::mlir::Operation *op) {\n";
  os.indent();
  os << "auto _inst = ::llvm::cast<" << qualClass << ">(op);\n";
  os << "(void)_inst;\n";
  os << "auto _encOrFailure = _inst.getEncoding(tgt);\n";
  os << "if (::mlir::failed(_encOrFailure))\n";
  os << "  return op->emitError(\"failed to get encoding\");\n";
  os << "auto _enc = *_encOrFailure;\n";

  emitEncodingDispatch(entries, os);

  os << "return ::mlir::failure();\n";
  os.unindent();
  os << "}\n";
  osOut << mlir::tblgen::tgfmt(strStream.str, &ctx);
}

//===----------------------------------------------------------------------===//
// Top-level generator
//===----------------------------------------------------------------------===//

static bool generateInstAsmPrinters(const llvm::RecordKeeper &records,
                                    raw_ostream &os) {
  llvm::SmallVector<const llvm::Record *> instRecs =
      llvm::to_vector(records.getAllDerivedDefinitions("AMDISAInstruction"));
  llvm::sort(instRecs, llvm::LessRecord());

  llvm::interleave(
      instRecs, os,
      [&](const llvm::Record *instRec) {
        ASMPrinterHandler handler(instRec);
        handler.genPrinter(os);
      },
      "\n");

  // Generate the TypeSwitch-based dispatch function.
  os << R"(
static ::mlir::LogicalResult printISAInstruction(
    ::mlir::aster::amdgcn::AsmPrinter &printer,
    ::mlir::aster::TargetAttrInterface tgt,
    ::mlir::Operation *op) {
  return ::llvm::TypeSwitch<::mlir::Operation *, ::mlir::LogicalResult>(op)
)";
  for (const llvm::Record *instRec : instRecs) {
    InstOp inst(instRec);
    os << "    .Case<" << inst.getQualCppClassName() << ">([&](auto) {\n";
    os << "      return print" << inst.getCppClassName()
       << "(printer, tgt, op);\n";
    os << "    })\n";
  }
  os << R"(    .Default([](::mlir::Operation *op) {
      return op->emitError("no ISA printer for instruction");
    });
}
)";
  return false;
}

//===----------------------------------------------------------------------===//
// TableGen Registration
//===----------------------------------------------------------------------===//

static GenRegistration
    generateInstAsmPrintersReg("gen-inst-asm-printers",
                               "Generate encoding-aware inst asm printers",
                               generateInstAsmPrinters);
