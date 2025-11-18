//===- OpGen.cpp --------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file generates the AMDGCN MLIR ops.
//
//===----------------------------------------------------------------------===//

#include "Instruction.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/Regex.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::aster::amdgcn;

static llvm::cl::OptionCategory opGenCat("Op Generator Options");

namespace {
struct StrStream {
  StrStream() : str(""), os(str) {}
  std::string str;
  llvm::raw_string_ostream os;
};
} // namespace

/// Generate a single argument for the given instruction.
static void genArgument(StringRef argName, const llvm::Init *arg,
                        raw_ostream &os) {
  os << "AnyType:$" << argName;
}

/// Generate the argument list for the given instruction.
static void genArguments(const Instruction &inst, raw_ostream &os) {
  const llvm::Record *instV = inst.getInstVariants().front();
  const llvm::DagInit *argOutsInit = instV->getValueAsDag("OutOperandList");
  ArrayRef<const llvm::Init *> argOuts = argOutsInit->getArgs();
  const llvm::DagInit *argInsInit = instV->getValueAsDag("InOperandList");
  ArrayRef<const llvm::Init *> argIns = argInsInit->getArgs();
  if (!argOutsInit->getArgs().empty()) {
    os << "/*OutOperands=*/";
    llvm::interleaveComma(llvm::enumerate(argOuts), os, [&](const auto iv) {
      auto [i, init] = iv;
      genArgument(argOutsInit->getArgNameStr(i), init, os);
    });
  }
  if (!argInsInit->getArgs().empty()) {
    if (!argOutsInit->getArgs().empty())
      os << ", ";
    os << "/*InOperands=*/";
    llvm::interleaveComma(llvm::enumerate(argIns), os, [&](const auto iv) {
      auto [i, init] = iv;
      genArgument(argInsInit->getArgNameStr(i), init, os);
    });
  }
}

template <typename T, typename U>
static void interleaveStringList(llvm::ArrayRef<T> container, raw_ostream &os,
                                 U &&getString) {
  SmallVector<std::string> strings;
  for (const T &elem : container)
    strings.push_back(getString(elem));
  llvm::sort(strings);
  strings.erase(std::unique(strings.begin(), strings.end()), strings.end());
  llvm::interleave(strings, os, ",\n");
}

/// Generate extra class members for the given instruction.
static void genExtraClassMembers(const Instruction &inst, raw_ostream &os) {
  os << "    static constexpr std::string_view asmStrings[] = {\n";
  interleaveStringList(inst.getInstVariants(), os,
                       [](const llvm::Record *record) -> std::string {
                         return ("      \"" +
                                 record->getValueAsString("AsmString") + "\"")
                             .str();
                       });
  os << "\n    };\n";
  os << "    static constexpr std::string_view decoders[] = {\n";
  interleaveStringList(inst.getInstVariants(), os,
                       [](const llvm::Record *record) -> std::string {
                         return ("      \"" +
                                 record->getValueAsString("DecoderNamespace") +
                                 "\"")
                             .str();
                       });
  os << "\n    };\n";
  os << "    static constexpr std::string_view tblgenNames[] = {\n";
  interleaveStringList(inst.getInstVariants(), os,
                       [](const llvm::Record *record) -> std::string {
                         return ("      \"" + record->getName() + "\"").str();
                       });
  os << "\n    };\n";
}

/// Generate the MLIR op definition for the given instruction.
static void genOp(const Instruction &inst, raw_ostream &os) {
  static constexpr std::string_view opFmt = R"(
def {0}Op : AMDGCN_Op<"{1}", [{2}]> {{
  let summary = [{{AMDGCN for `{1}`}];
  let arguments = (ins {3});
  let assemblyFormat = [{{
    $operands attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{{
{4}  }];
}
)";
  StrStream arguments, traits, extraClassMembers;
  genArguments(inst, arguments.os);
  genExtraClassMembers(inst, extraClassMembers.os);
  os << formatv(opFmt.data(), inst.getOpName(), inst.getOpMnemonic(),
                traits.str, arguments.str, extraClassMembers.str);
}

/// Generate MLIR ops for all instructions in the given record keeper.
static bool generateOps(const llvm::RecordKeeper &records, raw_ostream &os) {
  const llvm::Record *isGFX940Plus = records.getDef("isGFX940Plus");
  assert(isGFX940Plus && "couldn't find the isGFX940Plus predicate");
  FailureOr<InstMap> instMap =
      InstMap::get(records, [&](const llvm::Record *record) -> bool {
        std::vector<const llvm::Record *> predicates =
            record->getValueAsListOfDefs("Predicates");
        return llvm::count(predicates, isGFX940Plus) > 0;
      });
  if (failed(instMap))
    return true;
  for (const auto &[k, inst] : instMap->getInstructions())
    genOp(inst, os);
  return false;
}

// Generator that generates MLIR ops.
static GenRegistration
    generateOpsReg("amdgcn-op-gen", "Generate op definitions",
                   [](const llvm::RecordKeeper &records, raw_ostream &os) {
                     return generateOps(records, os);
                   });
