//===- Dump.cpp ---------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dump generator for the AMDGCN tblgen tool.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace llvm;

static cl::OptionCategory dumpGenCat("Dump Generator Options");

static cl::opt<std::string> filter("amdgcn-regex",
                                   cl::desc("Only match the instructions"),
                                   cl::cat(dumpGenCat));

static cl::opt<bool> dumpInsts("amdgcn-dump-insts",
                               cl::desc("Only match AMD instructions"),
                               cl::cat(dumpGenCat), cl::init(false));

static bool dumpRecords(const RecordKeeper &records, raw_ostream &os) {
  if (filter.empty()) {
    os << records;
    return false;
  }
  Regex regex(filter);
  std::string err;
  if (!regex.isValid(err))
    llvm::PrintFatalError(err);
  const llvm::Record *isGFX940Plus = records.getDef("isGFX940Plus");
  assert(isGFX940Plus && "couldn't find the isGFX940Plus predicate");
  const auto &defs = records.getDefs();
  const Record *instRecord = records.getClass("Instruction");
  assert(instRecord);
  auto isGFX940 = [&](const llvm::Record *record) -> bool {
    std::vector<const llvm::Record *> predicates =
        record->getValueAsListOfDefs("Predicates");
    return llvm::count(predicates, isGFX940Plus) > 0;
  };
  for (const auto &[k, v] : defs) {
    if (regex.match(k)) {
      if (dumpInsts) {
        auto rec = dyn_cast<llvm::Record>(v.get());
        if (!rec || !rec->isSubClassOf(instRecord))
          continue;
        if (!rec->getValue("AsmString") || !rec->getValue("Inst") ||
            !rec->getValue("DecoderNamespace"))
          continue;
      }
      if (!isGFX940(v.get()))
        continue;
      os << *v;
    }
  }
  return false;
}

// Generator that prints records.
static GenRegistration
    printRecords("print-records", "Print all records to stdout",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   return dumpRecords(records, os);
                 });
