//===- Instruction.cpp --------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Instruction and InstMap classes.
//
//===----------------------------------------------------------------------===//

#include "Instruction.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

LogicalResult Instruction::addInstVariant(StringRef asmStr,
                                          const llvm::Record &record,
                                          llvm::DenseSet<StringRef> &types) {
  if (asmString.empty()) {
    asmString = asmStr;
    opMnemonic = StringRef(asmString).take_while(
        [](char c) { return std::isalnum(c) || c == '_'; });
    opName = llvm::convertToCamelFromSnakeCase(opMnemonic, true);
  }
  // This is for debugging.
  for (const llvm::Init *argInit :
       record.getValueAsDag("InOperandList")->getArgs()) {
    auto record = llvm::cast<llvm::DefInit>(argInit);
    types.insert(record->getDef()->getName());
  }
  for (const llvm::Init *argInit :
       record.getValueAsDag("OutOperandList")->getArgs()) {
    auto record = llvm::cast<llvm::DefInit>(argInit);
    types.insert(record->getDef()->getName());
  }
  instVariants.push_back(&record);
  return success();
}

std::string Instruction::getAsmIdentifier(StringRef asmString) {
  return StringRef(asmString)
      .take_while([](char c) { return std::isalnum(c) || c == '_'; })
      .str();
}

FailureOr<InstMap>
InstMap::get(const llvm::RecordKeeper &records,
             function_ref<bool(const llvm::Record *)> filter) {
  const auto &defs = records.getDefs();
  const llvm::Record *instRecord = records.getClass("Instruction");
  assert(instRecord && "failed to found the `Instruction` class");
  InstMap instMap;
  for (const auto &[k, v] : defs) {
    auto record = dyn_cast<llvm::Record>(v.get());
    // Skip over records that are not instructions.
    if (!record || !record->isSubClassOf(instRecord) ||
        !record->getValue("AsmString") || !record->getValue("Inst") ||
        !record->getValue("DecoderNamespace"))
      continue;
    if (filter && !filter(record))
      continue;
    StringRef asmStr = record->getValueAsString("AsmString");
    Instruction &inst =
        instMap.instructions[Instruction::getAsmIdentifier(asmStr)];
    if (failed(inst.addInstVariant(asmStr, *record, instMap.usedTypes)))
      return failure();
  }
  return instMap;
}
