//===- Sched.cpp - AMDGCN scheduling labeler implementation ---------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/Sched.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNInst.h"
#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>

#define DEBUG_TYPE "amdgcn-sched-labeler"

using namespace mlir;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// YAML traits
//===----------------------------------------------------------------------===//

namespace {
/// YAML representation of a single pattern entry.
struct YAMLPatternEntry {
  std::string kind;
  int32_t label = 0;
  uint16_t benefit = 0;
  std::string name;
};
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(YAMLPatternEntry)

namespace llvm::yaml {
template <>
struct MappingTraits<YAMLPatternEntry> {
  static void mapping(IO &io, YAMLPatternEntry &entry) {
    io.mapRequired("kind", entry.kind);
    io.mapRequired("label", entry.label);
    io.mapOptional("benefit", entry.benefit, uint16_t{0});
    io.mapOptional("name", entry.name, std::string{});
  }
};
} // namespace llvm::yaml

//===----------------------------------------------------------------------===//
// SchedPattern
//===----------------------------------------------------------------------===//

bool SchedPattern::match(TypeID opId, const InstMetadata *metadata) const {
  if (instProp)
    return metadata && metadata->hasProp(*instProp);
  if (instMetadata)
    return instMetadata == metadata;
  if (opTypeID != TypeID())
    return opId == opTypeID;
  return true;
}

//===----------------------------------------------------------------------===//
// SchedLabeler
//===----------------------------------------------------------------------===//

FailureOr<SchedLabeler> SchedLabeler::getFromYAML(StringRef path,
                                                  MLIRContext *ctx) {
  Location loc = UnknownLoc::get(ctx);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrErr =
      llvm::MemoryBuffer::getFile(path);
  if (std::error_code ec = bufferOrErr.getError()) {
    emitError(loc) << "error opening YAML file '" << path
                   << "': " << ec.message();
    return failure();
  }

  llvm::SmallVector<YAMLPatternEntry> entries;
  llvm::yaml::Input yin((*bufferOrErr)->getBuffer());
  yin >> entries;
  if (yin.error()) {
    emitError(loc) << "error parsing YAML file '" << path << "'";
    return failure();
  }

  SchedLabeler labeler;
  labeler.name = path.str();
  SmallVector<SchedPattern> anyPatterns, opPatterns;
  for (const YAMLPatternEntry &entry : entries) {
    SchedPattern pattern;
    pattern.label = entry.label;
    pattern.benefit = entry.benefit;

    if (entry.kind == "opcode") {
      std::optional<OpCode> opcode = symbolizeOpCode(entry.name);
      if (!opcode) {
        emitError(loc) << "unknown opcode '" << entry.name << "' in '" << path
                       << "'";
        return failure();
      }
      // Look up the instruction metadata and op type ID for this opcode.
      pattern.instMetadata = InstAttr::get(ctx, *opcode).getMetadata();
      if (!pattern.instMetadata) {
        emitError(loc) << "no metadata for opcode '" << entry.name << "' in '"
                       << path << "'";
        return failure();
      }
      pattern.opTypeID = pattern.instMetadata->getOpTypeID();
    } else if (entry.kind == "inst_prop") {
      std::optional<InstProp> prop = symbolizeInstProp(entry.name);
      if (!prop) {
        emitError(loc) << "unknown inst_prop '" << entry.name << "' in '"
                       << path << "'";
        return failure();
      }
      pattern.instProp = *prop;
      anyPatterns.push_back(pattern);
      continue;
    } else if (entry.kind == "opname") {
      std::optional<RegisteredOperationName> regOp =
          RegisteredOperationName::lookup(entry.name, ctx);
      if (!regOp) {
        emitError(loc) << "operation '" << entry.name
                       << "' is not registered in context (from '" << path
                       << "')";
        return failure();
      }
      pattern.opTypeID = regOp->getTypeID();
    } else {
      anyPatterns.push_back(pattern);
      continue;
    }

    opPatterns.push_back(pattern);
  }

  // Sort the patterns by benefit (descending) to ensure deterministic
  // tie-breaking.
  llvm::stable_sort(anyPatterns,
                    [](const SchedPattern &a, const SchedPattern &b) {
                      return a.benefit > b.benefit;
                    });
  llvm::stable_sort(opPatterns,
                    [](const SchedPattern &a, const SchedPattern &b) {
                      return std::make_tuple(a.opTypeID.getAsOpaquePointer(),
                                             a.instMetadata, a.benefit) >
                             std::make_tuple(b.opTypeID.getAsOpaquePointer(),
                                             b.instMetadata, b.benefit);
                    });

  // Add any patterns.
  llvm::append_range(labeler.patterns, anyPatterns);
  labeler.metadataToPatternPos[{TypeID(), nullptr}] = {0, anyPatterns.size()};

  // For each unique (opTypeID, instMetadata) group in opPatterns, produce a
  // merged range with anyPatterns sorted by benefit descending, and record the
  // range in metadataToPatternPos.
  int64_t currentPos = static_cast<int64_t>(labeler.patterns.size());
  int64_t groupStart = 0;
  while (groupStart < static_cast<int64_t>(opPatterns.size())) {
    TypeID groupTypeId = opPatterns[groupStart].opTypeID;
    const InstMetadata *groupMeta = opPatterns[groupStart].instMetadata;

    // Find the end of this group (same opTypeID and instMetadata).
    int64_t groupEnd = groupStart + 1;
    while (groupEnd < static_cast<int64_t>(opPatterns.size()) &&
           opPatterns[groupEnd].opTypeID == groupTypeId &&
           opPatterns[groupEnd].instMetadata == groupMeta)
      ++groupEnd;

    // Merge the group with anyPatterns in descending benefit order. Both
    // sequences are already sorted by benefit descending.
    llvm::SmallVector<SchedPattern> merged;
    merged.reserve((groupEnd - groupStart) + anyPatterns.size());
    std::merge(opPatterns.begin() + groupStart, opPatterns.begin() + groupEnd,
               anyPatterns.begin(), anyPatterns.end(),
               std::back_inserter(merged),
               [](const SchedPattern &a, const SchedPattern &b) {
                 return a.benefit > b.benefit;
               });

    int64_t count = static_cast<int64_t>(merged.size());
    labeler.metadataToPatternPos[{groupTypeId, groupMeta}] = {currentPos,
                                                              count};
    llvm::append_range(labeler.patterns, merged);
    currentPos += count;
    groupStart = groupEnd;
  }

  LDBG_OS([&](llvm::raw_ostream &os) {
    os << "loaded '" << path << "': " << labeler.patterns.size()
       << " pattern(s)\n";
    for (auto [i, p] : llvm::enumerate(labeler.patterns)) {
      os << "  [" << i << "] label=" << p.label << " benefit=" << p.benefit
         << " opcode=";
      if (p.instMetadata)
        os << stringifyOpCode(p.instMetadata->getOpCode());
      else
        os << "none";
      os << " inst_prop=";
      if (p.instProp)
        os << stringifyInstProp(*p.instProp);
      else
        os << "none";
    }
  });

  return labeler;
}

int32_t SchedLabeler::getLabel(Operation *op) const {
  const InstMetadata *metadata = nullptr;
  TypeID opId = op->getName().getTypeID();
  if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op))
    metadata = instOp.getInstMetadata();
  auto [pos, num] = metadataToPatternPos.lookup({opId, metadata});
  if (num == 0) {
    auto [anyPos, anyNum] = metadataToPatternPos.lookup({TypeID(), nullptr});
    pos = anyPos;
    num = anyNum;
  }
  if (num == 0)
    return -1;
  for (const auto &[i, pattern] : llvm::enumerate(llvm::make_range(
           patterns.begin() + pos, patterns.begin() + pos + num))) {
    if (!pattern.match(opId, metadata))
      continue;
    LDBG() << "Applying pattern: " << i << " with label " << pattern.label
           << " to op: " << *op;
    if (pattern.label >= 0)
      return pattern.label;
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// AsmPrinter / FieldParser support
//===----------------------------------------------------------------------===//

namespace mlir::aster::amdgcn {

AsmPrinter &operator<<(AsmPrinter &printer, const SchedLabeler &labeler) {
  printer.printString(labeler.getName());
  return printer;
}

} // namespace mlir::aster::amdgcn

namespace mlir {
FailureOr<aster::amdgcn::SchedLabeler>
FieldParser<aster::amdgcn::SchedLabeler, aster::amdgcn::SchedLabeler>::parse(
    AsmParser &parser) {
  std::string path;
  if (parser.parseString(&path))
    return failure();
  FailureOr<aster::amdgcn::SchedLabeler> result =
      aster::amdgcn::SchedLabeler::getFromYAML(path, parser.getContext());
  if (failed(result))
    return failure();
  return std::move(*result);
}
} // namespace mlir
