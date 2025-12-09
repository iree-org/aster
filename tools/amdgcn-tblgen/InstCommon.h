//===- InstCommon.h -----------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the common classes and functions for the AMDGCN tblgen
// tool.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_TBLGEN_COMMON_H
#define AMDGCN_TBLGEN_COMMON_H

#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Builder.h"
#include "mlir/TableGen/Constraint.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#include <cassert>
#include <optional>
#include <string_view>
#include <utility>

namespace mlir::aster {
namespace amdgcn {
namespace tblgen {
//===----------------------------------------------------------------------===//
// Tablegen record wrappers
//===----------------------------------------------------------------------===//
using Argument = mlir::tblgen::Argument;
using Builder = mlir::tblgen::Builder;
using Constraint = mlir::tblgen::Constraint;
using Operator = mlir::tblgen::Operator;

/// Class representing a DAG operand with an associated name.
struct DagArg {
  DagArg(const llvm::Init *init, StringRef name) : init(init), name(name) {
    assert(init && "Init must be non-null");
  }

  /// Get the underlying Init.
  const llvm::Init *getInit() const { return init; }
  /// Get the name of the operand.
  StringRef getName() const { return name; }
  /// Get the underlying Init.
  const llvm::Record *getAsRecord() const {
    return cast<llvm::DefInit>(init)->getDef();
  }

private:
  const llvm::Init *init;
  StringRef name;
};

/// Wrapper class for the DagInit TableGen class.
struct Dag {
  Dag(const llvm::DagInit *dagInit) : dagInit(dagInit) {
    assert(dagInit && "DagInit must be non-null");
  }

  /// Get the number of arguments in the DAG.
  unsigned getNumArgs() const { return dagInit->getNumArgs(); }

  /// Get the argument at the given index.
  DagArg getArg(unsigned index) const {
    return DagArg(dagInit->getArg(index), dagInit->getArgNameStr(index));
  }

  /// Get all arguments as a range of DagArg.
  auto getAsRange() const {
    return llvm::map_range(llvm::seq<unsigned>(0, getNumArgs()),
                           [this](unsigned i) { return getArg(i); });
  }

private:
  const llvm::DagInit *dagInit;
};

// Wrapper class for the Record TableGen class.
struct Record {
  Record(const llvm::Record *def, llvm::StringRef classType) : def(def) {
    assert(def && "record must be non-null");
    assert(!classType.empty() && "class type must be non-empty");
    if (!def->isSubClassOf(classType)) {
      llvm::PrintFatalError(def,
                            "expected record of class type: " + classType +
                                ", but instead received: " + def->getName());
    }
  }

  /// Returns the underlying def.
  const llvm::Record &getDef() const { return *def; }

  /// Returns the name of the def.
  llvm::StringRef getName() const { return def->getName(); }

  /// Returns the field value as a `RecordTy` or failure if it is not
  /// convertible.
  template <typename RecordTy>
  RecordTy getDefAs(StringRef field) const {
    return RecordTy(def->getValueAsDef(field));
  }
  template <typename RecordTy, typename... Args>
  std::optional<RecordTy> getOptionalDefAs(StringRef field,
                                           Args &&...args) const {
    const llvm::Record *rec = def->getValueAsOptionalDef(field);
    return rec ? std::optional<RecordTy>(
                     RecordTy(rec, std::forward<Args>(args)...))
               : std::nullopt;
  }

  /// Returns the string corresponding to `field`.
  StringRef getStringRef(StringRef field) const {
    if (def->isValueUnset(field))
      return "";
    return def->getValueAsString(field);
  }

  /// Returns the bit corresponding to `field`.
  bool getBit(StringRef field) const { return def->getValueAsBit(field); }

  /// Returns the Dag corresponding to `field`.
  Dag getDag(StringRef field) const { return Dag(def->getValueAsDag(field)); }

  /// Returns the list of records of type `RecordTy` corresponding to `field`.
  template <typename RecordTy>
  SmallVector<RecordTy> getRecordList(StringRef field) const {
    return llvm::map_to_vector(
        def->getValueAsListInit(field)->getValues(),
        [&](const llvm::Init *init) {
          auto record = dyn_cast<llvm::DefInit>(init);
          if (!record) {
            llvm::PrintFatalError(
                def,
                "unable to convert to the specified record type in field: " +
                    field);
          }
          return RecordTy(record->getDef());
        });
  }

protected:
  const llvm::Record *def;
};

/// Helper class for defining Record subclasses.
template <typename RecordTy>
class RecordMixin : public Record {
public:
  using Base = RecordMixin;
  RecordMixin(const llvm::Record *def) : Record(def, RecordTy::ClassType) {
    assert(isa(def) && "Invalid record type");
  }

  /// Returns true if the record is of subclass `RecordTy`
  static bool isa(const llvm::Record *def) {
    return def->isSubClassOf(RecordTy::ClassType);
  }
};

/// Enum case information definition.
struct EnumCaseInfo : public RecordMixin<EnumCaseInfo> {
  using Base::Base;
  static constexpr llvm::StringRef ClassType = "EnumCaseInfo";

  /// Get the symbol of the enum case.
  StringRef getSymbol() const { return getStringRef("symbol"); }

  /// Get the identifier of the enum case.
  StringRef getIdentifier() const { return getStringRef("identifier"); }
};

/// AMDGCN target definition.
struct Target : public RecordMixin<Target> {
  using Base::Base;
  static constexpr llvm::StringRef ClassType = "Target";
  /// Get the ISA version associated with this target.
  EnumCaseInfo getISAVersion() const {
    return getDefAs<EnumCaseInfo>("isaVersion");
  }
  /// Get as enum case.
  EnumCaseInfo getAsEnumCase() const { return EnumCaseInfo(def); }
};

/// AMDGCN isa version definition.
struct ISAVersion : public RecordMixin<ISAVersion> {
  using Base::Base;
  static constexpr llvm::StringRef ClassType = "ISAVersion";
  /// Get as enum case.
  EnumCaseInfo getAsEnumCase() const { return EnumCaseInfo(def); }
};

/// AMDGCN instruction op type name.
static constexpr std::string_view InstOpClassType = "InstOp";

/// AMDGCN instruction argument definition.
struct InstConstraint : public RecordMixin<InstConstraint> {
  using Base::Base;
  static constexpr llvm::StringRef ClassType = "InstConstraint";

  /// Get the constraints list.
  llvm::SmallVector<mlir::tblgen::Constraint> getConstraints() const {
    return getRecordList<mlir::tblgen::Constraint>("constraints");
  }
};

/// AMDGCN instruction assembly argument format.
struct AsmArgFormat : public RecordMixin<AsmArgFormat> {
  using Base::Base;
  static constexpr llvm::StringRef ClassType = "AsmArgFormat";

  /// Get the parser of the argument.
  StringRef getParser() const { return getStringRef("asmParser"); }

  /// Get the printer of the argument.
  StringRef getPrinter() const { return getStringRef("asmPrinter"); }
};

/// AMDGCN instruction assembly variant.
struct AsmVariant : public RecordMixin<AsmVariant> {
  using Base::Base;
  static constexpr llvm::StringRef ClassType = "AsmVariant";

  /// Get the predicate.
  Constraint getPredicate() const { return getDefAs<Constraint>("predicate"); }

  /// Get the assembly format string.
  StringRef getAsmFormat() const { return getStringRef("asmFormat"); }
};

/// AMDGCN instruction definition.
struct AMDInst : public RecordMixin<AMDInst> {
  AMDInst(llvm::Record const *def)
      : Base(def), instOp(def->getValueAsDef("instOp")) {}
  static constexpr llvm::StringRef ClassType = "AMDInst";
  /// Get the name of the instruction.
  llvm::StringRef getName() const { return getAsEnumCase().getIdentifier(); }

  /// Get the mnemonic of the instruction.
  StringRef getMnemonic() const { return getStringRef("mnemonic"); }

  /// Get the summary of the instruction.
  StringRef getSummary() const { return getStringRef("summary"); }

  /// Get the description of the instruction.
  StringRef getDescription() const { return getStringRef("description"); }

  /// Get the C++ namespace.
  StringRef getCppNamespace() const { return getStringRef("cppNamespace"); }

  /// Get the instruction op.
  const Operator &getInstOp() const { return instOp; }

  /// Get the input operands of the instruction.
  Dag getConstraints() const { return getDag("constraints"); }

  /// Get the list of ISA versions this instruction is available on.
  SmallVector<ISAVersion> getISAVersions() const {
    return getRecordList<ISAVersion>("isa");
  }

  /// Get the assembly format variants.
  SmallVector<AsmVariant> getAsmFormat() const {
    return getRecordList<AsmVariant>("asmFormat");
  }

  /// Get the extra class declaration code.
  StringRef getExtraClassDeclaration() const {
    return getStringRef("extraClassDeclaration");
  }

  /// Get the extra class definition code.
  StringRef getExtraClassDefinition() const {
    return getStringRef("extraClassDefinition");
  }

  /// Get the C++ builder code.
  std::optional<Builder> getCppBuilder() const {
    const llvm::Record *record = def->getValueAsOptionalDef("cppBuilder");
    if (!record)
      return std::nullopt;
    return Builder(record, {});
  }

  /// Get the Python builder code.
  std::optional<Builder> getPythonBuilder() const {
    const llvm::Record *record = def->getValueAsOptionalDef("pythonBuilder");
    if (!record)
      return std::nullopt;
    return Builder(record, {});
  }

  /// Check if the instruction has an initialize method.
  bool hasInit() const { return getBit("hasInit"); }

  /// Get as enum case.
  EnumCaseInfo getAsEnumCase() const { return EnumCaseInfo(def); }

private:
  Operator instOp;
};

//===----------------------------------------------------------------------===//
// Utility functions and classes
//===----------------------------------------------------------------------===//

/// Helper class for building strings with a raw_ostream.
struct StrStream {
  StrStream() : str(""), os(str) {}
  std::string str;
  llvm::raw_string_ostream os;
};

/// Get the qualified C++ name.
std::string getQualName(StringRef cppNamespace, StringRef className);
/// Get the full C++ name of the instruction operation.
std::string getInstOpName(const mlir::tblgen::Operator &instOp,
                          bool addNamespace = true);

/// Get the C++ opcode name of the given instruction.
std::string getOpCode(const AMDInst &inst, bool isPython = false);

/// Generate parameter list for the given builder.
std::string genParamList(const Builder &b, mlir::tblgen::FmtContext &ctx,
                         bool isCpp, bool isDecl, bool prefixComma = true,
                         bool postfixComma = false);

/// Generate the argument list for the given builder.
std::string genArgList(const Builder &b, mlir::tblgen::FmtContext &ctx,
                       bool isCpp, bool useKwArgs = true);

/// Get the list of isa version names for the given instruction.
std::string getISAVersionList(const AMDInst &inst);

/// Populate the format context with common substitutions.
void populateFmtContext(const AMDInst &inst, mlir::tblgen::FmtContext &ctx);
} // namespace tblgen
} // namespace amdgcn
} // namespace mlir::aster

#endif // AMDGCN_TBLGEN_COMMON_H
