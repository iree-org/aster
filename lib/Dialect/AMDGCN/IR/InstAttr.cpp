
//===- InstAttr.cpp - AMDGCN Instruction Attribute -----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNInst.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Helper to get either the type of a Value or return the type itself.
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T>, int> = 0>
static auto getTypeOrValue(T value) {
  using Type = decltype(value.getType());
  if (value == nullptr)
    return Type();
  return value.getType();
}
/// Helper to passthrough values that are not MLIR Values.
template <typename T, std::enable_if_t<!std::is_base_of_v<Value, T>, int> = 0>
static T &&getTypeOrValue(T &&value) {
  return std::forward<T>(value);
}

//===----------------------------------------------------------------------===//
// AMDGCNDialect
//===----------------------------------------------------------------------===//

/// These are to avoid compiler warning because MLIR unconditionally generates
/// these functions.
[[maybe_unused]] static OptionalParseResult
generatedAttributeParser(AsmParser &parser, StringRef *mnemonic, Type type,
                         Attribute &value);
[[maybe_unused]] static LogicalResult
generatedAttributePrinter(Attribute def, AsmPrinter &printer);

void AMDGCNDialect::initializeInstAttr() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aster/Dialect/AMDGCN/IR/InstAttr.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// InstAttrStorage
//===----------------------------------------------------------------------===//

/// Helper to allocate instruction metadata.
template <typename ConcreteTy>
static mlir::aster::amdgcn::InstMetadata *
allocateInstMetadata(AttributeStorageAllocator &allocator) {
  return ::new (allocator.allocate<ConcreteTy>()) ConcreteTy();
}

/// Instruction Attribute Storage
struct amdgcn::detail::InstAttrStorage : public mlir::AttributeStorage {
  using KeyTy = OpCode;
  InstAttrStorage(InstMetadata *metadata) : metadata(metadata) {}

  /// Equality operator.
  bool operator==(const KeyTy &key) const {
    return key == metadata->getOpCode();
  }

  /// Hashing function.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Get the key from an OpCode.
  static KeyTy getKey(OpCode opCode) { return KeyTy(opCode); }

  /// Get the key from this instance.
  KeyTy getAsKey() const { return KeyTy(metadata->getOpCode()); }

  /// Construct the storage instance.
  static InstAttrStorage *construct(AttributeStorageAllocator &allocator,
                                    const KeyTy &key);

  /// Initialize the metadata.
  void initialize(MLIRContext *context) { metadata->initialize(context); }

  /// The instruction metadata.
  InstMetadata *metadata;
};

OpCode InstAttr::getValue() const { return getImpl()->metadata->getOpCode(); }

/// Get the instruction metadata.
const mlir::aster::amdgcn::InstMetadata *InstAttr::getMetadata() const {
  return getImpl()->metadata;
}

//===----------------------------------------------------------------------===//
// InstMetadata
//===----------------------------------------------------------------------===//

/// Verify that a register type matches the expected size and alignment.
static bool isValidRegisterType(Type tR, int32_t size, int32_t alignment) {
  auto type = dyn_cast<RegisterTypeInterface>(getTypeOrValue(tR));
  if (type == nullptr)
    return size == 0;
  RegisterRange range = type.getAsRange();
  return range.size() == size &&
         (alignment <= 0 || range.alignment() == alignment);
}

/// Verify that a type is either a register type of given size, or an immediate.
static bool isValidRegImmType(Type tR, int32_t size) {
  auto type = dyn_cast<RegisterTypeInterface>(getTypeOrValue(tR));
  if (type == nullptr)
    return tR.isSignlessInteger(32) || tR.isF32();
  RegisterRange range = type.getAsRange();
  return range.size() == size;
}

/// Helper to get the size of a register range type, or -1 if not a register
/// type.
static int64_t getRangeSize(Type type) {
  if (auto rTy = dyn_cast<RegisterTypeInterface>(type))
    return rTy.getAsRange().size();
  return -1;
}

// Instruction definitions, generated from TableGen.
#define AMDGCN_GEN_INST_DEFS
#include "aster/Dialect/AMDGCN/IR/AMDGCNInsts.cpp.inc"

// This needs to be defined here because `getMetadataForOpCode` is defined in
// AMDGCNInsts.cpp.inc.
amdgcn::detail::InstAttrStorage *
amdgcn::detail::InstAttrStorage::construct(AttributeStorageAllocator &allocator,
                                           const KeyTy &key) {
  return new (allocator.allocate<InstAttrStorage>())
      InstAttrStorage(getMetadataForOpCode(allocator, key));
}

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/AMDGCN/IR/InstAttr.cpp.inc"
