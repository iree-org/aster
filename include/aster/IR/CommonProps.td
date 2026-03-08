//===- CommonProps.td - Common property definitions ---------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common property definitions shared across Aster dialects.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_COMMONPROPS_TD
#define ASTER_IR_COMMONPROPS_TD


//===----------------------------------------------------------------------===//
// Ptr add flags property (uses ptr::PtrAddFlags).
//===----------------------------------------------------------------------===//

def PtrAddFlagsProp : Property<"::mlir::ptr::PtrAddFlags", "Pointer add flags"> {
  let predicate = CPred<"static_cast<uint32_t>($_self) <= mlir::ptr::getMaxEnumValForPtrAddFlags() && static_cast<uint32_t>($_self) >= 0">;
  let convertFromAttribute = [{
    auto intAttr = ::mlir::dyn_cast_if_present<::mlir::IntegerAttr>($_attr);
    if (!intAttr) {
      return $_diag() << "expected IntegerAttr storage for ptr::PtrAddFlags";
    }
    $_storage = static_cast<::mlir::ptr::PtrAddFlags>(intAttr.getValue().getZExtValue());
    return ::mlir::success();
  }];
  let convertToAttribute = [{
    return ::mlir::IntegerAttr::get(::mlir::IntegerType::get($_ctxt, 32), static_cast<uint32_t>($_storage));
  }];
  let writeToMlirBytecode = [{
    $_writer.writeVarInt(static_cast<uint64_t>($_storage));
  }];
  let readFromMlirBytecode = [{
    uint64_t rawValue;
    if (::mlir::failed($_reader.readVarInt(rawValue)))
      return ::mlir::failure();
    if (rawValue > std::numeric_limits<uint32_t>::max())
      return ::mlir::failure();
    $_storage = static_cast<::mlir::ptr::PtrAddFlags>(rawValue);
  }];
  let optionalParser = [{
    auto value = ::mlir::FieldParser<std::optional<::mlir::ptr::PtrAddFlags>>::parse($_parser);
    if (::mlir::failed(value))
      return ::mlir::failure();
    if (!(value->has_value()))
      return std::nullopt;
    $_storage = std::move(**value);
  }];
}

#endif // ASTER_IR_COMMONPROPS_TD
