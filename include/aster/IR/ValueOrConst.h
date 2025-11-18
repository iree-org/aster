//===- ValueOrConst.h - ValueOrConst ----------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_VALUEORCONST_H
#define ASTER_IR_VALUEORCONST_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include <cstdint>
#include <type_traits>

namespace mlir::aster {
/// Utility to get the attribute value from a Value if constant, or nullptr if
/// not present.
inline Attribute getAttrOrNull(Value value) {
  if (Operation *op = value.getDefiningOp(); op && m_Constant().match(op)) {
    SmallVector<OpFoldResult, 1> foldRes;
    LogicalResult result = op->fold(/*operands=*/{}, foldRes);
    (void)result;
    assert(succeeded(result) && "expected ConstantLike op to be foldable");
    assert(foldRes.size() == 1 &&
           "expected single result from folding constant op");
    return cast<Attribute>(foldRes.front());
  }
  return nullptr;
}

namespace detail {
/// Helper to get a constant from a Value.
template <typename ConstTy, typename Sfinae, typename... ValueTypes>
struct ConstValueTraits {
  using ConstantType = ConstTy;
  /// Get the constant attribute from the value.
  template <typename CTy = ConstantType,
            std::enable_if_t<std::is_base_of<Attribute, CTy>::value, int> = 0>
  static ConstantType get(Value value) {
    if constexpr (sizeof...(ValueTypes) > 0) {
      assert(llvm::isa<ValueTypes...>(value.getType()) &&
             "Value type does not match expected types");
    }
    if constexpr (std::is_same_v<CTy, Attribute>) {
      return getAttrOrNull(value);
    } else {
      return llvm::dyn_cast_if_present<CTy>(getAttrOrNull(value));
    }
  }
};

/// Specialization for Values that are also of a specific type.
template <typename T>
struct ConstValueTraits<T, std::enable_if_t<std::is_integral_v<T>, void>,
                        IntegerType> {
  using ConstantType = std::optional<T>;
  /// Get the constant attribute from the value.
  static ConstantType get(Value value) {
    assert(isa<IntegerType>(value.getType()) &&
           "Value type does not match expected types");
    if (auto attr =
            llvm::dyn_cast_if_present<IntegerAttr>(getAttrOrNull(value))) {
      return attr.getValue().getSExtValue();
    }
    return std::nullopt;
  }
};

template <>
struct ConstValueTraits<double, void, FloatType> {
  using ConstantType = std::optional<double>;
  /// Get the constant attribute from the value.
  static ConstantType get(Value value) {
    assert(isa<FloatType>(value.getType()) &&
           "Value type does not match expected types");
    if (auto attr =
            llvm::dyn_cast_if_present<FloatAttr>(getAttrOrNull(value))) {
      return attr.getValueAsDouble();
    }
    return std::nullopt;
  }
};

/// Template class for values that can be constants.
template <typename ConstTy, typename Sfinae, typename... ValueTypes>
class ValueOrConstImpl : public Value {
public:
  using ConstTraits = ConstValueTraits<ConstTy, Sfinae, ValueTypes...>;
  using ConstantType = typename ConstTraits::ConstantType;
  using Base = ValueOrConstImpl;
  constexpr ValueOrConstImpl(mlir::detail::ValueImpl *impl = nullptr)
      : Value(impl) {
    cVal = ConstTraits::get(*this);
  }

  /// Allow dynamic casting.
  static bool classof(Value value) {
    if constexpr (sizeof...(ValueTypes) > 0)
      return isa<ValueTypes...>(value.getType());
    return true;
  }

  /// Return the attribute value.
  ConstantType getConst() const { return cVal; }

  /// Static helper to get the constant attribute from a Value.
  static ConstantType getConstant(Value value) {
    return ConstTraits::get(value);
  }

private:
  ConstantType cVal;
};
} // namespace detail

/// Integer attribute value wrapper.
class ValueOrIntAttr : public detail::ValueOrConstImpl<IntegerAttr, void,
                                                       IntegerType, IndexType> {
public:
  using Base::Base;
};
class ValueOrI32 : public detail::ValueOrConstImpl<int32_t, void, IntegerType> {
public:
  using Base::Base;
};
class ValueOrI64
    : public detail::ValueOrConstImpl<int64_t, void, IntegerType, IndexType> {
public:
  using Base::Base;
};

/// Float attribute value wrapper.
class ValueOrFloatAttr
    : public detail::ValueOrConstImpl<FloatAttr, void, FloatType> {
public:
  using Base::Base;
};
class ValueOrDouble : public detail::ValueOrConstImpl<double, void, FloatType> {
public:
  using Base::Base;
};
} // namespace mlir::aster

#endif // ASTER_IR_VALUEORCONST_H
