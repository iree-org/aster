//===- InstImpl.cpp - Instruction Implementation ----------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/IR/InstImpl.h"

using namespace mlir;
using namespace mlir::aster;

void aster::detail::getWriteEffectsImpl(
    OpOperand &addr,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &addr);
}

void aster::detail::getReadEffectsImpl(
    OpOperand &addr,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &addr);
}
