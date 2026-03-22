//===- SchedOps.h - Sched dialect ops ----------------------------*- C++
//-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operations for the Sched dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_SCHED_IR_SCHEDOPS_H
#define ASTER_DIALECT_SCHED_IR_SCHEDOPS_H

#include "aster/Dialect/Sched/IR/SchedDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "aster/Dialect/Sched/IR/SchedOps.h.inc"

#endif // ASTER_DIALECT_SCHED_IR_SCHEDOPS_H
