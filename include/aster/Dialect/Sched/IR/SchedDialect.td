//===- SchedDialect.td - Sched dialect ---------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_SCHED_IR_SCHEDDIALECT_TD
#define ASTER_DIALECT_SCHED_IR_SCHEDDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Sched dialect definition
//===----------------------------------------------------------------------===//

def Sched_Dialect : Dialect {
  let name = "sched";
  let summary = "Aster scheduling dialect";
  let description = [{
    The Sched dialect provides operations for expressing loop scheduling
    structure, including units of work with stage annotations and loop
    resources with explicit memory lifecycle.
  }];
  let cppNamespace = "::mlir::aster::sched";
}

//===----------------------------------------------------------------------===//
// Base operation definition
//===----------------------------------------------------------------------===//

class Sched_Op<string mnemonic, list<Trait> traits = []>
    : Op<Sched_Dialect, mnemonic, traits>;

#endif // ASTER_DIALECT_SCHED_IR_SCHEDDIALECT_TD
