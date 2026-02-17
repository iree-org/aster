//===- AMDGCNOps.h - AMDGCN Operations --------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_AMDGCNOPS_H
#define ASTER_DIALECT_AMDGCN_IR_AMDGCNOPS_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNInterfaces.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNVerifiers.h"
#include "aster/IR/InstImpl.h"
#include "aster/Interfaces/AllocaOpInterface.h"
#include "aster/Interfaces/DependentOpInterface.h"
#include "aster/Interfaces/GPUFuncInterface.h"
#include "aster/Interfaces/ModuleOpInterface.h"
#include "aster/Interfaces/OperandBundleOpInterface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class PatternRewriter;
}

#define GET_OP_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h.inc"

#define AMDGCN_GEN_INST_DECLS
#include "aster/Dialect/AMDGCN/IR/AMDGCNInsts.h.inc"

#endif // ASTER_DIALECT_AMDGCN_IR_AMDGCNOPS_H
