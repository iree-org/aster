//===- DSOps.cpp - AMDGCN DS instructions ---------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstOpsCommon.h"
#include "aster/Dialect/AMDGCN/IR/OpAsmUtils.h"

#define GET_OP_CLASSES
#include "aster/Dialect/AMDGCN/IR/DS.cpp.inc"

#define AMDGCN_GEN_INST_METHODS
#include "aster/Dialect/AMDGCN/IR/DSInst.cpp.inc"
