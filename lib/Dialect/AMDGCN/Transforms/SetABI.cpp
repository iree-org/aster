//===- SetABI.cpp --------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Transforms/Passes.h"

#include "aster/Analysis/ABIAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Interfaces/GPUFuncInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Transforms/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_SETABI
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// SetABI pass
//===----------------------------------------------------------------------===//
struct SetABI : public amdgcn::impl::SetABIBase<SetABI> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// SetABI pass
//===----------------------------------------------------------------------===//

static Type getRegTy(Type type, const ABIAnalysis &abiAnalysis) {
  if (auto ty = dyn_cast<RegisterTypeInterface>(type))
    return ty;
  llvm::TypeSize sz = abiAnalysis.getTypeSize(type);
  int32_t words = (sz.getFixedValue() + 3) / 4;
  return amdgcn::GGPRType::get(type.getContext(),
                               RegisterRange(Register(), words), false);
}

static Type getRegTy(Value value, const ABIAnalysis &abiAnalysis,
                     bool useSGPR = false) {
  if (auto ty = dyn_cast<RegisterTypeInterface>(value.getType()))
    return ty;
  llvm::TypeSize sz = abiAnalysis.getTypeSize(value.getType());
  int32_t words = (sz.getFixedValue() + 3) / 4;
  if (useSGPR) {
    if (words == 1)
      return amdgcn::SGPRType::get(value.getContext(), Register());
    return amdgcn::SGPRRangeType::get(value.getContext(),
                                      RegisterRange(Register(), words));
  }
  return amdgcn::GGPRType::get(value.getContext(),
                               RegisterRange(Register(), words),
                               abiAnalysis.isThreadUniform(value));
}

static void handleFunction(FunctionOpInterface funcOp,
                           const ABIAnalysis &abiAnalysis) {
  if (isa<amdgcn::KernelOp>(funcOp.getOperation()))
    return;
  auto visibility = SymbolTable::Visibility::Private;
  auto fnTy = cast<FunctionType>(funcOp.getFunctionType());
  SmallVector<Type> ins, outs;
  ins.reserve(fnTy.getNumInputs());
  outs.reserve(fnTy.getNumResults());

  bool useSGPR = false;
  // If the function is a GPU kernel, set public visibility.
  if (auto gFn = dyn_cast<aster::GPUFuncInterface>(funcOp.getOperation());
      gFn && gFn.isGPUKernel()) {
    visibility = SymbolTable::Visibility::Public;
    useSGPR = true;
    SmallVector<int32_t> sizeInBytes;
    SmallVector<int32_t> alignInBytes;
    for (Type type : fnTy.getInputs()) {
      llvm::TypeSize sz = abiAnalysis.getTypeSize(type);
      sizeInBytes.push_back(static_cast<int32_t>(sz.getFixedValue()));
      alignInBytes.push_back(
          static_cast<int32_t>(abiAnalysis.getAlignment(type)));
    }
    gFn.setHostABI(fnTy, sizeInBytes, alignInBytes);
  }

  // Set the visibility.
  funcOp.setVisibility(visibility);

  // Determine the input types.
  for (Value v : funcOp.getArguments())
    ins.push_back(getRegTy(v, abiAnalysis, useSGPR));

  if (ins.empty()) {
    for (Type type : fnTy.getInputs())
      ins.push_back(getRegTy(type, abiAnalysis));
  }

  // If there are no results, set the ABI attribute directly.
  if (fnTy.getNumResults() == 0) {
    setABI(funcOp, getABI(funcOp, fnTy.clone(ins, outs)));
    return;
  }

  // Otherwise, analyze the return operations to determine the result types.
  SmallVector<func::ReturnOp> returns;
  for (Block &block : funcOp.getFunctionBody()) {
    auto retOp = dyn_cast<func::ReturnOp>(block.getTerminator());
    if (!retOp)
      continue;
    returns.push_back(retOp);

    // If outs is empty, initialize it with the return operand types.
    if (outs.empty()) {
      for (Value v : retOp.getOperands())
        outs.push_back(getRegTy(v, abiAnalysis));
      continue;
    }

    // Otherwise, ensure the types are consistent across all return ops.
    for (auto [v, type] : llvm::zip_equal(retOp.getOperands(), outs)) {
      Type nTy = getRegTy(v, abiAnalysis);
      if (type == nTy)
        continue;

      // This type promotion is guranteed to be safe since the type must be a
      // ggpr. Prefer non-uniform if there is a conflict.
      auto lhs = cast<amdgcn::GGPRType>(type);
      if (!lhs.getIsUniform())
        nTy = lhs;
      outs.push_back(nTy);
    }
  }

  // If outs is still empty, set the result types based on the function type.
  if (outs.empty()) {
    for (Type type : fnTy.getResults())
      outs.push_back(getRegTy(type, abiAnalysis));
  }

  // Set the ABI attribute on the function and return ops.
  FunctionType abyType = getABI(funcOp, fnTy.clone(ins, outs));
  for (func::ReturnOp retOp : returns)
    setABI(retOp, abyType);
  setABI(funcOp, abyType);
}

void SetABI::runOnOperation() {
  Operation *op = getOperation();
  auto &abiAnalysis = getAnalysis<aster::ABIAnalysis>();
  SymbolTableCollection symTable;
  SmallVector<CallOpInterface> callOps;
  op->walk([&](Operation *op) {
    if (auto funcOp = dyn_cast<FunctionOpInterface>(op))
      return handleFunction(funcOp, abiAnalysis);
    if (auto callOp = dyn_cast<CallOpInterface>(op))
      callOps.push_back(callOp);
  });
  for (CallOpInterface callOp : callOps) {
    auto callee = cast<FunctionOpInterface>(
        call_interface_impl::resolveCallable(callOp, &symTable));
    setABI(callOp, getABI(callee, nullptr));
  }
  markAllAnalysesPreserved();
}
