//===- PreloadLibrary.cpp - Preload AMDGCN library functions --------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Transforms/Passes.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_PRELOADLIBRARY
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// PreloadLibrary pass
//===----------------------------------------------------------------------===//

struct PreloadLibrary
    : public amdgcn::impl::PreloadLibraryBase<PreloadLibrary> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  /// Parse library files and return parsed modules (keeps them alive).
  LogicalResult
  parseLibraries(MLIRContext *ctx,
                 SmallVectorImpl<OwningOpRef<Operation *>> &parsedModules,
                 llvm::StringMap<func::FuncOp> &libraryFunctions);

  /// Process a single amdgcn.module, importing needed library functions.
  /// Returns true if any changes were made.
  bool processModule(amdgcn::ModuleOp module,
                     const llvm::StringMap<func::FuncOp> &libraryFunctions);
};
} // namespace

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

LogicalResult PreloadLibrary::parseLibraries(
    MLIRContext *ctx, SmallVectorImpl<OwningOpRef<Operation *>> &parsedModules,
    llvm::StringMap<func::FuncOp> &libraryFunctions) {

  for (const std::string &path : libraryPaths) {
    // Read the file.
    std::string errorMessage;
    auto file = mlir::openInputFile(path, &errorMessage);
    if (!file) {
      emitError(UnknownLoc::get(ctx))
          << "failed to open library file '" << path << "': " << errorMessage;
      return failure();
    }

    // Parse the file.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    OwningOpRef<Operation *> parsed = parseSourceFile(sourceMgr, ctx);
    if (!parsed) {
      emitError(UnknownLoc::get(ctx))
          << "failed to parse library file '" << path << "'";
      return failure();
    }

    // Walk the parsed module and collect functions from amdgcn.library ops.
    parsed.get()->walk([&](amdgcn::LibraryOp library) {
      for (Operation &op : library.getBodyRegion().front()) {
        if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
          StringRef name = funcOp.getSymName();
          if (!libraryFunctions.contains(name)) {
            libraryFunctions[name] = funcOp;
          }
        }
      }
    });

    // Keep the parsed module alive so the FuncOp references remain valid.
    parsedModules.push_back(std::move(parsed));
  }

  return success();
}

bool PreloadLibrary::processModule(
    amdgcn::ModuleOp module,
    const llvm::StringMap<func::FuncOp> &libraryFunctions) {

  // Build symbol table for checking existing functions.
  SymbolTable symbolTable(module);

  // Collect all unresolved function calls that can be satisfied by libraries.
  llvm::StringSet<> neededFunctions;
  module.walk([&](func::CallOp callOp) {
    StringRef callee = callOp.getCallee();
    // If the function is not in the module but is in libraries, we need it.
    if (!symbolTable.lookup(callee) && libraryFunctions.contains(callee)) {
      neededFunctions.insert(callee);
    }
  });

  // Also check for declarations to replace.
  SmallVector<func::FuncOp> declarationsToReplace;
  for (auto funcOp : module.getOps<func::FuncOp>()) {
    if (funcOp.isDeclaration()) {
      StringRef name = funcOp.getSymName();
      if (libraryFunctions.contains(name)) {
        declarationsToReplace.push_back(funcOp);
      }
    }
  }

  // No changes needed?
  if (neededFunctions.empty() && declarationsToReplace.empty())
    return false;

  // Import needed library functions.
  OpBuilder builder(module.getContext());
  Block *body = module.getBody();
  for (const auto &name : neededFunctions) {
    func::FuncOp libFunc = libraryFunctions.lookup(name.getKey());
    builder.setInsertionPointToStart(body);
    IRMapping mapping;
    builder.clone(*libFunc.getOperation(), mapping);
  }

  // Replace each declaration with the library implementation.
  for (func::FuncOp decl : declarationsToReplace) {
    StringRef name = decl.getSymName();
    func::FuncOp libFunc = libraryFunctions.lookup(name);

    // Clone the library function.
    builder.setInsertionPoint(decl);
    IRMapping mapping;
    builder.clone(*libFunc.getOperation(), mapping);

    // Erase the original declaration.
    decl.erase();
  }

  return true;
}

void PreloadLibrary::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  // Parse all library files and collect functions.
  // Keep parsed modules alive so function references remain valid.
  SmallVector<OwningOpRef<Operation *>> parsedModules;
  llvm::StringMap<func::FuncOp> libraryFunctions;
  if (failed(parseLibraries(ctx, parsedModules, libraryFunctions)))
    return signalPassFailure();

  // Process each amdgcn.module in the input.
  for (amdgcn::ModuleOp amdgcnModule : moduleOp.getOps<amdgcn::ModuleOp>()) {
    // Iterate until fixed point (libraries may depend on other libraries).
    while (processModule(amdgcnModule, libraryFunctions)) {
    }

    // Check for remaining private function declarations without definitions.
    SmallVector<func::FuncOp> unresolvedDecls;
    for (auto funcOp : amdgcnModule.getOps<func::FuncOp>()) {
      if (funcOp.isDeclaration() && funcOp.isPrivate()) {
        unresolvedDecls.push_back(funcOp);
      }
    }

    if (!unresolvedDecls.empty()) {
      auto diag = amdgcnModule.emitError()
                  << "unresolved private function declaration(s): ";
      llvm::interleaveComma(unresolvedDecls, diag, [&](func::FuncOp f) {
        diag << "'" << f.getSymName() << "'";
      });
      return signalPassFailure();
    }
  }
}
