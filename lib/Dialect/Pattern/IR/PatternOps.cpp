//===- PatternOps.cpp - Pattern dialect ops implementation ----------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/Pattern/IR/PatternOps.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::aster::pattern;

#define GET_OP_CLASSES
#include "aster/Dialect/Pattern/IR/PatternOps.cpp.inc"

//===----------------------------------------------------------------------===//
// RewritePatternOp
//===----------------------------------------------------------------------===//

LogicalResult RewritePatternOp::verify() {
  // Verify the fields region if present.
  Region &fieldsRegion = getFieldsRegion();
  if (!fieldsRegion.empty()) {
    Block &fieldsBlock = fieldsRegion.front();
    for (Operation &op : fieldsBlock.without_terminator())
      if (!isa<FieldOp>(op))
        return emitOpError("fields region must only contain pattern.field ops");

    // The terminator must be a YieldOp whose operands match the fields.
    auto yieldOp = cast<YieldOp>(fieldsBlock.getTerminator());
    unsigned numFields = llvm::range_size(fieldsBlock.without_terminator());

    if (yieldOp.getValues().size() != numFields)
      return emitOpError("yield in fields region must have the same number of "
                         "operands as field declarations");

    unsigned idx = 0;
    for (Operation &op : fieldsBlock.without_terminator()) {
      auto fieldOp = cast<FieldOp>(op);
      if (yieldOp.getValues()[idx].getType() != fieldOp.getResult().getType())
        return emitOpError()
               << "yield operand #" << idx << " type mismatch with field @"
               << fieldOp.getSymName();
      ++idx;
    }
  }

  // Verify the body region terminates with an ActionOp.
  Block &bodyBlock = getBodyRegion().front();
  Operation *terminator = bodyBlock.getTerminator();
  if (!isa<ActionOp>(terminator))
    return emitOpError("body region must terminate with pattern.action");

  return success();
}

LogicalResult RewritePatternOp::emitDecl(::mlir::emitc::EmitCContext &ctx) {
  raw_indented_ostream &os = ctx.ostream();
  StringRef name = getSymName();
  StringRef opName = getOpName();

  os << "struct " << name << " : OpRewrite<" << opName << "> {\n";
  os.indent();

  // Emit field declarations.
  Region &fieldsRegion = getFieldsRegion();
  if (!fieldsRegion.empty()) {
    Block &fieldsBlock = fieldsRegion.front();
    for (Operation &op : fieldsBlock.without_terminator()) {
      auto fieldOp = cast<FieldOp>(op);
      if (failed(ctx.emitType(fieldOp.getLoc(), fieldOp.getResult().getType())))
        return failure();
      os << " " << fieldOp.getSymName() << ";\n";
    }
  }

  // Emit the matchRewrite method.
  os << "LogicalResult matchRewrite(" << opName
     << " op, PatternRewriter &rewriter) {\n";
  os.indent();

  // Emit the body region. Regular ops get a trailing semicolon; the
  // terminator (ActionOp) handles its own formatting.
  Block &bodyBlock = getBodyRegion().front();
  for (Operation &op : bodyBlock) {
    bool isTerm = op.hasTrait<OpTrait::IsTerminator>();
    if (failed(ctx.emitOperation(op, /*trailingSemicolon=*/!isTerm)))
      return failure();
  }

  os << "return llvm::success();\n";
  os.unindent();
  os << "}\n";

  os.unindent();
  os << "};\n";
  return success();
}

//===----------------------------------------------------------------------===//
// ActionOp
//===----------------------------------------------------------------------===//

LogicalResult ActionOp::verify() {
  // The body region terminator must be a YieldOp with no operands.
  Block &bodyBlock = getBodyRegion().front();
  auto yieldOp = cast<YieldOp>(bodyBlock.getTerminator());
  if (!yieldOp.getValues().empty())
    return emitOpError("action body yield must have no operands");
  return success();
}

LogicalResult ActionOp::emitStmt(::mlir::emitc::EmitCContext &ctx) {
  raw_indented_ostream &os = ctx.ostream();

  os << "if (!";
  if (failed(ctx.emitOperand(getCondition())))
    return failure();
  os << ")\n";
  os.indent();
  os << "return failure();\n";
  os.unindent();

  // Emit the body (excluding the yield terminator).
  Block &bodyBlock = getBodyRegion().front();
  for (Operation &op : bodyBlock.without_terminator())
    if (failed(ctx.emitOperation(op, /*trailingSemicolon=*/true)))
      return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// GetFieldOp
//===----------------------------------------------------------------------===//

LogicalResult GetFieldOp::emitExpr(::mlir::emitc::EmitCContext &ctx) {
  raw_indented_ostream &os = ctx.ostream();
  os << getFieldName();
  return success();
}

//===----------------------------------------------------------------------===//
// MethodCallOp
//===----------------------------------------------------------------------===//

LogicalResult MethodCallOp::emitExpr(::mlir::emitc::EmitCContext &ctx) {
  raw_indented_ostream &os = ctx.ostream();

  if (failed(ctx.emitOperand(getObject())))
    return failure();

  // Use -> for pointer types, . otherwise.
  Type objectType = getObject().getType();
  if (isa<emitc::PointerType>(objectType))
    os << "->";
  else
    os << ".";

  os << getCallee() << "(";

  // Emit arguments.
  bool first = true;
  for (Value arg : getArgs()) {
    if (!first)
      os << ", ";
    if (failed(ctx.emitOperand(arg)))
      return failure();
    first = false;
  }

  os << ")";
  return success();
}
