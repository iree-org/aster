//===- SchedulerCostModel.cpp - AMDGCN scheduler cost model ---------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Scheduler/SchedulerCostModel.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

using namespace mlir;
using namespace mlir::aster::amdgcn;

namespace mlir::aster::amdgcn {

namespace {

/// Parse sched.queue attr: "valu", "xdl", "salu", "vmem", "lgkm".
// TODO: put this in instruction definition directly in tablegen.
std::optional<QueueType> parseQueueAttr(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>("sched.queue");
  if (!attr)
    return std::nullopt;
  return StringSwitch<std::optional<QueueType>>(attr.getValue())
      .Case("valu", QueueType::VALU)
      .Case("xdl", QueueType::XDL)
      .Case("salu", QueueType::SALU)
      .Case("vmem", QueueType::VMEM)
      .Case("lgkm", QueueType::LGKM)
      .Default(std::nullopt);
}

/// Preset 1: wider LGKM burst lookback (6) keeps the LGKM penalty active longer
struct CDNA3PresetMfmaHiding : CDNA3Latencies {
  CDNA3PresetMfmaHiding() { lgkmBurstLookback = 6; }
};

/// CDNA4 (gfx950 / MI350). Structurally identical to CDNA3 today.
struct CDNA4Latencies : CDNA3Latencies {
  CDNA4Latencies() {}
};

/// Per-opcode XDL exec latency from CDNA3 ISA Table 28 (MI300 manual p.42)
/// and CDNA4 ISA Table 28 (gfx950 manual p.43).
/// Returns 0 if the opcode is not an XDL instruction we model specifically.
int64_t getXdlExecLatency(OpCode op, const CDNA3Latencies &L) {
  switch (op) {
  // 4-pass: 16-cycle MFMAs (16x16x16 and 16x16x32 family).
  case OpCode::v_mfma_f32_16x16x16_f16:
  case OpCode::v_mfma_f32_16x16x16_bf16:
  case OpCode::v_mfma_f32_16x16x32_f16:
  case OpCode::v_mfma_f32_16x16x32_bf16:
  case OpCode::v_mfma_f32_16x16x32_fp8_fp8:
  case OpCode::v_mfma_f32_16x16x32_fp8_bf8:
  case OpCode::v_mfma_f32_16x16x32_bf8_fp8:
  case OpCode::v_mfma_f32_16x16x32_bf8_bf8:
  // CDNA4-only 4-pass: i32 16x16x64.
  case OpCode::v_mfma_i32_16x16x64_i8:
    return L.xdlExec4Pass;
  // 8-pass: 32-cycle MFMAs (32x32x8 / 32x32x16 family).
  case OpCode::v_mfma_f32_32x32x8_f16:
  case OpCode::v_mfma_f32_32x32x8_bf16:
  case OpCode::v_mfma_f32_32x32x16_f16:
  case OpCode::v_mfma_f32_32x32x16_bf16:
  case OpCode::v_mfma_f32_32x32x16_fp8_fp8:
  case OpCode::v_mfma_f32_32x32x16_fp8_bf8:
  case OpCode::v_mfma_f32_32x32x16_bf8_fp8:
  case OpCode::v_mfma_f32_32x32x16_bf8_bf8:
  case OpCode::v_mfma_i32_16x16x32_i8:
  case OpCode::v_mfma_i32_32x32x16_i8:
  // F8F6F4 16x16x128: 16cy (F6/F4) or 32cy (FP8). Conservative: 32cy.
  case OpCode::v_mfma_f32_16x16x128_f8f6f4:
  case OpCode::v_mfma_scale_f32_16x16x128_f8f6f4:
  // CDNA4-only 8-pass: i32 32x32x32.
  case OpCode::v_mfma_i32_32x32x32_i8:
    return L.xdlExec8Pass;
  // 16-pass: 64-cycle MFMAs (32x32x{1_2B,2,4_2B}, F64 16x16x4, F8F6F4
  // 32x32x64).
  case OpCode::v_mfma_f32_32x32x1_2b_f32:
  case OpCode::v_mfma_f32_32x32x2_f32:
  case OpCode::v_mfma_f32_32x32x4_2b_f16:
  case OpCode::v_mfma_f32_32x32x4_2b_bf16:
  case OpCode::v_mfma_i32_32x32x4_2b_i8:
  case OpCode::v_mfma_f64_16x16x4_f64:
  // F8F6F4 32x32x64: 32cy (F6/F4) or 64cy (FP8). Conservative: 64cy.
  case OpCode::v_mfma_f32_32x32x64_f8f6f4:
  case OpCode::v_mfma_scale_f32_32x32x64_f8f6f4:
    return L.xdlExec16Pass;
  default:
    return 0;
  }
}

} // namespace

QueueType classifyOp(Operation *op) {
  // sched.queue overrides InstProp classification (useful for test_inst).
  if (auto qt = parseQueueAttr(op))
    return *qt;

  auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
  if (!instOp || instOp.getOpCode() == OpCode::Invalid)
    return QueueType::Unknown;

  // SOPP (s_waitcnt, s_barrier, branches) must be scheduling barriers.
  if (instOp.hasProp(InstProp::Sopp))
    return QueueType::Unknown;
  // Any LDS op or SMEM load -> LGKM bucket.
  if (instOp.hasAnyProps({InstProp::Ds, InstProp::Smem}))
    return QueueType::LGKM;
  if (instOp.hasProp(InstProp::IsVmem))
    return QueueType::VMEM;
  // Check before VALU: MFMA ops carry both Mma and IsValu props.
  if (instOp.hasAnyProps({InstProp::Mma, InstProp::ScaledMma}))
    return QueueType::XDL;
  if (instOp.hasProp(InstProp::Salu))
    return QueueType::SALU;
  if (instOp.hasProp(InstProp::IsValu))
    return QueueType::VALU;

  return QueueType::Unknown;
}

/// Active per-architecture latency / policy table.
///
/// `preset` selects a pre-tuned magic-number set:
///   1 = mfma-hiding default
///   2..N = future presets (add a new struct above + a new case here).
CDNA3Latencies latencies(ISAVersion isa, int preset) {
  if (isa == ISAVersion::CDNA4)
    return CDNA4Latencies();
  switch (preset) {
  case 1:
    return CDNA3PresetMfmaHiding();
  default:
    return CDNA3PresetMfmaHiding();
  }
}

int64_t getExecLatency(Operation *op, QueueType qt, const CDNA3Latencies &L) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("sched.exec_latency"))
    return attr.getInt();
  switch (qt) {
  case QueueType::VALU:
    return L.valuExec;
  case QueueType::XDL:
    if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op))
      if (int64_t lat = getXdlExecLatency(instOp.getOpCode(), L))
        return lat;
    return L.xdlExec4Pass; // default to 4-pass
  case QueueType::SALU:
    return L.saluExec;
  case QueueType::VMEM: {
    // Pick per-width latency based on op mnemonic suffix.
    StringRef name = op->getName().getStringRef();
    bool isStore =
        isa<GlobalStoreDwordInstOpInterface, BufferStoreInstOpInterface>(op);
    if (name.contains("dwordx4"))
      return isStore ? L.vmemStoreDwordx4Exec : L.vmemLoadDwordx4Exec;
    if (name.contains("dwordx3"))
      return isStore ? L.vmemStoreDwordx3Exec : L.vmemLoadDwordx3Exec;
    if (name.contains("dwordx2"))
      return isStore ? L.vmemStoreDwordx2Exec : L.vmemLoadDwordx2Exec;
    if (name.contains("dword"))
      return isStore ? L.vmemStoreDwordExec : L.vmemLoadDwordExec;
    return L.vmemExec;
  }
  case QueueType::LGKM:
    if (isa<DSWriteInstOpInterface>(op))
      return L.dsWriteExec;
    if (isa<DSReadInstOpInterface>(op))
      return L.dsReadExec;
    return L.lgkmDefaultExec;
  case QueueType::Unknown:
    return L.unknownExec;
  }
  llvm_unreachable("unhandled queue type");
}

int64_t getQueueDepth(QueueType qt, const CDNA3Latencies &L) {
  // TODO: 4 is an approximation here, it depends on predication + stagger (e.g.
  // in ping-pong or wave-specialized schedules).
  int64_t numSimdActive = 4;
  return qt == QueueType::VMEM ? L.vmemQueueDepth / numSimdActive
                               : L.defaultQueueDepth;
}

int64_t getIssueCost(Operation *op, QueueType qt, const CDNA3Latencies &L) {
  if (qt == QueueType::LGKM && isa<DSWriteInstOpInterface>(op))
    return L.dsWriteIssueCost;
  return L.defaultIssueCost;
}

NodeCostInfo classifyGraph(ArrayRef<Operation *> ops, const CDNA3Latencies &L) {
  NodeCostInfo info;
  int64_t n = static_cast<int64_t>(ops.size());
  info.queueTypes.resize(n);
  info.execLatencies.resize(n);
  info.issueCosts.resize(n);
  for (auto [i, op] : llvm::enumerate(ops)) {
    info.queueTypes[i] = classifyOp(op);
    info.execLatencies[i] = getExecLatency(op, info.queueTypes[i], L);
    info.issueCosts[i] = getIssueCost(op, info.queueTypes[i], L);
  }
  return info;
}

} // namespace mlir::aster::amdgcn
