//===- QueueTypes.h - AMD GPU hardware queue classification -----*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the QueueType enum, queue classification helpers, and QueueSimulator
// shared by the pre-RA scheduler and the sched attribute implementations.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_IR_QUEUETYPES_H
#define ASTER_AMDGCN_IR_QUEUETYPES_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNInst.h"
#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNInstOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include <algorithm>
#include <cstdint>
#include <optional>

namespace mlir::aster::amdgcn {

//===----------------------------------------------------------------------===//
// QueueType
//===----------------------------------------------------------------------===//

/// AMD GPU hardware execution queue classification.
enum class QueueType : uint8_t { VALU, XDL, SALU, VMEM, LGKM, Unknown };

/// Issue cost in hardware cycles (1 quad = 4 hw cycles).
inline constexpr int64_t kIssueCost = 4;

/// Parse sched.queue attr: "valu", "xdl", "salu", "vmem", "lgkm".
inline std::optional<QueueType> parseQueueAttr(mlir::Operation *op) {
  auto attr = op->getAttrOfType<mlir::StringAttr>("sched.queue");
  if (!attr)
    return std::nullopt;
  return llvm::StringSwitch<std::optional<QueueType>>(attr.getValue())
      .Case("valu", QueueType::VALU)
      .Case("xdl", QueueType::XDL)
      .Case("salu", QueueType::SALU)
      .Case("vmem", QueueType::VMEM)
      .Case("lgkm", QueueType::LGKM)
      .Default(std::nullopt);
}

// TODO: put this in instruction definition directly in tablegen.
/// Classify an operation into a hardware execution queue.
/// Returns QueueType::Unknown for unrecognized operations.
inline QueueType classifyOp(mlir::Operation *op) {
  // sched.queue overrides InstProp classification (useful for test_inst).
  if (auto qt = parseQueueAttr(op))
    return *qt;

  auto instOp = mlir::dyn_cast<AMDGCNInstOpInterface>(op);
  if (!instOp)
    return QueueType::Unknown;
  const InstMetadata *md = instOp.getInstMetadata();
  if (!md)
    return QueueType::Unknown;

  // SOPP (s_waitcnt, s_barrier, branches) must be scheduling barriers.
  if (md->hasProp(InstProp::Sopp))
    return QueueType::Unknown;
  if (md->hasProp(InstProp::Dsmem))
    return QueueType::LGKM;
  if (md->hasProp(InstProp::Smem))
    return QueueType::LGKM;
  if (md->hasProp(InstProp::IsVmem))
    return QueueType::VMEM;
  // Check before VALU: MFMA ops carry both Mma and IsValu props.
  if (md->hasAnyProps({InstProp::Mma, InstProp::ScaledMma}))
    return QueueType::XDL;
  if (md->hasProp(InstProp::Salu))
    return QueueType::SALU;
  if (md->hasProp(InstProp::IsValu))
    return QueueType::VALU;

  return QueueType::Unknown;
}

/// Returns exec latency in hw cycles. sched.exec_latency overrides defaults.
// TODO: put this in instruction definition directly in tablegen.
inline int64_t getExecLatency(mlir::Operation *op, QueueType qt) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("sched.exec_latency"))
    return attr.getInt();
  switch (qt) {
  case QueueType::VALU:
    return 4;
  case QueueType::XDL:
    return 16;
  case QueueType::SALU:
    return 4;
  case QueueType::VMEM:
    return 128;
  case QueueType::LGKM:
    return 32;
  case QueueType::Unknown:
    return 4;
  }
  llvm_unreachable("unhandled queue type");
}

/// Returns the queue depth (number of in-flight slots).
/// VMEM is 2-deep (shared per CU across ~4 waves).
/// All per-SIMD queues are 8-deep.
inline int64_t getQueueDepth(QueueType qt) {
  switch (qt) {
  case QueueType::VMEM:
    return 2;
  default:
    return 8;
  }
}

inline llvm::StringRef getQueueName(QueueType qt) {
  switch (qt) {
  case QueueType::VALU:
    return "valu";
  case QueueType::XDL:
    return "xdl";
  case QueueType::SALU:
    return "salu";
  case QueueType::VMEM:
    return "vmem";
  case QueueType::LGKM:
    return "lgkm";
  case QueueType::Unknown:
    return "unknown";
  }
  llvm_unreachable("unhandled queue type");
}

//===----------------------------------------------------------------------===//
// QueueSimulator
//===----------------------------------------------------------------------===//

/// Models the hardware queue state for stall detection.
/// Each queue has `capacity` slots; issuing an op occupies one slot for
/// `execLatency` cycles. A stall occurs when all slots are busy.
struct QueueSimulator {
  llvm::DenseMap<QueueType, llvm::SmallVector<int64_t, 8>> slotFreeAt;
  int64_t currentCycle = 0;
  QueueSimulator() = default;

  /// Query how many hw cycles issuing to `qt` would stall.
  int64_t wouldStall(QueueType qt) const {
    if (qt == QueueType::Unknown)
      return 0;
    auto it = slotFreeAt.find(qt);
    if (it == slotFreeAt.end())
      return 0;
    int64_t depth = getQueueDepth(qt);
    int64_t occupied = 0;
    for (int64_t t : it->second) {
      if (t > currentCycle)
        occupied++;
    }
    if (occupied < depth)
      return 0;
    int64_t earliest = *std::min_element(it->second.begin(), it->second.end());
    return std::max(int64_t{0}, earliest - currentCycle);
  }

  /// Issue an op. Returns stall in hw cycles (always a multiple of 4).
  int64_t issue(QueueType qt, int64_t execLatency) {
    if (qt == QueueType::Unknown)
      return 0;

    auto &slots = slotFreeAt[qt];
    llvm::erase_if(slots, [&](int64_t t) { return t <= currentCycle; });

    int64_t depth = getQueueDepth(qt);
    int64_t stallCycles = 0;
    if (static_cast<int64_t>(slots.size()) >= depth) {
      int64_t earliest = *std::min_element(slots.begin(), slots.end());
      stallCycles = std::max(int64_t{0}, earliest - currentCycle);
      currentCycle += stallCycles;
      llvm::erase_if(slots, [&](int64_t t) { return t <= currentCycle; });
    }

    slots.push_back(currentCycle + execLatency);
    currentCycle += kIssueCost;
    return stallCycles;
  }
};

} // namespace mlir::aster::amdgcn

#endif // ASTER_AMDGCN_IR_QUEUETYPES_H
