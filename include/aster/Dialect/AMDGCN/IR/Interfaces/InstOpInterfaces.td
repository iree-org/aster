//===- InstOpInterfaces.td - Inst Op Interfaces ------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN instruction operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_INTERFACES_INSTOPINTERFACES_TD
#define ASTER_DIALECT_AMDGCN_IR_INTERFACES_INSTOPINTERFACES_TD

include "mlir/IR/OpBase.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNInstOpInterface.td"
include "aster/Interfaces/DependentOpInterface.td"

//===----------------------------------------------------------------------===//
// Memory instruction interfaces
//===----------------------------------------------------------------------===//

def LoadOpInterface :
    OpInterface<"LoadOpInterface", [AMDGCNInstOpInterface, FutureOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for load operations providing uniform access to operands and
    results.
  }];
  let methods = [
    InterfaceMethod<"Get the destination operand.",
      "::mlir::aster::Operand", "getDst", (ins), [{}], [{
        return $_op.getDestMutable();
      }]>,
    InterfaceMethod<"Get the address operand.",
      "::mlir::aster::Operand", "getAddress", (ins), [{}], [{
        return $_op.getAddrMutable();
      }]>
  ];
  let extraTraitClassDeclaration = [{
    ::mlir::aster::Future getFuture() {
      return ::mlir::aster::Future::read($_op.getToken(), $_op.getDst().getValue());
    }
  }];
}

def StoreOpInterface :
    OpInterface<"StoreOpInterface", [AMDGCNInstOpInterface, FutureOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for store operations providing uniform access to operands and
    results.
  }];
  let methods = [
    InterfaceMethod<"Get the data operand.",
      "::mlir::aster::Operand", "getDataOperand", (ins), [{}], [{
        return $_op.getDataMutable();
      }]>,
    InterfaceMethod<"Get the address operand.",
      "::mlir::aster::Operand", "getAddress", (ins), [{}], [{
        return $_op.getAddrMutable();
      }]>,
    InterfaceMethod<"Get the dynamic offset operand.",
      "::mlir::aster::Operand", "getDynamicOff", (ins), [{}], [{
        ::mlir::MutableOperandRange r = $_op.getDynamicOffsetMutable();
        return !r.empty() ? ::mlir::aster::Operand(r[0]) : ::mlir::aster::Operand();
      }]>
  ];
  let extraTraitClassDeclaration = [{
    ::mlir::aster::Future getFuture() {
      return ::mlir::aster::Future::write($_op.getToken(), $_op.getDataOperand().getValue());
    }
  }];
}

//===----------------------------------------------------------------------===//
// DS instruction interfaces
//===----------------------------------------------------------------------===//

def DSReadInstOpInterface :
    OpInterface<"DSReadInstOpInterface", [AMDGCNInstOpInterface, LoadOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for DS read instructions providing uniform access to operands,
    results, and modifiers.
  }];
  let methods = [
    InterfaceMethod<"Get the constant offset operand.",
      "::mlir::aster::Operand", "getConstOffsetOperand", (ins), [{}], [{
        return $_op.getConstOffsetMutable();
      }]>,
    InterfaceMethod<"Get the GDS modifier.",
      "::mlir::UnitAttr&", "getGdsMod", (ins), [{}], [{
        return $_op.getProperties().gds;
      }]>,
    InterfaceMethod<"Get the destination result.",
      "::mlir::OpResult", "getDestResult", (ins), [{}], [{
        return ::mlir::cast<::mlir::OpResult>(
            ::mlir::Value($_op.getDestRes()));
      }]>
  ];
}

def DSRead2InstOpInterface :
    OpInterface<"DSRead2InstOpInterface", [AMDGCNInstOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for DS read2 instructions providing uniform access to operands,
    results, and modifiers.
  }];
  let methods = [
    InterfaceMethod<"Get the dst0 operand.",
      "::mlir::aster::Operand", "getDestOperand", (ins), [{
        return $_op.getDestMutable();
      }]>,
    InterfaceMethod<"Get the address operand.",
      "::mlir::aster::Operand", "getAddrOperand", (ins), [{
        return $_op.getAddrMutable();
      }]>,
    InterfaceMethod<"Get the first constant offset operand.",
      "::mlir::aster::Operand", "getConstOffset0Operand", (ins), [{
        return $_op.getConstOffset0Mutable();
      }]>,
    InterfaceMethod<"Get the second constant offset operand.",
      "::mlir::aster::Operand", "getConstOffset1Operand", (ins), [{
        return $_op.getConstOffset1Mutable();
      }]>,
    InterfaceMethod<"Get the GDS modifier.",
      "::mlir::UnitAttr&", "getGdsMod", (ins), [{
        return $_op.getProperties().gds;
      }]>,
    InterfaceMethod<"Get the dst0 result.",
      "::mlir::OpResult", "getDestResult", (ins), [{
        return ::mlir::cast<::mlir::OpResult>(
            ::mlir::Value($_op.getDestRes()));
      }]>,
    InterfaceMethod<"Get the token result.",
      "::mlir::OpResult", "getTokenResult", (ins), [{
        return ::mlir::cast<::mlir::OpResult>(
            ::mlir::Value($_op.getToken()));
      }]>
  ];
}

def DSWriteInstOpInterface :
    OpInterface<"DSWriteInstOpInterface", [AMDGCNInstOpInterface, StoreOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for DS write instructions providing uniform access to operands,
    results, and modifiers.
  }];
  let methods = [
    InterfaceMethod<"Get the constant offset operand.",
      "::mlir::aster::Operand", "getConstOffsetOperand", (ins), [{}], [{
        return $_op.getConstOffsetMutable();
      }]>,
    InterfaceMethod<"Get the GDS modifier.",
      "::mlir::UnitAttr&", "getGdsMod", (ins), [{}], [{
        return $_op.getProperties().gds;
      }]>
  ];
  let extraTraitClassDeclaration = [{
    // TODO: Remove this once we finish the transition.
    ::mlir::MutableOperandRange getDynamicOffsetMutable() {
      return ::mlir::MutableOperandRange($_op.getOperation(), 0, 0);
    }
  }];
}

def DSWrite2InstOpInterface :
    OpInterface<"DSWrite2InstOpInterface", [AMDGCNInstOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for DS write2 instructions providing uniform access to operands,
    results, and modifiers.
  }];
  let methods = [
    InterfaceMethod<"Get the address operand.",
      "::mlir::aster::Operand", "getAddrOperand", (ins), [{}], [{
        return $_op.getAddrMutable();
      }]>,
    InterfaceMethod<"Get the first data operand.",
      "::mlir::aster::Operand", "getData0Operand", (ins), [{}], [{
        return $_op.getData0Mutable();
      }]>,
    InterfaceMethod<"Get the second data operand.",
      "::mlir::aster::Operand", "getData1Operand", (ins), [{}], [{
        return $_op.getData1Mutable();
      }]>,
    InterfaceMethod<"Get the first constant offset operand.",
      "::mlir::aster::Operand", "getConstOffset0Operand", (ins), [{}], [{
        return $_op.getConstOffset0Mutable();
      }]>,
    InterfaceMethod<"Get the second constant offset operand.",
      "::mlir::aster::Operand", "getConstOffset1Operand", (ins), [{}], [{
        return $_op.getConstOffset1Mutable();
      }]>,
    InterfaceMethod<"Get the GDS modifier.",
      "::mlir::UnitAttr&", "getGdsMod", (ins), [{}], [{
        return $_op.getProperties().gds;
      }]>,
    InterfaceMethod<"Get the token result.",
      "::mlir::OpResult", "getTokenResult", (ins), [{}], [{
        return ::mlir::cast<::mlir::OpResult>(
            ::mlir::Value($_op.getToken()));
      }]>
  ];
}

//===----------------------------------------------------------------------===//
// Scalar memory instruction interfaces
//===----------------------------------------------------------------------===//

def SLoadDwordInstOpInterface :
    OpInterface<"SLoadDwordInstOpInterface", [AMDGCNInstOpInterface, LoadOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for scalar load dword instructions providing uniform access to
    operands, results, and modifiers.
  }];
  let methods = [
    InterfaceMethod<"Get the soffset operand.",
      "::mlir::aster::MutableOperand", "getSoffsetOperand", (ins), [{}], [{
        return ::mlir::aster::MutableOperand::get(
            $_op, ConcreteOp::odsIndex_soffset);
      }]>,
    InterfaceMethod<"Get the constant offset operand.",
      "::mlir::aster::MutableOperand", "getConstOffsetOperand", (ins), [{}], [{
        return ::mlir::aster::MutableOperand::get(
            $_op, ConcreteOp::odsIndex_const_offset);
      }]>,
    InterfaceMethod<"Get the GLC modifier.",
      "::mlir::UnitAttr&", "getGlcMod", (ins), [{}], [{
        return $_op.getProperties().glc;
      }]>,
    InterfaceMethod<"Get the destination result.",
      "::mlir::OpResult", "getDestResult", (ins), [{}], [{
        return ::mlir::cast<::mlir::OpResult>(
            ::mlir::Value($_op.getDestRes()));
      }]>
  ];
}

def SStoreDwordInstOpInterface :
    OpInterface<"SStoreDwordInstOpInterface", [AMDGCNInstOpInterface, StoreOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for scalar store dword instructions providing uniform access to
    operands, results, and modifiers.
  }];
  let methods = [
    InterfaceMethod<"Get the soffset operand.",
      "::mlir::aster::MutableOperand", "getSoffsetOperand", (ins), [{}], [{
        return ::mlir::aster::MutableOperand::get(
            $_op, ConcreteOp::odsIndex_soffset);
      }]>,
    InterfaceMethod<"Get the constant offset operand.",
      "::mlir::aster::MutableOperand", "getConstOffsetOperand", (ins), [{}], [{
        return ::mlir::aster::MutableOperand::get(
            $_op, ConcreteOp::odsIndex_const_offset);
      }]>,
    InterfaceMethod<"Get the GLC modifier.",
      "::mlir::UnitAttr&", "getGlcMod", (ins), [{}], [{
        return $_op.getProperties().glc;
      }]>
  ];
  let extraTraitClassDeclaration = [{
    // TODO: Remove this once we finish the transition.
    ::mlir::MutableOperandRange getDynamicOffsetMutable() {
      return ::mlir::MutableOperandRange($_op.getOperation(), 0, 0);
    }
  }];
}

//===----------------------------------------------------------------------===//
// Buffer instruction interfaces
//===----------------------------------------------------------------------===//

def BufferLoadInstOpInterface :
    OpInterface<"BufferLoadInstOpInterface", [AMDGCNInstOpInterface, LoadOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for buffer load instructions providing uniform access to
    operands, results, and modifiers.
  }];
  let methods = [
    InterfaceMethod<"Get the offset or index operand.",
      "::mlir::aster::MutableOperand", "getOffOrIdxOperand", (ins), [{}], [{
        return ::mlir::aster::MutableOperand::get(
            $_op, ConcreteOp::odsIndex_off_or_idx);
      }]>,
    InterfaceMethod<"Get the scalar offset operand.",
      "::mlir::aster::Operand", "getSoffsetOperand", (ins), [{}], [{
        return $_op.getSoffsetMutable();
      }]>,
    InterfaceMethod<"Get the IDXEN modifier.",
      "::mlir::UnitAttr&", "getIdxenMod", (ins), [{}], [{
        return $_op.getProperties().idxen;
      }]>,
    InterfaceMethod<"Get whether the operation is a load-to-LDS operation.",
      "bool", "getLds">,
    InterfaceMethod<"Get the NT modifier.",
      "::mlir::UnitAttr&", "getNtMod", (ins), [{}], [{
        return $_op.getProperties().nt;
      }]>,
    InterfaceMethod<"Get the OFFEN modifier.",
      "::mlir::UnitAttr&", "getOffenMod", (ins), [{}], [{
        return $_op.getProperties().offen;
      }]>,
    InterfaceMethod<"Get the SC0 modifier.",
      "::mlir::UnitAttr&", "getSc0Mod", (ins), [{}], [{
        return $_op.getProperties().sc0;
      }]>,
    InterfaceMethod<"Get the SC1 modifier.",
      "::mlir::UnitAttr&", "getSc1Mod", (ins), [{}], [{
        return $_op.getProperties().sc1;
      }]>,
    InterfaceMethod<"Get the constant offset operand.",
      "::mlir::aster::Operand", "getConstOffsetOperand", (ins), [{}], [{
        return $_op.getConstOffsetMutable();
      }]>,
    InterfaceMethod<"Get the destination result.",
      "::mlir::OpResult", "getDestResult", (ins), [{}], [{
        return ::mlir::cast<::mlir::OpResult>(
            ::mlir::Value($_op.getDestRes()));
      }]>
  ];
}

def BufferStoreInstOpInterface :
    OpInterface<"BufferStoreInstOpInterface", [AMDGCNInstOpInterface, StoreOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for buffer store instructions providing uniform access to
    operands, results, and modifiers.
  }];
  let methods = [
    InterfaceMethod<"Get the offset or index operand.",
      "::mlir::aster::MutableOperand", "getOffOrIdxOperand", (ins), [{}], [{
        return ::mlir::aster::MutableOperand::get(
            $_op, ConcreteOp::odsIndex_off_or_idx);
      }]>,
    InterfaceMethod<"Get the scalar offset operand.",
      "::mlir::aster::Operand", "getSoffsetOperand", (ins), [{}], [{
        return $_op.getSoffsetMutable();
      }]>,
    InterfaceMethod<"Get the IDXEN modifier.",
      "::mlir::UnitAttr&", "getIdxenMod", (ins), [{}], [{
        return $_op.getProperties().idxen;
      }]>,
    InterfaceMethod<"Get the NT modifier.",
      "::mlir::UnitAttr&", "getNtMod", (ins), [{}], [{
        return $_op.getProperties().nt;
      }]>,
    InterfaceMethod<"Get the OFFEN modifier.",
      "::mlir::UnitAttr&", "getOffenMod", (ins), [{}], [{
        return $_op.getProperties().offen;
      }]>,
    InterfaceMethod<"Get the SC0 modifier.",
      "::mlir::UnitAttr&", "getSc0Mod", (ins), [{}], [{
        return $_op.getProperties().sc0;
      }]>,
    InterfaceMethod<"Get the SC1 modifier.",
      "::mlir::UnitAttr&", "getSc1Mod", (ins), [{}], [{
        return $_op.getProperties().sc1;
      }]>,
    InterfaceMethod<"Get the constant offset operand.",
      "::mlir::aster::Operand", "getConstOffsetOperand", (ins), [{}], [{
        return $_op.getConstOffsetMutable();
      }]>
  ];
  let extraTraitClassDeclaration = [{
    // TODO: Remove this once we finish the transition.
    ::mlir::MutableOperandRange getDynamicOffsetMutable() {
      return $_op.getAddrMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Global memory instruction interfaces
//===----------------------------------------------------------------------===//

def GlobalLoadInstOpInterface :
    OpInterface<"GlobalLoadInstOpInterface", [AMDGCNInstOpInterface, LoadOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for global load instructions providing uniform access to
    operands, results, and modifiers.
  }];
  let methods = [
    InterfaceMethod<"Get the offset operand.",
      "::mlir::aster::MutableOperand", "getOffsetOperand", (ins), [{}], [{
        return ::mlir::aster::MutableOperand::get(
            $_op, ConcreteOp::odsIndex_offset);
      }]>,
    InterfaceMethod<"Get the NT modifier.",
      "::mlir::UnitAttr&", "getNtMod", (ins), [{}], [{
        return $_op.getProperties().nt;
      }]>,
    InterfaceMethod<"Get the SC0 modifier.",
      "::mlir::UnitAttr&", "getSc0Mod", (ins), [{}], [{
        return $_op.getProperties().sc0;
      }]>,
    InterfaceMethod<"Get the SC1 modifier.",
      "::mlir::UnitAttr&", "getSc1Mod", (ins), [{}], [{
        return $_op.getProperties().sc1;
      }]>,
    InterfaceMethod<"Get the constant offset operand.",
      "::mlir::aster::Operand", "getConstOffsetOperand", (ins), [{}], [{
        return $_op.getConstOffsetMutable();
      }]>,
    InterfaceMethod<"Get the destination result.",
      "::mlir::OpResult", "getDestResult", (ins), [{}], [{
        return ::mlir::cast<::mlir::OpResult>(
            ::mlir::Value($_op.getDestRes()));
      }]>
  ];
}

def GlobalStoreDwordInstOpInterface :
    OpInterface<"GlobalStoreDwordInstOpInterface", [AMDGCNInstOpInterface, StoreOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for global store dword instructions providing uniform access to
    operands, results, and modifiers.
  }];
  let methods = [
    InterfaceMethod<"Get the offset operand.",
      "::mlir::aster::MutableOperand", "getOffsetOperand", (ins), [{}], [{
        return ::mlir::aster::MutableOperand::get(
            $_op, ConcreteOp::odsIndex_offset);
      }]>,
    InterfaceMethod<"Get the NT modifier.",
      "::mlir::UnitAttr&", "getNtMod", (ins), [{}], [{
        return $_op.getProperties().nt;
      }]>,
    InterfaceMethod<"Get the SC0 modifier.",
      "::mlir::UnitAttr&", "getSc0Mod", (ins), [{}], [{
        return $_op.getProperties().sc0;
      }]>,
    InterfaceMethod<"Get the SC1 modifier.",
      "::mlir::UnitAttr&", "getSc1Mod", (ins), [{}], [{
        return $_op.getProperties().sc1;
      }]>,
    InterfaceMethod<"Get the constant offset operand.",
      "::mlir::aster::Operand", "getConstOffsetOperand", (ins), [{}], [{
        return $_op.getConstOffsetMutable();
      }]>
  ];
  let extraTraitClassDeclaration = [{
    // TODO: Remove this once we finish the transition.
    ::mlir::MutableOperandRange getDynamicOffsetMutable() {
      return $_op.getOffsetMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// MMA instruction interfaces
//===----------------------------------------------------------------------===//

def MMAOpInterface :
    OpInterface<"MMAOpInterface", [AMDGCNInstOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for matrix multiply-accumulate (MMA) instructions providing
    uniform access to the source and destination operands.
  }];
  let methods = [
    InterfaceMethod<"Get the source A operand (first matrix input).",
      "::mlir::OpOperand &", "getSrcA", (ins), [{}], [{
        return $_op->getOpOperand(ConcreteOp::odsIndex_src0);
      }]>,
    InterfaceMethod<"Get the source B operand (second matrix input).",
      "::mlir::OpOperand &", "getSrcB", (ins), [{}], [{
        return $_op->getOpOperand(ConcreteOp::odsIndex_src1);
      }]>,
    InterfaceMethod<"Get the source C operand (accumulator input).",
      "::mlir::OpOperand &", "getSrcC", (ins), [{}], [{
        return $_op->getOpOperand(ConcreteOp::odsIndex_src2);
      }]>,
    InterfaceMethod<"Get the vdst operand (destination output).",
      "::mlir::OpOperand &", "getVdst", (ins), [{}], [{
        return $_op->getOpOperand(ConcreteOp::odsIndex_dst0);
      }]>
  ];
}

def ScaledMMAOpInterface :
    OpInterface<"ScaledMMAOpInterface", [MMAOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for scaled matrix multiply-accumulate (MFMA_SCALE) instructions,
    extending MMAOpInterface with access to per-lane scale factor operands.
  }];
  let methods = [
    InterfaceMethod<"Get the scale A operand (scale factor for source A).",
      "::mlir::OpOperand &", "getScaleA", (ins), [{}], [{
        return $_op->getOpOperand(ConcreteOp::odsIndex_src3);
      }]>,
    InterfaceMethod<"Get the scale B operand (scale factor for source B).",
      "::mlir::OpOperand &", "getScaleB", (ins), [{}], [{
        return $_op->getOpOperand(ConcreteOp::odsIndex_src4);
      }]>
  ];
}

#endif // ASTER_DIALECT_AMDGCN_IR_INTERFACES_INSTOPINTERFACES_TD
