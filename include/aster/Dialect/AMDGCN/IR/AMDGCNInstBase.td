//===- AMDGCNInstBase.td - AMDGCN Instruction base ---------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the base file for AMDGCN instructions.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INSTBASE_TD
#define AMDGCN_INSTBASE_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNTypeConstraints.td"
include "aster/IR/Inst.td"

//===----------------------------------------------------------------------===//
// Base instruction class for AMDGCN instructions.
//===----------------------------------------------------------------------===//

def AMDInstTrait : InstructionTrait {
  let extraConcreteClassDeclaration = [{
    /// Returns whether the instruction is valid for the given target and encoding.
    static LogicalResult isValid(TargetAttrInterface tgt, AMDGCNEncoding encoding, Adaptor adaptor);

    /// Returns the encoding of the instruction.
    FailureOr<Encoding> getEncoding(TargetAttrInterface tgt);

    /// Get the defined values of the instruction.
    ::mlir::LogicalResult livenessTransferFunction(
        LivenessCallback insertCallback,
        LivenessCallback removeCallback,
        IsLiveCallback isLiveCallback) {
      return ::mlir::aster::detail::livenessTransferFunctionImpl(
        *this, insertCallback, removeCallback, isLiveCallback);
    }
    ResultRange getInstResults() {
      return getInstInfo().getInstResults(getOperation()->getResults());
    }

    /// Returns the instruction properties.
    static const ::mlir::aster::amdgcn::InstructionProps *getInstProps();

    ::mlir::aster::amdgcn::InstAttr getOpcodeAttr() {
      return nullptr;
    }

    bool hasInstProps(ArrayRef<InstProp> props, bool all) {
      const InstructionProps *instProps = getInstProps();
      assert(instProps && "Instruction props are not set");
      return all ? instProps->hasProps(props) : instProps->hasAnyProps(props);
    }

    bool isNewInst() {
      return true;
    }

    ArrayRef<ISAVersion> getISAVersions();
  }];
  let extraConcreteClassDefinition = [{
    OperandRange $cppClass::getInstOuts() {
      return getInstInfo().getInstOuts(getOperation()->getOperands());
    }
    OperandRange $cppClass::getInstIns() {
      return getInstInfo().getInstIns(getOperation()->getOperands());
    }
    /// Returns the speculatability of the instruction.
    Speculation::Speculatability $cppClass::getSpeculatability() {
      return mlir::aster::detail::getInstSpeculatabilityImpl(*this);
    }
    InstOpInterface $cppClass::cloneInst(OpBuilder &builder, ValueRange outs, ValueRange ins) {
      return ::mlir::aster::detail::cloneInstructionImpl<$cppClass>(*this, builder, outs, ins);
    }

    /// FIXME: This method should be removed.
    ArrayRef<ISAVersion> $cppClass::getISAVersions() {
      static ISAVersion isas[] = {ISAVersion::CDNA3, ISAVersion::CDNA4};
      return isas;
    }
  }];
}

class AMDISAInstruction<string mnemonic, list<Trait> opTraits = []>
    : Instruction<AMDGCN_Dialect, mnemonic, opTraits # [
        AMDInstTrait,
        AMDGCNInstOpInterface,
      ]>, AMDInstOpCode<"_" # mnemonic, "_" # mnemonic> {
  list<InstProp> props = [];
  code instExtraClassDeclaration = "";
  code opcodeDecl = StrSubst<[{
    ::mlir::aster::amdgcn::OpCode getOpCode() {
      return ::mlir::aster::amdgcn::OpCode::_$mnemonic;
    }
  }], [VarRepl<"mnemonic", mnemonic>]>.result;
  let extraClassDecls = [opcodeDecl, instExtraClassDeclaration];
}

//===----------------------------------------------------------------------===//
// AMDGCN supported architectures.
//===----------------------------------------------------------------------===//

def CDNA3Isa : Arch<"CDNA3">;
def CDNA4Isa : Arch<"CDNA4">;

//===----------------------------------------------------------------------===//
// Encoding name records
//===----------------------------------------------------------------------===//

def ENC_DS : Encoding<"ds">;
def ENC_FLAT_GLBL : Encoding<"flat_glbl">;
def ENC_MUBUF : Encoding<"mubuf">;
def ENC_SMEM : Encoding<"smem">;
def ENC_SOP1 : Encoding<"sop1">;
def ENC_SOP2 : Encoding<"sop2">;
def ENC_SOPC : Encoding<"sopc">;
def ENC_SOPK : Encoding<"sopk">;
def ENC_SOPP : Encoding<"sopp">;
def ENC_VOP1 : Encoding<"vop1">;
def ENC_VOP2 : Encoding<"vop2">;
def ENC_VOP3 : Encoding<"vop3">;
def ENC_VOP3P : Encoding<"vop3p">;
def ENC_VOP3PX2 : Encoding<"vop3px2">;
def ENC_VOPC : Encoding<"vopc">;
def VOP3P_MFMA : Encoding<"vop3p_mfma">;

//===----------------------------------------------------------------------===//
// EncodedArch records
//===----------------------------------------------------------------------===//

def ENC_DS_CDNA3 : EncodedArch<ENC_DS, CDNA3Isa>;
def ENC_DS_CDNA4 : EncodedArch<ENC_DS, CDNA4Isa>;
def ENC_FLAT_GLBL_CDNA3 : EncodedArch<ENC_FLAT_GLBL, CDNA3Isa>;
def ENC_FLAT_GLBL_CDNA4 : EncodedArch<ENC_FLAT_GLBL, CDNA4Isa>;
def ENC_MUBUF_CDNA3 : EncodedArch<ENC_MUBUF, CDNA3Isa>;
def ENC_MUBUF_CDNA4 : EncodedArch<ENC_MUBUF, CDNA4Isa>;
def ENC_SMEM_CDNA3 : EncodedArch<ENC_SMEM, CDNA3Isa>;
def ENC_SMEM_CDNA4 : EncodedArch<ENC_SMEM, CDNA4Isa>;
def ENC_SOP1_CDNA3 : EncodedArch<ENC_SOP1, CDNA3Isa>;
def ENC_SOP1_CDNA4 : EncodedArch<ENC_SOP1, CDNA4Isa>;
def ENC_SOP2_CDNA3 : EncodedArch<ENC_SOP2, CDNA3Isa>;
def ENC_SOP2_CDNA4 : EncodedArch<ENC_SOP2, CDNA4Isa>;
def ENC_SOPC_CDNA3 : EncodedArch<ENC_SOPC, CDNA3Isa>;
def ENC_SOPC_CDNA4 : EncodedArch<ENC_SOPC, CDNA4Isa>;
def ENC_SOPK_CDNA3 : EncodedArch<ENC_SOPK, CDNA3Isa>;
def ENC_SOPK_CDNA4 : EncodedArch<ENC_SOPK, CDNA4Isa>;
def ENC_SOPP_CDNA3 : EncodedArch<ENC_SOPP, CDNA3Isa>;
def ENC_SOPP_CDNA4 : EncodedArch<ENC_SOPP, CDNA4Isa>;
def ENC_VOP1_CDNA3 : EncodedArch<ENC_VOP1, CDNA3Isa>;
def ENC_VOP1_CDNA4 : EncodedArch<ENC_VOP1, CDNA4Isa>;
def ENC_VOP2_CDNA3 : EncodedArch<ENC_VOP2, CDNA3Isa>;
def ENC_VOP2_CDNA4 : EncodedArch<ENC_VOP2, CDNA4Isa>;
def ENC_VOP3PX2_CDNA4 : EncodedArch<ENC_VOP3PX2, CDNA4Isa>;
def ENC_VOP3P_CDNA3 : EncodedArch<ENC_VOP3P, CDNA3Isa>;
def ENC_VOP3P_CDNA4 : EncodedArch<ENC_VOP3P, CDNA4Isa>;
def ENC_VOP3_CDNA3 : EncodedArch<ENC_VOP3, CDNA3Isa>;
def ENC_VOP3_CDNA4 : EncodedArch<ENC_VOP3, CDNA4Isa>;
def ENC_VOPC_CDNA3 : EncodedArch<ENC_VOPC, CDNA3Isa>;
def ENC_VOPC_CDNA4 : EncodedArch<ENC_VOPC, CDNA4Isa>;
def VOP3P_MFMA_CDNA3 : EncodedArch<VOP3P_MFMA, CDNA3Isa>;
def VOP3P_MFMA_CDNA4 : EncodedArch<VOP3P_MFMA, CDNA4Isa>;

//===----------------------------------------------------------------------===//
// Modifier attribute constraints
//===----------------------------------------------------------------------===//

def GDS : UnitModifier<"gds", "1=GDS, 0=LDS">;
def GLC : UnitModifier<"glc", "If set, operation is globally coherent.">;
def IDXEN : UnitModifier<"idxen", "If set, send VADDR as an index. If unset, treat the index as zero.">;
def LDS : UnitModifier<"lds", "If set, data is written to LDS memory. If unset, data is read from/written to a VGPR.">;
def NT : UnitModifier<"nt", "Non-Temporal: data is streamed or uncached.">;
def OFFEN : UnitModifier<"offen", "If set, send VADDR as an offset. If unset, send the instruction offset stored in OFFSET. Only one of these offsets may be sent.">;
def SC0 : UnitModifier<"sc0", "For regular load and store operations, SC0 and SC1 indicate memory scope such as wave, group, device and system. For atomics, SC0 means Atomic with return. 1 = Atomic with return, 0 = Atomic with no return.">;
def SC1 : UnitModifier<"sc1", "For regular load and store operations, SC0 and SC1 indicate memory scope such as wave, group, device and system.">;

//===----------------------------------------------------------------------===//
// Effects
//===----------------------------------------------------------------------===//

def GlobalRead : Effect {
  let summary = "Reads from global memory.";
  let body = [{
    // TODO: implement
  }];
}

#endif // AMDGCN_INSTBASE_TD
