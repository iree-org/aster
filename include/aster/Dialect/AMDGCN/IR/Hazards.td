//===- Hazards.td - AMDGCN hazard definitions -----------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines AMDGCN hazards.
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_HAZARDS_TD
#define ASTER_AMDGCN_HAZARDS_TD

include "mlir/IR/Utils.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/HazardAttrInterface.td"
include "aster/Support/Utils.td"


/// Helper class to define a hazard case.
class HazardCase<int _vWaits, int _sWaits, int _dWaits = 0> {
  int vWaits = _vWaits;
  int sWaits = _sWaits;
  int dWaits = _dWaits;
  string instCounts = StrSubst<[{InstCounts($vWaits, $sWaits, $dWaits)}], [
    IntRepl<"vWaits", vWaits>, IntRepl<"sWaits", sWaits>, IntRepl<"dWaits", dWaits>
  ]>.result;
}

/// Base class for hazard attributes.
class HazardAttrBase<string name, string attrMnemonic> :
  AMDGCN_Attr<name, attrMnemonic, [
    DeclareAttrInterfaceMethods<HazardRaiserAttrInterface>,
    DeclareAttrInterfaceMethods<HazardCheckerAttrInterface>
  ]> {
}

/// A hazard attribute with a list of hazard cases, each case has a unique
/// required instruction count. Therefore, the attribute is parameterized by
/// the index of the hazard case to use.
class HazardAttr<string name, string attrMnemonic, list<HazardCase> _hazardCases = []> :
    HazardAttrBase<name, attrMnemonic> {
  // The index of the hazard case to use.
  let parameters = (ins DefaultValuedParameter<"int16_t", "0">:$case_num);
  let assemblyFormat = "(`<` $case_num^ `>`)?";

  // The list of hazard cases.
  list<HazardCase> hazardCases = _hazardCases;

  // Create an array with all the required instruction counts for each case.
  string casesArray = !interleave(!foreach(case, hazardCases, case.instCounts), ", ");
  string cases = StrSubst<[{
    /// An array with all the required instruction counts for each case.
    static constexpr std::array<InstCounts, $numCases> kRequiredInstCounts
      = {$cases};

    /// Get the required instruction counts for the given case index.
    static constexpr InstCounts getInstCounts(int16_t caseNum = 0) {
      if (caseNum < 0 || caseNum >= static_cast<int16_t>(kRequiredInstCounts.size()))
        return {};
      return kRequiredInstCounts[caseNum];
    }
  }], [
    VarRepl<"cases", casesArray>, IntRepl<"numCases", !size(hazardCases)>
  ]>.result;

  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins CArg<"int16_t", "0">:$caseNum), [{
      return $_get($_ctxt, caseNum);
    }]>
  ];

  // Set the extra class declaration to the member declaration of the array.
  let extraClassDeclaration = cases;

  // Verify the case index is valid.
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult $cppClass::verify(
      function_ref<InFlightDiagnostic()> emitError, int16_t caseNum) {
      if (caseNum < 0 || caseNum >= static_cast<int16_t>(kRequiredInstCounts.size())) {
        return emitError() << "invalid hazard case index: " << caseNum;
      }
      return success();
    }
  }];
}

//===----------------------------------------------------------------------===//
// AllHazardAttr
//===----------------------------------------------------------------------===//

def AllHazardAttr : HazardAttrBase<"AllHazard", "all_hazard"> {
  let summary = "Hazard that matches all instructions";
  let parameters = (ins "int16_t":$v_nops, "int16_t":$s_nops);
  let assemblyFormat = "`<` struct($v_nops, $s_nops) `>`";
}

//===----------------------------------------------------------------------===//
// CDNA3 Hazards (CDNA3 ISA Section 4.5 - Manually Inserted Wait States)
//===----------------------------------------------------------------------===//

//--- Case 1: S_SETREG <*> -> S_GETREG (2 S_NOPs) ---
// TODO: S_SETREG and S_GETREG instructions not implemented in AMDGCN.
def CDNA3_SetRegGetRegHazard : HazardAttr<"CDNA3SetRegGetRegHazard",
    "cdna3_setreg_getreg_hazard"> {
  let summary = "CDNA3 S_SETREG followed by S_GETREG (same reg)";
  let description = [{
    S_SETREG followed by S_GETREG reading the same register requires 2 wait states.
    TODO: S_SETREG and S_GETREG instructions not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/0, /*s_nops=*/2, /*ds_nops=*/0>
  ];
}

//--- Case 2: S_SETREG <*> -> S_SETREG (same reg) (2 S_NOPs) ---
// TODO: S_SETREG instruction not implemented in AMDGCN.
def CDNA3_SetRegSetRegHazard : HazardAttr<"CDNA3SetRegSetRegHazard",
    "cdna3_setreg_setreg_hazard"> {
  let summary = "CDNA3 S_SETREG followed by S_SETREG (same reg)";
  let description = [{
    S_SETREG followed by S_SETREG writing the same register requires 2 wait states.
    TODO: S_SETREG instruction not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/0, /*s_nops=*/2, /*ds_nops=*/0>
  ];
}

//--- Case 3: SET_VSKIP -> S_GETREG MODE (2 S_NOPs) ---
// TODO: SET_VSKIP and S_GETREG instructions not implemented in AMDGCN.
def CDNA3_SetVskipGetRegHazard : HazardAttr<"CDNA3SetVskipGetRegHazard",
    "cdna3_setvskip_getreg_hazard"> {
  let summary = "CDNA3 SET_VSKIP followed by S_GETREG MODE";
  let description = [{
    SET_VSKIP followed by S_GETREG reading VSKIP from MODE requires 2 wait states.
    TODO: SET_VSKIP and S_GETREG instructions not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/0, /*s_nops=*/2, /*ds_nops=*/0>
  ];
}

//--- Case 4: S_SETREG MODE.vskip -> any vector op (2 S_NOPs) ---
// TODO: S_SETREG instruction not implemented in AMDGCN.
def CDNA3_SetRegVskipVectorHazard : HazardAttr<"CDNA3SetRegVskipVectorHazard",
    "cdna3_setreg_vskip_vector_hazard"> {
  let summary = "CDNA3 S_SETREG MODE.vskip followed by vector op";
  let description = [{
    S_SETREG MODE.vskip followed by any vector op requires 2 nops or non-vector
    instructions. TODO: S_SETREG instruction not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/0, /*s_nops=*/2, /*ds_nops=*/0>
  ];
}

//--- Case 5: VALU that sets VCC or EXEC -> VALU that uses EXECZ or VCCZ (5 V_NOPs) ---
def CDNA3_VccExecVcczExeczHazard : HazardAttr<"CDNA3VccExecVcczExeczHazard",
    "cdna3_vcc_exec_vccz_execz_hazard"> {
  let summary = "CDNA3 VALU sets VCC/EXEC followed by VALU uses EXECZ/VCCZ";
  let description = [{
    VALU that sets VCC or EXEC followed by VALU that uses EXECZ or VCCZ as data
    source requires 5 wait states.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/5, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Case 6: VALU writes SGPR/VCC -> V_[READ,WRITE]LANE lane select (4 V_NOPs) ---
// TODO: V_READLANE, V_WRITELANE instructions not implemented in AMDGCN.
def CDNA3_ValuSgprReadlaneHazard : HazardAttr<"CDNA3ValuSgprReadlaneHazard",
    "cdna3_valu_sgpr_readlane_hazard"> {
  let summary = "CDNA3 VALU writes SGPR/VCC followed by V_READLANE/V_WRITELANE";
  let description = [{
    VALU writes SGPR/VCC (readlane, cmp, add/sub, div_scale) followed by
    V_READLANE/V_WRITELANE using that SGPR/VCC as lane select requires 4 wait states.
    TODO: V_READLANE, V_WRITELANE instructions not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/4, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Case 7: VALU writes VCC -> V_DIV_FMAS (4 V_NOPs) ---
// TODO: V_DIV_FMAS instruction not implemented in AMDGCN.
def CDNA3_VccDivFmasHazard : HazardAttr<"CDNA3VccDivFmasHazard",
    "cdna3_vcc_div_fmas_hazard"> {
  let summary = "CDNA3 VALU writes VCC followed by V_DIV_FMAS";
  let description = [{
    VALU writes VCC (including v_div_scale) followed by V_DIV_FMAS requires 4 wait
    states. TODO: V_DIV_FMAS instruction not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/4, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Case 8: Store/atomic X3/X4 -> Write VGPRs holding writedata (1 V_NOP) ---
// Implemented for GLOBAL_STORE_DWORDX3/X4, BUFFER_STORE_DWORDX3/X4.
// TODO: FLAT_ATOMIC_[F]CMPSWAP_X2, BUFFER_STORE_FORMAT_XYZ/XYZW,
// BUFFER_ATOMIC_[F]CMPSWAP_X2 not implemented in AMDGCN.
def CDNA3_StoreWriteDataHazard : HazardAttr<"CDNA3StoreWriteDataHazard",
    "cdna3_store_write_data_hazard"> {
  let summary = "CDNA3 store/atomic X3/X4 followed by write to writedata VGPRs";
  let description = [{
    FLAT_STORE_X3/X4, BUFFER_STORE_*_X3/X4, FLAT_ATOMIC_[F]CMPSWAP_X2, etc.
    followed by instruction that writes VGPRs holding writedata requires 1 wait
    state. BUFFER_STORE_* with SGPR offset does not require wait states.
    TODO: FLAT_ATOMIC, BUFFER_STORE_FORMAT, BUFFER_ATOMIC not implemented.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/1, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Case 9: Store/atomic X3/X4 -> VALU writes VGPRs holding writedata (2 V_NOPs) ---
// Implemented: CDNA3_StoreHazard (GLOBAL_STORE, BUFFER_STORE_DWORDX3/X4).
def CDNA3_StoreHazard : HazardAttr<"CDNA3StoreHazard",
    "cdna3_store_hazard"> {
  let summary = "CDNA3 store hazard followed by VALU write";
  let description = [{
    This hazard is raised when a store operation (FLAT_STORE_X3/X4, BUFFER_STORE_*,
    etc.) is followed by a VALU instruction overwriting the write data of the store.
    TODO: FLAT_ATOMIC, BUFFER_STORE_FORMAT, BUFFER_ATOMIC not implemented.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/2, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Case 10: VALU writes SGPR -> VMEM reads that SGPR (5 V_NOPs) ---
def CDNA3_ValuSgprVmemHazard : HazardAttr<"CDNA3ValuSgprVmemHazard",
    "cdna3_valu_sgpr_vmem_hazard"> {
  let summary = "CDNA3 VALU writes SGPR followed by VMEM reads that SGPR";
  let description = [{
    VALU writes SGPR followed by VMEM that reads that SGPR requires 5 wait states.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/5, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Case 11: SALU writes M0 -> GDS, S_SENDMSG (1 S_NOP) ---
// TODO: GDS and S_SENDMSG instructions not implemented in AMDGCN.
def CDNA3_SaluM0GdsSendmsgHazard : HazardAttr<"CDNA3SaluM0GdsSendmsgHazard",
    "cdna3_salu_m0_gds_sendmsg_hazard"> {
  let summary = "CDNA3 SALU writes M0 followed by GDS or S_SENDMSG";
  let description = [{
    SALU writes M0 followed by GDS or S_SENDMSG requires 1 wait state.
    TODO: GDS and S_SENDMSG instructions not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/0, /*s_nops=*/1, /*ds_nops=*/0>
  ];
}

//--- Case 12: VALU writes VGPR -> VALU DPP reads that VGPR (2 V_NOPs) ---
// TODO: DPP (Data Parallel Primitives) modifier not implemented in AMDGCN.
def CDNA3_ValuVgprDppHazard : HazardAttr<"CDNA3ValuVgprDppHazard",
    "cdna3_valu_vgpr_dpp_hazard"> {
  let summary = "CDNA3 VALU writes VGPR followed by VALU DPP reads that VGPR";
  let description = [{
    VALU writes VGPR followed by VALU DPP that reads that VGPR requires 2 wait
    states. TODO: DPP modifier not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/2, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Case 13: VALU writes EXEC -> VALU DPP op (5 V_NOPs) ---
// TODO: DPP modifier and EXEC forwarding not implemented in AMDGCN.
def CDNA3_ExecDppHazard : HazardAttr<"CDNA3ExecDppHazard",
    "cdna3_exec_dpp_hazard"> {
  let summary = "CDNA3 VALU writes EXEC followed by VALU DPP op";
  let description = [{
    VALU writes EXEC followed by VALU DPP op requires 5 wait states. ALU does not
    forward EXEC to DPP. TODO: DPP modifier not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/5, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Case 14: Mixed VCC use -> VALU reads VCC as constant (1 V_NOP) ---
// TODO: VCC alias vs SGPR# tracking not implemented in AMDGCN.
def CDNA3_MixedVccConstantHazard : HazardAttr<"CDNA3MixedVccConstantHazard",
    "cdna3_mixed_vcc_constant_hazard"> {
  let summary = "CDNA3 Mixed VCC use followed by VALU reads VCC as constant";
  let description = [{
    Mixed use of VCC (alias vs SGPR#) from v_readlane, v_cmp, v_add, v_sub,
    v_div_scale followed by VALU reading VCC as constant requires 1 wait state.
    Carry-in is 0 wait states. TODO: VCC/SGPR alias tracking not implemented.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/1, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Case 15: S_SETREG TRAPSTS -> RFE, RFE_restore (1 S_NOP) ---
// TODO: S_SETREG, RFE, RFE_restore instructions not implemented in AMDGCN.
def CDNA3_SetRegTrapstsRfeHazard : HazardAttr<"CDNA3SetRegTrapstsRfeHazard",
    "cdna3_setreg_trapsts_rfe_hazard"> {
  let summary = "CDNA3 S_SETREG TRAPSTS followed by RFE/RFE_restore";
  let description = [{
    S_SETREG TRAPSTS followed by RFE or RFE_restore requires 1 wait state.
    TODO: S_SETREG, RFE, RFE_restore instructions not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/0, /*s_nops=*/1, /*ds_nops=*/0>
  ];
}

//--- Case 16: SALU writes M0 -> LDS add-TID, buffer_store_LDS, scratch/global LDS=1 (1 S_NOP) ---
// TODO: LDS add-TID, buffer_store_LDS_dword not implemented in AMDGCN.
def CDNA3_SaluM0LdsHazard : HazardAttr<"CDNA3SaluM0LdsHazard",
    "cdna3_salu_m0_lds_hazard"> {
  let summary = "CDNA3 SALU writes M0 followed by LDS add-TID or buffer_store_LDS";
  let description = [{
    SALU writes M0 followed by LDS add-TID instruction, buffer_store_LDS_dword,
    or scratch/global with LDS=1 requires 1 wait state.
    TODO: LDS add-TID, buffer_store_LDS_dword not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/0, /*s_nops=*/1, /*ds_nops=*/0>
  ];
}

//--- Case 17: SALU writes M0 -> S_MOVEREL (1 S_NOP) ---
// TODO: S_MOVEREL instruction not implemented in AMDGCN.
def CDNA3_SaluM0MoverelHazard : HazardAttr<"CDNA3SaluM0MoverelHazard",
    "cdna3_salu_m0_moverel_hazard"> {
  let summary = "CDNA3 SALU writes M0 followed by S_MOVEREL";
  let description = [{
    SALU writes M0 followed by S_MOVEREL requires 1 wait state.
    TODO: S_MOVEREL instruction not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/0, /*s_nops=*/1, /*ds_nops=*/0>
  ];
}

//--- Case 18: VALU writes SGPR/VCC -> VALU reads SGPR (2 or 4 V_NOPs) ---
// Subcases: constant 2, lane-select 4, carry-in 0, v_cmpx->EXEC 2, readlane 4.
// TODO: VALU SGPR/VCC write tracking and v_readlane/v_writelane not implemented.
def CDNA3_ValuSgprValuReadHazard : HazardAttr<"CDNA3ValuSgprValuReadHazard",
    "cdna3_valu_sgpr_valu_read_hazard"> {
  let summary = "CDNA3 VALU writes SGPR/VCC followed by VALU reads SGPR";
  let description = [{
    VALU writes SGPR/VCC (readlane, cmp, add/sub, div_scale) followed by VALU
    reading SGPR: as constant 2 waits, as lane-select 4 waits, carry-in 0 waits.
    TODO: SGPR/VCC dependency tracking not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/2, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Case 19: VALU writes VGPRn -> v_readlane vsrc0 reads VGPRn (1 V_NOP) ---
// TODO: V_READLANE instruction not implemented in AMDGCN.
def CDNA3_ValuVgprReadlaneHazard : HazardAttr<"CDNA3ValuVgprReadlaneHazard",
    "cdna3_valu_vgpr_readlane_hazard"> {
  let summary = "CDNA3 VALU writes VGPRn followed by v_readlane vsrc0 reads VGPRn";
  let description = [{
    VALU writes VGPRn followed by v_readlane with vsrc0 reading VGPRn requires 1
    wait state. TODO: V_READLANE instruction not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/1, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Case 20: VALU OPSEL/SDWA -> VALU consumes result (1 V_NOP) ---
// TODO: OPSEL and SDWA modifiers not implemented in AMDGCN.
def CDNA3_OpselSdwaHazard : HazardAttr<"CDNA3OpselSdwaHazard",
    "cdna3_opsel_sdwa_hazard"> {
  let summary = "CDNA3 VALU OPSEL/SDWA followed by VALU consumes result";
  let description = [{
    VALU op using OPSEL or SDWA that changes result bit position, followed by
    VALU op consuming that result, requires 1 wait state.
    TODO: OPSEL and SDWA modifiers not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/1, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Case 21: VALU Trans op -> Non-trans VALU consumes result (1 V_NOP) ---
// TODO: Trans op detection (V_EXP_F32, V_LOG_F32, etc.) not implemented.
def CDNA3_TransOpHazard : HazardAttr<"CDNA3TransOpHazard",
    "cdna3_trans_op_hazard"> {
  let summary = "CDNA3 VALU Trans op followed by Non-trans VALU consumes result";
  let description = [{
    VALU Trans op (V_EXP_F32, V_LOG_F32, V_RCP_F32, etc.) followed by non-Trans
    VALU op consuming the result requires 1 wait state.
    TODO: Trans op detection not implemented in AMDGCN.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/1, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//===----------------------------------------------------------------------===//
// CDNA3 Section 7.5 - Dependency Resolution: Required Independent Instructions
// (Matrix operations - VOP3P-Matrix, XDL, SGEMM, DGEMM)
//===----------------------------------------------------------------------===//

//--- Table 37: Non-DLops VALU Write VGPR -> V_MFMA* read VGPR OR V_SMFMA* read VGPR (2 V_NOPs) ---
def CDNA3_NonDLOpsValuMfmaHazard : HazardAttr<"CDNA3NonDLOpsValuMfmaHazard",
    "cdna3_nondlops_valu_mfma_hazard"> {
  let summary = "CDNA3 Non-DLops VALU Write VGPR followed by V_MFMA*/V_SMFMA* read VGPR";
  let description = [{
    Non-DLops VALU Write VGPR followed by V_MFMA* read VGPR OR V_SMFMA* read VGPR
    requires 2 wait states. No internal 4 & 8 cycle forwarding path.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/2, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Table 37: DL ops Write VGPR (3 subcases: SrcC same opcode 0, SrcA/B same opcode 3, RAW+WAW diff opcode 3) ---
def CDNA3_DLOpsWriteVgprHazard : HazardAttr<"CDNA3DLOpsWriteVgprHazard",
    "cdna3_dlops_write_vgpr_hazard"> {
  let summary = "CDNA3 DL ops Write VGPR followed by DLops read/write VGPR";
  let description = [{
    DL ops Write VGPR followed by: (case 0) DLops read VGPR as SrcC, same opcode - 0 waits;
    (case 1) DLops read VGPR as SrcA/B, same opcode - 3 waits; (case 2) Any opcode read/write
    VGPR RAW+WAW different from 1st DLops - 3 waits.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/0, /*s_nops=*/0, /*ds_nops=*/0>,   // SrcC same opcode
    HazardCase</*v_nops=*/3, /*s_nops=*/0, /*ds_nops=*/0>,   // SrcA/B same opcode
    HazardCase</*v_nops=*/3, /*s_nops=*/0, /*ds_nops=*/0>    // RAW+WAW diff opcode
  ];
}

//--- Table 37: XDL Write VGPR or V_SMFMA* Write VGPR -> XDL read VGPR as Source C exactly same (2/0/0/0 by passes) ---
def CDNA3_XdlWriteVgprXdlReadSrcCExactHazard : HazardAttr<"CDNA3XdlWriteVgprXdlReadSrcCExactHazard",
    "cdna3_xdl_write_vgpr_xdl_read_srcc_exact_hazard"> {
  let summary = "CDNA3 XDL/V_SMFMA* Write VGPR followed by XDL read VGPR as SrcC exactly same";
  let description = [{
    XDL Write VGPR or V_SMFMA* Write VGPR followed by XDL read VGPR as Source C exactly
    same with 1st vDst. Case 0: 2 passes - 2 waits; Case 1: 4 passes - 0; Case 2: 8 - 0; Case 3: 16 - 0.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/2, /*s_nops=*/0, /*ds_nops=*/0>,   // 2 passes
    HazardCase</*v_nops=*/0, /*s_nops=*/0, /*ds_nops=*/0>,   // 4 passes
    HazardCase</*v_nops=*/0, /*s_nops=*/0, /*ds_nops=*/0>,   // 8 passes
    HazardCase</*v_nops=*/0, /*s_nops=*/0, /*ds_nops=*/0>    // 16 passes
  ];
}

//--- Table 37: XDL Write VGPR or V_SMFMA* Write VGPR -> XDL read VGPR as Source C overlapped (3/5/9/17 by passes) ---
def CDNA3_XdlWriteVgprXdlReadSrcCOverlapHazard : HazardAttr<"CDNA3XdlWriteVgprXdlReadSrcCOverlapHazard",
    "cdna3_xdl_write_vgpr_xdl_read_srcc_overlap_hazard"> {
  let summary = "CDNA3 XDL/V_SMFMA* Write VGPR followed by XDL read VGPR as SrcC overlapped";
  let description = [{
    XDL Write VGPR or V_SMFMA* Write VGPR followed by XDL read VGPR as Source C overlapped
    with 1st vDst. Case 0: 2 passes - 3 waits; Case 1: 4 - 5; Case 2: 8 - 9; Case 3: 16 - 17.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/3, /*s_nops=*/0, /*ds_nops=*/0>,   // 2 passes
    HazardCase</*v_nops=*/5, /*s_nops=*/0, /*ds_nops=*/0>,   // 4 passes
    HazardCase</*v_nops=*/9, /*s_nops=*/0, /*ds_nops=*/0>,   // 8 passes
    HazardCase</*v_nops=*/17, /*s_nops=*/0, /*ds_nops=*/0>  // 16 passes
  ];
}

//--- Table 37: XDL Write VGPR or V_SMFMA* Write VGPR -> S/DGEMM read VGPR as Source C (3/5/9/17 by passes) ---
def CDNA3_XdlWriteVgprSdgemmReadSrcCHazard : HazardAttr<"CDNA3XdlWriteVgprSdgemmReadSrcCHazard",
    "cdna3_xdl_write_vgpr_sdgemm_read_srcc_hazard"> {
  let summary = "CDNA3 XDL/V_SMFMA* Write VGPR followed by S/DGEMM read VGPR as Source C";
  let description = [{
    XDL Write VGPR or V_SMFMA* Write VGPR followed by S/DGEMM read VGPR as Source C overlapped.
    Case 0: 2 passes - 3; Case 1: 4 - 5; Case 2: 8 - 9; Case 3: 16 - 17.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/3, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/5, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/9, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/17, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Table 37: XDL Write VGPR or V_SMFMA* Write VGPR -> V_MFMA read VGPR as SrcA/SrcB (5/7/11/19 by passes) ---
def CDNA3_XdlWriteVgprMfmaReadSrcABHazard : HazardAttr<"CDNA3XdlWriteVgprMfmaReadSrcABHazard",
    "cdna3_xdl_write_vgpr_mfma_read_srcab_hazard"> {
  let summary = "CDNA3 XDL/V_SMFMA* Write VGPR followed by V_MFMA read VGPR as SrcA/SrcB";
  let description = [{
    XDL Write VGPR or V_SMFMA* Write VGPR followed by V_MFMA read VGPR as SrcA or SrcB
    OR V_SMFMA* read VGPR as SrcA or SrcB or Index SrcC. No internal forwarding path.
    Case 0: 2 passes - 5; Case 1: 4 - 7; Case 2: 8 - 11; Case 3: 16 - 19.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/5, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/7, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/11, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/19, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Table 37: XDL Write VGPR or V_SMFMA* Write VGPR -> VM/L/GDS/FLAT/Export Read VGPR or VALU RAW+WAW (5/7/11/19 by passes) ---
def CDNA3_XdlWriteVgprVmemValuHazard : HazardAttr<"CDNA3XdlWriteVgprVmemValuHazard",
    "cdna3_xdl_write_vgpr_vmem_valu_hazard"> {
  let summary = "CDNA3 XDL/V_SMFMA* Write VGPR followed by VM/LDS/FLAT/Export Read or VALU RAW+WAW";
  let description = [{
    XDL Write VGPR or V_SMFMA* Write VGPR followed by (1) VM, L/GDS, FLAT, Export Read VGPR
    overlapped with 1st vDst, or (2) VALU read/write VGPR (RAW+WAW). Case 0: 2 passes - 5;
    Case 1: 4 - 7; Case 2: 8 - 11; Case 3: 16 - 19.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/5, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/7, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/11, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/19, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Table 37: SGEMM Write VGPR -> XDL read VGPR as Source C exactly same (0 waits) ---
def CDNA3_SgemmWriteVgprXdlReadSrcCExactHazard : HazardAttr<"CDNA3SgemmWriteVgprXdlReadSrcCExactHazard",
    "cdna3_sgemm_write_vgpr_xdl_read_srcc_exact_hazard"> {
  let summary = "CDNA3 SGEMM Write VGPR followed by XDL read VGPR as SrcC exactly same";
  let description = [{
    SGEMM Write VGPR followed by XDL read VGPR as Source C exactly same with 1st vDst.
    The two V_MFMA must be same number passes, vDst and vSrc from same offset and VGPR size.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/0, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Table 37: SGEMM Write VGPR -> XDL read VGPR as Source C overlapped (2/4/8/16 by passes) ---
def CDNA3_SgemmWriteVgprXdlReadSrcCOverlapHazard : HazardAttr<"CDNA3SgemmWriteVgprXdlReadSrcCOverlapHazard",
    "cdna3_sgemm_write_vgpr_xdl_read_srcc_overlap_hazard"> {
  let summary = "CDNA3 SGEMM Write VGPR followed by XDL read VGPR as SrcC overlapped";
  let description = [{
    SGEMM Write VGPR followed by XDL read VGPR as Source C overlapped with 1st vDst.
    Case 0: 2 passes - 2; Case 1: 4 - 4; Case 2: 8 - 8; Case 3: 16 - 16.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/2, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/4, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/8, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/16, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Table 37: SGEMM Write VGPR -> S/DGEMM read VGPR as Source C (2/4/8/16 by passes) ---
def CDNA3_SgemmWriteVgprSdgemmReadSrcCHazard : HazardAttr<"CDNA3SgemmWriteVgprSdgemmReadSrcCHazard",
    "cdna3_sgemm_write_vgpr_sdgemm_read_srcc_hazard"> {
  let summary = "CDNA3 SGEMM Write VGPR followed by S/DGEMM read VGPR as Source C";
  let description = [{
    SGEMM Write VGPR followed by S/DGEMM read VGPR as Source C overlapped.
    Case 0: 2 passes - 2; Case 1: 4 - 4; Case 2: 8 - 8; Case 3: 16 - 16.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/2, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/4, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/8, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/16, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Table 37: SGEMM Write VGPR -> V_MFMA read VGPR as SrcA/SrcB (4/6/10/18 by passes) ---
def CDNA3_SgemmWriteVgprMfmaReadSrcABHazard : HazardAttr<"CDNA3SgemmWriteVgprMfmaReadSrcABHazard",
    "cdna3_sgemm_write_vgpr_mfma_read_srcab_hazard"> {
  let summary = "CDNA3 SGEMM Write VGPR followed by V_MFMA read VGPR as SrcA/SrcB";
  let description = [{
    SGEMM Write VGPR followed by V_MFMA read VGPR as SrcA or SrcB OR V_SMFMA* read VGPR
    as SrcA or SrcB or Index SrcC. No internal forwarding path. Case 0: 2 - 4; Case 1: 4 - 6;
    Case 2: 8 - 10; Case 3: 16 - 18.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/4, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/6, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/10, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/18, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Table 37: SGEMM Write VGPR -> VM/L/GDS/FLAT/Export Read VGPR or VALU RAW+WAW (4/6/10/18 by passes) ---
def CDNA3_SgemmWriteVgprVmemValuHazard : HazardAttr<"CDNA3SgemmWriteVgprVmemValuHazard",
    "cdna3_sgemm_write_vgpr_vmem_valu_hazard"> {
  let summary = "CDNA3 SGEMM Write VGPR followed by VM/LDS/FLAT/Export Read or VALU RAW+WAW";
  let description = [{
    SGEMM Write VGPR followed by (1) VM, L/GDS, FLAT, Export Read VGPR overlapped with 1st
    vDst, or (2) VALU read/write VGPR (RAW+WAW). Case 0: 2 - 4; Case 1: 4 - 6; Case 2: 8 - 10;
    Case 3: 16 - 18.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/4, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/6, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/10, /*s_nops=*/0, /*ds_nops=*/0>,
    HazardCase</*v_nops=*/18, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Table 37: V_MFMA_16x16x4_F64 Write VGPR (multiple subcases) ---
def CDNA3_Mfma16x16x4F64WriteVgprHazard : HazardAttr<"CDNA3Mfma16x16x4F64WriteVgprHazard",
    "cdna3_mfma_16x16x4_f64_write_vgpr_hazard"> {
  let summary = "CDNA3 V_MFMA_16x16x4_F64 Write VGPR followed by various consumers";
  let description = [{
    V_MFMA_16x16x4_F64 Write VGPR followed by: (0) same MFMA read SrcC exactly same - 0;
    (1) S/DGEMM read SrcC overlapped - 9; (2) XDL read SrcC overlapped - 0; (3) V_SMFMA* Matrix C overlapped - 0;
    (4) S/DGEMM read SrcA/SrcB - 11; (5) XDL read SrcA/SrcB - 11; (6) V_SMFMA* SrcA/SrcB/Index SrcC - 11;
    (7) VALU RAW+WAW - 11; (8) VM/L/GDS/FLAT/Export Read overlapped - 18.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/0, /*s_nops=*/0, /*ds_nops=*/0>,   // same MFMA SrcC exact
    HazardCase</*v_nops=*/9, /*s_nops=*/0, /*ds_nops=*/0>,   // S/DGEMM SrcC overlap
    HazardCase</*v_nops=*/0, /*s_nops=*/0, /*ds_nops=*/0>,   // XDL SrcC overlap
    HazardCase</*v_nops=*/0, /*s_nops=*/0, /*ds_nops=*/0>,   // V_SMFMA* Matrix C overlap
    HazardCase</*v_nops=*/11, /*s_nops=*/0, /*ds_nops=*/0>,  // S/DGEMM SrcA/SrcB
    HazardCase</*v_nops=*/11, /*s_nops=*/0, /*ds_nops=*/0>,  // XDL SrcA/SrcB
    HazardCase</*v_nops=*/11, /*s_nops=*/0, /*ds_nops=*/0>,  // V_SMFMA* SrcA/SrcB/Index
    HazardCase</*v_nops=*/11, /*s_nops=*/0, /*ds_nops=*/0>,  // VALU RAW+WAW
    HazardCase</*v_nops=*/18, /*s_nops=*/0, /*ds_nops=*/0>   // VM/L/GDS/FLAT/Export
  ];
}

//--- Table 37: V_MFMA_4x4x4_F64 Write VGPR (multiple subcases) ---
def CDNA3_Mfma4x4x4F64WriteVgprHazard : HazardAttr<"CDNA3Mfma4x4x4F64WriteVgprHazard",
    "cdna3_mfma_4x4x4_f64_write_vgpr_hazard"> {
  let summary = "CDNA3 V_MFMA_4x4x4_F64 Write VGPR followed by various consumers";
  let description = [{
    V_MFMA_4x4x4_F64 Write VGPR followed by: (0) same MFMA read SrcC exactly same - 4;
    (1) S/DGEMM read SrcC overlapped - 4; (2) XDL read SrcC overlapped - 0; (3) V_SMFMA Matrix C overlapped - 0;
    (4) S/DGEMM read SrcA/SrcB - 6; (5) XDL read SrcA/SrcB - 6; (6) V_SMFMA* SrcA/SrcB/Index - 6;
    (7) VALU RAW+WAW - 6; (8) VM/L/GDS/FLAT/Export Read overlapped - 9.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/4, /*s_nops=*/0, /*ds_nops=*/0>,   // same MFMA SrcC exact
    HazardCase</*v_nops=*/4, /*s_nops=*/0, /*ds_nops=*/0>,   // S/DGEMM SrcC overlap
    HazardCase</*v_nops=*/0, /*s_nops=*/0, /*ds_nops=*/0>,   // XDL SrcC overlap
    HazardCase</*v_nops=*/0, /*s_nops=*/0, /*ds_nops=*/0>,   // V_SMFMA Matrix C overlap
    HazardCase</*v_nops=*/6, /*s_nops=*/0, /*ds_nops=*/0>,   // S/DGEMM SrcA/SrcB
    HazardCase</*v_nops=*/6, /*s_nops=*/0, /*ds_nops=*/0>,  // XDL SrcA/SrcB
    HazardCase</*v_nops=*/6, /*s_nops=*/0, /*ds_nops=*/0>,  // V_SMFMA* SrcA/SrcB/Index
    HazardCase</*v_nops=*/6, /*s_nops=*/0, /*ds_nops=*/0>,  // VALU RAW+WAW
    HazardCase</*v_nops=*/9, /*s_nops=*/0, /*ds_nops=*/0>   // VM/L/GDS/FLAT/Export
  ];
}

//--- Table 37: V_CMPX* write EXEC MASK -> V_MFMA* (4 V_NOPs) ---
def CDNA3_VcmpxExecMfmaHazard : HazardAttr<"CDNA3VcmpxExecMfmaHazard",
    "cdna3_vcmpx_exec_mfma_hazard"> {
  let summary = "CDNA3 V_CMPX* write EXEC MASK followed by V_MFMA*";
  let description = [{
    V_CMPX* write EXEC MASK followed by V_MFMA* requires 4 wait states.
    CDNA does not support execution mask forwarding with XDL/DGEMM.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/4, /*s_nops=*/0, /*ds_nops=*/0>
  ];
}

//--- Table 37: XDL/SMFMA Read VGPR SrcC -> VALU write VGPR (WAR, Co-Execution Anti-Dependency) (1/3/7/15 by passes) ---
def CDNA3_XdlSmfmaReadSrcCValuWriteHazard : HazardAttr<"CDNA3XdlSmfmaReadSrcCValuWriteHazard",
    "cdna3_xdl_smfma_read_srcc_valu_write_hazard"> {
  let summary = "CDNA3 XDL/SMFMA Read VGPR SrcC followed by VALU write VGPR (WAR)";
  let description = [{
    XDL/SMFMA Read VGPR SrcC followed by VALU write VGPR (WAR), Co-Execution Anti-Dependency
    for overlapping with 1st SrcC. Case 0: 2 passes - 1; Case 1: 4 - 3; Case 2: 8 - 7; Case 3: 16 - 15.
  }];
  let hazardCases = [
    HazardCase</*v_nops=*/1, /*s_nops=*/0, /*ds_nops=*/0>,   // 2 passes
    HazardCase</*v_nops=*/3, /*s_nops=*/0, /*ds_nops=*/0>,  // 4 passes
    HazardCase</*v_nops=*/7, /*s_nops=*/0, /*ds_nops=*/0>,  // 8 passes
    HazardCase</*v_nops=*/15, /*s_nops=*/0, /*ds_nops=*/0>  // 16 passes
  ];
}

#endif // ASTER_AMDGCN_HAZARDS_TD
