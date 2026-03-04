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
include "aster/Support/EnumBase.td"
include "aster/Support/Utils.td"

class HazardAttr<string name, string attrMnemonic> :
  AMDGCN_Attr<name, attrMnemonic, [
    DeclareAttrInterfaceMethods<HazardAttrInterface>
  ]> {
}

/// Hazard attribute with configurable V (vector) and S (scalar) wait counts.
/// CDNA3 ISA Table 11: Most hazards use V_NOPs; SALU-related hazards use S_NOPs.
class HazardAttrWithWaits<string name, string attrMnemonic, int vWaits, int sWaits> :
  AMDGCN_Attr<name, attrMnemonic, [
    DeclareAttrInterfaceMethods<HazardAttrInterface>
  ]> {
  let extraClassDeclaration = StrSubst<[{
    static constexpr int32_t requiredVWaits = $vWaits;
    static constexpr int32_t requiredSWaits = $sWaits;
  }], [
    IntRepl<"vWaits", vWaits>, IntRepl<"sWaits", sWaits>
  ]>.result;
}

//===----------------------------------------------------------------------===//
// CDNA3 Hazards (CDNA3 ISA Section 4.5 - Manually Inserted Wait States)
//===----------------------------------------------------------------------===//

//--- Case 1: S_SETREG <*> -> S_GETREG (2 S_NOPs) ---
// TODO: S_SETREG and S_GETREG instructions not implemented in AMDGCN.
def CDNA3_SetRegGetRegHazard : HazardAttrWithWaits<"CDNA3SetRegGetRegHazard",
    "cdna3_setreg_getreg_hazard", 0, 2> {
  let summary = "CDNA3 S_SETREG followed by S_GETREG (same reg)";
  let description = [{
    S_SETREG followed by S_GETREG reading the same register requires 2 wait states.
    TODO: S_SETREG and S_GETREG instructions not implemented in AMDGCN.
  }];
}

//--- Case 2: S_SETREG <*> -> S_SETREG (same reg) (2 S_NOPs) ---
// TODO: S_SETREG instruction not implemented in AMDGCN.
def CDNA3_SetRegSetRegHazard : HazardAttrWithWaits<"CDNA3SetRegSetRegHazard",
    "cdna3_setreg_setreg_hazard", 0, 2> {
  let summary = "CDNA3 S_SETREG followed by S_SETREG (same reg)";
  let description = [{
    S_SETREG followed by S_SETREG writing the same register requires 2 wait states.
    TODO: S_SETREG instruction not implemented in AMDGCN.
  }];
}

//--- Case 3: SET_VSKIP -> S_GETREG MODE (2 S_NOPs) ---
// TODO: SET_VSKIP and S_GETREG instructions not implemented in AMDGCN.
def CDNA3_SetVskipGetRegHazard : HazardAttrWithWaits<"CDNA3SetVskipGetRegHazard",
    "cdna3_setvskip_getreg_hazard", 0, 2> {
  let summary = "CDNA3 SET_VSKIP followed by S_GETREG MODE";
  let description = [{
    SET_VSKIP followed by S_GETREG reading VSKIP from MODE requires 2 wait states.
    TODO: SET_VSKIP and S_GETREG instructions not implemented in AMDGCN.
  }];
}

//--- Case 4: S_SETREG MODE.vskip -> any vector op (2 S_NOPs) ---
// TODO: S_SETREG instruction not implemented in AMDGCN.
def CDNA3_SetRegVskipVectorHazard : HazardAttrWithWaits<"CDNA3SetRegVskipVectorHazard",
    "cdna3_setreg_vskip_vector_hazard", 0, 2> {
  let summary = "CDNA3 S_SETREG MODE.vskip followed by vector op";
  let description = [{
    S_SETREG MODE.vskip followed by any vector op requires 2 nops or non-vector
    instructions. TODO: S_SETREG instruction not implemented in AMDGCN.
  }];
}

//--- Case 5: VALU that sets VCC or EXEC -> VALU that uses EXECZ or VCCZ (5 V_NOPs) ---
def CDNA3_VccExecVcczExeczHazard : HazardAttrWithWaits<"CDNA3VccExecVcczExeczHazard",
    "cdna3_vcc_exec_vccz_execz_hazard", 5, 0> {
  let summary = "CDNA3 VALU sets VCC/EXEC followed by VALU uses EXECZ/VCCZ";
  let description = [{
    VALU that sets VCC or EXEC followed by VALU that uses EXECZ or VCCZ as data
    source requires 5 wait states.
  }];
}

//--- Case 6: VALU writes SGPR/VCC -> V_[READ,WRITE]LANE lane select (4 V_NOPs) ---
// TODO: V_READLANE, V_WRITELANE instructions not implemented in AMDGCN.
def CDNA3_ValuSgprReadlaneHazard : HazardAttrWithWaits<"CDNA3ValuSgprReadlaneHazard",
    "cdna3_valu_sgpr_readlane_hazard", 4, 0> {
  let summary = "CDNA3 VALU writes SGPR/VCC followed by V_READLANE/V_WRITELANE";
  let description = [{
    VALU writes SGPR/VCC (readlane, cmp, add/sub, div_scale) followed by
    V_READLANE/V_WRITELANE using that SGPR/VCC as lane select requires 4 wait states.
    TODO: V_READLANE, V_WRITELANE instructions not implemented in AMDGCN.
  }];
}

//--- Case 7: VALU writes VCC -> V_DIV_FMAS (4 V_NOPs) ---
// TODO: V_DIV_FMAS instruction not implemented in AMDGCN.
def CDNA3_VccDivFmasHazard : HazardAttrWithWaits<"CDNA3VccDivFmasHazard",
    "cdna3_vcc_div_fmas_hazard", 4, 0> {
  let summary = "CDNA3 VALU writes VCC followed by V_DIV_FMAS";
  let description = [{
    VALU writes VCC (including v_div_scale) followed by V_DIV_FMAS requires 4 wait
    states. TODO: V_DIV_FMAS instruction not implemented in AMDGCN.
  }];
}

//--- Case 8: Store/atomic X3/X4 -> Write VGPRs holding writedata (1 V_NOP) ---
// Implemented for GLOBAL_STORE_DWORDX3/X4, BUFFER_STORE_DWORDX3/X4.
// TODO: FLAT_ATOMIC_[F]CMPSWAP_X2, BUFFER_STORE_FORMAT_XYZ/XYZW,
// BUFFER_ATOMIC_[F]CMPSWAP_X2 not implemented in AMDGCN.
def CDNA3_StoreWriteDataHazard : HazardAttrWithWaits<"CDNA3StoreWriteDataHazard",
    "cdna3_store_write_data_hazard", 1, 0> {
  let summary = "CDNA3 store/atomic X3/X4 followed by write to writedata VGPRs";
  let description = [{
    FLAT_STORE_X3/X4, BUFFER_STORE_*_X3/X4, FLAT_ATOMIC_[F]CMPSWAP_X2, etc.
    followed by instruction that writes VGPRs holding writedata requires 1 wait
    state. BUFFER_STORE_* with SGPR offset does not require wait states.
    TODO: FLAT_ATOMIC, BUFFER_STORE_FORMAT, BUFFER_ATOMIC not implemented.
  }];
}

//--- Case 9: Store/atomic X3/X4 -> VALU writes VGPRs holding writedata (2 V_NOPs) ---
// Implemented: CDNA3_StoreHazard (GLOBAL_STORE, BUFFER_STORE_DWORDX3/X4).
def CDNA3_StoreHazard : HazardAttrWithWaits<"CDNA3StoreHazard",
    "cdna3_store_hazard", 2, 0> {
  let summary = "CDNA3 store hazard followed by VALU write";
  let description = [{
    This hazard is raised when a store operation (FLAT_STORE_X3/X4, BUFFER_STORE_*,
    etc.) is followed by a VALU instruction overwriting the write data of the store.
    TODO: FLAT_ATOMIC, BUFFER_STORE_FORMAT, BUFFER_ATOMIC not implemented.
  }];
}

//--- Case 10: VALU writes SGPR -> VMEM reads that SGPR (5 V_NOPs) ---
def CDNA3_ValuSgprVmemHazard : HazardAttrWithWaits<"CDNA3ValuSgprVmemHazard",
    "cdna3_valu_sgpr_vmem_hazard", 5, 0> {
  let summary = "CDNA3 VALU writes SGPR followed by VMEM reads that SGPR";
  let description = [{
    VALU writes SGPR followed by VMEM that reads that SGPR requires 5 wait states.
  }];
}

//--- Case 11: SALU writes M0 -> GDS, S_SENDMSG (1 S_NOP) ---
// TODO: GDS and S_SENDMSG instructions not implemented in AMDGCN.
def CDNA3_SaluM0GdsSendmsgHazard : HazardAttrWithWaits<"CDNA3SaluM0GdsSendmsgHazard",
    "cdna3_salu_m0_gds_sendmsg_hazard", 0, 1> {
  let summary = "CDNA3 SALU writes M0 followed by GDS or S_SENDMSG";
  let description = [{
    SALU writes M0 followed by GDS or S_SENDMSG requires 1 wait state.
    TODO: GDS and S_SENDMSG instructions not implemented in AMDGCN.
  }];
}

//--- Case 12: VALU writes VGPR -> VALU DPP reads that VGPR (2 V_NOPs) ---
// TODO: DPP (Data Parallel Primitives) modifier not implemented in AMDGCN.
def CDNA3_ValuVgprDppHazard : HazardAttrWithWaits<"CDNA3ValuVgprDppHazard",
    "cdna3_valu_vgpr_dpp_hazard", 2, 0> {
  let summary = "CDNA3 VALU writes VGPR followed by VALU DPP reads that VGPR";
  let description = [{
    VALU writes VGPR followed by VALU DPP that reads that VGPR requires 2 wait
    states. TODO: DPP modifier not implemented in AMDGCN.
  }];
}

//--- Case 13: VALU writes EXEC -> VALU DPP op (5 V_NOPs) ---
// TODO: DPP modifier and EXEC forwarding not implemented in AMDGCN.
def CDNA3_ExecDppHazard : HazardAttrWithWaits<"CDNA3ExecDppHazard",
    "cdna3_exec_dpp_hazard", 5, 0> {
  let summary = "CDNA3 VALU writes EXEC followed by VALU DPP op";
  let description = [{
    VALU writes EXEC followed by VALU DPP op requires 5 wait states. ALU does not
    forward EXEC to DPP. TODO: DPP modifier not implemented in AMDGCN.
  }];
}

//--- Case 14: Mixed VCC use -> VALU reads VCC as constant (1 V_NOP) ---
// TODO: VCC alias vs SGPR# tracking not implemented in AMDGCN.
def CDNA3_MixedVccConstantHazard : HazardAttrWithWaits<"CDNA3MixedVccConstantHazard",
    "cdna3_mixed_vcc_constant_hazard", 1, 0> {
  let summary = "CDNA3 Mixed VCC use followed by VALU reads VCC as constant";
  let description = [{
    Mixed use of VCC (alias vs SGPR#) from v_readlane, v_cmp, v_add, v_sub,
    v_div_scale followed by VALU reading VCC as constant requires 1 wait state.
    Carry-in is 0 wait states. TODO: VCC/SGPR alias tracking not implemented.
  }];
}

//--- Case 15: S_SETREG TRAPSTS -> RFE, RFE_restore (1 S_NOP) ---
// TODO: S_SETREG, RFE, RFE_restore instructions not implemented in AMDGCN.
def CDNA3_SetRegTrapstsRfeHazard : HazardAttrWithWaits<"CDNA3SetRegTrapstsRfeHazard",
    "cdna3_setreg_trapsts_rfe_hazard", 0, 1> {
  let summary = "CDNA3 S_SETREG TRAPSTS followed by RFE/RFE_restore";
  let description = [{
    S_SETREG TRAPSTS followed by RFE or RFE_restore requires 1 wait state.
    TODO: S_SETREG, RFE, RFE_restore instructions not implemented in AMDGCN.
  }];
}

//--- Case 16: SALU writes M0 -> LDS add-TID, buffer_store_LDS, scratch/global LDS=1 (1 S_NOP) ---
// TODO: LDS add-TID, buffer_store_LDS_dword not implemented in AMDGCN.
def CDNA3_SaluM0LdsHazard : HazardAttrWithWaits<"CDNA3SaluM0LdsHazard",
    "cdna3_salu_m0_lds_hazard", 0, 1> {
  let summary = "CDNA3 SALU writes M0 followed by LDS add-TID or buffer_store_LDS";
  let description = [{
    SALU writes M0 followed by LDS add-TID instruction, buffer_store_LDS_dword,
    or scratch/global with LDS=1 requires 1 wait state.
    TODO: LDS add-TID, buffer_store_LDS_dword not implemented in AMDGCN.
  }];
}

//--- Case 17: SALU writes M0 -> S_MOVEREL (1 S_NOP) ---
// TODO: S_MOVEREL instruction not implemented in AMDGCN.
def CDNA3_SaluM0MoverelHazard : HazardAttrWithWaits<"CDNA3SaluM0MoverelHazard",
    "cdna3_salu_m0_moverel_hazard", 0, 1> {
  let summary = "CDNA3 SALU writes M0 followed by S_MOVEREL";
  let description = [{
    SALU writes M0 followed by S_MOVEREL requires 1 wait state.
    TODO: S_MOVEREL instruction not implemented in AMDGCN.
  }];
}

//--- Case 18: VALU writes SGPR/VCC -> VALU reads SGPR (2 or 4 V_NOPs) ---
// Subcases: constant 2, lane-select 4, carry-in 0, v_cmpx->EXEC 2, readlane 4.
// TODO: VALU SGPR/VCC write tracking and v_readlane/v_writelane not implemented.
def CDNA3_ValuSgprValuReadHazard : HazardAttrWithWaits<"CDNA3ValuSgprValuReadHazard",
    "cdna3_valu_sgpr_valu_read_hazard", 2, 0> {
  let summary = "CDNA3 VALU writes SGPR/VCC followed by VALU reads SGPR";
  let description = [{
    VALU writes SGPR/VCC (readlane, cmp, add/sub, div_scale) followed by VALU
    reading SGPR: as constant 2 waits, as lane-select 4 waits, carry-in 0 waits.
    TODO: SGPR/VCC dependency tracking not implemented in AMDGCN.
  }];
}

//--- Case 19: VALU writes VGPRn -> v_readlane vsrc0 reads VGPRn (1 V_NOP) ---
// TODO: V_READLANE instruction not implemented in AMDGCN.
def CDNA3_ValuVgprReadlaneHazard : HazardAttrWithWaits<"CDNA3ValuVgprReadlaneHazard",
    "cdna3_valu_vgpr_readlane_hazard", 1, 0> {
  let summary = "CDNA3 VALU writes VGPRn followed by v_readlane vsrc0 reads VGPRn";
  let description = [{
    VALU writes VGPRn followed by v_readlane with vsrc0 reading VGPRn requires 1
    wait state. TODO: V_READLANE instruction not implemented in AMDGCN.
  }];
}

//--- Case 20: VALU OPSEL/SDWA -> VALU consumes result (1 V_NOP) ---
// TODO: OPSEL and SDWA modifiers not implemented in AMDGCN.
def CDNA3_OpselSdwaHazard : HazardAttrWithWaits<"CDNA3OpselSdwaHazard",
    "cdna3_opsel_sdwa_hazard", 1, 0> {
  let summary = "CDNA3 VALU OPSEL/SDWA followed by VALU consumes result";
  let description = [{
    VALU op using OPSEL or SDWA that changes result bit position, followed by
    VALU op consuming that result, requires 1 wait state.
    TODO: OPSEL and SDWA modifiers not implemented in AMDGCN.
  }];
}

//--- Case 21: VALU Trans op -> Non-trans VALU consumes result (1 V_NOP) ---
// TODO: Trans op detection (V_EXP_F32, V_LOG_F32, etc.) not implemented.
def CDNA3_TransOpHazard : HazardAttrWithWaits<"CDNA3TransOpHazard",
    "cdna3_trans_op_hazard", 1, 0> {
  let summary = "CDNA3 VALU Trans op followed by Non-trans VALU consumes result";
  let description = [{
    VALU Trans op (V_EXP_F32, V_LOG_F32, V_RCP_F32, etc.) followed by non-Trans
    VALU op consuming the result requires 1 wait state.
    TODO: Trans op detection not implemented in AMDGCN.
  }];
}

#endif // ASTER_AMDGCN_HAZARDS_TD
