//===-- AMDGCN.td - AMDGCN target definitions -----------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines AMDGCN target definitions.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_TD
#define ASTER_AMDGCN_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNEnumAttrs.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNModifiers.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNTypeConstraints.td"
include "aster/Interfaces/TargetAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/Constraints.td"

/// The optional value must not be present.
def NotPresent : InstConstraint<[
  TypeConstraint<
    CPred<"$_self == nullptr">, "expected `$_argName` to not be present">
]>;

/// A register or immediate operand that can be either SGPR or VGPR type
def VIntSrc : RegOrImm<[SGPRType, VGPRType], [I32]>;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

def InvalidCase : EnumCaseInfo<"invalid", "Invalid">;

//===----------------------------------------------------------------------===//
// ISA Versions
//===----------------------------------------------------------------------===//

def CDNA3 : ISAVersion<"cdna3"> { let description = "CDNA 3 ISA version"; }

def CDNA4 : ISAVersion<"cdna4"> { let description = "CDNA 4 ISA version"; }

def RDNA4 : ISAVersion<"rdna4"> { let description = "RDNA 4 ISA version"; }

// TODO: Ideally call this CDNA5 / RDNA5 or UDNA ...
// don't know what the right teminology is here but this sits at the same level
// as CDNA4/CDNA4/RDNA4 and should not conflict with
// GFX940/GFX950/GFX1201/GFX1250/GFX1251 terminology.
def GFX12_50 : ISAVersion<"gfx12_50"> { let description = "GFX12.5 ISA version (gfx1250/gfx1251)"; }

def IsaVersions
    : AutoI32EnumCases<"ISAVersion",
                       "AMDGCN ISA Versions",
                       [InvalidCase, CDNA3, CDNA4, RDNA4, GFX12_50]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

def IsaVersionAttr : AMDGCN_EnumAttr<"IsaVersion", "isa", IsaVersions>;

//===----------------------------------------------------------------------===//
// Target definitions
//===----------------------------------------------------------------------===//

def GFX940 : Target<"gfx940", CDNA3> { let description = "GFX940 target"; }

def GFX942 : Target<"gfx942", CDNA3> { let description = "GFX942 target"; }

def GFX950 : Target<"gfx950", CDNA4> { let description = "GFX950 target"; }

def GFX1201 : Target<"gfx1201", RDNA4> { let description = "GFX1201 target"; }

def GFX1250 : Target<"gfx1250", GFX12_50> { let description = "GFX1250 target (gfx1250)"; }

def GFX1251 : Target<"gfx1251", GFX12_50> { let description = "GFX1251 target (gfx1251)"; }

def AMDGCNTargets
    : AutoI32EnumCases<
          "Target", "AMDGCN Targets",
          [InvalidCase, GFX940, GFX942, GFX950, GFX1201, GFX1250, GFX1251]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

def AMDGCNTargetAttr : AMDGCN_EnumAttr<"Target", "target", AMDGCNTargets,
    [DeclareAttrInterfaceMethods<TargetAttrInterface>]>;

//===----------------------------------------------------------------------===//
// Instruction properties definitions
//===----------------------------------------------------------------------===//

def IsVALU : InstProp<"is_valu">;

def IsVMem : InstProp<"is_vmem">;

// Scalar ALU encoding families
def IsSALU : InstProp<"salu">;
def IsSOP1 : InstProp<"sop1">;
def IsSOP2 : InstProp<"sop2">;
def IsSOPC : InstProp<"sopc">;
def IsSOPK : InstProp<"sopk">;
def IsSOPP : InstProp<"sopp">;

// Memory encoding families
def IsSMEM : InstProp<"smem">;
def IsDS : InstProp<"ds">;
def IsBuffer : InstProp<"buffer">;
def IsGlobal : InstProp<"global">;

// Control flow
def IsControlFlow : InstProp<"control_flow">;

// Matrix multiply-accumulate
def IsMMA : InstProp<"mma">;
def IsScaledMMA : InstProp<"scaled_mma">;
// Wave matrix multiply-accumulate (gfx1250 WMMA, wave32 VGPR-only)
def IsWMMA : InstProp<"wmma">;

// Vector ALU encoding families
def IsVOP1 : InstProp<"vop1">;
def IsVOP2 : InstProp<"vop2">;
def IsVOP3 : InstProp<"vop3">;
def IsVOP3P : InstProp<"vop3p">;
def IsVOPC : InstProp<"vopc">;

#endif // ASTER_AMDGCN_TD
