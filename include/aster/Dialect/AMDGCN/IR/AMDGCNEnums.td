//===-- AMDGCNEnums.td - AMDGCN dialect enumerations -------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_ENUMS
#define ASTER_AMDGCN_ENUMS

include "mlir/IR/Utils.td"
include "aster/Support/EnumBase.td"

//===----------------------------------------------------------------------===//
// AMDGCN register variants.
//===----------------------------------------------------------------------===//

class SpecialRegister<string identifier, string cppTypeName = ""> {
  string name = identifier;
  string typeName = !if(!empty(cppTypeName), identifier, cppTypeName);
  string typeMnemonic = !tolower(identifier);
}

defvar SpecialRegisters = [
  SpecialRegister<"VCC">,
  SpecialRegister<"VCC_LO", "VCCLo">,
  SpecialRegister<"VCC_HI", "VCCHi">,
  SpecialRegister<"VCCZ">,
  SpecialRegister<"EXEC">,
  SpecialRegister<"EXEC_LO", "EXECLo">,
  SpecialRegister<"EXEC_HI", "EXECHi">,
  SpecialRegister<"EXECZ">,
  SpecialRegister<"M0">,
  SpecialRegister<"SCC">,
];

def RegisterKind : AutoI32EnumCases<"RegisterKind", "AMDGCN register kind", [
    EnumCaseInfo<"Unknown", "Unknown">,
    EnumCaseInfo<"AGPR">,
    EnumCaseInfo<"SGPR">,
    EnumCaseInfo<"VGPR">,
    EnumCaseInfo<"SREG">
  ] # !foreach(sreg, SpecialRegisters, EnumCaseInfo<sreg.name>)> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

//===----------------------------------------------------------------------===//
// AMDGCN access kinds.
//===----------------------------------------------------------------------===//

def AccessKind : AutoI32EnumCases<"AccessKind",
  "AMDGCN buffer memory access kind", [
    PrettyEnumCaseInfo<"unspecified">,
    PrettyEnumCaseInfo<"read_only">,
    PrettyEnumCaseInfo<"write_only">,
    PrettyEnumCaseInfo<"read_write">,
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

//===----------------------------------------------------------------------===//
// AMDGCN ptr addr space kinds.
//===----------------------------------------------------------------------===//

def AddressSpaceKind : AutoI32EnumCases<"AddressSpaceKind",
  "AMDGCN pointer address space", [
    PrettyEnumCaseInfo<"invalid">,
    PrettyEnumCaseInfo<"private">,
    PrettyEnumCaseInfo<"global">,
    PrettyEnumCaseInfo<"constant">,
    PrettyEnumCaseInfo<"local">,
    PrettyEnumCaseInfo<"generic">,
    PrettyEnumCaseInfo<"region">,
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

//===----------------------------------------------------------------------===//
// AMDGCN kernel argument kinds.
//===----------------------------------------------------------------------===//

def ArgumentValueKind : AutoI32EnumCases<"ArgumentValueKind",
  "AMDGCN kernel argument kind", [
    PrettyEnumCaseInfo<"invalid">,
    PrettyEnumCaseInfo<"by_value">,
    PrettyEnumCaseInfo<"global_buffer">,
    PrettyEnumCaseInfo<"dynamic_shared_pointer">,
    PrettyEnumCaseInfo<"sampler">,
    PrettyEnumCaseInfo<"image">,
    PrettyEnumCaseInfo<"pipe">,
    PrettyEnumCaseInfo<"queue">,
    PrettyEnumCaseInfo<"hidden_block_count_x">,
    PrettyEnumCaseInfo<"hidden_block_count_y">,
    PrettyEnumCaseInfo<"hidden_block_count_z">,
    PrettyEnumCaseInfo<"hidden_group_size_x">,
    PrettyEnumCaseInfo<"hidden_group_size_y">,
    PrettyEnumCaseInfo<"hidden_group_size_z">,
    PrettyEnumCaseInfo<"hidden_none">,
    PrettyEnumCaseInfo<"hidden_printf_buffer">,
    PrettyEnumCaseInfo<"hidden_hostcall_buffer">,
    PrettyEnumCaseInfo<"hidden_default_queue">,
    PrettyEnumCaseInfo<"hidden_completion_action">,
    PrettyEnumCaseInfo<"hidden_multigrid_sync_arg">,
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

//===----------------------------------------------------------------------===//
// AMDGCN kernel argument flags.
//===----------------------------------------------------------------------===//

def KernelArgumentFlag : I8BitEnum<"KernelArgumentFlags",
  "AMDGCN kernel argument flags", [
    I8BitEnumCase<"None", 0, "none">,
    I8BitEnumCase<"Const", 1, "const">,
    I8BitEnumCase<"Volatile", 2, "volatile">,
    I8BitEnumCase<"Restrict", 4, "restrict">,
    I8BitEnumCase<"Pipe", 8, "pipe">
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

//===----------------------------------------------------------------------===//
// AMDGCN VGPR workitem ID mode.
//===----------------------------------------------------------------------===//

def WorkitemIDMode : AutoI32EnumCases<"WorkitemIDMode",
  "AMDGCN VGPR workitem mode", [
    PrettyEnumCaseInfo<"x">,
    PrettyEnumCaseInfo<"x_y">,
    PrettyEnumCaseInfo<"x_y_z">
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

//===----------------------------------------------------------------------===//
// AMDGCN floating point rounding modes.
//===----------------------------------------------------------------------===//

def FloatRoundMode : AutoI32EnumCases<"FloatRoundMode",
  "AMDGCN floating point rounding mode", [
    PrettyEnumCaseInfo<"near_even">,
    PrettyEnumCaseInfo<"plus_infinity">,
    PrettyEnumCaseInfo<"minus_infinity">,
    PrettyEnumCaseInfo<"zero">,
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

//===----------------------------------------------------------------------===//
// AMDGCN floating point denorm modes.
//===----------------------------------------------------------------------===//

def FloatDenormMode : AutoI32EnumCases<"FloatDenormMode",
  "AMDGCN floating point denorm mode", [
    PrettyEnumCaseInfo<"src_dst">,
    PrettyEnumCaseInfo<"dst">,
    PrettyEnumCaseInfo<"src">,
    PrettyEnumCaseInfo<"none">,
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

//===----------------------------------------------------------------------===//
// AMDGCN dim identifier.
//===----------------------------------------------------------------------===//

def DimEnum : AutoI32EnumCases<"Dim",
  "AMDGCN dimension identifier", [
    PrettyEnumCaseInfo<"x">,
    PrettyEnumCaseInfo<"y">,
    PrettyEnumCaseInfo<"z">,
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

//===----------------------------------------------------------------------===//
// AMDGCN SREG variants.
//===----------------------------------------------------------------------===//

def SregKind : AutoI32EnumCases<"SregKind",
  "AMDGCN Sreg kind", [
    PrettyEnumCaseInfo<"scc">,
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

//===----------------------------------------------------------------------===//
// AMDGCN memory instruction kind.
//===----------------------------------------------------------------------===//

def MemoryInstructionKind : AutoI8EnumCases<"MemoryInstructionKind",
  "AMDGCN memory instruction kind", [
    PrettyEnumCaseInfo<"none">,
    PrettyEnumCaseInfo<"flat">,
    PrettyEnumCaseInfo<"constant">,
    PrettyEnumCaseInfo<"shared">,
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

#endif // ASTER_AMDGCN_ENUMS
