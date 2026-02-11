//===- Comparison.td - Comparison related operations -------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_COMPARISON_TD
#define AMDGCN_INST_COMPARISON_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"
include "aster/Interfaces/DependentOpInterface.td"

//===----------------------------------------------------------------------===//
// Comparison operand type
//===----------------------------------------------------------------------===//

/// A comparison operand.
def CmpIOpOperand : RegOrImm<[
    VGPRType, SGPRType, SReg<"VCC">, SReg<"VCCLo">, SReg<"VCCHi">, SReg<"EXEC">,
    SReg<"EXECLo">, SReg<"EXECHi">, SReg<"SCC">, SReg<"VCCZ">, SReg<"EXECZ">
  ], [I32]>;

/// A comparison result.
def CmpIOpResult : AnyRegOf<[SReg<"VCC">, SReg<"SCC">, SGPRType]>;

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

def AMDGCN_CmpIOp : InstOp<"cmpi", [
    AMDGCNInstOpInterface,
    ConditionallySpeculatable,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let summary = "AMDGCN integer comparison operation";
  let description = [{
    This operation represents an AMDGCN integer comparison instruction (e.g.,
    s_cmp_eq_i32, v_cmp_lt_i32, etc.). It compares two operands and writes the
    result to a special register (VCC or SCC).
  }];
  let arguments = (ins
    AMDGCN_InstAttr:$opcode,
    CmpIOpResult:$dest,
    OptionalReg<SReg<"EXEC">>:$exec,
    CmpIOpOperand:$lhs,
    CmpIOpOperand:$rhs
  );
  let results = (outs Optional<CmpIOpResult>:$result);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "OpCode":$opcode, "Value":$dest, "Value":$lhs, "Value":$rhs)>,
    OpBuilder<(ins "OpCode":$opcode, "Value":$dest, "Value":$exec, "Value":$lhs, "Value":$rhs)>
  ];
  let assemblyFormat = [{
    custom<Opcode>($opcode) `outs` $dest (`,` $exec^)? `ins` $lhs `,` $rhs
    attr-dict `:` custom<CmpOutTypes>(type($dest), ref($exec), type($exec))
    `ins` `(` type($lhs) `,` type($rhs) `)`
  }];
  let extraClassDeclaration = [{
    /// Returns the number of output operands.
    int32_t getNumOuts() {
      return getExec() ? 2 : 1;
    }

    /// Returns true if this instruction is in DPS format.
    bool isDPSInstruction() {
      return getResult() != nullptr;
    }

    /// Returns the speculatability of the operation.
    Speculation::Speculatability getSpeculatability() {
      return aster::detail::getInstSpeculatabilityImpl(*this);
    }

    //===------------------------------------------------------------------===//
    // AMDGCNInstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getOperation()->getOpOperands().take_front(getNumOuts());
    }
    /// Get the instruction input operands.
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getOperation()->getOpOperands().drop_front(getNumOuts());
    }
    /// Get the instruction results.
    ResultRange getInstResults() {
      if (isDPSInstruction())
        return getOperation()->getResults().slice(0, 1);
      return getOperation()->getResults().slice(0, 0);
    }
  }];
}

//===----------------------------------------------------------------------===//
// SAlu Instructions - 32-bit integer comparisons
//===----------------------------------------------------------------------===//

def SCmpI32Operand :  RegOrImm<[
    SGPRType, SReg<"VCCLo">, SReg<"VCCHi">, SReg<"EXECLo">, SReg<"EXECHi">,
    SReg<"SCC">, SReg<"VCCZ">, SReg<"EXECZ">
  ], [I32]>;

class SOPCmpIInst<string _mnemonic, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, AMDGCN_CmpIOp, _isa> {
  let summary = "SOPC compare instruction setting SCC";
  // Default constraints - 2 scalar inputs, 1 SCC output.
  let constraints = (ins
    /*ins=*/
      RegImmOperand<SCmpI32Operand, 1>:$lhs,
      RegImmOperand<SCmpI32Operand, 1>:$rhs,
    /*outs=*/
      RegImmOperand<SReg<"SCC">, 1>:$dest,
      NotPresent:$exec
  );
  // Default builders.
  let cppBuilder = DefaultCppBuilder<(ins
    CppValue:$dest, CppValue:$lhs, CppValue:$rhs)>;
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$dest, PyValue:$lhs, PyValue:$rhs)>;
  // ASM format: s_cmp_* src0, src1 (SCC output is implicit in hardware encoding)
  let asmFormat = [SingleAsmVariant<"$lhs, $rhs">];
}

def SCmpEqI32 : SOPCmpIInst<"s_cmp_eq_i32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "SOPC compare equal (signed 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is equal to the second.
    SCC = (S0.i32 == S1.i32)
  }];
}

def SCmpLgI32 : SOPCmpIInst<"s_cmp_lg_i32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "SOPC compare not equal (signed 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is not equal to the second.
    SCC = (S0.i32 != S1.i32)
  }];
}

def SCmpGtI32 : SOPCmpIInst<"s_cmp_gt_i32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "SOPC compare greater than (signed 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is greater than the second.
    SCC = (S0.i32 > S1.i32)
  }];
}

def SCmpGeI32 : SOPCmpIInst<"s_cmp_ge_i32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "SOPC compare greater than or equal (signed 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is greater than or equal to the second.
    SCC = (S0.i32 >= S1.i32)
  }];
}

def SCmpLtI32 : SOPCmpIInst<"s_cmp_lt_i32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "SOPC compare less than (signed 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is less than the second.
    SCC = (S0.i32 < S1.i32)
  }];
}

def SCmpLeI32 : SOPCmpIInst<"s_cmp_le_i32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "SOPC compare less than or equal (signed 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is less than or equal to the second.
    SCC = (S0.i32 <= S1.i32)
  }];
}

//===----------------------------------------------------------------------===//
// SOPC Instructions - Unsigned 32-bit integer comparisons
//===----------------------------------------------------------------------===//

def SCmpEqU32 : SOPCmpIInst<"s_cmp_eq_u32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "SOPC compare equal (unsigned 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input equals the second scalar input.
    SCC = (S0.u32 == S1.u32)
  }];
}

def SCmpLgU32 : SOPCmpIInst<"s_cmp_lg_u32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "SOPC compare not equal (unsigned 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is not equal to the second.
    SCC = (S0.u32 != S1.u32)
  }];
}

def SCmpGtU32 : SOPCmpIInst<"s_cmp_gt_u32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "SOPC compare greater than (unsigned 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is greater than the second.
    SCC = (S0.u32 > S1.u32)
  }];
}

def SCmpGeU32 : SOPCmpIInst<"s_cmp_ge_u32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "SOPC compare greater than or equal (unsigned 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is greater than or equal to the second.
    SCC = (S0.u32 >= S1.u32)
  }];
}

def SCmpLtU32 : SOPCmpIInst<"s_cmp_lt_u32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "SOPC compare less than (unsigned 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is less than the second.
    SCC = (S0.u32 < S1.u32)
  }];
}

def SCmpLeU32 : SOPCmpIInst<"s_cmp_le_u32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "SOPC compare less than or equal (unsigned 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is less than or equal to the second.
    SCC = (S0.u32 <= S1.u32)
  }];
}

//===----------------------------------------------------------------------===//
//
// VAlu Instructions - 32-bit integer comparisons
//
//===----------------------------------------------------------------------===//

def VOPCCmpI32Operand :  RegOrImm<[
    VGPRType, SGPRType, SReg<"VCCLo">, SReg<"VCCHi">, SReg<"EXECLo">,
    SReg<"EXECHi">, SReg<"SCC">, SReg<"VCCZ">, SReg<"EXECZ">
  ], [I32]>;

//===----------------------------------------------------------------------===//
// 32 bit encoding
//===----------------------------------------------------------------------===//
// Base class for VAlu compare instructions on 32 bits.
class VOPCE32CmpIInst<string _mnemonic, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, AMDGCN_CmpIOp, _isa> {
  let summary = "VOPC compare instruction setting VCC";
  // Default constraints - 2 scalar inputs, 1 VCC output.
  let constraints = (ins
    /*ins=*/
      RegImmOperand<VOPCCmpI32Operand, 1>:$lhs,
      RegImmOperand<VGPRType, 1>:$rhs,
    /*outs=*/
      RegImmOperand<SReg<"VCC">, 1>:$dest,
      NotPresent:$exec
  );
  // Default builders.
  let cppBuilder = DefaultCppBuilder<(ins
    CppValue:$dest, CppValue:$lhs, CppValue:$rhs)>;
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$dest, PyValue:$lhs, PyValue:$rhs)>;
  // ASM format: v_cmp_* src0, src1 (VCC output is implicit in hardware encoding)
  let asmFormat = [SingleAsmVariant<"$lhs, $rhs">];
}

def VCmpEqI32 : VOPCE32CmpIInst<"v_cmp_eq_i32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "VOPC compare equal (signed 32-bit)";
}

//===----------------------------------------------------------------------===//
// 64 bit encoding
//===----------------------------------------------------------------------===//
// VAlu compare instructions on 64 bit encoding.
class VOPCE64CmpIInst<string _mnemonic, list<ISAVersion> _isa> : VOPCE32CmpIInst<_mnemonic, _isa> {
  let summary = "VOPC compare instruction (64-bit)";
  // Override constraints - 2 scalar inputs, 1 VCC output.
  let constraints = (ins
    /*ins=*/
      RegImmOperand<VOPCCmpI32Operand, 1>:$lhs,
      RegImmOperand<VOPCCmpI32Operand, 1>:$rhs,
    /*outs=*/
      RegImmOperand<AnyRegOf<[SReg<"VCC">, SGPRType]>, 1>:$dest,
      NotPresent:$exec
  );
}

def VCmpEqI32E64 : VOPCE64CmpIInst<"v_cmp_eq_i32_e64", [CDNA3, CDNA4, RDNA4]> {
  let summary = "VOPC compare equal (signed 32-bit) E64";
}

#endif // AMDGCN_INST_COMPARISON_TD
