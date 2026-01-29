//===- SOPC.td - SOPC Operations ---------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SOPC specific operations.
//
// SOPC instructions compare two scalar values and set the SCC (Scalar Condition
// Code) register. They have no destination register in the encoding - only SCC
// is modified.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_SOPC_TD
#define AMDGCN_INST_SOPC_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"

//===----------------------------------------------------------------------===//
// SOPCOp Operation
//===----------------------------------------------------------------------===//

// SOPC source can be SGPR or 32-bit immediate
def SOPCSSrc : RegOrImm<[SGPRType], [I32]>;

def SOPCOp : AutoInstOp<"sopc", [AMDGCNInstTraits]> {
  let summary = "SOPC (Scalar Operations Compare) operation";
  let description = [{
    The SOPC operation compares two scalar source operands and sets the SCC
    (Scalar Condition Code) register. The result of the comparison is stored
    in SCC, not in a general-purpose register.

    Example:
    ```mlir
    // Compare two SGPR values, set SCC
    %scc = amdgcn.sopc s_cmp_eq_i32 outs %scc ins %s0, %s1
      : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sreg<scc>
    ```
  }];
  let inputs = (ins SOPCSSrc:$src0, SOPCSSrc:$src1);
  let outputs = (ins SCCReg:$scc);
  let assemblyFormat = [{
    custom<Opcode>($opcode) `outs` $scc `ins` $src0 `,` $src1 attr-dict
    `:` type(operands)
  }];
}

class SOPCInst<string _mnemonic, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, SOPCOp, _isa> {
  let summary = "SOPC compare instruction setting SCC";
  // Default constraints - 2 scalar inputs, 1 SCC output.
  let constraints = (ins
    /*ins=*/
      RegImmOperand<SOPCSSrc, 1>:$src0,
      RegImmOperand<SOPCSSrc, 1>:$src1,
    /*outs=*/
      RegImmOperand<SCCReg, 1>:$scc
  );
  // Default builders.
  let cppBuilder = DefaultCppBuilder<(ins
    CppValue:$scc, CppValue:$src0, CppValue:$src1)>;
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$scc, PyValue:$src0, PyValue:$src1)>;
  // ASM format: s_cmp_* src0, src1 (SCC output is implicit in hardware encoding)
  let asmFormat = [SingleAsmVariant<"$src0, $src1">];
}

//===----------------------------------------------------------------------===//
// SOPC Instructions - Signed 32-bit integer comparisons
//===----------------------------------------------------------------------===//


def SCmpLgI32 : SOPCInst<"s_cmp_lg_i32", [CDNA3, RDNA4]> {
  let summary = "SOPC compare not equal (signed 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is not equal to the second.
    SCC = (S0.i32 != S1.i32)
  }];
}

def SCmpGtI32 : SOPCInst<"s_cmp_gt_i32", [CDNA3, RDNA4]> {
  let summary = "SOPC compare greater than (signed 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is greater than the second.
    SCC = (S0.i32 > S1.i32)
  }];
}

def SCmpGeI32 : SOPCInst<"s_cmp_ge_i32", [CDNA3, RDNA4]> {
  let summary = "SOPC compare greater than or equal (signed 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is greater than or equal to the second.
    SCC = (S0.i32 >= S1.i32)
  }];
}

def SCmpLtI32 : SOPCInst<"s_cmp_lt_i32", [CDNA3, RDNA4]> {
  let summary = "SOPC compare less than (signed 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is less than the second.
    SCC = (S0.i32 < S1.i32)
  }];
}

def SCmpLeI32 : SOPCInst<"s_cmp_le_i32", [CDNA3, RDNA4]> {
  let summary = "SOPC compare less than or equal (signed 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is less than or equal to the second.
    SCC = (S0.i32 <= S1.i32)
  }];
}

//===----------------------------------------------------------------------===//
// SOPC Instructions - Unsigned 32-bit integer comparisons
//===----------------------------------------------------------------------===//

def SCmpEqU32 : SOPCInst<"s_cmp_eq_u32", [CDNA3, RDNA4]> {
  let summary = "SOPC compare equal (unsigned 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input equals the second scalar input.
    SCC = (S0.u32 == S1.u32)
  }];
}

def SCmpLgU32 : SOPCInst<"s_cmp_lg_u32", [CDNA3, RDNA4]> {
  let summary = "SOPC compare not equal (unsigned 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is not equal to the second.
    SCC = (S0.u32 != S1.u32)
  }];
}

def SCmpGtU32 : SOPCInst<"s_cmp_gt_u32", [CDNA3, RDNA4]> {
  let summary = "SOPC compare greater than (unsigned 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is greater than the second.
    SCC = (S0.u32 > S1.u32)
  }];
}

def SCmpGeU32 : SOPCInst<"s_cmp_ge_u32", [CDNA3, RDNA4]> {
  let summary = "SOPC compare greater than or equal (unsigned 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is greater than or equal to the second.
    SCC = (S0.u32 >= S1.u32)
  }];
}

def SCmpLtU32 : SOPCInst<"s_cmp_lt_u32", [CDNA3, RDNA4]> {
  let summary = "SOPC compare less than (unsigned 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is less than the second.
    SCC = (S0.u32 < S1.u32)
  }];
}

def SCmpLeU32 : SOPCInst<"s_cmp_le_u32", [CDNA3, RDNA4]> {
  let summary = "SOPC compare less than or equal (unsigned 32-bit)";
  let description = [{
    Set SCC to 1 if the first scalar input is less than or equal to the second.
    SCC = (S0.u32 <= S1.u32)
  }];
}

#endif // AMDGCN_INST_SOPC_TD
