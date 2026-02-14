//===- SOP1.td - SOP1 Operations ---------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SOP1 specific operations.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_SOP1_TD
#define AMDGCN_INST_SOP1_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"

//===----------------------------------------------------------------------===//
// SOP1Op Operation
//===----------------------------------------------------------------------===//

def SOP1SSrc : RegOrImm<[SGPRType], [I32]>;
def SOP1SDst : AnyRegOf<[SGPRType]>;

def SOP1Op : AMDGCN_InstOp<"sop1", [PureInst, InstInferType]> {
  let summary = "SOP1 (Scalar Operations with 1 operands) operation";
  let description = [{
    The SOP1 operation performs scalar operations with one source operands and
    one dps operand.
  }];
  let leadingArguments = (ins AMDGCN_InstAttr:$opcode);
  let inputs = (ins SOP1SSrc:$src0);
  let outputs = (ins SOP1SDst:$sdst);
  // TODO: add modifiers.
  let assemblyFormat = [{
    custom<Opcode>($opcode) `outs` $sdst `ins` $src0 attr-dict
    `:` type(operands)
  }];
}

class SOP1Constraints {
  dag constraints = (ins
    /*ins=*/
      RegImmOperand<SOP1SSrc, 1>:$src0,
    /*outs=*/
      RegisterOperand<[SOP1SDst], 1>:$sdst
  );
}

class SOP1Inst<string _mnemonic, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, SOP1Op, _isa> {
  // Default constraints.
  let constraints = SOP1Constraints<>.constraints;
  // Default builders.
  let cppBuilder = DefaultCppBuilder<(ins
    CppValue:$sdst, CppValue:$src0)>;
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$sdst, PyValue:$src0)>;
  // Default ASM.
  let asmFormat = [SingleAsmVariant<"$sdst, $src0">];
}

//===----------------------------------------------------------------------===//
// SOP1Op instructions
//===----------------------------------------------------------------------===//

def SMovB32 : SOP1Inst<"s_mov_b32", [CDNA3, CDNA4, RDNA4]> {
  let summary = "SOP1 move 32-bit value to destination register.";
}

#endif // AMDGCN_INST_SOP1_TD
