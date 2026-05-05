//===- ControlFlow.td - Control Flow Operations ------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines control flow operations for AMDGCN.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_CONTROLFLOW_TD
#define AMDGCN_INST_CONTROLFLOW_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNInstBase.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/InstOpInterfaces.td"

#ifndef SENDPGM_DEF
#define SENDPGM_DEF
def EndKernelOp : AMDISAInstruction<"end_kernel", [
    Terminator, ReturnLike, HasParent<"KernelOp">
  ]> {
  let props = [IsControlFlow, IsSOPP];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    End of program; terminate wavefront.
  }];
  let asm_mnemonic = "s_endpgm";
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0000001>];
  let asm_syntax = [
    ASMString<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], "${mnemonic}">
  ];
}
#endif // SENDPGM_DEF

#ifndef SBRANCH_DEF
#define SBRANCH_DEF
def SBranch : AMDISAInstruction<"s_branch", [
    AMDGCNInstOpInterface, Terminator, BranchOpInterface
  ]> {
  let props = [IsControlFlow, IsSOPP];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Jump to a constant offset relative to the current PC.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0000010>];
  let successors = (successor BranchLabel:$dest);
  let asm_syntax = [
    ASMString<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], "${mnemonic} $dest">
  ];
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    $dest attr-dict
  }];
  let instExtraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // BranchOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Returns the operands forwarded to the given successor. No arguments are
    /// forwarded by this branch.
    SuccessorOperands getSuccessorOperands(unsigned index) {
      assert(index == 0 && "Only one successor is supported for s_branch");
      return SuccessorOperands(MutableOperandRange(getOperation(), 0, 0));
    }
  }];
}
#endif // SBRANCH_DEF

#ifndef SCBRANCHINST_DEF
#define SCBRANCHINST_DEF
class SCbranchInst<string mnemonic, list<Type> srcType> :
    AMDISAInstruction<mnemonic, [Terminator, CBranchInstOpInterface]> {
  let props = [IsControlFlow, IsSOPP];
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins);
  let inputs = (ins
    AnyRegOf<srcType>:$cond
  );
  let successors = (successor BranchLabel:$true_dest, BranchLabel:$false_dest);
  let asm_syntax = [
    ASMString<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], "${mnemonic} @printCondBr($true_dest, $false_dest)">
  ];
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    $cond `,` `true` `(` $true_dest `)` ` ` `false` `(` $false_dest `)`
    attr-dict `:` type($cond)
  }];
  let instExtraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // BranchInstOpInterface Methods
    //===------------------------------------------------------------------===//
    Operand getCondOperand() {
      return getCondMutable();
    }
    BlockOperand* getTrueBlock() {
      return &getOperation()->getBlockOperands()[0];
    }
    BlockOperand* getFalseBlock() {
      return &getOperation()->getBlockOperands()[1];
    }
  }];
}
#endif // SCBRANCHINST_DEF

#ifndef SCBRANCHEXECNZ_DEF
#define SCBRANCHEXECNZ_DEF
def SCbranchExecnz : SCbranchInst<"s_cbranch_execnz",
    /*src=*/[EXECType]
  > {
  let description = [{
    If EXECZ is 0 then jump to a constant offset relative to the current PC.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0001001>];
}
#endif // SCBRANCHEXECNZ_DEF

#ifndef SCBRANCHEXECZ_DEF
#define SCBRANCHEXECZ_DEF
def SCbranchExecz : SCbranchInst<"s_cbranch_execz",
    /*src=*/[EXECType]
  > {
  let description = [{
    If EXECZ is 1 then jump to a constant offset relative to the current PC.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0001000>];
}
#endif // SCBRANCHEXECZ_DEF

#ifndef SCBRANCHSCC0_DEF
#define SCBRANCHSCC0_DEF
def SCbranchScc0 : SCbranchInst<"s_cbranch_scc0",
    /*src=*/[SCCType]
  > {
  let description = [{
    If SCC is 0 then jump to a constant offset relative to the current PC.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0000100>];
}
#endif // SCBRANCHSCC0_DEF

#ifndef SCBRANCHSCC1_DEF
#define SCBRANCHSCC1_DEF
def SCbranchScc1 : SCbranchInst<"s_cbranch_scc1",
    /*src=*/[SCCType]
  > {
  let description = [{
    If SCC is 1 then jump to a constant offset relative to the current PC.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0000101>];
}
#endif // SCBRANCHSCC1_DEF

#ifndef SCBRANCHVCCNZ_DEF
#define SCBRANCHVCCNZ_DEF
def SCbranchVccnz : SCbranchInst<"s_cbranch_vccnz",
    /*src=*/[VCCType]
  > {
  let description = [{
    If VCCZ is 0 then jump to a constant offset relative to the current PC.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0000111>];
}
#endif // SCBRANCHVCCNZ_DEF

#ifndef SCBRANCHVCCZ_DEF
#define SCBRANCHVCCZ_DEF
def SCbranchVccz : SCbranchInst<"s_cbranch_vccz",
    /*src=*/[VCCType]
  > {
  let description = [{
    If VCCZ is 1 then jump to a constant offset relative to the current PC.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0000110>];
}
#endif // SCBRANCHVCCZ_DEF

#endif // AMDGCN_INST_CONTROLFLOW_TD
