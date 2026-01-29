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

//===----------------------------------------------------------------------===//
// EndKernelOp
//===----------------------------------------------------------------------===//

def AMDGCN_EndKernelOp : InstOp<"end_kernel", [
    Terminator, ReturnLike, HasParent<"KernelOp">,
    AMDGCNInstOpInterface
  ]> {
  let summary = "AMDGCN end kernel operation";
  let description = [{Terminates an AMDGCN kernel function.}];
  let arguments = (ins
    DefaultValuedAttr<AMDGCN_InstAttr, "mlir::aster::amdgcn::OpCode::S_ENDPGM">:$opcode);
  let assemblyFormat = "attr-dict";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return {};
    }
    /// Get the instruction input operands.
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return {};
    }
  }];
  let cppNamespace = "::mlir::aster::amdgcn";
}

def EndKernelInst : AMDInst<
  /*mnemonic=*/"s_endpgm", /*symbol=*/"s_endpgm",
  /*instOp=*/AMDGCN_EndKernelOp, /*isa=*/[]> {
  let summary = "End kernel instruction";
  let description = [{
    End kernel instruction that terminates a kernel function.
  }];
  let pythonBuilder = DefaultPyBuilder<(ins)>;
  let asmFormat = [SingleAsmVariant<"">];
}

//===----------------------------------------------------------------------===//
// BranchOp/CondBranchOp
//===----------------------------------------------------------------------===//

def BranchOperand : AnyRegOf<[SReg<"SCC">, SReg<"VCCZ">, SReg<"EXECZ">]>;

def AMDGCN_BranchOp : InstOp<"branch", [
    Pure, Terminator, AMDGCNInstOpInterface
  ]> {
  let summary = "AMDGCN unconditional branch operation";
  let description = [{
    Unconditional branching operation.
  }];
  let arguments = (ins AMDGCN_InstAttr:$opcode);
  let successors = (successor BranchLabel:$dest);
  let assemblyFormat = "custom<Opcode>($opcode) $dest attr-dict";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return {};
    }
    /// Get the instruction input operands.
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return {};
    }
  }];
  let cppNamespace = "::mlir::aster::amdgcn";
}

def AMDGCN_CBranchOp : InstOp<"cbranch", [
  Terminator, AMDGCNInstOpInterface, HasParent<"KernelOp">]> {
  let summary = "AMDGCN conditional branch operation";
  let description = [{
    Conditional branching operation that branches based on a condition register.

    Example:
    ```mlir
    // Branch to ^loop if SCC is 1, otherwise fall through to ^exit
    amdgcn.cbranch s_cbranch_scc1 %cond ^loop fallthrough (^exit)
      : !amdgcn.scc
    ```
  }];
  let arguments = (ins AMDGCN_InstAttr:$opcode, BranchOperand:$cond);
  let successors = (successor BranchLabel:$dest, BranchLabel:$fallthrough);
  let assemblyFormat = [{
    custom<Opcode>($opcode)  $cond $dest `fallthrough` `(` $fallthrough `)`
    attr-dict `:` type($cond)
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return {};
    }
    /// Get the instruction input operands.
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getCondMutable();
    }
  }];
  let cppNamespace = "::mlir::aster::amdgcn";
}

def BranchInst : AMDInst</*mnemonic=*/"s_branch", /*symbol=*/"s_branch",
    /*instOp=*/AMDGCN_BranchOp, /*targets=*/[]> {
  let summary = "Unconditional Branch instruction";
  let description = [{
    Branch instruction that performs unconditional branching.
  }];
  let cppBuilder = OpBuilder<(ins "Block *":$dest), [{
    return $_create($_builder, $_loc, $_opcode, dest);
  }]>;
  let pythonBuilder = OpBuilder<(ins PyBlock:$dest), [{
    return $_create(opcode=$_opcode, dest=dest);
  }]>;
  let asmFormat = [SingleAsmVariant<"$dest">];
}

class CBranchInst<string _mnemonic, string sym>
  : AMDInst<_mnemonic, sym, /*instOp=*/AMDGCN_CBranchOp, /*isa=*/[]> {
  let cppBuilder = OpBuilder<(ins "Value":$cond, "Block *":$dest, "Block *":$fallthrough), [{
    return $_create($_builder, $_loc, $_opcode, cond, dest, fallthrough);
  }]>;
  let pythonBuilder = OpBuilder<(ins PyValue:$cond, PyBlock:$dest, PyBlock:$fallthrough), [{
    return $_create(opcode=$_opcode, cond=cond, dest=dest, fallthrough=fallthrough);
  }]>;
  // Assembly output: s_cbranch_cond* label (SCC is implicit in hardware encoding)
  let asmFormat = [SingleAsmVariant<"$dest">];
}

def S_CBRANCH_SCC0_INST : CBranchInst<
  /*mnemonic=*/"s_cbranch_scc0", /*symbol=*/"s_cbranch_scc0"> {
  let summary = "Conditional Branch if SCC is 0";
  let description = [{
    Conditional branch instruction that branches if SCC is 0.
  }];
}

def S_CBRANCH_SCC1_INST : CBranchInst<
  /*mnemonic=*/"s_cbranch_scc1", /*symbol=*/"s_cbranch_scc1"> {
  let summary = "Conditional Branch if SCC is 1";
  let description = [{
    Conditional branch instruction that branches if SCC is 1.
  }];
}

#endif // AMDGCN_INST_CONTROLFLOW_TD
