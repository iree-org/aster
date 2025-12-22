//===- SOPP.td - SOPP Operations ---------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SOPP specific operations.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_SOPP_TD
#define AMDGCN_INST_SOPP_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"

//===----------------------------------------------------------------------===//
// EndKernelOp
//===----------------------------------------------------------------------===//

def AMDGCN_EndKernelOp : InstOp<"end_kernel", [
    Terminator, ReturnLike, HasParent<"KernelOp">,
    AMDGCNInstOpInterface
  ]> {
  let summary = "AMDGCN end kernel operation";
  let description = [{
    Terminates an AMDGCN kernel function.
  }];
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

def AMDGCN_BranchOp : InstOp<"branch", [
  Pure, Terminator, AMDGCNInstOpInterface, HasParent<"KernelOp">]> {
  let summary = "AMDGCN unconditional branch operation";
  let description = [{
    unconditional branching operation.
  }];
  let arguments = (ins AMDGCN_InstAttr:$opcode);
  let successors = (successor BranchLabel:$dest );
  let assemblyFormat = "$opcode $dest attr-dict";
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
    Conditional branching operation that branches based on the SCC register value.

    Example:
    ```mlir
    // Branch to ^loop if SCC is 1, otherwise fall through to ^exit
    amdgcn.cbranch #amdgcn.inst<s_cbranch_scc1> %scc ^loop fallthrough (^exit)
      : !amdgcn.sreg<scc>
    ```
  }];
  let arguments = (ins AMDGCN_InstAttr:$opcode, SREGType:$scc);
  let successors = (successor BranchLabel:$dest, BranchLabel:$fallthrough);
  let assemblyFormat = "$opcode $scc $dest `fallthrough` `(` $fallthrough `)` attr-dict `:` type($scc)";
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
      return getSccMutable();
    }
  }];
  let cppNamespace = "::mlir::aster::amdgcn";
}

def BranchInst : AMDInst<
  /*mnemonic=*/"s_branch", /*symbol=*/"s_branch",
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
  let cppBuilder = OpBuilder<(ins "Value":$scc, "Block *":$dest, "Block *":$fallthrough), [{
    return $_create($_builder, $_loc, $_opcode, scc, dest, fallthrough);
  }]>;
  let pythonBuilder = OpBuilder<(ins PyValue:$scc, PyBlock:$dest, PyBlock:$fallthrough), [{
    return $_create(opcode=$_opcode, scc=scc, dest=dest, fallthrough=fallthrough);
  }]>;
  // Assembly output: s_cbranch_scc* label (SCC is implicit in hardware encoding)
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

//===----------------------------------------------------------------------===//
// SOPP Operation
//===----------------------------------------------------------------------===//

def SOPPOp : SOPP_Op<"sopp", [
    AMDGCNInstOpInterface
  ]> {
  let summary = "SOPP (Scalar Operations with Pairs) operation";
  let description = [{
    The SOPP operation performs scalar operations with pairs for CDNA architecture.
    These operations typically have no register operands and use immediate values.
  }];

  let arguments = (ins
    AMDGCN_InstAttr:$opcode,
    Imm16Attr:$imm
  );

  let results = (outs);

  let assemblyFormat = [{
    $opcode
    (`,` `imm` `=` $imm^)?
    attr-dict
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
      return {};
    }
  }];
}

class SOPPInst<string _mnemonic, string sym, InstOp _instOp, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, sym, _instOp, _isa> {
  let pythonBuilder = DefaultPyBuilder<(ins)>;
  let asmFormat = [SingleAsmVariant<"$imm">];
}

def SOPP_TRAP_INST : SOPPInst<
  /*mnemonic=*/"s_trap", /*symbol=*/"s_trap",
  /*instOp=*/SOPPOp, /*isa=*/[CDNA3]> {
  let summary = "SOPP trap";
  let description = [{
    SOPP instruction that generates a trap exception.
  }];
  let pythonBuilder = DefaultPyBuilder<(ins IntArg<0>:$imm)>;
}

def SOPP_BARRIER_INST : SOPPInst<
  /*mnemonic=*/"s_barrier", /*symbol=*/"s_barrier",
  /*instOp=*/SOPPOp, /*isa=*/[CDNA3]> {
  let summary = "SOPP barrier";
  let description = [{
    SOPP instruction that creates a barrier synchronization point.
  }];
}

def SOPP_NOP_INST : SOPPInst<
  /*mnemonic=*/"s_nop", /*symbol=*/"s_nop",
  /*instOp=*/SOPPOp, /*isa=*/[CDNA3]> {
  let summary = "SOPP NOP";
  let description = [{
    Do nothing. Delay issue of next instruction by a small, fixed amount.
    0x0 means the next instruction can issue on the next clock, 0xf means the
    next instruction can issue 16 clocks later.
  }];
  let cppBuilder = DefaultCppBuilder<(ins
    IntArg<0>:$imm
  )>;
  let pythonBuilder = DefaultPyBuilder<(ins IntArg<0>:$imm)>;
}

//===----------------------------------------------------------------------===//
// S_WAITCNT Operation
//===----------------------------------------------------------------------===//

def SWaitcntOp : SOPP_Op<"s_waitcnt", [
    AMDGCNInstOpInterface
  ]> {
  let summary = "s_waitcnt operation";
  let description = [{
    The s_waitcnt operation waits for the counts of outstanding local data share,
    vector memory and export instructions to be at or below the specified levels.

    Example:
    ```mlir
    // Wait for all counters to reach 0
    s_waitcnt vmcnt = 0 expcnt = 0 lgkmcnt = 0

    // Wait for specific counters
    s_waitcnt vmcnt = 5 expcnt = 2 lgkmcnt = 1
    ```
  }];

  let arguments = (ins
    DefaultValuedAttr<AMDGCN_InstAttr, "mlir::aster::amdgcn::OpCode::S_WAITCNT">:$opcode,
    VmcntAttr:$vmcnt,
    ExpcntAttr:$expcnt,
    LgkmcntAttr:$lgkmcnt,
    UnitAttr:$immutable
  );
  let results = (outs);
  let assemblyFormat = [{
    $opcode
    (`vmcnt` `=` $vmcnt^)?
    (`expcnt` `=` $expcnt^)?
    (`lgkmcnt` `=` $lgkmcnt^)?
    (`immutable` $immutable^)?
    attr-dict
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
      return {};
    }
  }];
}

def S_WAITCNT_INST : AMDInst<
  /*mnemonic=*/"s_waitcnt", /*symbol=*/"s_waitcnt",
  /*instOp=*/SWaitcntOp, /*isa=*/[CDNA3]> {
  let summary = "S_WAITCNT instruction";
  let description = [{
    Wait for counts of outstanding lds, vector-memory and export/vmem-write-data to be at or below specified levels.
  }];
  let asmFormat = [SingleAsmVariant<"$vmcnt $expcnt $lgkmcnt">];
  let cppBuilder = DefaultCppBuilder<(ins
    IntArg<0>:$vmcnt,
    IntArg<0>:$expcnt,
    IntArg<0>:$lgkmcnt
  )>;
  let pythonBuilder = DefaultPyBuilder<(ins
    IntArg<0>:$vmcnt,
    IntArg<0>:$expcnt,
    IntArg<0>:$lgkmcnt
  )>;
}

#endif // AMDGCN_INST_SOPP_TD
