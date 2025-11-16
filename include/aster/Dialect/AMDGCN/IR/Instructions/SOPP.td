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
  /*instOp=*/AMDGCN_EndKernelOp, /*targets=*/[]> {
  let summary = "End kernel instruction";
  let description = [{
    End kernel instruction that terminates a kernel function.
  }];
  let pythonBuilder = DefaultPyBuilder<(ins)>;
  let asmFormat = [SingleAsmVariant<"">];
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

class SOPPInst<string _mnemonic, string sym, InstOp _instOp, list<Target> _targets>
    : AMDInst<_mnemonic, sym, _instOp, _targets> {
  let pythonBuilder = DefaultPyBuilder<(ins)>;
  let asmFormat = [SingleAsmVariant<"$imm">];
}

def SOPP_TRAP_INST : SOPPInst<
  /*mnemonic=*/"s_trap", /*symbol=*/"s_trap",
  /*instOp=*/SOPPOp, /*targets=*/[GFX942]> {
  let summary = "SOPP trap";
  let description = [{
    SOPP instruction that generates a trap exception.
  }];
  let pythonBuilder = DefaultPyBuilder<(ins IntArg<0>:$imm)>;
}

def SOPP_BARRIER_INST : SOPPInst<
  /*mnemonic=*/"s_barrier", /*symbol=*/"s_barrier",
  /*instOp=*/SOPPOp, /*targets=*/[GFX942]> {
  let summary = "SOPP barrier";
  let description = [{
    SOPP instruction that creates a barrier synchronization point.
  }];
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
    LgkmcntAttr:$lgkmcnt
  );

  let results = (outs);

  let assemblyFormat = [{
    $opcode
    (`vmcnt` `=` $vmcnt^)?
    (`expcnt` `=` $expcnt^)?
    (`lgkmcnt` `=` $lgkmcnt^)?
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
  /*instOp=*/SWaitcntOp, /*targets=*/[GFX942]> {
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
