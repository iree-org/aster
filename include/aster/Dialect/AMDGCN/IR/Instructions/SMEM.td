//===- SMEM.td - SMEM Operations ---------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SMEM specific operations.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_SMEM_TD
#define AMDGCN_INST_SMEM_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"

//===----------------------------------------------------------------------===//
// SMEM Load Operation
//===----------------------------------------------------------------------===//

def SMEMLoadOp : SMEM_Op<"load", [
    AMDGCNInstOpInterface,
    AMDGCNMemoryReadInstTraits<GlobalMemoryResource>,
    AllTypesMatch<["sdst", "result"]>
  ]> {
  let summary = "SMEM load operation";
  let description = [{
    The SMEM load operation loads data from scalar memory using SGPRs as the address.

    Operands:
    - sdst: SGPR range destination for the loaded value
    - addr: SGPR range of size 2 containing the address to load from

    Results:
    - result: SGPR range containing the loaded value

    Example:
    ```mlir
    // Load 32-bit value
    %result = smem_load #amdgcn.smem_load_opcode<s_load_dword> %sdst, %addr offset = 0
      : !amdgcn.sgpr_range<[12 : 13]>, !amdgcn.sgpr_range<[10 : 12]> -> !amdgcn.sgpr_range<[12 : 13]>
    ```
  }];

  let arguments = (ins
    AMDGCN_InstAttr:$opcode,
    AnyRegOf<[SGPRRangeType, SGPRType]>:$sdst,
    OptionalReg<SGPRRangeType>:$addr,
    OffsetImm32Attr:$offset
  );

  let results = (outs AnyRegOf<[SGPRRangeType, SGPRType]>:$result);

  let assemblyFormat = [{
    $opcode
    $sdst
    (`,` $addr^)?
    (`offset` `=` $offset^)?
    attr-dict
    `:` type($sdst) (`,` type($addr)^)? `->` type($result)
  }];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getSdstMutable();
    }
    /// Get the instruction input operands.
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getAddrMutable();
    }
  }];
}

class SMEMLoadInst<string _mnemonic, string sym, int addrRegs, int resultRegs,
                   InstOp _instOp, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, sym, _instOp, _isa> {
  let constraints = (ins SGPROperand<resultRegs>:$sdst,
                         SGPROperand<addrRegs, 1>:$addr);
  let cppBuilder = DefaultCppBuilder<(ins
    CppValue:$sdst, CppValue:$addr, IntArg<0>:$offset)>;
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$sdst, PyValue:$addr, IntArg<0>:$offset)>;
  let asmFormat = [SingleAsmVariant<"$sdst, $addr, $offset">];
}

def SMEM_LOAD_DWORD_INST : SMEMLoadInst<
  /*mnemonic=*/"s_load_dword", /*symbol=*/"s_load_dword",
  /*addrRegs=*/2, /*resultRegs=*/1, /*instOp=*/SMEMLoadOp, /*isa=*/[CDNA3]> {
  let summary = "SMEM load 32-bit";
  let description = [{
    SMEM load instruction that loads a 32-bit value from scalar memory.
  }];
}

def SMEM_LOAD_DWORDX2_INST : SMEMLoadInst<
  /*mnemonic=*/"s_load_dwordx2", /*symbol=*/"s_load_dwordx2",
  /*addrRegs=*/2, /*resultRegs=*/2, /*instOp=*/SMEMLoadOp, /*isa=*/[CDNA3]> {
  let summary = "SMEM load 64-bit";
  let description = [{
    SMEM load instruction that loads a 64-bit value from scalar memory.
  }];
}

def SMEM_LOAD_DWORDX4_INST : SMEMLoadInst<
  /*mnemonic=*/"s_load_dwordx4", /*symbol=*/"s_load_dwordx4",
  /*addrRegs=*/2, /*resultRegs=*/4, /*instOp=*/SMEMLoadOp, /*isa=*/[CDNA3]> {
  let summary = "SMEM load 128-bit";
  let description = [{
    SMEM load instruction that loads a 128-bit value from scalar memory.
  }];
}

def SMEM_LOAD_DWORDX8_INST : SMEMLoadInst<
  /*mnemonic=*/"s_load_dwordx8", /*symbol=*/"s_load_dwordx8",
  /*addrRegs=*/2, /*resultRegs=*/8, /*instOp=*/SMEMLoadOp, /*isa=*/[CDNA3]> {
  let summary = "SMEM load 256-bit";
  let description = [{
    SMEM load instruction that loads a 256-bit value from scalar memory.
  }];
}

def SMEM_LOAD_DWORDX16_INST : SMEMLoadInst<
  /*mnemonic=*/"s_load_dwordx16", /*symbol=*/"s_load_dwordx16",
  /*addrRegs=*/2, /*resultRegs=*/16, /*instOp=*/SMEMLoadOp, /*isa=*/[CDNA3]> {
  let summary = "SMEM load 512-bit";
  let description = [{
    SMEM load instruction that loads a 512-bit value from scalar memory.
  }];
}

// TODO: this requires extra verifiers:
//   - no offset
//   - no addr operand
//   - etc
def SMEM_MEMTIME_INST : SMEMLoadInst<
  /*mnemonic=*/"s_memtime", /*symbol=*/"s_memtime",
  /*addrRegs=*/0, /*resultRegs=*/2, /*instOp=*/SMEMLoadOp, /*isa=*/[CDNA3]> {
  let summary = "SMEM load 128-bit";
  let description = [{
    SMEM time instruction.
  }];
  let asmFormat = [SingleAsmVariant<"$sdst">];
}

//===----------------------------------------------------------------------===//
// SMEM Store Operation
//===----------------------------------------------------------------------===//

def SMEMStoreOp : SMEM_Op<"store", [
    AMDGCNInstOpInterface,
    AMDGCNMemoryWriteInstTraits<GlobalMemoryResource>
  ]> {
  let summary = "SMEM store operation";
  let description = [{
    The SMEM store operation stores data to scalar memory using SGPRs as the address.

    Operands:
    - addr: SGPR range of size 2 containing the address to store to
    - data: SGPR range containing the data to store

    Example:
    ```mlir
    // Store 32-bit value
    smem_store #amdgcn.smem_store_opcode<s_store_dword> %data, %addr offset = 0
      : !amdgcn.sgpr_range<[12 : 13]>, !amdgcn.sgpr_range<[10 : 12]>
    ```
  }];

  let arguments = (ins
    AMDGCN_InstAttr:$opcode,
    SGPRRangeType:$data,
    SGPRRangeType:$addr,
    OffsetImm32Attr:$offset
  );

  let results = (outs);

  let assemblyFormat = [{
    $opcode $data `,` $addr
    (`,` `offset` `=` $offset^)?
    attr-dict `:` type($data) `,` type($addr)
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
      return getOperation()->getOpOperands();
    }
  }];
}

class SMEMStoreInst<string _mnemonic, string sym, int dataRegs, int addrRegs,
                    InstOp _instOp, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, sym, _instOp, _isa> {
  let constraints = (ins SGPROperand<dataRegs>:$data,
                         SGPROperand<addrRegs>:$addr);
  let cppBuilder = DefaultCppBuilder<(ins
    CppValue:$data, CppValue:$addr, IntArg<0>:$offset)>;
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$data, PyValue:$addr, IntArg<0>:$offset)>;
  let asmFormat = [SingleAsmVariant<"$data, $addr $offset">];
}

def SMEM_STORE_DWORD_INST : SMEMStoreInst<
  /*mnemonic=*/"s_store_dword", /*symbol=*/"s_store_dword",
  /*dataRegs=*/1, /*addrRegs=*/2, /*instOp=*/SMEMStoreOp, /*isa=*/[CDNA3]> {
  let summary = "SMEM store 32-bit";
  let description = [{
    SMEM store instruction that stores a 32-bit value to scalar memory.
  }];
}

def SMEM_STORE_DWORDX2_INST : SMEMStoreInst<
  /*mnemonic=*/"s_store_dwordx2", /*symbol=*/"s_store_dwordx2",
  /*dataRegs=*/2, /*addrRegs=*/2, /*instOp=*/SMEMStoreOp, /*isa=*/[CDNA3]> {
  let summary = "SMEM store 64-bit";
  let description = [{
    SMEM store instruction that stores a 64-bit value to scalar memory.
  }];
}

def SMEM_STORE_DWORDX4_INST : SMEMStoreInst<
  /*mnemonic=*/"s_store_dwordx4", /*symbol=*/"s_store_dwordx4",
  /*dataRegs=*/4, /*addrRegs=*/2, /*instOp=*/SMEMStoreOp, /*isa=*/[CDNA3]> {
  let summary = "SMEM store 128-bit";
  let description = [{
    SMEM store instruction that stores a 128-bit value to scalar memory.
  }];
}

#endif // AMDGCN_INST_SMEM_TD
