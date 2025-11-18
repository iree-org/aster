//===- FLAT.td - FLAT Operations ---------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines FLAT specific operations.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_FLAT_TD
#define AMDGCN_INST_FLAT_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"


//===----------------------------------------------------------------------===//
// Global Load Operation
//===----------------------------------------------------------------------===//

def GlobalLoadOp : FLAT_Op<"global_load", [
    AMDGCNInstOpInterface,
    AMDGCNMemoryReadInstTraits<GlobalMemoryResource>,
    AllTypesMatch<["vdst", "result"]>
  ]> {
  let summary = "Global load operation";
  let description = [{
    The Global load operation loads data from global memory using a VGPR or SGPR as the address.

    Operands:
    - vdst: VGPR range destination for the loaded value
    - addr: VGPR or SGPR range of size 2 containing the address to load from

    Results:
    - result: VGPR range containing the loaded value
  }];
  let arguments = (ins
    AMDGCN_InstAttr:$opcode,
    AnyRegOf<[VGPRRangeType, VGPRType]>:$vdst,
    AnyRegOf<[VGPRRangeType, SGPRRangeType]>:$addr,
    OptionalReg<VGPRType>:$vgpr_offset,
    Offset32Attr:$offset
  );
  let results = (outs VGPRRangeType:$result);
  let assemblyFormat = [{
    $opcode $vdst `,` $addr
    (`[` $vgpr_offset^ `]`)?
    (`,` `offset` `=` $offset^)?
    attr-dict `:` type($vdst) `,` type($addr) (`[` type($vgpr_offset)^ `]`)? `->` type($result)
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // Builders
    //===------------------------------------------------------------------===//
    /// Builder with vgpr_offset
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ::mlir::aster::amdgcn::OpCode opcode, ::mlir::Value vdst,
                     ::mlir::Value addr, ::mlir::Value vgpr_offset,
                     int32_t offset);
    /// Builder with static offset
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ::mlir::aster::amdgcn::OpCode opcode, ::mlir::Value vdst,
                     ::mlir::Value addr, int32_t offset = 0);

    //===------------------------------------------------------------------===//
    // AMDGCNInstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getVdstMutable();
    }
    /// Get the instruction input operands.
    MutableArrayRef<OpOperand> getInstInsMutable() {
      if (getVgprOffset())
        return getOperation()->getOpOperands().drop_front();
      return getOperation()->getOpOperands().drop_front();
    }
  }];
}

class GlobalLoadInst<string _mnemonic, string sym, int addrRegs, int resultRegs,
                     list<Target> _targets>
    : AMDInst<_mnemonic, sym, GlobalLoadOp, _targets> {
  let constraints = (ins VGPROperand<resultRegs>:$vdst,
                         AddrOperand<addrRegs>:$addr,
                         VGPROperand<1, 1>:$vgpr_offset);
  let cppBuilder = DefaultCppBuilder<(ins
    CppValue:$vdst, CppValue:$addr, CppValue:$vgpr_offset, IntArg<0>:$offset)>;
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$vdst, PyValue:$addr, IntArg<0>:$offset)>;
  let asmFormat = [
    AsmVariant<CPred<"isa<VGPRRangeType>($_self.getAddr().getType())">, "$vdst, $addr, off $offset">,
    AsmVariant<CPred<"isa<SGPRRangeType>($_self.getAddr().getType())">, "$vdst, $vgpr_offset, $addr">
  ];
}

def GLOBAL_LOAD_DWORD_INST : GlobalLoadInst<
  /*mnemonic=*/"global_load_dword", /*symbol=*/"global_load_dword",
  /*addrRegs=*/2, /*resultRegs=*/1, /*targets=*/[GFX942]> {
  let summary = "Global load 32-bit";
  let description = [{
    Global load instruction that loads a 32-bit value from global memory.
  }];
}

def GLOBAL_LOAD_DWORDX2_INST : GlobalLoadInst<
  /*mnemonic=*/"global_load_dwordx2", /*symbol=*/"global_load_dwordx2",
  /*addrRegs=*/2, /*resultRegs=*/2, /*targets=*/[GFX942]> {
  let summary = "Global load 64-bit";
  let description = [{
    Global load instruction that loads a 64-bit value from global memory.
  }];
}

def GLOBAL_LOAD_DWORDX3_INST : GlobalLoadInst<
  /*mnemonic=*/"global_load_dwordx3", /*symbol=*/"global_load_dwordx3",
  /*addrRegs=*/2, /*resultRegs=*/3, /*targets=*/[GFX942]> {
  let summary = "Global load 96-bit";
  let description = [{
    Global load instruction that loads a 96-bit value from global memory.
  }];
}

def GLOBAL_LOAD_DWORDX4_INST : GlobalLoadInst<
  /*mnemonic=*/"global_load_dwordx4", /*symbol=*/"global_load_dwordx4",
  /*addrRegs=*/2, /*resultRegs=*/4, /*targets=*/[GFX942]> {
  let summary = "Global load 128-bit";
  let description = [{
    Global load instruction that loads a 128-bit value from global memory.
  }];
}

//===----------------------------------------------------------------------===//
// Global Store Operation
//===----------------------------------------------------------------------===//

def GlobalStoreOp : FLAT_Op<"global_store", [
    AMDGCNInstOpInterface,
    AMDGCNMemoryWriteInstTraits<GlobalMemoryResource>
  ]> {
  let summary = "Global store operation";
  let description = [{
    The Global store operation stores data to global memory using a VGPR or SGPR as the address.

    Operands:
    - data: VGPR or AGPR range containing the data to store
    - addr: VGPR or SGPR range of size 2 containing the address to store to

    Example:
    ```mlir
    // Store 32-bit value from VGPR
    global_store #amdgcn.global_store_opcode<global_store_dword> %data, %addr offset = 0
      : !amdgcn.vgpr_range<[12 : 13]>, !amdgcn.vgpr_range<[10 : 12]>
    // Store 32-bit value from AGPR
    global_store #amdgcn.global_store_opcode<global_store_dword> %data, %addr offset = 0
      : !amdgcn.agpr_range<[12 : 13]>, !amdgcn.vgpr_range<[10 : 12]>
    ```
  }];

  let arguments = (ins
    AMDGCN_InstAttr:$opcode,
    AnyRegOf<[VGPRRangeType, AGPRRangeType, VGPRType]>:$data,
    AnyRegOf<[VGPRRangeType, SGPRRangeType]>:$addr,
    // TODO: we really need a verifier here: if sgpr then vgpr_offset must be present.
    OptionalReg<VGPRType>:$vgpr_offset,
    Offset32Attr:$offset
  );

  let results = (outs);

  let assemblyFormat = [{
    $opcode $data `,` $addr
    (`[` $vgpr_offset^ `]`)?
    (`,` `offset` `=` $offset^)?
    attr-dict `:` type($data) `,` type($addr) (`[` type($vgpr_offset)^ `]`)?
  }];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // Builders
    //===------------------------------------------------------------------===//
    /// Builder with vgpr_offset
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ::mlir::aster::amdgcn::OpCode opcode, ::mlir::Value data,
                     ::mlir::Value addr, ::mlir::Value vgpr_offset,
                     int32_t offset);
    /// Builder with static offset
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ::mlir::aster::amdgcn::OpCode opcode, ::mlir::Value data,
                     ::mlir::Value addr, int32_t offset = 0);

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

class GlobalStoreInst<string _mnemonic, string sym, int dataRegs, int addrRegs,
                      InstOp _instOp, list<Target> _targets>
    : AMDInst<_mnemonic, sym, _instOp, _targets> {
  let constraints = (ins AccOperand<dataRegs>:$data,
                         AddrOperand<addrRegs>:$addr,
                         VGPROperand<1, 1>:$vgpr_offset);
  let cppBuilder = DefaultCppBuilder<(ins
    CppValue:$data, CppValue:$addr, CppValue:$vgpr_offset, IntArg<0>:$offset)>;
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$data, PyValue:$addr, IntArg<0>:$offset)>;
  let asmFormat = [
    AsmVariant<CPred<"isa<VGPRRangeType>($_self.getAddr().getType())">, "$addr, $data, off $offset">,
    // TODO: we really need a verifier here: if sgpr then vgpr_offset must be present.
    AsmVariant<CPred<"isa<SGPRRangeType>($_self.getAddr().getType())">, "$vgpr_offset, $data, $addr $offset">
  ];
}

def GLOBAL_STORE_DWORD_INST : GlobalStoreInst<
  /*mnemonic=*/"global_store_dword", /*symbol=*/"global_store_dword",
  /*dataRegs=*/1, /*addrRegs=*/2, /*instOp=*/GlobalStoreOp, /*targets=*/[GFX942]> {
  let summary = "Global store 32-bit";
  let description = [{
    Global store instruction that stores a 32-bit value to global memory.
  }];
}

def GLOBAL_STORE_DWORDX2_INST : GlobalStoreInst<
  /*mnemonic=*/"global_store_dwordx2", /*symbol=*/"global_store_dwordx2",
  /*dataRegs=*/2, /*addrRegs=*/2, /*instOp=*/GlobalStoreOp, /*targets=*/[GFX942]> {
  let summary = "Global store 64-bit";
  let description = [{
    Global store instruction that stores a 64-bit value to global memory.
  }];
}

def GLOBAL_STORE_DWORDX3_INST : GlobalStoreInst<
  /*mnemonic=*/"global_store_dwordx3", /*symbol=*/"global_store_dwordx3",
  /*dataRegs=*/3, /*addrRegs=*/2, /*instOp=*/GlobalStoreOp, /*targets=*/[GFX942]> {
  let summary = "Global store 96-bit";
  let description = [{
    Global store instruction that stores a 96-bit value to global memory.
  }];
}

def GLOBAL_STORE_DWORDX4_INST : GlobalStoreInst<
  /*mnemonic=*/"global_store_dwordx4", /*symbol=*/"global_store_dwordx4",
  /*dataRegs=*/4, /*addrRegs=*/2, /*instOp=*/GlobalStoreOp, /*targets=*/[GFX942]> {
  let summary = "Global store 128-bit";
  let description = [{
    Global store instruction that stores a 128-bit value to global memory.
  }];
}

#endif // AMDGCN_INST_FLAT_TD
