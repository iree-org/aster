//===- DS.td - DS Operations -------------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines DS specific operations.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_DS_TD
#define AMDGCN_INST_DS_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"


//===----------------------------------------------------------------------===//
// DS Read Operation
//===----------------------------------------------------------------------===//

def DSReadOp : DS_Op<"read", [
    AMDGCNInstOpInterface,
    AMDGCNMemoryReadInstTraits<LDSMemoryResource>,
    AllTypesMatch<["vdst", "result"]>
  ]> {
  let summary = "DS read operation";
  let description = [{
    The DS read operation reads data from LDS (Local Data Share).

    Operands:
    - addr: VGPR containing the address to read from

    Results:
    - result: VGPR containing the read value
  }];

  let arguments = (ins
    AMDGCN_InstAttr:$opcode,
    VGPRRangeType:$vdst,
    VGPRType:$addr,
    // Note: offset0 and offset1 are concatenated 8b into a single 16b attr.
    Offset16Attr:$offset
  );

  let results = (outs VGPRRangeType:$result);

  let assemblyFormat = [{
    $opcode $vdst `,` $addr
    (`,` `offset` `=` $offset^)?
    attr-dict `:` type($addr) `->` type($result)
  }];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getVdstMutable();
    }
    /// Get the instruction input operands.
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getAddrMutable();
    }
  }];
}

class DSReadInst<string _mnemonic, string sym, int dataRegs, InstOp _instOp,
      list<Target> _targets>
    : AMDInst<_mnemonic, sym, _instOp, _targets> {
  let constraints = (ins VGPROperand<dataRegs>:$vdst,
                         VGPROperand<1>:$addr);
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$vdst, PyValue:$addr, IntArg<0>:$offset)>;
  let asmFormat = [SingleAsmVariant<"$vdst, $addr $offset">];
}

def DS_READ_B32_INST : DSReadInst<
  /*mnemonic=*/"ds_read_b32", /*symbol=*/"ds_read_b32",
  /*dataRegs=*/1, /*instOp=*/DSReadOp, /*targets=*/[GFX942]> {
  let summary = "DS read 32-bit";
  let description = [{
    DS read instruction that loads a 32-bit value from LDS.
  }];
}

def DS_READ_B64_INST : DSReadInst<
  /*mnemonic=*/"ds_read_b64", /*symbol=*/"ds_read_b64",
  /*dataRegs=*/2, /*instOp=*/DSReadOp, /*targets=*/[GFX942]> {
  let summary = "DS read 64-bit";
  let description = [{
    DS read instruction that loads a 64-bit value from LDS.
  }];
}

def DS_READ_B96_INST : DSReadInst<
  /*mnemonic=*/"ds_read_b96", /*symbol=*/"ds_read_b96",
  /*dataRegs=*/3, /*instOp=*/DSReadOp, /*targets=*/[GFX942]> {
  let summary = "DS read 96-bit";
  let description = [{
    DS read instruction that loads a 96-bit value from LDS.
  }];
}

def DS_READ_B128_INST : DSReadInst<
  /*mnemonic=*/"ds_read_b128", /*symbol=*/"ds_read_b128",
  /*dataRegs=*/4, /*instOp=*/DSReadOp, /*targets=*/[GFX942]> {
  let summary = "DS read 128-bit";
  let description = [{
    DS read instruction that loads a 128-bit value from LDS.
  }];
}

//===----------------------------------------------------------------------===//
// DS Write Operation
//===----------------------------------------------------------------------===//

def DSWriteOp : DS_Op<"write", [
    AMDGCNInstOpInterface,
    AMDGCNMemoryWriteInstTraits<LDSMemoryResource>,
  ]> {
  let summary = "DS write operation";
  let description = [{
    The DS write operation writes data to LDS (Local Data Share).

    Operands:
    - data: VGPR range containing the data to write
    - addr: VGPR containing the address to write to
  }];
  let arguments = (ins
    AMDGCN_InstAttr:$opcode,
    VGPRRangeType:$data,
    VGPRType:$addr,
    Offset16Attr:$offset
  );
  let assemblyFormat = [{
    $opcode $data `,` $addr
    (`,` `offset` `=` $offset^)?
    attr-dict `:` type($data) `,` type($addr)
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // AMDGCNInstOpInterface Methods
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

class DSWriteInst<string _mnemonic, string sym, int dataRegs, InstOp _instOp,
               list<Target> _targets> : AMDInst<_mnemonic, sym, _instOp, _targets> {
  let constraints = (ins VGPROperand<dataRegs>:$data,
                         VGPROperand<1>:$addr);
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$data, PyValue:$addr, IntArg<0>:$offset)>;
  let asmFormat = [SingleAsmVariant<"$addr, $data $offset">];
}

def DS_WRITE_B32_INST : DSWriteInst<
  /*mnemonic=*/"ds_write_b32", /*symbol=*/"ds_write_b32",
  /*dataRegs=*/1, /*instOp=*/DSWriteOp, /*targets=*/[GFX942]> {
  let summary = "DS write 32-bit";
  let description = [{
    DS write instruction that stores a 32-bit value to LDS.
  }];
}

def DS_WRITE_B64_INST : DSWriteInst<
  /*mnemonic=*/"ds_write_b64", /*symbol=*/"ds_write_b64",
  /*dataRegs=*/2, /*instOp=*/DSWriteOp, /*targets=*/[GFX942]> {
  let summary = "DS write 64-bit";
  let description = [{
    DS write instruction that stores a 64-bit value to LDS.
  }];
}

def DS_WRITE_B96_INST : DSWriteInst<
  /*mnemonic=*/"ds_write_b96", /*symbol=*/"ds_write_b96",
  /*dataRegs=*/3, /*instOp=*/DSWriteOp, /*targets=*/[GFX942]> {
  let summary = "DS write 96-bit";
  let description = [{
    DS write instruction that stores a 96-bit value to LDS.
  }];
}

def DS_WRITE_B128_INST : DSWriteInst<
  /*mnemonic=*/"ds_write_b128", /*symbol=*/"ds_write_b128",
  /*dataRegs=*/4, /*instOp=*/DSWriteOp, /*targets=*/[GFX942]> {
  let summary = "DS write 128-bit";
  let description = [{
    DS write instruction that stores a 128-bit value to LDS.
  }];
}

#endif // AMDGCN_INST_DS_TD
