//===- Memory.td - Memory related operations ---------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines memory related operations.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_MEMORY_TD
#define AMDGCN_INST_MEMORY_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"
include "aster/Interfaces/DependentOpInterface.td"

//===----------------------------------------------------------------------===//
// Memory address operand type
//===----------------------------------------------------------------------===//

/// A load or store address operand.
/// DS ops use a single VGPR for the address, while global ops use a two
/// word VGPR or SGPR range, and buffer ops use an SGPR range of size 4.
def AddrOperand :
  AnyRegOf<[VGPRType, SizedGPRRange<[VGPRRangeType, SGPRRangeType], [2, 4]>]>;

/// A load or store offset operand.
def UniformOffsetOperand : AnyRegOf<[SGPRType]>;

/// A load or store offset operand.
def DynamicOffsetOperand : AnyRegOf<[VGPRType]>;

/// A load or store data operand.
def DataOperand : AnyRegOf<[
  AGPRType, VGPRType, SGPRType, AGPRRangeType, VGPRRangeType, SGPRRangeType
]>;

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def AMDGCN_LoadOp : InstOp<"load", [
    AttrSizedOperandSegments,
    AMDGCNInstOpInterface,
    ConditionallySpeculatable,
    DeclareOpInterfaceMethods<DependentOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let summary = "AMDGCN memory load operation";
  let description = [{
    The `load` operation loads data from memory into registers. The specific
    instruction used by this operation is determined by the opcode argument.

    Operands:
    - opcode: The instruction opcode
    - dest: Destination register (VGPR or SGPR, or ranges of them)
    - addr: Address register (A two word VGPR or SGPR range)
    - uniform_offset[optional]: Uniform offset register (An SGPR)
    - dynamic_offset[optional]: Dynamic offset register (A VGPR)
    - constant_offset[optional]: Constant offset (i32)

    Results:
    - result[optional]: The loaded value (same type as dest)
    - token: A read token indicating the memory operation

    Syntax:

    ```mlir
    amdgcn.load <opcode> dest <dest> addr <addr>
    offset u(<uniform_offset>)? + d(<dynamic_offset>)? + c(<constant_offset>)?
    : out-types in-types -> token
    ```

    Example:

    ```mlir
    %result, %token = amdgcn.load global_load_b32 dest %dest addr %addr
      : dps(!amdgcn.vgpr_range<[0:4]>) ins(!amdgcn.sgpr_range<[0:2]>)
        -> !amdgcn.read_token<flat>
    ```
  }];
  let arguments = (ins
    AMDGCN_InstAttr:$opcode,
    DataOperand:$dest,
    AddrOperand:$addr,
    OptionalReg<UniformOffsetOperand>:$uniform_offset,
    OptionalReg<DynamicOffsetOperand>:$dynamic_offset,
    OptionalOffsetType:$constant_offset
  );
  let results = (outs
    Optional<DataOperand>:$result,
    ReadTokenType:$token
  );
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "OpCode":$opcode, "Value":$dest, "Value":$addr,
      CArg<"Value", "nullptr">:$uniform_offset,
      CArg<"Value", "nullptr">:$dynamic_offset,
      CArg<"Value", "nullptr">:$constant_offset)>
  ];
  let assemblyFormat = [{
    custom<Opcode>($opcode) `dest` $dest `addr` $addr ``
    custom<Offsets>($uniform_offset, $dynamic_offset, $constant_offset)
    attr-dict `:`
    custom<OutTypes>(type($dest), type($result))
    `ins` `` `(` type($addr) ``
    custom<OffsetTypes>(
      ref($uniform_offset), ref($dynamic_offset), ref($constant_offset),
      type($uniform_offset), type($dynamic_offset), type($constant_offset))
    `)` `->` qualified(type($token))
  }];
  let extraClassDeclaration = [{
    MemoryInstructionKind getInstKind() {
      return getMemoryInstructionKind(getOpcode());
    }
    /// Returns the speculatability of the operation.
    Speculation::Speculatability getSpeculatability() {
      return aster::detail::getInstSpeculatabilityImpl(*this);
    }

    //===------------------------------------------------------------------===//
    // AMDGCNInstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    OperandRange getInstOuts() {
      return getAsOperandRange(getDestMutable());
    }
    /// Get the instruction input operands.
    OperandRange getInstIns();
    /// Get the instruction results.
    ResultRange getInstResults() {
      return getOperation()->getResult(0);
    }
  }];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def AMDGCN_StoreOp : InstOp<"store", [
    AttrSizedOperandSegments,
    AMDGCNInstOpInterface,
    ConditionallySpeculatable,
    DeclareOpInterfaceMethods<DependentOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let summary = "AMDGCN memory store operation";
  let description = [{
    The `store` operation stores data from a register into memory. The specific
    instruction used by this operation is determined by the opcode argument.

    Operands:
    - opcode: The instruction opcode
    - data: Data to store (VGPR or SGPR, or ranges of them)
    - addr: Address register (A two word VGPR or SGPR range)
    - uniform_offset[optional]: Uniform offset register (An SGPR)
    - dynamic_offset[optional]: Dynamic offset register (A VGPR)
    - constant_offset[optional]: Constant offset (i32)

    Results:
    - token: A write token indicating the memory operation

    Syntax:

    ```mlir
    amdgcn.store <opcode> data <data> addr <addr>
    offset u(<uniform_offset>)? + d(<dynamic_offset>)? + c(<constant_offset>)?
    : in-types -> token
    ```

    Example:
    ```mlir
    %token = amdgcn.store ds_read_b128 data %data addr %addr offset c(%off)
      : ins(!amdgcn.vgpr_range<[0:16]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<shared>
    ```
  }];
  let arguments = (ins
    AMDGCN_InstAttr:$opcode,
    DataOperand:$data,
    AddrOperand:$addr,
    OptionalReg<UniformOffsetOperand>:$uniform_offset,
    OptionalReg<DynamicOffsetOperand>:$dynamic_offset,
    OptionalOffsetType:$constant_offset
  );
  let results = (outs WriteTokenType:$token);
  let builders = [
    OpBuilder<(ins
      "OpCode":$opcode, "Value":$data, "Value":$addr,
      CArg<"Value", "nullptr">:$uniform_offset,
      CArg<"Value", "nullptr">:$dynamic_offset,
      CArg<"Value", "nullptr">:$constant_offset)>
  ];
  let skipDefaultBuilders = 1;
  let assemblyFormat = [{
    custom<Opcode>($opcode) `data` $data `addr` $addr ``
    custom<Offsets>($uniform_offset, $dynamic_offset, $constant_offset)
    attr-dict `:`
    `ins` `` `(` type($data) ``  `,`  type($addr) ``
    custom<OffsetTypes>(
      ref($uniform_offset), ref($dynamic_offset), ref($constant_offset),
      type($uniform_offset), type($dynamic_offset), type($constant_offset))
    `)` `->` qualified(type($token))
  }];
  let extraClassDeclaration = [{
    MemoryInstructionKind getInstKind() {
      return getMemoryInstructionKind(getOpcode());
    }
    /// Returns the speculatability of the operation.
    Speculation::Speculatability getSpeculatability() {
      return aster::detail::getInstSpeculatabilityImpl(*this);
    }

    //===------------------------------------------------------------------===//
    // AMDGCNInstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    OperandRange getInstOuts() {
      return getAsOperandRange({});
    }
    /// Get the instruction input operands.
    OperandRange getInstIns();
    /// Get the instruction results.
    ResultRange getInstResults() {
      return getOperation()->getResults().slice(0, 0);
    }
  }];
}

//===----------------------------------------------------------------------===//
// Load instructions
//===----------------------------------------------------------------------===//

class SLoadInst<string _mnemonic, int resultRegs, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, AMDGCN_LoadOp, _isa> {
  let constraints = (ins SGPROperand<resultRegs>:$dest,
                         SGPROperand<2>:$addr,
                         SGPROperand<1, 1>:$uniform_offset,
                         NotPresent:$dynamic_offset);
  let cppBuilder = OpBuilder<(ins
    CppValue:$dest, CppValue:$addr, CppValue:$uniform_offset,
    CppValue:$constant_offset), [{
    return $_create($_builder, $_loc, $_opcode, dest, addr, uniform_offset,
      nullptr, constant_offset);
    }]>;
  let asmFormat = [
    SingleAsmVariant<"$dest, $addr, @printLSOffset($uniform_offset, $constant_offset)">
  ];
}

def S_LOAD_DWORD_INST : SLoadInst<
  /*mnemonic=*/"s_load_dword", /*resultRegs=*/1, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "SMEM load 32-bit";
  let description = [{
    SMEM load instruction that loads a 32-bit value from scalar memory.
  }];
}

def S_LOAD_DWORDX2_INST : SLoadInst<
  /*mnemonic=*/"s_load_dwordx2", /*resultRegs=*/2, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "SMEM load 64-bit";
  let description = [{
    SMEM load instruction that loads a 64-bit value from scalar memory.
  }];
}

def S_LOAD_DWORDX4_INST : SLoadInst<
  /*mnemonic=*/"s_load_dwordx4", /*resultRegs=*/4, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "SMEM load 128-bit";
  let description = [{
    SMEM load instruction that loads a 128-bit value from scalar memory.
  }];
}

def S_LOAD_DWORDX8_INST : SLoadInst<
  /*mnemonic=*/"s_load_dwordx8", /*resultRegs=*/8, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "SMEM load 256-bit";
  let description = [{
    SMEM load instruction that loads a 256-bit value from scalar memory.
  }];
}

def S_LOAD_DWORDX16_INST : SLoadInst<
  /*mnemonic=*/"s_load_dwordx16", /*resultRegs=*/16, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "SMEM load 512-bit";
  let description = [{
    SMEM load instruction that loads a 512-bit value from scalar memory.
  }];
}

//===----------------------------------------------------------------------===//
// Global Load Instructions
//===----------------------------------------------------------------------===//

class GLoadInst<string _mnemonic, int resultRegs,
                     list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, AMDGCN_LoadOp, _isa> {
  let constraints = (ins AccOperand<resultRegs>:$dest,
                         AddrOperand<2>:$addr,
                         NotPresent:$uniform_offset,
                         VGPROperand<1, 1>:$dynamic_offset);
  let cppBuilder = OpBuilder<(ins
    CppValue:$dest, CppValue:$addr, CppValue:$dynamic_offset,
    CppValue:$constant_offset), [{
    return $_create($_builder, $_loc, $_opcode, dest, addr, nullptr,
      dynamic_offset, constant_offset);
    }]>;
  let asmFormat = [
    AsmVariant<CPred<"isa<VGPRRangeType>($_self.getAddr().getType())">,
      "$dest, $addr, off $constant_offset">,
    AsmVariant<CPred<"isa<SGPRRangeType>($_self.getAddr().getType())">,
      "$dest, $dynamic_offset, $addr $constant_offset">
  ];
}

def GLOBAL_LOAD_DWORD_INST : GLoadInst<
  /*mnemonic=*/"global_load_dword", /*resultRegs=*/1, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Global load 32-bit";
  let description = [{
    Global load instruction that loads a 32-bit value from global memory.
  }];
}

def GLOBAL_LOAD_DWORDX2_INST : GLoadInst<
  /*mnemonic=*/"global_load_dwordx2", /*resultRegs=*/2, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Global load 64-bit";
  let description = [{
    Global load instruction that loads a 64-bit value from global memory.
  }];
}

def GLOBAL_LOAD_DWORDX3_INST : GLoadInst<
  /*mnemonic=*/"global_load_dwordx3", /*resultRegs=*/3, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Global load 96-bit";
  let description = [{
    Global load instruction that loads a 96-bit value from global memory.
  }];
}

def GLOBAL_LOAD_DWORDX4_INST : GLoadInst<
  /*mnemonic=*/"global_load_dwordx4", /*resultRegs=*/4, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Global load 128-bit";
  let description = [{
    Global load instruction that loads a 128-bit value from global memory.
  }];
}

//===----------------------------------------------------------------------===//
// DS Read Instructions
//===----------------------------------------------------------------------===//

class DReadInst<string _mnemonic, int dataRegs, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, AMDGCN_LoadOp, _isa> {
  let constraints = (ins AccOperand<dataRegs>:$dest,
                         VGPROperand<1>:$addr,
                         NotPresent:$uniform_offset,
                         NotPresent:$dynamic_offset);
  let cppBuilder = OpBuilder<(ins
    CppValue:$dest, CppValue:$addr, CppValue:$constant_offset), [{
    return $_create($_builder, $_loc, $_opcode, dest, addr, nullptr,
      nullptr, constant_offset);
    }]>;
  let asmFormat = [SingleAsmVariant<"$dest, $addr $constant_offset">];
}

def DS_READ_B32_INST : DReadInst<
  /*mnemonic=*/"ds_read_b32", /*dataRegs=*/1, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "DS read 32-bit";
  let description = [{
    DS read instruction that loads a 32-bit value from LDS.
  }];
}

def DS_READ_B64_INST : DReadInst<
  /*mnemonic=*/"ds_read_b64", /*dataRegs=*/2, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "DS read 64-bit";
  let description = [{
    DS read instruction that loads a 64-bit value from LDS.
  }];
}

def DS_READ_B96_INST : DReadInst<
  /*mnemonic=*/"ds_read_b96", /*dataRegs=*/3, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "DS read 96-bit";
  let description = [{
    DS read instruction that loads a 96-bit value from LDS.
  }];
}

def DS_READ_B128_INST : DReadInst<
  /*mnemonic=*/"ds_read_b128", /*dataRegs=*/4, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "DS read 128-bit";
  let description = [{
    DS read instruction that loads a 128-bit value from LDS.
  }];
}

//===----------------------------------------------------------------------===//
// Store instructions
//===----------------------------------------------------------------------===//

class SStoreInst<string _mnemonic, int dataRegs, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, AMDGCN_StoreOp, _isa> {
  let constraints = (ins SGPROperand<dataRegs>:$data,
                         SGPROperand<2>:$addr,
                         SGPROperand<1, 1>:$uniform_offset,
                         NotPresent:$dynamic_offset);
  let cppBuilder = OpBuilder<(ins
    CppValue:$data, CppValue:$addr, CppValue:$uniform_offset,
    CppValue:$constant_offset), [{
    return $_create($_builder, $_loc, $_opcode, data, addr, uniform_offset,
      nullptr, constant_offset);
    }]>;
  let asmFormat = [
    SingleAsmVariant<"$data, $addr, @printLSOffset($uniform_offset, $constant_offset)">
  ];
}

def S_STORE_DWORD_INST : SStoreInst<
  /*mnemonic=*/"s_store_dword", /*dataRegs=*/1, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "SMEM store 32-bit";
  let description = [{
    SMEM store instruction that stores a 32-bit value to scalar memory.
  }];
}

def S_STORE_DWORDX2_INST : SStoreInst<
  /*mnemonic=*/"s_store_dwordx2", /*dataRegs=*/2, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "SMEM store 64-bit";
  let description = [{
    SMEM store instruction that stores a 64-bit value to scalar memory.
  }];
}

def S_STORE_DWORDX4_INST : SStoreInst<
  /*mnemonic=*/"s_store_dwordx4", /*dataRegs=*/4, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "SMEM store 128-bit";
  let description = [{
    SMEM store instruction that stores a 128-bit value to scalar memory.
  }];
}

//===----------------------------------------------------------------------===//
// Global Store Instructions
//===----------------------------------------------------------------------===//

class GStoreInst<string _mnemonic, int dataRegs,
                      list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, AMDGCN_StoreOp, _isa> {
  let constraints = (ins AccOperand<dataRegs>:$data,
                         AddrOperand<2>:$addr,
                         NotPresent:$uniform_offset,
                         VGPROperand<1, 1>:$dynamic_offset);
  let cppBuilder = OpBuilder<(ins
    CppValue:$data, CppValue:$addr, CppValue:$dynamic_offset,
    CppValue:$constant_offset), [{
    return $_create($_builder, $_loc, $_opcode, data, addr, nullptr,
      dynamic_offset, constant_offset);
    }]>;
  let asmFormat = [
    AsmVariant<CPred<"isa<VGPRRangeType>($_self.getAddr().getType())">,
      "$addr, $data, off $constant_offset">,
    AsmVariant<CPred<"isa<SGPRRangeType>($_self.getAddr().getType())">,
      "$dynamic_offset, $data, $addr $constant_offset">
  ];
}

def GLOBAL_STORE_DWORD_INST : GStoreInst<
  /*mnemonic=*/"global_store_dword", /*dataRegs=*/1, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Global store 32-bit";
  let description = [{
    Global store instruction that stores a 32-bit value to global memory.
  }];
}

def GLOBAL_STORE_DWORDX2_INST : GStoreInst<
  /*mnemonic=*/"global_store_dwordx2", /*dataRegs=*/2, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Global store 64-bit";
  let description = [{
    Global store instruction that stores a 64-bit value to global memory.
  }];
}

def GLOBAL_STORE_DWORDX3_INST : GStoreInst<
  /*mnemonic=*/"global_store_dwordx3", /*dataRegs=*/3, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Global store 96-bit";
  let description = [{
    Global store instruction that stores a 96-bit value to global memory.
  }];
}

def GLOBAL_STORE_DWORDX4_INST : GStoreInst<
  /*mnemonic=*/"global_store_dwordx4", /*dataRegs=*/4, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Global store 128-bit";
  let description = [{
    Global store instruction that stores a 128-bit value to global memory.
  }];
}

//===----------------------------------------------------------------------===//
// Buffer Load Instructions (MUBUF)
//===----------------------------------------------------------------------===//

class BLoadInst<string _mnemonic, int resultRegs, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, AMDGCN_LoadOp, _isa> {
  let constraints = (ins AccOperand<resultRegs>:$dest,
                         SGPROperand<4, 0, 4>:$addr,
                         SGPROperand<1, 1>:$uniform_offset,
                         VGPROperand<1, 1>:$dynamic_offset);
  let cppBuilder = OpBuilder<(ins
    CppValue:$dest, CppValue:$addr,
    CppValue:$uniform_offset, CppValue:$dynamic_offset,
    CppValue:$constant_offset), [{
    return $_create($_builder, $_loc, $_opcode, dest, addr, uniform_offset,
      dynamic_offset, constant_offset);
    }]>;
  let asmFormat = [
    SingleAsmVariant<
      "$dest, $dynamic_offset, $addr, $uniform_offset offen $constant_offset">
  ];
}

def BUFFER_LOAD_DWORD_INST : BLoadInst<
  /*mnemonic=*/"buffer_load_dword", /*resultRegs=*/1, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer load 32-bit";
  let description = [{
    MUBUF load instruction that loads a 32-bit value via buffer descriptor.
  }];
}

def BUFFER_LOAD_DWORDX2_INST : BLoadInst<
  /*mnemonic=*/"buffer_load_dwordx2", /*resultRegs=*/2, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer load 64-bit";
  let description = [{
    MUBUF load instruction that loads a 64-bit value via buffer descriptor.
  }];
}

def BUFFER_LOAD_DWORDX3_INST : BLoadInst<
  /*mnemonic=*/"buffer_load_dwordx3", /*resultRegs=*/3, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer load 96-bit";
  let description = [{
    MUBUF load instruction that loads a 96-bit value via buffer descriptor.
  }];
}

def BUFFER_LOAD_DWORDX4_INST : BLoadInst<
  /*mnemonic=*/"buffer_load_dwordx4", /*resultRegs=*/4, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer load 128-bit";
  let description = [{
    MUBUF load instruction that loads a 128-bit value via buffer descriptor.
  }];
}

//===----------------------------------------------------------------------===//
// Buffer Store Instructions (MUBUF)
//===----------------------------------------------------------------------===//

class BStoreInst<string _mnemonic, int dataRegs, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, AMDGCN_StoreOp, _isa> {
  let constraints = (ins AccOperand<dataRegs>:$data,
                         SGPROperand<4, 0, 4>:$addr,
                         SGPROperand<1, 1>:$uniform_offset,
                         VGPROperand<1, 1>:$dynamic_offset);
  let cppBuilder = OpBuilder<(ins
    CppValue:$data, CppValue:$addr,
    CppValue:$uniform_offset, CppValue:$dynamic_offset,
    CppValue:$constant_offset), [{
    return $_create($_builder, $_loc, $_opcode, data, addr, uniform_offset,
      dynamic_offset, constant_offset);
    }]>;
  let asmFormat = [
    SingleAsmVariant<
      "$data, $dynamic_offset, $addr, $uniform_offset offen $constant_offset">
  ];
}

def BUFFER_STORE_DWORD_INST : BStoreInst<
  /*mnemonic=*/"buffer_store_dword", /*dataRegs=*/1, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer store 32-bit";
  let description = [{
    MUBUF store instruction that stores a 32-bit value via buffer descriptor.
  }];
}

def BUFFER_STORE_DWORDX2_INST : BStoreInst<
  /*mnemonic=*/"buffer_store_dwordx2", /*dataRegs=*/2, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer store 64-bit";
  let description = [{
    MUBUF store instruction that stores a 64-bit value via buffer descriptor.
  }];
}

def BUFFER_STORE_DWORDX3_INST : BStoreInst<
  /*mnemonic=*/"buffer_store_dwordx3", /*dataRegs=*/3, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer store 96-bit";
  let description = [{
    MUBUF store instruction that stores a 96-bit value via buffer descriptor.
  }];
}

def BUFFER_STORE_DWORDX4_INST : BStoreInst<
  /*mnemonic=*/"buffer_store_dwordx4", /*dataRegs=*/4, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer store 128-bit";
  let description = [{
    MUBUF store instruction that stores a 128-bit value via buffer descriptor.
  }];
}

//===----------------------------------------------------------------------===//
// Buffer Load Instructions (MUBUF, IDXEN mode)
//===----------------------------------------------------------------------===//

// MUBUF is a complex instruction and is currently split into different ops.

// MUBUF has four addressing modes based on the OFFEN/IDXEN encoding bits:
//   OFFSET (0,0): no VGPR, soffset + imm only
//   OFFEN  (1,0): 1 VGPR byte offset (BLoadInst/BStoreInst above)
//   IDXEN  (0,1): 1 VGPR element index, hw computes index * stride
//   BOTHEN (1,1): VGPR pair {index, offset} in consecutive registers
//
// These are separate op classes (not a mode attribute on one op) because the
// VGPR operand count and type differ per mode: OFFSET has none, OFFEN/IDXEN
// have one, BOTHEN has a pair. This matches LLVM's separate pseudo-instructions.
//
// TODO: add a BufferOpInterface for shared queries (get descriptor, soffset, etc).
class BLoadIdxInst<string _mnemonic, int resultRegs, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic # "_idxen", AMDGCN_LoadOp, _isa> {
  let constraints = (ins AccOperand<resultRegs>:$dest,
                         SGPROperand<4, 0, 4>:$addr,
                         SGPROperand<1, 1>:$uniform_offset,
                         VGPROperand<1, 1>:$dynamic_offset);
  let cppBuilder = OpBuilder<(ins
    CppValue:$dest, CppValue:$addr,
    CppValue:$uniform_offset, CppValue:$dynamic_offset,
    CppValue:$constant_offset), [{
    return $_create($_builder, $_loc, $_opcode, dest, addr, uniform_offset,
      dynamic_offset, constant_offset);
    }]>;
  let asmFormat = [
    SingleAsmVariant<
      "$dest, $dynamic_offset, $addr, $uniform_offset idxen $constant_offset">
  ];
}

def BUFFER_LOAD_DWORD_IDXEN_INST : BLoadIdxInst<
  /*mnemonic=*/"buffer_load_dword", /*resultRegs=*/1, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer load 32-bit (IDXEN)";
  let description = [{
    MUBUF load with IDXEN addressing. The VGPR provides an element index for
    structured buffer mode (stride > 0). Hardware computes index * stride.
  }];
}

def BUFFER_LOAD_DWORDX2_IDXEN_INST : BLoadIdxInst<
  /*mnemonic=*/"buffer_load_dwordx2", /*resultRegs=*/2, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer load 64-bit (IDXEN)";
  let description = [{
    MUBUF load with IDXEN addressing. The VGPR provides an element index for
    structured buffer mode (stride > 0). Hardware computes index * stride.
  }];
}

def BUFFER_LOAD_DWORDX3_IDXEN_INST : BLoadIdxInst<
  /*mnemonic=*/"buffer_load_dwordx3", /*resultRegs=*/3, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer load 96-bit (IDXEN)";
  let description = [{
    MUBUF load with IDXEN addressing. The VGPR provides an element index for
    structured buffer mode (stride > 0). Hardware computes index * stride.
  }];
}

def BUFFER_LOAD_DWORDX4_IDXEN_INST : BLoadIdxInst<
  /*mnemonic=*/"buffer_load_dwordx4", /*resultRegs=*/4, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer load 128-bit (IDXEN)";
  let description = [{
    MUBUF load with IDXEN addressing. The VGPR provides an element index for
    structured buffer mode (stride > 0). Hardware computes index * stride.
  }];
}

//===----------------------------------------------------------------------===//
// Buffer Store Instructions (MUBUF, IDXEN mode)
//===----------------------------------------------------------------------===//

class BStoreIdxInst<string _mnemonic, int dataRegs, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic # "_idxen", AMDGCN_StoreOp, _isa> {
  let constraints = (ins AccOperand<dataRegs>:$data,
                         SGPROperand<4, 0, 4>:$addr,
                         SGPROperand<1, 1>:$uniform_offset,
                         VGPROperand<1, 1>:$dynamic_offset);
  let cppBuilder = OpBuilder<(ins
    CppValue:$data, CppValue:$addr,
    CppValue:$uniform_offset, CppValue:$dynamic_offset,
    CppValue:$constant_offset), [{
    return $_create($_builder, $_loc, $_opcode, data, addr, uniform_offset,
      dynamic_offset, constant_offset);
    }]>;
  let asmFormat = [
    SingleAsmVariant<
      "$data, $dynamic_offset, $addr, $uniform_offset idxen $constant_offset">
  ];
}

def BUFFER_STORE_DWORD_IDXEN_INST : BStoreIdxInst<
  /*mnemonic=*/"buffer_store_dword", /*dataRegs=*/1, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer store 32-bit (IDXEN)";
  let description = [{
    MUBUF store with IDXEN addressing. The VGPR provides an element index for
    structured buffer mode (stride > 0). Hardware computes index * stride.
  }];
}

def BUFFER_STORE_DWORDX2_IDXEN_INST : BStoreIdxInst<
  /*mnemonic=*/"buffer_store_dwordx2", /*dataRegs=*/2, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer store 64-bit (IDXEN)";
  let description = [{
    MUBUF store with IDXEN addressing. The VGPR provides an element index for
    structured buffer mode (stride > 0). Hardware computes index * stride.
  }];
}

def BUFFER_STORE_DWORDX3_IDXEN_INST : BStoreIdxInst<
  /*mnemonic=*/"buffer_store_dwordx3", /*dataRegs=*/3, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer store 96-bit (IDXEN)";
  let description = [{
    MUBUF store with IDXEN addressing. The VGPR provides an element index for
    structured buffer mode (stride > 0). Hardware computes index * stride.
  }];
}

def BUFFER_STORE_DWORDX4_IDXEN_INST : BStoreIdxInst<
  /*mnemonic=*/"buffer_store_dwordx4", /*dataRegs=*/4, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "Buffer store 128-bit (IDXEN)";
  let description = [{
    MUBUF store with IDXEN addressing. The VGPR provides an element index for
    structured buffer mode (stride > 0). Hardware computes index * stride.
  }];
}

//===----------------------------------------------------------------------===//
// DS Write Instructions
//===----------------------------------------------------------------------===//

class DWriteInst<string _mnemonic, int dataRegs, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, AMDGCN_StoreOp, _isa> {
  let constraints = (ins VGPROperand<dataRegs>:$data,
                         VGPROperand<1>:$addr,
                         NotPresent:$uniform_offset,
                         NotPresent:$dynamic_offset);
  let cppBuilder = OpBuilder<(ins
    CppValue:$data, CppValue:$addr, CppValue:$constant_offset), [{
    return $_create($_builder, $_loc, $_opcode, data, addr, nullptr,
      nullptr, constant_offset);
    }]>;
  let asmFormat = [SingleAsmVariant<"$addr, $data $constant_offset">];
}

def DS_WRITE_B32_INST : DWriteInst<
  /*mnemonic=*/"ds_write_b32", /*dataRegs=*/1, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "DS write 32-bit";
  let description = [{
    DS write instruction that stores a 32-bit value to LDS.
  }];
}

def DS_WRITE_B64_INST : DWriteInst<
  /*mnemonic=*/"ds_write_b64", /*dataRegs=*/2, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "DS write 64-bit";
  let description = [{
    DS write instruction that stores a 64-bit value to LDS.
  }];
}

def DS_WRITE_B96_INST : DWriteInst<
  /*mnemonic=*/"ds_write_b96", /*dataRegs=*/3, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "DS write 96-bit";
  let description = [{
    DS write instruction that stores a 96-bit value to LDS.
  }];
}

def DS_WRITE_B128_INST : DWriteInst<
  /*mnemonic=*/"ds_write_b128", /*dataRegs=*/4, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "DS write 128-bit";
  let description = [{
    DS write instruction that stores a 128-bit value to LDS.
  }];
}

#endif // AMDGCN_INST_MEMORY_TD
