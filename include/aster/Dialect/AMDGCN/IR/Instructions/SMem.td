//===- SMem.td - AMDGCN SMem Instructions ------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_SMEM_TD
#define AMDGCN_SMEM_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNInstBase.td"
include "aster/Dialect/AMDGCN/IR/AMDGCN.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/MemoryOpInterfaces.td"

#ifndef SLOADDWORDINST_DEF
#define SLOADDWORDINST_DEF
class SLoadDwordInst<string mnemonic, list<Type> dataType> : AMDISAInstruction<mnemonic, [SLoadDwordInstOpInterface]> {
  let props = [IsSMEM];
  let archs = [CDNA3Isa, CDNA4Isa];
  let effects = [ConstantRead];
  let outputs = (ins
    AnyRegOf<dataType>:$dest
  );
  let inputs = (ins
    TwoSGPROrVCCType:$addr,
    OptionalReg<OneSGPROrVCCM0Type>:$soffset
  );
  let trailingArgs = (ins
    OptionalOffsetType<21>:$const_offset,
    GLC:$glc
  );
  let trailingResults = (outs ConstantReadToken:$token);
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    `dest` $dest `addr` $addr `offset` (`u` `(`$soffset^`)`)?
    (`c` `(`$const_offset^`)`)? attr-dict `:`
    `outs``(` type($dest) `)` `ins` `(` type($addr) (`,` type($soffset)^)? `)`
    (`mods` `(` type($const_offset)^ `)`)? `->`
    custom<LoadResults>(ref(type($dest)), type($dest_res), type($token))
  }];
  let asm_syntax = [
    ASMString<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], "${mnemonic} $dest, $addr, @printLSOffset($soffset, [const_offset]) [glc]">
  ];
  let hasVerifier = 1;
  let extraClassDefinition = [{
    LogicalResult $cppClass::verify() {
      bool hasConstOffset = getConstOffset() != nullptr;
      bool hasSoffset = getSoffset() != nullptr;
      if (!(hasConstOffset ^ hasSoffset))
        return emitError("only one of `const_offset` and `soffset` must be present");
      return success();
    }
  }];
}
#endif // SLOADDWORDINST_DEF

#ifndef SLOADDWORD_DEF
#define SLOADDWORD_DEF
def SLoadDword : SLoadDwordInst<"s_load_dword",
    /*dest=*/[OneSGPROrVCCType]
  > {
  let description = [{
    Load 32 bits of dest from the scalar memory into a scalar register.
  }];
  let encodings = [InstEnc<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], 0b00000000>];
}
#endif // SLOADDWORD_DEF

#ifndef SLOADDWORDX16_DEF
#define SLOADDWORDX16_DEF
def SLoadDwordx16 : SLoadDwordInst<"s_load_dwordx16",
    /*dest=*/[SGPROf<16>]
  > {
  let description = [{
    Load 512 bits of dest from the scalar memory into a scalar register.
  }];
  let encodings = [InstEnc<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], 0b00000100>];
}
#endif // SLOADDWORDX16_DEF

#ifndef SLOADDWORDX2_DEF
#define SLOADDWORDX2_DEF
def SLoadDwordx2 : SLoadDwordInst<"s_load_dwordx2",
    /*dest=*/[TwoSGPROrVCCType]
  > {
  let description = [{
    Load 64 bits of dest from the scalar memory into a scalar register.
  }];
  let encodings = [InstEnc<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], 0b00000001>];
}
#endif // SLOADDWORDX2_DEF

#ifndef SLOADDWORDX4_DEF
#define SLOADDWORDX4_DEF
def SLoadDwordx4 : SLoadDwordInst<"s_load_dwordx4",
    /*dest=*/[SGPROf<4>]
  > {
  let description = [{
    Load 128 bits of dest from the scalar memory into a scalar register.
  }];
  let encodings = [InstEnc<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], 0b00000010>];
}
#endif // SLOADDWORDX4_DEF

#ifndef SLOADDWORDX8_DEF
#define SLOADDWORDX8_DEF
def SLoadDwordx8 : SLoadDwordInst<"s_load_dwordx8",
    /*dest=*/[SGPROf<8>]
  > {
  let description = [{
    Load 256 bits of dest from the scalar memory into a scalar register.
  }];
  let encodings = [InstEnc<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], 0b00000011>];
}
#endif // SLOADDWORDX8_DEF

#ifndef SMEMREALTIME_DEF
#define SMEMREALTIME_DEF
def SMemrealtime : AMDISAInstruction<"s_memrealtime"> {
  let props = [IsSMEM];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Return current 64-bit RTC.
  }];
  let encodings = [InstEnc<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], 0b00100101>];
  let outputs = (ins
    TwoSGPROrVCCType:$dest
  );
  let inputs = (ins);
  let trailingArgs = (ins);
  let trailingResults = (outs ConstantReadToken:$token);
  let asm_syntax = [
    ASMString<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], "${mnemonic} $dest">
  ];
}
#endif // SMEMREALTIME_DEF

#ifndef SMEMTIME_DEF
#define SMEMTIME_DEF
def SMemtime : AMDISAInstruction<"s_memtime"> {
  let props = [IsSMEM];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Return current 64-bit timestamp.
  }];
  let encodings = [InstEnc<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], 0b00100100>];
  let outputs = (ins
    TwoSGPROrVCCType:$dest
  );
  let inputs = (ins);
  let trailingArgs = (ins);
  let trailingResults = (outs ConstantReadToken:$token);
  let asm_syntax = [
    ASMString<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], "${mnemonic} $dest">
  ];
}
#endif // SMEMTIME_DEF

#ifndef SSTOREDWORDINST_DEF
#define SSTOREDWORDINST_DEF
class SStoreDwordInst<string mnemonic, list<Type> dataType> : AMDISAInstruction<mnemonic, [SStoreDwordInstOpInterface]> {
  let props = [IsSMEM];
  let archs = [CDNA3Isa, CDNA4Isa];
  let effects = [ConstantWrite];
  let outputs = (ins);
  let inputs = (ins
    AnyRegOf<dataType>:$data,
    TwoSGPROrVCCType:$addr,
    OptionalReg<OneSGPROrVCCM0Type>:$soffset
  );
  let trailingArgs = (ins
    OptionalOffsetType<21>:$const_offset,
    GLC:$glc
  );
  let trailingResults = (outs ConstantWriteToken:$token);
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    `data` $data `addr` $addr `offset` (`u` `(`$soffset^`)`)?
    (`c` `(`$const_offset^`)`)? attr-dict `:`
    `ins` `(` type($data) `,` type($addr) (`,` type($soffset)^)? `)`
    (`mods` `(` type($const_offset)^ `)`)? `->` qualified(type($token))
  }];
  let asm_syntax = [
    ASMString<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], "${mnemonic} $data, $addr, @printLSOffset($soffset, [const_offset]) [glc]">
  ];
  let hasVerifier = 1;
  let extraClassDefinition = [{
    LogicalResult $cppClass::verify() {
      bool hasConstOffset = getConstOffset() != nullptr;
      bool hasSoffset = getSoffset() != nullptr;
      if (!(hasConstOffset ^ hasSoffset))
        return emitError("only one of `const_offset` and `soffset` must be present");
      return success();
    }
  }];
}
#endif // SSTOREDWORDINST_DEF

#ifndef SSTOREDWORD_DEF
#define SSTOREDWORD_DEF
def SStoreDword : SStoreDwordInst<"s_store_dword",
    /*data=*/[OneSGPROrVCCType]
  > {
  let description = [{
    Store 32 bits of data from a scalar register into the scalar memory.
  }];
  let encodings = [InstEnc<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], 0b00010000>];
}
#endif // SSTOREDWORD_DEF

#ifndef SSTOREDWORDX2_DEF
#define SSTOREDWORDX2_DEF
def SStoreDwordx2 : SStoreDwordInst<"s_store_dwordx2",
    /*data=*/[TwoSGPROrVCCType]
  > {
  let description = [{
    Store 64 bits of data from a scalar register into the scalar memory.
  }];
  let encodings = [InstEnc<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], 0b00010001>];
}
#endif // SSTOREDWORDX2_DEF

#ifndef SSTOREDWORDX4_DEF
#define SSTOREDWORDX4_DEF
def SStoreDwordx4 : SStoreDwordInst<"s_store_dwordx4",
    /*data=*/[SGPROf<4>]
  > {
  let description = [{
    Store 128 bits of data from a scalar register into the scalar memory.
  }];
  let encodings = [InstEnc<[ENC_SMEM_CDNA3, ENC_SMEM_CDNA4], 0b00010010>];
}
#endif // SSTOREDWORDX4_DEF

#endif // AMDGCN_SMEM_TD
