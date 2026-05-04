//===- DS.td - AMDGCN DS Instructions ----------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_DS_TD
#define AMDGCN_DS_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNInstBase.td"
include "aster/Dialect/AMDGCN/IR/AMDGCN.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/MemoryOpInterfaces.td"

#ifndef DSBPERMUTEB32_DEF
#define DSBPERMUTEB32_DEF
def DsBpermuteB32 : AMDISAInstruction<"ds_bpermute_b32", [FutureOpInterface]> {
  let props = [IsDS];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Backward permute. This does not access LDS memory and may be called even
    if no LDS memory is allocated to the wave. It uses LDS hardware to
    implement an arbitrary swizzle across threads in a wavefront.
  }];
  let encodings = [
    InstEnc<[ENC_DS_CDNA3], 0b00111111, (pred
      NotPresent:$gds
    )>,
    InstEnc<[ENC_DS_CDNA4], 0b00111111>
  ];
  let outputs = (ins
    VGPROrAGPROf<1>:$dst0
  );
  let inputs = (ins
    VGPROf<1>:$addr,
    VGPROrAGPROf<1>:$data
  );
  let trailingArgs = (ins
    OffsetType<16>:$const_offset,
    GDS:$gds
  );
  let trailingResults = (outs DSReadToken:$token);
  let asm_syntax = [
    ASMString<[ENC_DS_CDNA3], "${mnemonic} $dst0, $addr, $data [const_offset]">,
    ASMString<[ENC_DS_CDNA4], "${mnemonic} $dst0, $addr, $data [const_offset] [gds]">
  ];
  let instExtraClassDeclaration = [{
    ::mlir::aster::Future getFuture() {
      return ::mlir::aster::Future::read(getToken(), getDst0());
    }
  }];
}
#endif // DSBPERMUTEB32_DEF

#ifndef DSPERMUTEB32_DEF
#define DSPERMUTEB32_DEF
def DsPermuteB32 : AMDISAInstruction<"ds_permute_b32", [FutureOpInterface]> {
  let props = [IsDS];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Forward permute. This does not access LDS memory and may be called even if
    no LDS memory is allocated to the wave. It uses LDS to implement an
    arbitrary swizzle across threads in a wavefront.
  }];
  let encodings = [
    InstEnc<[ENC_DS_CDNA3], 0b00111110, (pred
      NotPresent:$gds
    )>,
    InstEnc<[ENC_DS_CDNA4], 0b00111110>
  ];
  let outputs = (ins
    VGPROrAGPROf<1>:$dst0
  );
  let inputs = (ins
    VGPROf<1>:$addr,
    VGPROrAGPROf<1>:$data
  );
  let trailingArgs = (ins
    OffsetType<16>:$const_offset,
    GDS:$gds
  );
  let trailingResults = (outs DSReadToken:$token);
  let asm_syntax = [
    ASMString<[ENC_DS_CDNA3], "${mnemonic} $dst0, $addr, $data [const_offset]">,
    ASMString<[ENC_DS_CDNA4], "${mnemonic} $dst0, $addr, $data [const_offset] [gds]">
  ];
  let instExtraClassDeclaration = [{
    ::mlir::aster::Future getFuture() {
      return ::mlir::aster::Future::read(getToken(), getDst0());
    }
  }];
}
#endif // DSPERMUTEB32_DEF

#ifndef DSREAD2INST_DEF
#define DSREAD2INST_DEF
class DSRead2Inst<string mnemonic, list<Type> dst0Type> : AMDISAInstruction<mnemonic, [DSRead2InstOpInterface]> {
  let props = [IsDS];
  let archs = [CDNA3Isa, CDNA4Isa];
  let effects = [LDSRead];
  let outputs = (ins
    AnyRegOf<dst0Type>:$dest
  );
  let inputs = (ins
    VGPROf<1>:$addr
  );
  let trailingArgs = (ins
    OffsetType<8>:$const_offset0,
    OffsetType<8>:$const_offset1,
    GDS:$gds
  );
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    `dest` $dest `addr` $addr `offsets`
    `c0` `(`$const_offset0`)` ` ` `c1` `(`$const_offset1`)` attr-dict `:`
    `outs``(` type($dest) `)` `ins` `(` type($addr) `)`
    `mods` `(` type($const_offset0) `,` type($const_offset1) `)` `->`
    custom<LoadResults>(ref(type($dest)), type($dest_res), type($token))
  }];
  let trailingResults = (outs DSReadToken:$token);
  let asm_syntax = [
    ASMString<[ENC_DS_CDNA3, ENC_DS_CDNA4], "${mnemonic} $dest, $addr [const_offset0] [const_offset1] [gds]">
  ];
}
#endif // DSREAD2INST_DEF

#ifndef DSREAD2B32_DEF
#define DSREAD2B32_DEF
def DsRead2B32 : DSRead2Inst<"ds_read2_b32",
    /*dest=*/[VGPROrAGPROf<2>]
  > {
  let description = [{
    Load 32 bits of data from one location in a data share and then 32 bits of
    data from a second location in a data share and store the results into a
    64-bit vector register.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b00110111>];
}
#endif // DSREAD2B32_DEF

#ifndef DSREAD2B64_DEF
#define DSREAD2B64_DEF
def DsRead2B64 : DSRead2Inst<"ds_read2_b64",
    /*dest=*/[VGPROrAGPROf<4>]
  > {
  let description = [{
    Load 64 bits of data from one location in a data share and then 64 bits of
    data from a second location in a data share and store the results into a
    128-bit vector register.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b01110111>];
}
#endif // DSREAD2B64_DEF

#ifndef DSREADINST_DEF
#define DSREADINST_DEF
class DSReadInst<string mnemonic, list<Type> dst0Type> : AMDISAInstruction<mnemonic, [DSReadInstOpInterface]> {
  let props = [IsDS];
  let archs = [CDNA3Isa, CDNA4Isa];
  let effects = [LDSRead];
  let outputs = (ins
    AnyRegOf<dst0Type>:$dest
  );
  let inputs = (ins
    VGPROf<1>:$addr
  );
  let trailingArgs = (ins
    OffsetType<16>:$const_offset,
    GDS:$gds
  );
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    `dest` $dest `addr` $addr `offset` `c` `(`$const_offset`)` attr-dict `:`
    `outs``(` type($dest) `)` `ins` `(` type($addr) `)` `mods` `(` type($const_offset) `)` `->`
    custom<LoadResults>(ref(type($dest)), type($dest_res), type($token))
  }];
  let trailingResults = (outs DSReadToken:$token);
  let asm_syntax = [
    ASMString<[ENC_DS_CDNA3, ENC_DS_CDNA4], "${mnemonic} $dest, $addr [const_offset] [gds]">
  ];
}
#endif // DSREADINST_DEF

#ifndef DSREADB128_DEF
#define DSREADB128_DEF
def DsReadB128 : DSReadInst<"ds_read_b128",
    /*dest=*/[VGPROrAGPROf<4>]
  > {
  let description = [{
    Load 128 bits of data from a data share into a vector register.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b11111111>];
}
#endif // DSREADB128_DEF

#ifndef DSREADB32_DEF
#define DSREADB32_DEF
def DsReadB32 : DSReadInst<"ds_read_b32",
    /*dest=*/[VGPROrAGPROf<1>]
  > {
  let description = [{
    Load 32 bits of data from a data share into a vector register.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b00110110>];
}
#endif // DSREADB32_DEF

#ifndef DSREADB64_DEF
#define DSREADB64_DEF
def DsReadB64 : DSReadInst<"ds_read_b64",
    /*dest=*/[VGPROrAGPROf<2>]
  > {
  let description = [{
    Load 64 bits of data from a data share into a vector register.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b01110110>];
}
#endif // DSREADB64_DEF

#ifndef DSREADB64TRB16_DEF
#define DSREADB64TRB16_DEF
def DsReadB64TrB16 : DSReadInst<"ds_read_b64_tr_b16",
    /*dest=*/[VGPROrAGPROf<2>]
  > {
  let archs = [CDNA4Isa];
  let description = [{
    Read 64 bits of data per lane from data share. Interpret the data as a
    matrix with 16 bit elements and transpose the matrix. Store the result
    into vector registers.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA4], 0b11100011>];
  let asm_syntax = [
    ASMString<[ENC_DS_CDNA4], "${mnemonic} $dest, $addr [const_offset] [gds]">
  ];
}
#endif // DSREADB64TRB16_DEF

#ifndef DSREADB64TRB4_DEF
#define DSREADB64TRB4_DEF
def DsReadB64TrB4 : DSReadInst<"ds_read_b64_tr_b4",
    /*dest=*/[VGPROrAGPROf<2>]
  > {
  let archs = [CDNA4Isa];
  let description = [{
    Read 64 bits of data per lane from data share. Interpret the data as a
    matrix with 4 bit elements and transpose the matrix. Store the result into
    vector registers.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA4], 0b11100000>];
  let asm_syntax = [
    ASMString<[ENC_DS_CDNA4], "${mnemonic} $dest, $addr [const_offset] [gds]">
  ];
}
#endif // DSREADB64TRB4_DEF

#ifndef DSREADB64TRB8_DEF
#define DSREADB64TRB8_DEF
def DsReadB64TrB8 : DSReadInst<"ds_read_b64_tr_b8",
    /*dest=*/[VGPROrAGPROf<2>]
  > {
  let archs = [CDNA4Isa];
  let description = [{
    Read 64 bits of data per lane from data share. Interpret the data as a
    matrix with 8 bit elements and transpose the matrix. Store the result into
    vector registers.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA4], 0b11100010>];
  let asm_syntax = [
    ASMString<[ENC_DS_CDNA4], "${mnemonic} $dest, $addr [const_offset] [gds]">
  ];
}
#endif // DSREADB64TRB8_DEF

#ifndef DSREADB96_DEF
#define DSREADB96_DEF
def DsReadB96 : DSReadInst<"ds_read_b96",
    /*dest=*/[VGPROrAGPROf<3>]
  > {
  let description = [{
    Load 96 bits of data from a data share into a vector register.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b11111110>];
}
#endif // DSREADB96_DEF

#ifndef DSREADB96TRB6_DEF
#define DSREADB96TRB6_DEF
def DsReadB96TrB6 : DSReadInst<"ds_read_b96_tr_b6",
    /*dest=*/[VGPROrAGPROf<3>]
  > {
  let archs = [CDNA4Isa];
  let description = [{
    Read 96 bits of data per lane from data share. Interpret the data as a
    matrix with 6 bit elements and transpose the matrix. Store the result into
    vector registers.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA4], 0b11100001>];
  let asm_syntax = [
    ASMString<[ENC_DS_CDNA4], "${mnemonic} $dest, $addr [const_offset] [gds]">
  ];
}
#endif // DSREADB96TRB6_DEF

#ifndef DSWRITE2INST_DEF
#define DSWRITE2INST_DEF
class DSWrite2Inst<string mnemonic, list<Type> data0Type, list<Type> data1Type> : AMDISAInstruction<mnemonic, [DSWrite2InstOpInterface]> {
  let props = [IsDS];
  let archs = [CDNA3Isa, CDNA4Isa];
  let effects = [LDSWrite];
  let outputs = (ins);
  let inputs = (ins
    VGPROf<1>:$addr,
    AnyRegOf<data0Type>:$data0,
    AnyRegOf<data1Type>:$data1
  );
  let trailingArgs = (ins
    OffsetType<8>:$const_offset0,
    OffsetType<8>:$const_offset1,
    GDS:$gds
  );
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    `data0` $data0 `data1` $data1 `addr` $addr `offsets`
    `c0` `(`$const_offset0`)` ` ` `c1` `(`$const_offset1`)`  attr-dict `:`
    `ins` `(`type($data0) `,` type($data1) `,` type($addr) `)`
    `mods` `(` type($const_offset0) `,` type($const_offset1) `)` `->`
    qualified(type($token))
  }];
  let trailingResults = (outs DSWriteToken:$token);
  let asm_syntax = [
    ASMString<[ENC_DS_CDNA3, ENC_DS_CDNA4], "${mnemonic} $addr, $data0, $data1 [const_offset0] [const_offset1] [gds]">
  ];
}
#endif // DSWRITE2INST_DEF

#ifndef DSWRITE2B32_DEF
#define DSWRITE2B32_DEF
def DsWrite2B32 : DSWrite2Inst<"ds_write2_b32",
    /*data0=*/[VGPROrAGPROf<1>],
    /*data1=*/[VGPROrAGPROf<1>]
  > {
  let description = [{
    Store 32 bits of data from one vector input register and then 32 bits of
    data from a second vector input register into a data share.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b00001110>];
}
#endif // DSWRITE2B32_DEF

#ifndef DSWRITE2B64_DEF
#define DSWRITE2B64_DEF
def DsWrite2B64 : DSWrite2Inst<"ds_write2_b64",
    /*data0=*/[VGPROrAGPROf<2>],
    /*data1=*/[VGPROrAGPROf<2>]
  > {
  let description = [{
    Store 64 bits of data from one vector input register and then 64 bits of
    data from a second vector input register into a data share.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b01001110>];
}
#endif // DSWRITE2B64_DEF

#ifndef DSWRITEINST_DEF
#define DSWRITEINST_DEF
class DSWriteInst<string mnemonic, list<Type> dataType> : AMDISAInstruction<mnemonic, [DSWriteInstOpInterface]> {
  let props = [IsDS];
  let archs = [CDNA3Isa, CDNA4Isa];
  let effects = [LDSWrite];
  let outputs = (ins);
  let inputs = (ins
    VGPROf<1>:$addr,
    AnyRegOf<dataType>:$data
  );
  let trailingArgs = (ins
    OffsetType<16>:$const_offset,
    GDS:$gds
  );
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    `data` $data `addr` $addr `offset` `c` `(`$const_offset`)` attr-dict `:`
    `ins` `(`type($data) `,` type($addr) `)` `mods` `(` type($const_offset) `)` `->`
    qualified(type($token))
  }];
  let trailingResults = (outs DSWriteToken:$token);
  let asm_syntax = [
    ASMString<[ENC_DS_CDNA3, ENC_DS_CDNA4], "${mnemonic} $addr, $data [const_offset] [gds]">
  ];
}
#endif // DSWRITEINST_DEF

#ifndef DSWRITEB128_DEF
#define DSWRITEB128_DEF
def DsWriteB128 : DSWriteInst<"ds_write_b128",
    /*data=*/[VGPROrAGPROf<4>]
  > {
  let description = [{
    Store 128 bits of data from a vector input register into a data share.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b11011111>];
}
#endif // DSWRITEB128_DEF

#ifndef DSWRITEB16_DEF
#define DSWRITEB16_DEF
def DsWriteB16 : DSWriteInst<"ds_write_b16",
    /*data=*/[VGPROrAGPROf<1>]
  > {
  let description = [{
    Store 16 bits of data from a vector register into a data share.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b00011111>];
}
#endif // DSWRITEB16_DEF

#ifndef DSWRITEB32_DEF
#define DSWRITEB32_DEF
def DsWriteB32 : DSWriteInst<"ds_write_b32",
    /*data=*/[VGPROrAGPROf<1>]
  > {
  let description = [{
    Store 32 bits of data from a vector input register into a data share.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b00001101>];
}
#endif // DSWRITEB32_DEF

#ifndef DSWRITEB64_DEF
#define DSWRITEB64_DEF
def DsWriteB64 : DSWriteInst<"ds_write_b64",
    /*data=*/[VGPROrAGPROf<2>]
  > {
  let description = [{
    Store 64 bits of data from a vector input register into a data share.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b01001101>];
}
#endif // DSWRITEB64_DEF

#ifndef DSWRITEB8_DEF
#define DSWRITEB8_DEF
def DsWriteB8 : DSWriteInst<"ds_write_b8",
    /*data=*/[VGPROrAGPROf<1>]
  > {
  let description = [{
    Store 8 bits of data from a vector register into a data share.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b00011110>];
}
#endif // DSWRITEB8_DEF

#ifndef DSWRITEB8D16HI_DEF
#define DSWRITEB8D16HI_DEF
def DsWriteB8D16Hi : DSWriteInst<"ds_write_b8_d16_hi",
    /*data=*/[VGPROrAGPROf<1>]
  > {
  let description = [{
    Store 8 bits of data from the high bits of a vector register into a data
    share.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b01010100>];
}
#endif // DSWRITEB8D16HI_DEF

#ifndef DSWRITEB96_DEF
#define DSWRITEB96_DEF
def DsWriteB96 : DSWriteInst<"ds_write_b96",
    /*data=*/[VGPROrAGPROf<3>]
  > {
  let description = [{
    Store 96 bits of data from a vector input register into a data share.
  }];
  let encodings = [InstEnc<[ENC_DS_CDNA3, ENC_DS_CDNA4], 0b11011110>];
}
#endif // DSWRITEB96_DEF

#endif // AMDGCN_DS_TD
