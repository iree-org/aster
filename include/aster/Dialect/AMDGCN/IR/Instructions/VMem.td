//===- VMem.td - AMDGCN VMem Instructions ------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_VMEM_TD
#define AMDGCN_VMEM_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNInstBase.td"
include "aster/Dialect/AMDGCN/IR/AMDGCN.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/MemoryOpInterfaces.td"

#ifndef BUFFERLOADINST_DEF
#define BUFFERLOADINST_DEF
class BufferLoadInst<string mnemonic, list<Type> dataType> : AMDISAInstruction<mnemonic, [BufferLoadInstOpInterface]> {
  let props = [IsVMem, IsBuffer];
  let archs = [CDNA3Isa, CDNA4Isa];
  let effects = [GlobalRead];
  let outputs = (ins
    AnyRegOf<dataType>:$dest
  );
  let inputs = (ins
    SGPROf<4>:$addr,
    OptionalReg<AnyRegOf<[VGPROf<1>, VGPROf<2>]>>:$off_or_idx,
    OneSGPROrSRegOrIntConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>>:$soffset
  );
  let trailingArgs = (ins
    OffsetType<12>:$const_offset,
    IDXEN:$idxen,
    NT:$nt,
    OFFEN:$offen,
    SC0:$sc0,
    SC1:$sc1
  );
  let trailingResults = (outs FlatReadToken:$token);
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    `dest` $dest `addr` $addr `offset`
    `u` `(` $soffset `)` `+` (`off_idx` `(` $off_or_idx^ `)` `+`)? `c` `(`$const_offset`)`
    attr-dict `:`
    `outs``(` type($dest) `)`
    `ins` `(` type($addr) `,` type($soffset) (`,` type($off_or_idx)^)? `)`
    `mods` `(` type($const_offset) `)`
    `->` custom<LoadResults>(ref(type($dest)), type($dest_res), type($token))
  }];
  let asm_syntax = [
    ASMString<[ENC_MUBUF_CDNA3], "${mnemonic} $dest, $off_or_idx, $addr, $soffset [idxen] [offen] [const_offset] [sc0] [nt] [sc1]">,
    ASMString<[ENC_MUBUF_CDNA4], "${mnemonic} $dest, $off_or_idx, $addr, $soffset [offen] [idxen] [const_offset] [sc0] [sc1] [nt]">
  ];
  let hasVerifier = 1;
  let instExtraClassDeclaration = [{
    /// Returns true if the operation is a load-to-LDS operation.
    bool getLds() {
      return false;
    }
  }];
  let extraClassDefinition = [{
    LogicalResult $cppClass::verify() {
      Value off_or_idx = getOffOrIdx();
      bool offen = getOffen();
      bool idxen = getIdxen();
      if (!off_or_idx && (offen || idxen))
        return emitError("expects an address operand when offen or idxen is set");
      if (off_or_idx && !(offen || idxen))
        return emitError("expects no address operand when offen or idxen is not set");
      if (!off_or_idx)
        return success();
      auto regTy = cast<RegisterTypeInterface>(off_or_idx.getType());
      int32_t nRegs = regTy.getAsRange().size();
      int32_t expectedRegs = static_cast<int32_t>(offen) + static_cast<int32_t>(idxen);
      if (nRegs != expectedRegs)
        return emitError() << "expected " << expectedRegs << " register(s) for the address based on the offen and idxen flags, got " << nRegs;
      return success();
    }
  }];
}
#endif // BUFFERLOADINST_DEF

#ifndef BUFFERLOADDWORD_DEF
#define BUFFERLOADDWORD_DEF
def BufferLoadDword : BufferLoadInst<"buffer_load_dword",
    /*dest=*/[VGPROrAGPROf<1>]
  > {
  let description = [{
    Load 32 bits of dest from a buffer surface into a vector register.
  }];
  let encodings = [InstEnc<[ENC_MUBUF_CDNA3, ENC_MUBUF_CDNA4], 0b0010100>];
  let asm_syntax = [
    ASMString<[ENC_MUBUF_CDNA3], "${mnemonic} $dest, $off_or_idx, $addr, $soffset [idxen] [offen] [const_offset] [sc0] [nt] [sc1]">,
    ASMString<[ENC_MUBUF_CDNA4], "${mnemonic} $dest, $off_or_idx, $addr, $soffset [offen] [idxen] [const_offset] [sc0] [sc1] [nt]">
  ];
}
#endif // BUFFERLOADDWORD_DEF

#ifndef BUFFERLOADDWORDX2_DEF
#define BUFFERLOADDWORDX2_DEF
def BufferLoadDwordx2 : BufferLoadInst<"buffer_load_dwordx2",
    /*dest=*/[VGPROrAGPROf<2>]
  > {
  let description = [{
    Load 64 bits of dest from a buffer surface into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_MUBUF_CDNA3], 0b0010101>,
    InstEnc<[ENC_MUBUF_CDNA4], 0b0010101>
  ];
}
#endif // BUFFERLOADDWORDX2_DEF

#ifndef BUFFERLOADDWORDX3_DEF
#define BUFFERLOADDWORDX3_DEF
def BufferLoadDwordx3 : BufferLoadInst<"buffer_load_dwordx3",
    /*dest=*/[VGPROrAGPROf<3>]
  > {
  let description = [{
    Load 96 bits of dest from a buffer surface into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_MUBUF_CDNA3], 0b0010110>,
    InstEnc<[ENC_MUBUF_CDNA4], 0b0010110>
  ];
}
#endif // BUFFERLOADDWORDX3_DEF

#ifndef BUFFERLOADDWORDX4_DEF
#define BUFFERLOADDWORDX4_DEF
def BufferLoadDwordx4 : BufferLoadInst<"buffer_load_dwordx4",
    /*dest=*/[VGPROrAGPROf<4>]
  > {
  let description = [{
    Load 128 bits of dest from a buffer surface into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_MUBUF_CDNA3], 0b0010111>,
    InstEnc<[ENC_MUBUF_CDNA4], 0b0010111>
  ];
}
#endif // BUFFERLOADDWORDX4_DEF

#ifndef BUFFERLOADLDSINST_DEF
#define BUFFERLOADLDSINST_DEF
class BufferLoadLDSInst<string mnemonic> : AMDISAInstruction<mnemonic, [BufferLoadInstOpInterface]> {
  let props = [IsVMem, IsBuffer];
  let archs = [CDNA3Isa, CDNA4Isa];
  let effects = [GlobalRead, LDSWrite];
  let inputs = (ins
    SGPROf<4>:$addr,
    OptionalReg<AnyRegOf<[VGPROf<1>, VGPROf<2>]>>:$off_or_idx,
    OneSGPROrSRegOrIntConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>>:$soffset,
    M0Type:$m0_src
  );
  let trailingArgs = (ins
    OffsetType<12>:$const_offset,
    IDXEN:$idxen,
    NT:$nt,
    OFFEN:$offen,
    SC0:$sc0,
    SC1:$sc1
  );
  let trailingResults = (outs FlatReadToken:$token);
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    `addr` $addr `m0` $m0_src `offset`
    `u` `(` $soffset `)` `+` (`off_idx` `(` $off_or_idx^ `)` `+`)? `c` `(`$const_offset`)`
    attr-dict `:`
    `ins` `(` type($addr) `,` type($m0_src) `,` type($soffset) (`,` type($off_or_idx)^)? `)`
    `mods` `(` type($const_offset) `)`
    `->` qualified(type($token))
  }];
  let asm_syntax = [
    ASMString<[ENC_MUBUF_CDNA3], "${mnemonic} $off_or_idx, $addr, $soffset [idxen] [offen] [const_offset] [sc0] [nt] [sc1] lds">,
    ASMString<[ENC_MUBUF_CDNA4], "${mnemonic} $off_or_idx, $addr, $soffset [offen] [idxen] [const_offset] lds [sc0] [sc1] [nt]">
  ];
  let hasVerifier = 1;
  let instExtraClassDeclaration = [{
    /// Load-to-LDS has no register destination; data goes directly to LDS.
    /// Callers using BufferLoadInstOpInterface must check getLds() before
    /// accessing getDst() or getDestResult().
    Operand getDst() {
      return {};
    }
    /// See getDst().
    OpResult getDestResult() {
      return {};
    }
    /// Returns true if the operation is a load-to-LDS operation.
    bool getLds() {
      return true;
    }
  }];
  let extraClassDefinition = [{
    LogicalResult $cppClass::verify() {
      Value off_or_idx = getOffOrIdx();
      bool offen = getOffen();
      bool idxen = getIdxen();
      if (!off_or_idx && (offen || idxen))
        return emitError("expects an address operand when offen or idxen is set");
      if (off_or_idx && !(offen || idxen))
        return emitError("expects no address operand when offen or idxen is not set");
      if (!off_or_idx)
        return success();
      auto regTy = cast<RegisterTypeInterface>(off_or_idx.getType());
      int32_t nRegs = regTy.getAsRange().size();
      int32_t expectedRegs = static_cast<int32_t>(offen) + static_cast<int32_t>(idxen);
      if (nRegs != expectedRegs)
        return emitError() << "expected " << expectedRegs << " register(s) for the address based on the offen and idxen flags, got " << nRegs;
      return success();
    }
  }];
}
#endif // BUFFERLOADLDSINST_DEF

#ifndef BUFFERLOADLDSDWORD_DEF
#define BUFFERLOADLDSDWORD_DEF
def BufferLoadLDSDword : BufferLoadLDSInst<"buffer_load_lds_dword"> {
  let description = [{
    Load 32 bits of data from a buffer surface into LDS.
  }];
  let asm_mnemonic = "buffer_load_dword";
  let encodings = [InstEnc<[ENC_MUBUF_CDNA3, ENC_MUBUF_CDNA4], 0b0010100>];
}
#endif // BUFFERLOADLDSDWORD_DEF

#ifndef BUFFERLOADLDSDWORDX2_DEF
#define BUFFERLOADLDSDWORDX2_DEF
def BufferLoadLDSDwordx2 : BufferLoadLDSInst<"buffer_load_lds_dwordx2"> {
  let archs = [CDNA4Isa];
  let description = [{
    Load 64 bits of data from a buffer surface into LDS.
  }];
  let encodings = [
    InstEnc<[ENC_MUBUF_CDNA4], 0b0010101>
  ];
  let asm_mnemonic = "buffer_load_dwordx2";
  let asm_syntax = [
    ASMString<[ENC_MUBUF_CDNA4], "${mnemonic} $off_or_idx, $addr, $soffset [offen] [idxen] [const_offset] lds [sc0] [sc1] [nt]">
  ];
}
#endif // BUFFERLOADLDSDWORDX2_DEF

#ifndef BUFFERLOADLDSDWORDX3_DEF
#define BUFFERLOADLDSDWORDX3_DEF
def BufferLoadLDSDwordx3 : BufferLoadLDSInst<"buffer_load_lds_dwordx3"> {
  let archs = [CDNA4Isa];
  let description = [{
    Load 96 bits of data from a buffer surface into LDS.
  }];
  let encodings = [
    InstEnc<[ENC_MUBUF_CDNA4], 0b0010110>
  ];
  let asm_mnemonic = "buffer_load_dwordx3";
  let asm_syntax = [
    ASMString<[ENC_MUBUF_CDNA4], "${mnemonic} $off_or_idx, $addr, $soffset [offen] [idxen] [const_offset] lds [sc0] [sc1] [nt]">
  ];
}
#endif // BUFFERLOADLDSDWORDX3_DEF

#ifndef BUFFERLOADLDSDWORDX4_DEF
#define BUFFERLOADLDSDWORDX4_DEF
def BufferLoadLDSDwordx4 : BufferLoadLDSInst<"buffer_load_lds_dwordx4"> {
  let archs = [CDNA4Isa];
  let description = [{
    Load 128 bits of data from a buffer surface into LDS.
  }];
  let encodings = [
    InstEnc<[ENC_MUBUF_CDNA4], 0b0010111>
  ];
  let asm_mnemonic = "buffer_load_dwordx4";
  let asm_syntax = [
    ASMString<[ENC_MUBUF_CDNA4], "${mnemonic} $off_or_idx, $addr, $soffset [offen] [idxen] [const_offset] lds [sc0] [sc1] [nt]">
  ];
}
#endif // BUFFERLOADLDSDWORDX4_DEF

#ifndef BUFFERSTOREINST_DEF
#define BUFFERSTOREINST_DEF
class BufferStoreInst<string mnemonic, list<Type> dataType> : AMDISAInstruction<mnemonic, [BufferStoreInstOpInterface]> {
  let props = [IsVMem, IsBuffer];
  let archs = [CDNA3Isa, CDNA4Isa];
  let effects = [GlobalWrite];
  let outputs = (ins);
  let inputs = (ins
    AnyRegOf<dataType>:$data,
    SGPROf<4>:$addr,
    OptionalReg<AnyRegOf<[VGPROf<1>, VGPROf<2>]>>:$off_or_idx,
    OneSGPROrSRegOrIntConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>>:$soffset
  );
  let trailingArgs = (ins
    OffsetType<12>:$const_offset,
    IDXEN:$idxen,
    NT:$nt,
    OFFEN:$offen,
    SC0:$sc0,
    SC1:$sc1
  );
  let trailingResults = (outs FlatWriteToken:$token);
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    `data` $data `addr` $addr `offset`
    `u` `(` $soffset `)` `+` (`off_idx` `(` $off_or_idx^ `)` `+`)? `c` `(`$const_offset`)`
    attr-dict `:`
    `ins` `(` type($data) `,` type($addr) `,` type($soffset) (`,` type($off_or_idx)^)? `)`
    `mods` `(` type($const_offset) `)`
    `->` qualified(type($token))
  }];
  let asm_syntax = [
    ASMString<[ENC_MUBUF_CDNA3], "${mnemonic} $data, $off_or_idx, $addr, $soffset [idxen] [offen] [const_offset] [sc0] [nt] [sc1]">,
    ASMString<[ENC_MUBUF_CDNA4], "${mnemonic} $data, $off_or_idx, $addr, $soffset [offen] [idxen] [const_offset] [sc0] [sc1] [nt]">
  ];
  let hasVerifier = 1;
  let extraClassDefinition = [{
    LogicalResult $cppClass::verify() {
      Value off_or_idx = getOffOrIdx();
      bool offen = getOffen();
      bool idxen = getIdxen();
      if (!off_or_idx && (offen || idxen))
        return emitError("expects an address operand when offen or idxen is set");
      if (off_or_idx && !(offen || idxen))
        return emitError("expects no address operand when offen or idxen is not set");
      if (!off_or_idx)
        return success();
      auto regTy = cast<RegisterTypeInterface>(off_or_idx.getType());
      int32_t nRegs = regTy.getAsRange().size();
      int32_t expectedRegs = static_cast<int32_t>(offen) + static_cast<int32_t>(idxen);
      if (nRegs != expectedRegs)
        return emitError() << "expected " << expectedRegs << " register(s) for the address based on the offen and idxen flags, got " << nRegs;
      return success();
    }
  }];
}
#endif // BUFFERSTOREINST_DEF

#ifndef BUFFERSTOREDWORD_DEF
#define BUFFERSTOREDWORD_DEF
def BufferStoreDword : BufferStoreInst<"buffer_store_dword",
    /*data=*/[VGPROrAGPROf<1>]
  > {
  let description = [{
    Store 32 bits of data from vector input registers into a buffer surface.
  }];
  let encodings = [InstEnc<[ENC_MUBUF_CDNA3, ENC_MUBUF_CDNA4], 0b0011100>];
}
#endif // BUFFERSTOREDWORD_DEF

#ifndef BUFFERSTOREDWORDX2_DEF
#define BUFFERSTOREDWORDX2_DEF
def BufferStoreDwordx2 : BufferStoreInst<"buffer_store_dwordx2",
    /*data=*/[VGPROrAGPROf<2>]
  > {
  let description = [{
    Store 64 bits of data from vector input registers into a buffer surface.
  }];
  let encodings = [InstEnc<[ENC_MUBUF_CDNA3, ENC_MUBUF_CDNA4], 0b0011101>];
}
#endif // BUFFERSTOREDWORDX2_DEF

#ifndef BUFFERSTOREDWORDX3_DEF
#define BUFFERSTOREDWORDX3_DEF
def BufferStoreDwordx3 : BufferStoreInst<"buffer_store_dwordx3",
    /*data=*/[VGPROrAGPROf<3>]
  > {
  let description = [{
    Store 96 bits of data from vector input registers into a buffer surface.
  }];
  let encodings = [InstEnc<[ENC_MUBUF_CDNA3, ENC_MUBUF_CDNA4], 0b0011110>];
}
#endif // BUFFERSTOREDWORDX3_DEF

#ifndef BUFFERSTOREDWORDX4_DEF
#define BUFFERSTOREDWORDX4_DEF
def BufferStoreDwordx4 : BufferStoreInst<"buffer_store_dwordx4",
    /*data=*/[VGPROrAGPROf<4>]
  > {
  let description = [{
    Store 128 bits of data from vector input registers into a buffer surface.
  }];
  let encodings = [InstEnc<[ENC_MUBUF_CDNA3, ENC_MUBUF_CDNA4], 0b0011111>];
}
#endif // BUFFERSTOREDWORDX4_DEF

#ifndef GLOBALLOADINST_DEF
#define GLOBALLOADINST_DEF
class GlobalLoadInst<string mnemonic, list<Type> dst0Type> : AMDISAInstruction<mnemonic, [GlobalLoadInstOpInterface]> {
  let props = [IsVMem, IsGlobal];
  let archs = [CDNA3Isa, CDNA4Isa];
  let effects = [GlobalRead];
  let outputs = (ins
    AnyRegOf<dst0Type>:$dest
  );
  let inputs = (ins
    AnyRegOf<[VGPROf<2>, TwoSGPROrVCCType]>:$addr,
    OptionalReg<VGPROf<1>>:$offset
  );
  let trailingArgs = (ins
    OffsetType<13, /*signed=*/true>:$const_offset,
    NT:$nt,
    SC0:$sc0,
    SC1:$sc1
  );
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    `dest` $dest `addr` $addr
    `offset` (`d` `(`$offset^ `)` `+`)? `c` `(`$const_offset`)` attr-dict `:`
    `outs``(` type($dest) `)`
    `ins` `(` type($addr) (`,` type($offset)^)? `)` `mods` `(` type($const_offset) `)` `->`
    custom<LoadResults>(ref(type($dest)), type($dest_res), type($token))
  }];
  let trailingResults = (outs FlatReadToken:$token);
  let asm_syntax = [
    ASMString<[ENC_FLAT_GLBL_CDNA3],
      "${mnemonic} $dest, $addr, off [const_offset] [sc0] [nt] [sc1]",
      CPred<"isa<VGPRType>($_self.getAddr().getType())">>,
    ASMString<[ENC_FLAT_GLBL_CDNA3],
      "${mnemonic} $dest, $offset, $addr [const_offset] [sc0] [nt] [sc1]",
      CPred<"isa<SGPRType>($_self.getAddr().getType())">>,
    ASMString<[ENC_FLAT_GLBL_CDNA4],
      "${mnemonic} $dest, $addr, off [const_offset] [sc0] [sc1] [nt]",
      CPred<"isa<VGPRType>($_self.getAddr().getType())">>,
    ASMString<[ENC_FLAT_GLBL_CDNA4],
      "${mnemonic} $dest, $offset, $addr [const_offset] [sc0] [sc1] [nt]",
      CPred<"isa<SGPRType>($_self.getAddr().getType())">>,
  ];
  let hasVerifier = 1;
  let extraClassDefinition = [{
    LogicalResult $cppClass::verify() {
      if (isa<VGPRType>(getAddr().getType()) && getOffset())
        return emitError("expects no offset with VGPR address");
      if (isa<SGPRType>(getAddr().getType()) && !getOffset())
        return emitError("expects an offset with SGPR address");
      return success();
    }
  }];
}
#endif // GLOBALLOADINST_DEF

#ifndef GLOBALLOADDWORD_DEF
#define GLOBALLOADDWORD_DEF
def GlobalLoadDword : GlobalLoadInst<"global_load_dword",
    /*dest=*/[VGPROrAGPROf<1>]
  > {
  let description = [{
    Load 32 bits of data from the global aperture into a vector register.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA3, ENC_FLAT_GLBL_CDNA4], 0b0010100>];
}
#endif // GLOBALLOADDWORD_DEF

#ifndef GLOBALLOADDWORDX2_DEF
#define GLOBALLOADDWORDX2_DEF
def GlobalLoadDwordx2 : GlobalLoadInst<"global_load_dwordx2",
    /*dest=*/[VGPROrAGPROf<2>]
  > {
  let description = [{
    Load 64 bits of data from the global aperture into a vector register.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA3, ENC_FLAT_GLBL_CDNA4], 0b0010101>];
}
#endif // GLOBALLOADDWORDX2_DEF

#ifndef GLOBALLOADDWORDX3_DEF
#define GLOBALLOADDWORDX3_DEF
def GlobalLoadDwordx3 : GlobalLoadInst<"global_load_dwordx3",
    /*dest=*/[VGPROrAGPROf<3>]
  > {
  let description = [{
    Load 96 bits of data from the global aperture into a vector register.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA3, ENC_FLAT_GLBL_CDNA4], 0b0010110>];
}
#endif // GLOBALLOADDWORDX3_DEF

#ifndef GLOBALLOADDWORDX4_DEF
#define GLOBALLOADDWORDX4_DEF
def GlobalLoadDwordx4 : GlobalLoadInst<"global_load_dwordx4",
    /*dest=*/[VGPROrAGPROf<4>]
  > {
  let description = [{
    Load 128 bits of data from the global aperture into a vector register.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA3, ENC_FLAT_GLBL_CDNA4], 0b0010111>];
}
#endif // GLOBALLOADDWORDX4_DEF


#ifndef GLOBALLOADLDSINST_DEF
#define GLOBALLOADLDSINST_DEF
class GlobalLoadLdsInst<string mnemonic> : AMDISAInstruction<mnemonic, [GlobalLoadInstOpInterface]> {
  let props = [IsVMem, IsGlobal];
  let archs = [CDNA3Isa, CDNA4Isa];
  let effects = [GlobalRead, LDSWrite];
  let outputs = (ins);
  let inputs = (ins
    AnyRegOf<[VGPROf<2>, TwoSGPROrVCCType]>:$addr,
    OptionalReg<VGPROf<1>>:$offset,
    M0Type:$m0_src
  );
  let trailingArgs = (ins
    OffsetType<13, /*signed=*/true>:$const_offset,
    NT:$nt,
    SC0:$sc0,
    SC1:$sc1
  );
  let trailingResults = (outs FlatReadToken:$token);
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    `addr` $addr `m0` $m0_src
    `offset` (`d` `(`$offset^ `)` `+`)? `c` `(`$const_offset`)` attr-dict `:`
    `ins` `(` type($addr) `,` type($m0_src) (`,` type($offset)^)? `)` `mods` `(` type($const_offset) `)` `->`
    qualified(type($token))
  }];
  let asm_syntax = [
    ASMString<[ENC_FLAT_GLBL_CDNA3],
      "${mnemonic} $addr, off [const_offset] [sc0] [nt] [sc1]",
      CPred<"isa<VGPRType>($_self.getAddr().getType())">>,
    ASMString<[ENC_FLAT_GLBL_CDNA3],
      "${mnemonic} $offset, $addr [const_offset] [sc0] [nt] [sc1]",
      CPred<"isa<SGPRType>($_self.getAddr().getType())">>,
    ASMString<[ENC_FLAT_GLBL_CDNA4],
      "${mnemonic} $addr, off [const_offset] [sc0] [sc1] [nt]",
      CPred<"isa<VGPRType>($_self.getAddr().getType())">>,
    ASMString<[ENC_FLAT_GLBL_CDNA4],
      "${mnemonic} $offset, $addr [const_offset] [sc0] [sc1] [nt]",
      CPred<"isa<SGPRType>($_self.getAddr().getType())">>
  ];
  let hasVerifier = 1;
  let instExtraClassDeclaration = [{
    /// Load-to-LDS has no register destination; data goes directly to LDS.
    /// Callers using GlobalLoadInstOpInterface must check for null before
    /// accessing getDst() or getDestResult().
    Operand getDst() {
      return {};
    }
    /// See getDst().
    OpResult getDestResult() {
      return {};
    }
  }];
  let extraClassDefinition = [{
    LogicalResult $cppClass::verify() {
      if (isa<VGPRType>(getAddr().getType()) && getOffset())
        return emitError("expects no offset with VGPR address");
      return success();
    }
  }];
}
#endif // GLOBALLOADLDSINST_DEF

#ifndef GLOBALLOADLDSDWORD_DEF
#define GLOBALLOADLDSDWORD_DEF
def GlobalLoadLdsDword : GlobalLoadLdsInst<"global_load_lds_dword"> {
  let description = [{
    Load 32 bits of untyped data from the global aperture and store the result
    into a data share.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA3, ENC_FLAT_GLBL_CDNA4], 0b0101010>];
}
#endif // GLOBALLOADLDSDWORD_DEF

#ifndef GLOBALLOADLDSDWORDX3_DEF
#define GLOBALLOADLDSDWORDX3_DEF
def GlobalLoadLdsDwordx3 : GlobalLoadLdsInst<"global_load_lds_dwordx3"> {
  let archs = [CDNA4Isa];
  let description = [{
    Untyped buffer load 3 dwords, store result into data share.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA4], 0b1111110>];
  let asm_syntax = [
    ASMString<[ENC_FLAT_GLBL_CDNA4],
      "${mnemonic} $addr, off [const_offset] [sc0] [sc1] [nt]",
      CPred<"isa<VGPRType>($_self.getAddr().getType())">>,
    ASMString<[ENC_FLAT_GLBL_CDNA4],
      "${mnemonic} $offset, $addr [const_offset] [sc0] [sc1] [nt]",
      CPred<"isa<SGPRType>($_self.getAddr().getType())">>
  ];
}
#endif // GLOBALLOADLDSDWORDX3_DEF

#ifndef GLOBALLOADLDSDWORDX4_DEF
#define GLOBALLOADLDSDWORDX4_DEF
def GlobalLoadLdsDwordx4 : GlobalLoadLdsInst<"global_load_lds_dwordx4"> {
  let archs = [CDNA4Isa];
  let description = [{
    Untyped buffer load 4 dwords, store result into data share.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA4], 0b1111101>];
  let asm_syntax = [
    ASMString<[ENC_FLAT_GLBL_CDNA4],
      "${mnemonic} $addr, off [const_offset] [sc0] [sc1] [nt]",
      CPred<"isa<VGPRType>($_self.getAddr().getType())">>,
    ASMString<[ENC_FLAT_GLBL_CDNA4],
      "${mnemonic} $offset, $addr [const_offset] [sc0] [sc1] [nt]",
      CPred<"isa<SGPRType>($_self.getAddr().getType())">>
  ];
}
#endif // GLOBALLOADLDSDWORDX4_DEF

#ifndef GLOBALSTOREDWORDINST_DEF
#define GLOBALSTOREDWORDINST_DEF
class GlobalStoreDwordInst<string mnemonic, list<Type> dataType> : AMDISAInstruction<mnemonic, [GlobalStoreDwordInstOpInterface]> {
  let props = [IsVMem, IsGlobal];
  let archs = [CDNA3Isa, CDNA4Isa];
  let effects = [GlobalWrite];
  let outputs = (ins);
  let inputs = (ins
    AnyRegOf<dataType>:$data,
    AnyRegOf<[VGPROf<2>, TwoSGPROrVCCType]>:$addr,
    OptionalReg<VGPROf<1>>:$offset
  );
  let trailingArgs = (ins
    OffsetType<13, /*signed=*/true>:$const_offset,
    NT:$nt,
    SC0:$sc0,
    SC1:$sc1
  );
  let trailingResults = (outs FlatWriteToken:$token);
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    `data` $data `addr` $addr
    `offset` (`d` `(`$offset^ `)` `+`)? `c` `(`$const_offset`)` attr-dict `:`
    `ins` `(`type($data) `,` type($addr) (`,` type($offset)^)? `)` `mods` `(` type($const_offset) `)`
     `->` qualified(type($token))
  }];
  let asm_syntax = [
    ASMString<[ENC_FLAT_GLBL_CDNA3],
      "${mnemonic} $addr, $data, off [const_offset] [sc0] [nt] [sc1]",
      CPred<"isa<VGPRType>($_self.getAddr().getType())">>,
    ASMString<[ENC_FLAT_GLBL_CDNA3],
      "${mnemonic} $offset, $data, $addr [const_offset] [sc0] [nt] [sc1]",
      CPred<"isa<SGPRType>($_self.getAddr().getType())">>,
    ASMString<[ENC_FLAT_GLBL_CDNA4],
      "${mnemonic} $addr, $data, off [const_offset] [sc0] [sc1] [nt]",
      CPred<"isa<VGPRType>($_self.getAddr().getType())">>,
    ASMString<[ENC_FLAT_GLBL_CDNA4],
      "${mnemonic} $offset, $data, $addr [const_offset] [sc0] [sc1] [nt]",
      CPred<"isa<SGPRType>($_self.getAddr().getType())">>,
  ];
  let hasVerifier = 1;
  let extraClassDefinition = [{
    LogicalResult $cppClass::verify() {
      if (isa<VGPRType>(getAddr().getType()) && getOffset())
        return emitError("expects no offset with VGPR address");
      return success();
    }
  }];
}
#endif // GLOBALSTOREDWORDINST_DEF

#ifndef GLOBALSTOREDWORD_DEF
#define GLOBALSTOREDWORD_DEF
def GlobalStoreDword : GlobalStoreDwordInst<"global_store_dword",
    /*data=*/[VGPROrAGPROf<1>]
  > {
  let description = [{
    Store 32 bits of data from vector input registers into the global
    aperture.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA3, ENC_FLAT_GLBL_CDNA4], 0b0011100>];
}
#endif // GLOBALSTOREDWORD_DEF

#ifndef GLOBALSTOREDWORDX2_DEF
#define GLOBALSTOREDWORDX2_DEF
def GlobalStoreDwordx2 : GlobalStoreDwordInst<"global_store_dwordx2",
    /*data=*/[VGPROrAGPROf<2>]
  > {
  let description = [{
    Store 64 bits of data from vector input registers into the global
    aperture.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA3, ENC_FLAT_GLBL_CDNA4], 0b0011101>];
}
#endif // GLOBALSTOREDWORDX2_DEF

#ifndef GLOBALSTOREDWORDX3_DEF
#define GLOBALSTOREDWORDX3_DEF
def GlobalStoreDwordx3 : GlobalStoreDwordInst<"global_store_dwordx3",
    /*data=*/[VGPROrAGPROf<3>]
  > {
  let description = [{
    Store 96 bits of data from vector input registers into the global
    aperture.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA3, ENC_FLAT_GLBL_CDNA4], 0b0011110>];
}
#endif // GLOBALSTOREDWORDX3_DEF

#ifndef GLOBALSTOREDWORDX4_DEF
#define GLOBALSTOREDWORDX4_DEF
def GlobalStoreDwordx4 : GlobalStoreDwordInst<"global_store_dwordx4",
    /*data=*/[VGPROrAGPROf<4>]
  > {
  let description = [{
    Store 128 bits of data from vector input registers into the global
    aperture.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA3, ENC_FLAT_GLBL_CDNA4], 0b0011111>];
}
#endif // GLOBALSTOREDWORDX4_DEF

#endif // AMDGCN_VMEM_TD
