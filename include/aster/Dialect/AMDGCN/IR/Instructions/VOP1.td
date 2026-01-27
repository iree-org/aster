//===- VOP1.td - VOP1 Operations ---------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines VOP1 specific operations.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_VOP1_TD
#define AMDGCN_INST_VOP1_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"

//===----------------------------------------------------------------------===//
// VOP1 Nop Operation
//===----------------------------------------------------------------------===//

def VOP1NopOp : VOP1_Op<"v_nop", [AMDGCNInstOpInterface]> {
  let summary = "VOP1 NOP operation";
  let description = [{Do nothing.This operation is a vector no -
                      op that occupies a VALU slot.}];

  let arguments = (ins
    DefaultValuedAttr<AMDGCN_InstAttr, "mlir::aster::amdgcn::OpCode::V_NOP">:$opcode
  );

  let results = (outs);

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
MutableArrayRef<OpOperand> getInstInsMutable() { return {}; }
}];
}

def V_NOP_INST : AMDInst<
                     /*mnemonic=*/"v_nop", /*symbol=*/"v_nop",
                     /*instOp=*/VOP1NopOp, /*isa=*/[CDNA3]> {
  let summary = "VOP1 NOP";
  let description = [{Do nothing.This is a vector no -
                      op instruction that occupies a VALU slot.}];
  let pythonBuilder = DefaultPyBuilder<(ins)>;
  let asmFormat = [SingleAsmVariant<"">];
}

//===----------------------------------------------------------------------===//
// VOP1 Operation
//===----------------------------------------------------------------------===//

def VOP1Op : VOP1_Op<"vop1", [
  AMDGCNInstTraits, AMDGCNInstOpInterface, AllTypesMatch<["vdst", "result"]>
]> {
  let summary = "VOP1 (Vector Operations with 1 operand) operation";
  let description = [{
    The VOP1 operation performs vector operations with one source operand for CDNA
    architecture. It takes one source operand (src0) and a destination operand (vdst).
    The source operand (src0) can be a register (SGPR or VGPR) or an immediate value.
    The result type matches the vdst type.

    VOP1 Format fields:
    - VDST [24:17]: Destination VGPR
    - OP [30:25]: Opcode
    - SRC0 [8:0]: Source 0 operand (SGPR, VGPR, or immediate)
    - ENCODING [31]: Must be: 0

    Example:
    ```mlir
    %result = vop1 #amdgcn.inst<v_mov_b32_e32> %vdst, %src0
        : !amdgcn.vgpr<10>, !amdgcn.vgpr<11> -> !amdgcn.vgpr<10>
    ```
  }];
  let arguments = (ins
    AMDGCN_InstAttr:$opcode,
    VGPRType:$vdst,
    RegOrImm<[RegisterLike], [I32]>:$src0
  );
  let results = (outs VGPRType : $result);
  let assemblyFormat = [{
    $opcode $vdst `,` $src0
    attr-dict `:` functional-type(operands, results)
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
  return getOperation()->getOpOperands().drop_front();
}
}];
}

class VOP1Inst<string _mnemonic, string sym, InstOp _instOp,
               list<ISAVersion> _isa> : AMDInst<_mnemonic, sym, _instOp, _isa> {
  let constraints = (ins
      /*outs=*/VGPROperand<1>:$vdst);
  let pythonBuilder = DefaultPyBuilder<(ins PyValue : $vdst, PyValue : $src0)>;
  let cppBuilder = DefaultCppBuilder<(ins CppValue : $vdst, CppValue : $src0)>;
  let asmFormat = [SingleAsmVariant<"$vdst, $src0">];
}

def V_MOV_B32_E32_INST
    : VOP1Inst<
          /*mnemonic=*/"v_mov_b32_e32", /*symbol=*/"v_mov_b32_e32",
          /*instOp=*/VOP1Op, /*isa=*/[CDNA3]> {
  let summary = "VOP1 move 32-bit";
  let description = [{
    VOP1 instruction that moves a 32 -
    bit value to a destination register.Computes: vdst =
                                            src0 The source operand(src0)
                                                can be a register(SGPR or VGPR)
                                                    or an immediate value.
  }];
}

#endif // AMDGCN_INST_VOP1_TD
