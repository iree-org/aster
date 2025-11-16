//===- VOP2.td - VOP2 Operations ---------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines VOP2 specific operations.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_VOP2_TD
#define AMDGCN_INST_VOP2_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"

//===----------------------------------------------------------------------===//
// VOP2 Operation
//===----------------------------------------------------------------------===//

def VOP2Op : VOP2_Op<"vop2", [
    AMDGCNInstTraits,
    AMDGCNInstOpInterface,
    AllTypesMatch<["vdst", "result"]>
  ]> {
  let summary = "VOP2 (Vector Operations with 2 operands) operation";
  let description = [{
    The VOP2 operation performs vector operations with two source operands for CDNA
    architecture. It takes two source operands (src0, vsrc1) and a destination
    operand (vdst). The first operand (src0) can be a register (SGPR or VGPR) or
    an immediate value. The second operand (vsrc1) is always a VGPR. The result
    type matches the vdst type.

    VOP2 Format fields:
    - VDST [24:17]: Destination VGPR
    - OP [30:25]: Opcode
    - SRC0 [8:0]: Source 0 operand (SGPR, VGPR, or immediate)
    - VSRC1 [16:9]: Source 1 operand (VGPR)
    - ENCODING [31]: Must be: 0

    Example:
    ```mlir
    %result = vop2 #amdgcn.inst<v_lshrrev_b32_e32> %vdst, %src0, %vsrc1
        : !amdgcn.vgpr<10>, !amdgcn.vgpr<11> -> !amdgcn.vgpr<12>
    ```
  }];
  let arguments = (ins
    AMDGCN_InstAttr:$opcode,
    VGPRType:$vdst,
    RegOrImm<[RegisterLike], [I32]>:$src0,
    VGPRType:$vsrc1
  );
  let results = (outs VGPRType:$result);
  let assemblyFormat = [{
    $opcode $vdst `,` $src0 `,` $vsrc1
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

class VOP2Inst<string _mnemonic, string sym, InstOp _instOp, list<Target> _targets>
    : AMDInst<_mnemonic, sym, _instOp, _targets> {
  let constraints = (ins
      /*outs=*/VGPROperand<1>:$vdst,
      /*ins=*/VGPROperand<1>:$vsrc1);
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$vdst, PyValue:$src0, PyValue:$vsrc1
  )>;
  let asmFormat = [SingleAsmVariant<"$vdst, $src0, $vsrc1">];
}

def V_LSHRREV_B32_E32_INST : VOP2Inst<
  /*mnemonic=*/"v_lshrrev_b32_e32", /*symbol=*/"v_lshrrev_b32_e32",
  /*instOp=*/VOP2Op, /*targets=*/[GFX942]> {
  let summary = "VOP2 logical shift right reverse 32-bit";
  let description = [{
    VOP2 instruction that performs logical shift right reverse operation.
    Computes: vdst = vsrc1 >> src0
    The "rev" indicates that the operands are reversed - src0 is the shift amount
    (can be SGPR, VGPR, or immediate), vsrc1 is the value to shift (VGPR).
  }];
}

def V_LSHLREV_B32_E32_INST : VOP2Inst<
  /*mnemonic=*/"v_lshlrev_b32_e32", /*symbol=*/"v_lshlrev_b32_e32",
  /*instOp=*/VOP2Op, /*targets=*/[GFX942]> {
  let summary = "VOP2 logical shift left reverse 32-bit";
  let description = [{
    VOP2 instruction that performs logical shift left reverse operation.
    Computes: vdst = vsrc1 << src0
    The "rev" indicates that the operands are reversed - src0 is the shift amount
    (can be SGPR, VGPR, or immediate), vsrc1 is the value to shift (VGPR).
  }];
}

//===----------------------------------------------------------------------===//
// AddI operation
//===----------------------------------------------------------------------===//

defm VAddI : InstDef<
/*mnemonic=*/"vop.add",
/*summary=*/"VOP2/VOP3 add operation",
/*description=*/[{
  Addition operation that adds two source operands.
  Computes: dst = lhs + rhs
}],
/*ins=*/(ins VIntSrc:$lhs, VIntSrc:$rhs, OptionalReg<SGPRRangeType>:$carry_in),
/*outs=*/(ins VGPRType:$dst, OptionalReg<SGPRRangeType>:$carry_out),
/*attrs=*/(ins), /*traits=*/[AMDGCNInstTraits],
/*insts*/[
  InstCase<
    /*mnemonic=*/"v_add_co_u32",
    /*asmFormat=*/[SingleAsmVariant<"$dst, $carry_out, $lhs, $rhs">],
    /*constraints*/(ins NotPresent:$carry_in),
    /*targets=*/[GFX942],
    /*description=*/"VOP2 add with carry out",
    /*cppBuilder=*/OpBuilder<(ins
      CppValue:$dst, CppValue:$carry_out, CppValue:$lhs, CppValue:$rhs), [{
        return $_create(
          $_builder, $_loc, $_opcode, dst, carry_out, lhs, rhs, nullptr);
      }]>,
    /*pyBuilder=*/OpBuilder<(ins
      PyValue:$dst, PyValue:$carry_out, PyValue:$lhs, PyValue:$rhs), [{
        return $_create(
          opcode=$_opcode, dst=dst, lhs=lhs, rhs=rhs, carry_in=None, carry_out=carry_out, $_lastArgs);
      }]>
  >,
  InstCase<
    /*mnemonic=*/"v_add_u16",
    /*asmFormat=*/[SingleAsmVariant<"$dst, $lhs, $rhs">],
    /*constraints*/(ins NotPresent:$carry_in, NotPresent:$carry_out),
    /*targets=*/[GFX942],
    /*description=*/"VOP2 add on 16-bit unsigned operands"
  >,
  InstCase<
    /*mnemonic=*/"v_add_u32",
    /*asmFormat=*/[SingleAsmVariant<"$dst, $lhs, $rhs">],
    /*constraints*/(ins NotPresent:$carry_in, NotPresent:$carry_out),
    /*targets=*/[GFX942],
    /*description=*/"VOP2 add on 32-bit unsigned operands"
  >,
  InstCase<
    /*mnemonic=*/"v_add_i16",
    /*asmFormat=*/[SingleAsmVariant<"$dst, $lhs, $rhs">],
    /*constraints*/(ins NotPresent:$carry_in, NotPresent:$carry_out),
    /*targets=*/[GFX942],
    /*description=*/"VOP2 add on 16-bit signed operands"
  >,
  InstCase<
    /*mnemonic=*/"v_add_i32",
    /*asmFormat=*/[SingleAsmVariant<"$dst, $lhs, $rhs">],
    /*constraints*/(ins NotPresent:$carry_in, NotPresent:$carry_out),
    /*targets=*/[GFX942],
    /*description=*/"VOP2 add on 32-bit signed operands"
  >,
  InstCase<
    /*mnemonic=*/"v_addc_co_u32",
    /*asmFormat=*/[SingleAsmVariant<"$dst, $carry_out, $lhs, $rhs, $carry_in">],
    /*constraints*/(ins Present:$carry_in, Present:$carry_out),
    /*targets=*/[GFX942],
    /*description=*/"VOP add with carry in and out",
    /*cppBuilder=*/OpBuilder<(ins
      CppValue:$dst, CppValue:$carry_out, CppValue:$lhs, CppValue:$rhs, CppValue:$carry_in), [{
        return $_create(
          $_builder, $_loc, $_opcode, dst, carry_out, lhs, rhs, carry_in);
      }]>,
    /*pyBuilder=*/OpBuilder<(ins
      PyValue:$dst, PyValue:$carry_out, PyValue:$lhs, PyValue:$rhs, PyValue:$carry_in), [{
        return $_create(
          opcode=$_opcode, dst=dst, lhs=lhs, rhs=rhs, carry_in=carry_in, carry_out=carry_out, $_lastArgs)
      }]>
  >
],
// Default builders
/*cppBuilder=*/OpBuilder<(ins
  CppValue:$dst, CppValue:$lhs, CppValue:$rhs), [{
    return $_create(
      $_builder, $_loc, $_opcode, dst, nullptr, lhs, rhs, nullptr);
  }]>,
/*pyBuilder=*/OpBuilder<(ins
  PyValue:$dst, PyValue:$lhs, PyValue:$rhs), [{
    return $_create(
      opcode=$_opcode, dst=dst, lhs=lhs, rhs=rhs, carry_in=None, carry_out=None, $_lastArgs)
  }]>
>;

#endif // AMDGCN_INST_VOP2_TD
