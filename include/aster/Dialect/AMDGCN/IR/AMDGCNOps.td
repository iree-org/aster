//===- AMDGCNOps.td - AMDGCN Ops ----------------------------*- tablegen-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the amdgcn dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_OPS
#define AMDGCN_OPS

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.td"
include "aster/Dialect/AMDGCN/IR/AMDGCN.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// AMDGCN Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

def AMDGCN_AllocaOp : AMDGCN_Op<"alloca", [
    AMDGCNOpTraits
  ]> {
  let summary = "AMDGCN alloca operation";
  let description = [{
    Allocates a register.
  }];
  let arguments = (ins);
  let results = (outs RegisterLike:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// DeallocCastOp
//===----------------------------------------------------------------------===//

def AMDGCN_DeallocCastOp : AMDGCN_Op<"dealloc_cast", [
    Pure,
    TypesMatchWith<"unallocated result", "input", "output", [{
      [&]() -> Type {
        auto rTy = dyn_cast<RegisterTypeInterface>($_self);
        return rTy.cloneRegisterType(rTy.getAsRange().getAsRelocatableRange());
      }()
    }]>
  ]> {
  let summary = "AMDGCN dealloc cast operation";
  let description = [{
    Casts a register to an unallocated version of the same register type.

    Example:
    ```mlir
    %1 = dealloc_cast %0 : !amdgcn.vgpr<1>
    // typeof(%1) == !amdgcn.vgpr
    ```
  }];
  let arguments = (ins RegisterTypeInterface:$input);
  let results = (outs RegisterTypeInterface:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

class IntMDAttr<int value = 0> :
  DefaultValuedAttr<I32Attr, !cast<string>(value)>;

class BoolMDAttr<bit value = 0> :
  DefaultValuedAttr<BoolAttr, !cast<string>(value)>;

def ReqdWgpSize : ConfinedAttr<DenseI32ArrayAttr, [DenseArrayCount<3>]> {
  let defaultValue = "{0, 0, 0}";
}

def AMDGCN_KernelOp : AMDGCN_Op<"kernel", [
    AffineScope, IsolatedFromAbove, Symbol, OpAsmOpInterface,
    HasParent<"ModuleOp">, DeclareOpInterfaceMethods<CallableOpInterface>,
    NoRegionArguments, AutomaticAllocationScope
  ]> {
  let summary = "AMDGCN kernel operation";
  let description = [{
    Represents an AMDGCN kernel function with arguments of type ptr or POD.
    The first block of the body region can have block arguments corresponding
    to the kernel's function arguments.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    KernelArgumentsAttr:$arguments,
    IntMDAttr<0>:$shared_memory_size,  // Shared memory size
    IntMDAttr<0>:$private_memory_size, // Private memory size
    // SGPR initial values
    UnitAttr:$enable_private_segment_buffer, // Whether to enable the private segment buffer SGPR
    UnitAttr:$enable_dispatch_ptr,      // Whether to enable the dispatch pointer SGPR
    BoolMDAttr<1>:$enable_kernarg_segment_ptr, // Whether to enable the kernarg segment pointer SGPR
    // Wavefront size 32 if set, else 64
    UnitAttr:$wavefront_size32,
    // Workgroup SGPR enables
    BoolMDAttr<1>:$enable_workgroup_id_x,
    UnitAttr:$enable_workgroup_id_y,
    UnitAttr:$enable_workgroup_id_z,
    // VGPR Workitem mode
    WorkitemIDModeAttr:$workitem_id_mode,
    // Workgroup hints
    ConfinedAttr<IntMDAttr<1024>, [IntMaxValue<1024>]>:$max_flat_workgroup_size,
    ReqdWgpSize:$reqd_workgroup_size,
    // Math flags:
    FloatRoundModeAttr:$f32_round_mode,
    FloatRoundModeAttr:$f16_f64_round_mode,
    FloatDenormModeAttr:$f32_denorm_mode,
    DefaultValuedAttr<FloatDenormModeAttr, "FloatDenormMode::None">:$f16_f64_denorm_mode,
    BoolMDAttr<1>:$ieee_mode,
    UnitAttr:$exception_fp_ieee_invalid_op,
    UnitAttr:$exception_fp_denorm_src,
    UnitAttr:$exception_fp_ieee_div_zero,
    UnitAttr:$exception_fp_ieee_overflow,
    UnitAttr:$exception_fp_ieee_underflow,
    UnitAttr:$exception_fp_ieee_inexact,
    UnitAttr:$exception_int_div_zero
  );
  let regions = (region MinSizedRegion<1>:$body_region);
  let assemblyFormat = [{
    $sym_name (`arguments` $arguments^)? attr-dict-with-keyword $body_region
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    static ::llvm::StringRef getDefaultDialect() {
      return "amdgcn";
    }

    //===------------------------------------------------------------------===//
    // ArgAndResultAttrsOpInterface Methods
    //===------------------------------------------------------------------===//

    ::mlir::ArrayAttr getArgAttrsAttr();
    ::mlir::ArrayAttr getResAttrsAttr();
    void setArgAttrsAttr(::mlir::ArrayAttr attrs);
    void setResAttrsAttr(::mlir::ArrayAttr attrs);
    ::mlir::Attribute removeArgAttrsAttr();
    ::mlir::Attribute removeResAttrsAttr();
  }];
}

//===----------------------------------------------------------------------===//
// MakeRegisterRangeOp
//===----------------------------------------------------------------------===//

def AMDGCN_MakeRegisterRangeOp : AMDGCN_Op<"make_register_range", [
    DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
    Pure
  ]> {
  let summary = "AMDGCN make register range operation";
  let description = [{
    Creates a register range from a variadic number of registers.
  }];
  let arguments = (ins Variadic<RegisterLike>:$inputs);
  let results = (outs RegisterRangeLike:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}

//===----------------------------------------------------------------------===//
// SplitRegisterRangeOp
//===----------------------------------------------------------------------===//

def AMDGCN_SplitRegisterRangeOp : AMDGCN_Op<"split_register_range", [
    DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
    Pure
  ]> {
  let summary = "AMDGCN split register range operation";
  let description = [{
    Splits a register range into individual registers.
  }];
  let arguments = (ins RegisterRangeLike:$input);
  let results = (outs Variadic<RegisterLike>:$results);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

def AMDGCN_ModuleOp : AMDGCN_Op<"module", [
    AffineScope, IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
    OpAsmOpInterface, SingleBlock, NoTerminator
  ]> {
  let summary = "AMDGCN module operation";
  let description = [{
    An AMDGCN module.
  }];
  let arguments = (ins AMDGCNTargetAttr:$target, IsaVersionAttr:$isa_version,
      SymbolNameAttr:$sym_name, OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region SizedRegion<1>:$body_region);
  let assemblyFormat = [{
    $sym_name `target` `=` $target `isa` `=` $isa_version
    attr-dict-with-keyword $body_region
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// The symbol is not optional.
    bool isOptionalSymbol() { return false; }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    static ::llvm::StringRef getDefaultDialect() {
      return "amdgcn";
    }
  }];
}

//===----------------------------------------------------------------------===//
// TestInst
//===----------------------------------------------------------------------===//

def AMDGCN_TestInstOp : AMDGCN_Op<"test_inst", [
    AMDGCNInstOpInterface,
    AttrSizedOperandSegments
  ]> {
  let summary = "AMDGCN test instruction";
  let description = [{
    A generic test instruction for AMDGCN. This operation can model almost
    all instructions in AMDGCN.

    TODO: Add side-effect modelling.

    Example:
    ```mlir
    %r = test_inst outs %0 ins %2, %4 : (!amdgcn.vgpr_range<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr_range<[? + 4]>
    ```
  }];
  let arguments = (ins Variadic<RegisterTypeInterface>:$outs,
                       Variadic<RegisterTypeInterface>:$ins);
  let results = (outs Variadic<RegisterTypeInterface>:$results);
  let assemblyFormat = [{
    (`outs` $outs^)? (`ins` $ins^)? attr-dict `:`
    functional-type(operands, results)
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Get the opcode of the instruction.
    InstAttr getOpcodeAttr() {
      assert(false && "not yet implemented");
      return InstAttr();
    }
    /// Get the instruction output operands.
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getOutsMutable();
    }
    /// Get the instruction input operands.
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getInsMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// AMDGCN Target Operations
//===----------------------------------------------------------------------===//

include "aster/Dialect/AMDGCN/IR/Instructions/DS.td"
include "aster/Dialect/AMDGCN/IR/Instructions/FLAT.td"
include "aster/Dialect/AMDGCN/IR/Instructions/MTBUF.td"
include "aster/Dialect/AMDGCN/IR/Instructions/MUBUF.td"
include "aster/Dialect/AMDGCN/IR/Instructions/SMEM.td"
include "aster/Dialect/AMDGCN/IR/Instructions/SOP1.td"
include "aster/Dialect/AMDGCN/IR/Instructions/SOP2.td"
include "aster/Dialect/AMDGCN/IR/Instructions/SOPC.td"
include "aster/Dialect/AMDGCN/IR/Instructions/SOPK.td"
include "aster/Dialect/AMDGCN/IR/Instructions/SOPP.td"
include "aster/Dialect/AMDGCN/IR/Instructions/VOP1.td"
include "aster/Dialect/AMDGCN/IR/Instructions/VOP2.td"
include "aster/Dialect/AMDGCN/IR/Instructions/VOP3.td"
include "aster/Dialect/AMDGCN/IR/Instructions/VOP3P.td"
include "aster/Dialect/AMDGCN/IR/Instructions/VOPC.td"

def OpCodeEnum : AutoI64EnumCases<"OpCode", "AMDGCN Operation Codes",
    [InvalidCase] # !instances<AMDInst>()
  > {
  let cppNamespace = "::mlir::aster::amdgcn";
}

#endif // AMDGCN_OPS
