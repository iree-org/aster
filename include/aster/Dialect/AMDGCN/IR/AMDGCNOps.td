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

#ifndef ASTER_AMDGCN_OPS
#define ASTER_AMDGCN_OPS

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNVerifiers.td"
include "aster/Dialect/AMDGCN/IR/AMDGCN.td"
include "aster/Interfaces/AllocaOpInterface.td"
include "aster/Interfaces/DependentOpInterface.td"
include "aster/Interfaces/GPUFuncInterface.td"
include "aster/Interfaces/LivenessOpInterface.td"
include "aster/Interfaces/ModuleOpInterface.td"
include "aster/Interfaces/OperandBundleOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
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
    AllocaOpInterface,
    DeclareOpInterfaceMethods<ConditionallySpeculatable>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "AMDGCN alloca operation";
  let description = [{
    Allocates a register.
  }];
  let arguments = (ins);
  let results = (outs RegisterLike:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let extraClassDeclaration = [{
    ::mlir::Value getAlloca() { return getResult(); }
  }];
}

//===----------------------------------------------------------------------===//
// AllocLDS
//===----------------------------------------------------------------------===//

def AMDGCN_AllocLDSOp : AMDGCN_Op<"alloc_lds"> {
  let summary = "AMDGCN allocate LDS operation";
  let description = [{
    Allocates a buffer in the Local Data Store (LDS) with the specified size
    and optional alignment in bytes. The allocated buffer is considered
    live until it is deallocated with `amdgcn.dealloc_lds` or the value
    stops being reachable.

    If offset is provided, it specifies the byte offset of the allocated buffer
    in LDS.

    Example:
    ```mlir
    // Allocate 16 bytes of LDS with 4-byte alignment.
    %1 = amdgcn.alloc_lds 16 alignment 4
    ```
  }];
  let arguments = (ins
    Optional<Index>:$dynamic_size,
    I64Attr:$static_size,
    DefaultValuedAttr<IntValidAlignment<I64Attr>, "16">:$alignment,
    OptionalAttr<ConfinedAttr<I32Attr, [IntNonNegative]>>:$offset
  );
  let results = (outs LDSBufferType:$buffer);
  let assemblyFormat = [{
    custom<AllocSize>($dynamic_size, $static_size) (`alignment` $alignment^)?
    (`offset` $offset^)? attr-dict
  }];
  let extraClassDeclaration = [{
    /// Get the size of the allocated LDS buffer, either static or dynamic.
    OpFoldResult getSize() {
      return ShapedType::isDynamic(getStaticSize()) ?
        OpFoldResult(getDynamicSize()) : OpFoldResult(getStaticSizeAttr());
    }
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BlockIdOp
//===----------------------------------------------------------------------===//

def AMDGCN_BlockIdOp : AMDGCN_Op<"block_id", [Pure]> {
  let summary = "AMDGCN block ID operation";
  let description = [{
    Returns the block (workgroup) ID in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs SGPRType:$result);
  let assemblyFormat = "$dim attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// BlockDimOp
//===----------------------------------------------------------------------===//

def AMDGCN_BlockDimOp : AMDGCN_Op<"block_dim", [Pure]> {
  let summary = "AMDGCN block dimension operation";
  let description = [{
    Returns the block dimension in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs SGPRType:$result);
  let assemblyFormat = "$dim attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// DeallocCastOp
//===----------------------------------------------------------------------===//

def AMDGCN_DeallocCastOp : AMDGCN_Op<"dealloc_cast", [
    Pure,
    TypesMatchWith<"unallocated result", "input", "output", [{
      [&]() -> Type {
        auto rTy = dyn_cast<RegisterTypeInterface>($_self);
        return rTy.cloneRegisterType(rTy.getAsRange().getAsValueRange());
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
// DeallocLDSOp
//===----------------------------------------------------------------------===//

def AMDGCN_DeallocLDSOp : AMDGCN_Op<"dealloc_lds"> {
  let summary = "AMDGCN deallocate LDS operation";
  let description = [{
    Deallocates a buffer in the Local Data Store (LDS). It's considered
    undefined behavior to use a buffer after deallocation.

    Example:
    ```mlir
    amdgcn.dealloc_lds %0
    ```
  }];
  let arguments = (ins LDSBufferType:$buffer);
  let assemblyFormat = "$buffer attr-dict";
}

//===----------------------------------------------------------------------===//
// GetLDSOffsetOp
//===----------------------------------------------------------------------===//

def AMDGCN_GetLDSOffsetOp : AMDGCN_Op<"get_lds_offset", [Pure]> {
  let summary = "AMDGCN get LDS offset operation";
  let description = [{
    Returns the offset to the start of shared memory for the buffer. Its value
    is considered poison if used after the buffer's lifetime has ended, or if
    the buffer never had a valid lifetime.

    Example:
    ```mlir
    %ptr = amdgcn.get_lds_offset %0 : i32
    ```
  }];
  let arguments = (ins LDSBufferType:$buffer);
  let results = (outs AnyTypeOf<[I32, Index]>:$result);
  let assemblyFormat = "$buffer attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// GridDimOp
//===----------------------------------------------------------------------===//

def AMDGCN_GridDimOp : AMDGCN_Op<"grid_dim", [Pure]> {
  let summary = "AMDGCN grid dimension operation";
  let description = [{
    Returns the grid dimension in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs SGPRType:$result);
  let assemblyFormat = "$dim attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// LoadArgOp
//===----------------------------------------------------------------------===//

def AMDGCN_LoadArgOp : AMDGCN_Op<"load_arg", [Pure]> {
  let summary = "AMDGCN load argument operation";
  let description = [{
    Loads a kernel argument by index.
  }];
  let arguments = (ins I64Attr:$index);
  let results = (outs AnyRegOf<[SGPRType]>:$result);
  let assemblyFormat = "$index attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// MakeBufferRsrcOp
//===----------------------------------------------------------------------===//

def AMDGCN_MakeBufferRsrcOp : AMDGCN_Op<"make_buffer_rsrc", [Pure]> {
  let summary = "Construct a buffer resource descriptor";
  let description = [{
    Constructs a 128-bit (4-SGPR) buffer resource descriptor from component
    fields. The buffer resource descriptor describes a buffer in memory and is
    used by buffer load/store instructions (MUBUF/MTBUF).

    The descriptor is specified in four consecutive SGPRs (4-SGPR aligned) and
    sent to the texture cache with each buffer instruction. Its layout is:
      - dword 0: base_addr[31:0]
      - dword 1: base_addr[47:32] | stride[13:0] | cache_swizzle | swizzle_en
      - dword 2: num_records[31:0]
      - dword 3: dst_sel | num_format | data_format | user_vm | index_stride | ...

    The `base_addr` operand provides the 48-bit buffer base address (2 SGPRs).
    The `num_records` operand provides the 32-bit number of accessible records.
    The `stride` operand is the byte stride between elements (i32, must be in
    [0, 16383] if constant). This goes into bits [29:16] of dword 1.
    The `cache_swizzle` attribute enables TC L1 cache bank swizzling.
    The `swizzle_enable` attribute enables AOS swizzle (vs linear addressing).
    The `flags` attribute encodes dword 3 fields (dst_sel, num_format,
    data_format, etc.) as a packed 32-bit immediate.

    Example:
    ```mlir
    %stride = arith.constant 4 : i32
    %rsrc = amdgcn.make_buffer_rsrc %base, %num_records, %stride,
      cache_swizzle = false, swizzle_enable = false, flags = 0
      : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr_range<[? + 4]>
    ```
  }];
  let arguments = (ins
    AnyRegOf<[SGPRType]>:$base_addr,
    SGPRType:$num_records,
    I32:$stride,
    BoolAttr:$cache_swizzle,
    BoolAttr:$swizzle_enable,
    I32Attr:$flags
  );
  let results = (outs SGPRType:$result);
  let assemblyFormat = [{
    $base_addr `,` $num_records `,` $stride `,`
    `cache_swizzle` `=` $cache_swizzle `,`
    `swizzle_enable` `=` $swizzle_enable `,`
    `flags` `=` $flags
    attr-dict `:` `(` type($base_addr) `,` type($num_records) `,` type($stride) `)` `->` type($result)
  }];
  let hasVerifier = 1;
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

def ArgResAttr : DefaultValuedAttr<DictArrayAttr, "::ArrayRef<Attribute>()"> {
  let constBuilderCall = "$_builder.getArrayAttr($0)";
  let isOptional = true;
}

def Dim3DAttr : DefaultValuedAttr<DenseI32ArrayAttr, "ArrayRef<int32_t>()"> {
  let constBuilderCall = "$_builder.getDenseI32ArrayAttr($0)";
  let isOptional = true;
}

def AMDGCN_KernelOp : AMDGCN_Op<"kernel", [
    AffineScope, AutomaticAllocationScope, FunctionOpInterface,
    IsolatedFromAbove, NoRegionArguments, GPUFuncInterface,
    OpAsmOpInterface
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
    UnitAttr:$exception_int_div_zero,
    Dim3DAttr:$grid_dims,
    Dim3DAttr:$block_dims,
    ArgResAttr:$arg_attrs,
    ArgResAttr:$res_attrs
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
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//
    ::mlir::FunctionType getFunctionType() {
      return ::mlir::FunctionType::get(getContext(), {}, {});
    }
    void setFunctionTypeAttr(TypeAttr type) {
      assert(false &&
        "AMDGCN KernelOp does not have a mutable function type attribute");
    }

    /// Get the callable body.
    Region *getCallableRegion() { return &getBodyRegion(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return {}; }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return {}; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the visibility of the symbol.
    SymbolTable::Visibility getVisibility() {
      return SymbolTable::Visibility::Public;
    }
    /// Set the visibility of the symbol.
    void setVisibility(SymbolTable::Visibility visibility) {
      assert(visibility == SymbolTable::Visibility::Public &&
             "AMDGCN KernelOp symbols are always public");
    }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//
    bool isDeclaration() { return false; }

    //===------------------------------------------------------------------===//
    // GPUFuncInterface Methods
    //===------------------------------------------------------------------===//
    /// Return whether the function is a GPU kernel.
    bool isGPUKernel() { return true; }
    /// Mark the function as a GPU kernel.
    void setGPUKernel(bool isKernel) {
      assert(isKernel && "amdgcn.kernel is always a kernel");
    }
    /// Get the host ABI function type.
    std::tuple<FunctionType, ArrayRef<int32_t>, ArrayRef<int32_t>> getHostABI() {
      assert(false && "not yet implemented");
      return {};
    }
    /// Set the host ABI function type.
    void setHostABI(FunctionType type,
                    ArrayRef<int32_t> sizeInBytes,
                    ArrayRef<int32_t> alignInBytes) {
      assert(false && "not yet implemented");
    }
  }];
}

//===----------------------------------------------------------------------===//
// MakeRegisterRangeOp
//===----------------------------------------------------------------------===//

def AMDGCN_MakeRegisterRangeOp : AMDGCN_Op<"make_register_range", [
    DeclareOpInterfaceMethods<LivenessOpInterface>,
    OperandBundleOpInterface,
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
  let extraClassDeclaration = [{
    ::mlir::ValueRange unpackBundle() {
      return getInputs();
    }
  }];
}

//===----------------------------------------------------------------------===//
// RegInterferenceOp
//===----------------------------------------------------------------------===//

def InterferanceOperands : Variadic<RegisterTypeInterface> {
  let minSize = 2;
}

def AMDGCN_RegInterferenceOp : AMDGCN_Op<"reg_interference", [
    DeclareOpInterfaceMethods<LivenessOpInterface>
  ]> {
  let summary = "Add register interference between values";
  let description = [{
    The `amdgcn.reg_interference` operation adds register interference between
    values, indicating that they should not be allocated to the same register.

    Example:
    ```mlir
    amdgcn.reg_interference %r0, %r1, %r2 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    ```
  }];
  let arguments = (ins InterferanceOperands:$inputs);
  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];
}

//===----------------------------------------------------------------------===//
// SplitRegisterRangeOp
//===----------------------------------------------------------------------===//

def AMDGCN_SplitRegisterRangeOp : AMDGCN_Op<"split_register_range", [
    DeclareOpInterfaceMethods<LivenessOpInterface>,
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
    OpAsmOpInterface, SingleBlock, NoTerminator, ModuleOpInterface
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
// LibraryOp
//===----------------------------------------------------------------------===//

def AMDGCN_LibraryOp : AMDGCN_Op<"library", [
    IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
    OpAsmOpInterface, SingleBlock, NoTerminator
  ]> {
  let summary = "AMDGCN library operation";
  let description = [{
    A container for reusable functions that can be preloaded into an amdgcn.module.
    Libraries cannot contain amdgcn.kernel operations.

    The optional `isa` attribute specifies which ISA targets the library supports:

    - When `isa` is absent: the library is fully target-agnostic. No AMDGCN
      target-specific operations are allowed inside nested functions.

    - When `isa` is present: the library supports the listed ISA targets.
      Target-agnostic operations are allowed, as well as AMDGCN target-specific
      instructions that are valid for **all** the listed ISA targets.

    Example (target-agnostic library):
    ```mlir
    amdgcn.library @indexing_lib {
      func.func @compute_offset(%bidx: index, %tidx: index) -> index {
        // Only target-agnostic ops allowed (arith, affine, lsir, etc.)
      }
    }
    ```

    Example (target-specific library for cdna3):
    ```mlir
    amdgcn.library @mfma_lib isa = [#amdgcn.isa<cdna3>] {
      func.func @mfma_f32(...) {
        // CDNA3-specific AMDGCN instructions allowed
      }
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<ArrayAttr>:$isa
  );
  let regions = (region SizedRegion<1>:$body_region);
  let assemblyFormat = [{
    $sym_name (`isa` `=` $isa^)? attr-dict-with-keyword $body_region
  }];
  let hasVerifier = 1;
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

def InstTraitAMDGCN : NativeOpTrait<"InstTrait"> {
  let cppNamespace = "::mlir::aster";
}

def AMDGCN_TestInstOp : AMDGCN_Op<"test_inst", [
    AMDGCNInstOpInterface,
    AttrSizedOperandSegments,
    InstTraitAMDGCN
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
    static constexpr int32_t kLeadingOperandsSize = 0;
    static constexpr int32_t kOutsSize = 1;
    static constexpr int32_t kInsSize = 1;
    static constexpr int32_t kLeadingResultsSize = 0;
    static constexpr int32_t kTrailingResultsSize = 0;

    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Get the opcode of the instruction.
    InstAttr getOpcodeAttr() {
      assert(false && "not yet implemented");
      return InstAttr();
    }
    /// Get the instruction output operands.
    OperandRange getInstOuts() {
      return getOutsMutable();
    }
    /// Get the instruction input operands.
    OperandRange getInstIns() {
      return getAsOperandRange(getInsMutable());
    }

    /// Clones the instruction operation with new operands and results.
    InstOpInterface cloneInst(OpBuilder &builder, ValueRange outs,
                            ValueRange ins) {
      return aster::detail::cloneInstImpl(*this, builder, outs, ins);
    }

    /// Get the operand and result information for the instruction.
    InstOpInfo getInstInfo() {
      return getInstInfoImpl(*this);
    }

    /// Computes liveness transfer function for the instruction.
    ::mlir::LogicalResult livenessTransferFunction(
        LivenessCallback insertCallback,
        LivenessCallback removeCallback,
        IsLiveCallback isLiveCallback) {
      return ::mlir::aster::detail::livenessTransferFunctionImpl(
        *this, insertCallback, removeCallback, isLiveCallback);
    }
  }];
}

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

def AMDGCN_ThreadIdOp : AMDGCN_Op<"thread_id", [Pure]> {
  let summary = "AMDGCN thread ID operation";
  let description = [{
    Returns the thread ID in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs VGPRType:$result);
  let assemblyFormat = "$dim attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// PtrAddOp
//===----------------------------------------------------------------------===//

def AMDGCN_PtrAddOp : AMDGCN_Op<"ptr_add", [
    Pure,
    AllTypesMatch<["ptr", "result"]>
  ]> {
  let summary = "Add register offsets to a pointer";
  let description = [{
    The `amdgcn.ptr_add` operation computes a new pointer by adding register
    offsets to an existing pointer. The operation supports multiple offset
    components:
    - `offset`: A dynamic (potentially non-uniform) register offset (VGPR)
    - `uniform_offset`: An optional uniform register offset (SGPR)
    - `const_offset`: A constant offset attribute (defaults to 0)

    The final address is computed as:
      result = ptr + offset + uniform_offset + const_offset

    This operation is similar to `aster_utils.ptr_add` but operates on
    AMDGCN register types instead of integer values.

    Example:
    ```mlir
    %new_ptr = amdgcn.ptr_add %ptr, %vgpr_offset : !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr
    %new_ptr = amdgcn.ptr_add %ptr, %vgpr_offset, %sgpr_offset : !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr, !amdgcn.sgpr
    %new_ptr = amdgcn.ptr_add %ptr, %vgpr_offset const_offset = 16 : !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr
    ```
  }];
  let arguments = (ins
    AnyRegOf<[SGPRRangeType, VGPRType, VGPRRangeType]>:$ptr,
    VGPRType:$offset,
    Optional<SGPRType>:$uniform_offset,
    DefaultValuedAttr<I64Attr, "0">:$const_offset
  );
  let results = (outs RegisterRangeLike:$result);
  let assemblyFormat = [{
    $ptr `,` $offset (`,` $uniform_offset^)? (`const_offset` `=` $const_offset^)?
    attr-dict `:` type($ptr) `,` type($offset) (`,` type($uniform_offset)^)?
  }];
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

def WaitCountAttr :
  DefaultValuedAttr<I16Attr, "std::numeric_limits<uint16_t>::max()">;

def AMDGCN_WaitOp : AMDGCN_Op<"wait", [
    DeclareOpInterfaceMethods<DependentOpInterface, ["getOutDependency"]>
  ]> {
  let summary = "AMDGCN wait operation";
  let description = [{
    Waits for the completion of operations represented by the given dependency
    tokens, or attributes. This operation is used to express ordering
    constraints between operations.

    The attributes control which memory subsystems to wait for:
    - `vm_cnt`: Wait for vector memory operations
    - `lgkm_cnt`: Wait for lds, gws and constant memory operations

    NOTE: This operations is compatiable only with CDNA3 and CDNA4 targets.
    TODO: Support RDNA4 counts, and other targets.

    Example:
    ```mlir
    %tok0 = amdgcn.store global_store_dword data %data addr %addr
        : ins(!amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>)
          -> !amdgcn.write_token<flat>
    %result, %tok1 = amdgcn.load global_load_dword dest %dest addr %addr2
        : dps(!amdgcn.vgpr_range<[? + 1]>) ins(!amdgcn.vgpr_range<[? + 2]>)
          -> !amdgcn.read_token<flat>
    amdgcn.wait %tok0, %tok1 : !amdgcn.write_token<flat>, !amdgcn.read_token<flat>
    ```
  }];
  let arguments = (ins
    Variadic<TokenDependencyTypeInterface>:$dependencies,
    WaitCountAttr:$vm_cnt, WaitCountAttr:$lgkm_cnt
  );
  let assemblyFormat = [{
    oilist(`vm_cnt` $vm_cnt | `lgkm_cnt` $lgkm_cnt) (`deps` $dependencies^)?
    attr-dict (`:` type($dependencies)^)?
  }];
  let extraClassDeclaration = [{
    static constexpr uint16_t kNoWaitCount = std::numeric_limits<uint16_t>::max();

    /// Canonicalize the wait operation, this method removes duplicate operands,
    /// merges consecutive wait operations if possible. Returns an iterator to
    /// the next operation if changes were made, or failure otherwise.
    static FailureOr<Block::iterator> canonicalizeWait(WaitOp op,
      RewriterBase &rewriter, llvm::SetVector<Value>& uniqueDeps);

    /// Add dependencies to the wait operation.
    bool addDependencies(ValueRange deps);

    /// Remove dependencies from the wait operation. Returns true if any changes
    /// were made.
    bool removeDependencies(ValueRange deps);

    /// Set the dependencies of the wait operation.
    void setDependencies(ValueRange deps);

    /// Check if the wait operation waits for any memory subsystem.
    bool hasAnyCount() {
      return getVmCnt() != kNoWaitCount || getLgkmCnt() != kNoWaitCount;
    }

    /// Check if the wait operation is a no-wait (has no dependencies).
    bool isNowait() {
      return !hasAnyCount() && getDependencies().empty();
    }
  }];
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// AMDGCN Target Operations
//===----------------------------------------------------------------------===//

include "aster/Dialect/AMDGCN/IR/Instructions/Comparison.td"
include "aster/Dialect/AMDGCN/IR/Instructions/ControlFlow.td"
include "aster/Dialect/AMDGCN/IR/Instructions/Memory.td"
include "aster/Dialect/AMDGCN/IR/Instructions/SOP1.td"
include "aster/Dialect/AMDGCN/IR/Instructions/SOP2.td"
include "aster/Dialect/AMDGCN/IR/Instructions/SOPP.td"
include "aster/Dialect/AMDGCN/IR/Instructions/VOP1.td"
include "aster/Dialect/AMDGCN/IR/Instructions/VOP2.td"
include "aster/Dialect/AMDGCN/IR/Instructions/VOP3.td"
include "aster/Dialect/AMDGCN/IR/Instructions/VOP3P.td"

def OpCodeEnum : AutoI64EnumCases<"OpCode", "AMDGCN Operation Codes",
    [InvalidCase] # !instances<AMDInst>()
  > {
  let cppNamespace = "::mlir::aster::amdgcn";
}

#endif // ASTER_AMDGCN_OPS
