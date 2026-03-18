# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""High-level Python API for building AMDGCN kernel IR.

Usage::

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        b = KernelBuilder("my_mod", "my_kernel", target="gfx942", isa="cdna3")
        b.add_ptr_arg(AccessKind.ReadOnly)
        b.add_ptr_arg(AccessKind.WriteOnly)
        [a_ptr, b_ptr] = b.load_args()
        tid = b.thread_id_x()
        ...
        module = b.build()

The returned module can be passed directly to aster.testing compile_kernel_module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from aster import ir

if TYPE_CHECKING:
    from aster.layout import Layout
from aster.dialects import arith
from aster.dialects import affine as affined
from aster.dialects import func as funcd
from aster.dialects._gpu_ops_gen import (
    ThreadIdOp as _GPUThreadIdOp,
    BlockIdOp as _GPUBlockIdOp,
)
from aster.dialects._amdgcn_ops_gen import (
    AllocaOp,
    EndKernelOp,
    KernelOp,
    LoadArgOp,
    MakeBufferRsrcOp,
    MakeRegisterRangeOp,
    ModuleOp,
    SWaitcntOp,
    LoadOp,
    StoreOp,
)
from aster._mlir_libs._amdgcn import (
    AGPRRangeType,
    AGPRType,
    SGPRRangeType,
    SGPRType,
    VGPRRangeType,
    VGPRType,
)
from aster.dialects import _amdgcn_inst_gen as _inst
from aster.dialects import lsir as lsird
from aster.dialects.amdgcn import (
    AccessKind,
    AddressSpaceKind,
    KernelArgumentFlags,
    get_buffer_argument,
    get_kernel_arguments,
    vop2 as _amdgcn_vop2,
)
from aster.dialects import ptr as ptrd


def _i8(value: int, ctx: ir.Context) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(8, ctx), value)


def _i32(value: int, ctx: ir.Context) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32, ctx), value)


def _i64(value: int, ctx: ir.Context) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64, ctx), value)


class KernelBuilder:
    """Builds an amdgcn kernel IR module.

    Must be created inside an active ``with ctx:`` block. The caller does NOT
    need to manage InsertionPoint or Location -- the builder handles both.
    """

    def __init__(
        self,
        module_name: str,
        kernel_name: str,
        target: str = "gfx942",
        isa: str = "cdna3",
    ):
        ctx = ir.Context.current
        self._ctx = ctx
        self._loc = ir.Location.unknown(ctx)
        self._module_name = module_name
        self._kernel_name = kernel_name
        self._ptr_args: List[AccessKind] = []
        self._kernel_attrs = {}

        # Build outer builtin.module container.
        self._outer = ir.Module.create(self._loc)

        # Build amdgcn.module inside the outer module.
        outer_ip = ir.InsertionPoint(self._outer.body)
        target_attr = ir.Attribute.parse(f"#amdgcn.target<{target}>")
        isa_attr = ir.Attribute.parse(f"#amdgcn.isa<{isa}>")
        self._amdgcn_mod = ModuleOp(
            target=target_attr,
            isa_version=isa_attr,
            sym_name=module_name,
            loc=self._loc,
            ip=outer_ip,
        )

        # Create the module body block (where func.func and amdgcn.kernel live).
        self._mod_block = ir.Block.create_at_start(self._amdgcn_mod.body_region, [])

        # Create amdgcn.kernel (arguments attr set later in load_args / build).
        mod_ip = ir.InsertionPoint(self._mod_block)
        self._kernel_op = KernelOp(
            sym_name=kernel_name,
            loc=self._loc,
            ip=mod_ip,
        )

        # Create the kernel body block. All instruction methods insert here.
        self._kernel_block = ir.Block.create_at_start(self._kernel_op.body_region, [])
        self._kip = ir.InsertionPoint(self._kernel_block)

    # ---------------------------------------------------------------------------
    # Kernel arguments
    # ---------------------------------------------------------------------------

    def add_ptr_arg(
        self,
        access: AccessKind = AccessKind.ReadWrite,
    ) -> int:
        """Record a pointer argument. Returns the argument index.

        Call load_args() after all args are registered to get the MLIR Values.
        """
        idx = len(self._ptr_args)
        self._ptr_args.append(access)
        return idx

    def load_args(self) -> List[ir.Value]:
        """Create the load-args helper and return MLIR Values for each pointer.

        Inserts ``func.func private @load_N_ptrs()`` before the kernel in the
        module body, and emits ``func.call`` inside the kernel body.
        Returns one SGPRRangeType(size=2) Value per registered pointer argument.
        """
        n = len(self._ptr_args)
        fn_name = f"load_{n}_ptrs"
        sgpr2_type = SGPRRangeType.get(self._ctx, size=2)
        ret_types = [sgpr2_type] * n

        # Set kernel arguments attribute.
        buf_args = [
            get_buffer_argument(
                address_space=AddressSpaceKind.Global,
                access=access,
                flags=KernelArgumentFlags.None_,
                ctx=self._ctx,
            )
            for access in self._ptr_args
        ]
        self._kernel_op.operation.attributes["arguments"] = get_kernel_arguments(
            buf_args, ctx=self._ctx
        )

        # Build func.func private before the kernel.
        # Insert at the beginning of the module block -- kernel is always last.
        func_type = ir.FunctionType.get([], ret_types)
        before_ip = ir.InsertionPoint.at_block_begin(self._mod_block)
        fn_op = funcd.FuncOp(
            fn_name, func_type, visibility="private", loc=self._loc, ip=before_ip
        )
        func_block = ir.Block.create_at_start(fn_op.body, [])
        func_ip = ir.InsertionPoint(func_block)
        loaded = []
        for i in range(n):
            la = LoadArgOp(
                result=sgpr2_type,
                index=_i64(i, self._ctx),
                loc=self._loc,
                ip=func_ip,
            )
            loaded.append(la.result)
        SWaitcntOp(lgkmcnt=_i8(0, self._ctx), loc=self._loc, ip=func_ip)
        funcd.ReturnOp(loaded, loc=self._loc, ip=func_ip)

        # Emit func.call inside the kernel body.
        call_op = funcd.CallOp(ret_types, fn_name, [], loc=self._loc, ip=self._kip)
        return list(call_op.results)

    # ---------------------------------------------------------------------------
    # Thread/block IDs
    # ---------------------------------------------------------------------------

    def thread_id_x(self) -> ir.Value:
        """Get thread ID x as index.

        Emits gpu.thread_id x which returns index type and is lowered through the ASTER
        pass pipeline (aster-to-int-arith -> aster-codegen -> expand-md-ops).
        """
        dim_x = ir.Attribute.parse("#gpu<dim x>")
        return _GPUThreadIdOp(dim_x, loc=self._loc, ip=self._kip).result

    def block_id_x(self) -> ir.Value:
        """Get workgroup (block) ID x as index.

        Emits gpu.block_id x which returns index type and is lowered through the ASTER
        pass pipeline (aster-to-int-arith -> aster-codegen -> expand-md-ops).
        """
        dim_x = ir.Attribute.parse("#gpu<dim x>")
        return _GPUBlockIdOp(dim_x, loc=self._loc, ip=self._kip).result

    @staticmethod
    def _as_value(v) -> ir.Value:
        """Extract an ir.Value from either a Value or an OpView."""
        if isinstance(v, ir.Value):
            return v
        return v.results[0]

    def _index_to_vgpr(self, val: ir.Value) -> ir.Value:
        """Convert an index value to a VGPR via affine.apply + arith.index_cast + lsir.to_reg."""
        identity_map = ir.Attribute.parse("affine_map<(d0) -> (d0)>")
        idx_val = affined.apply(identity_map, [val], loc=self._loc, ip=self._kip)
        i32_type = ir.IntegerType.get_signless(32, self._ctx)
        i32_val = arith.index_cast(i32_type, idx_val, loc=self._loc, ip=self._kip)
        vgpr_type = VGPRType.get(self._ctx, reg=None)
        return lsird.to_reg(vgpr_type, i32_val, loc=self._loc, ip=self._kip)

    def _emit_vop2(self, opcode_str: str, dest: ir.Value, src0, src1) -> ir.Value:
        """Emit amdgcn.vop2 via module-level _amdgcn_vop2 helper.

        Uses a dedicated helper because the ODS-generated VOP2Op has incorrect
        _ODS_RESULT_SEGMENTS for DPS-style ops.
        """
        dest_v = self._as_value(dest)
        src0_v = self._as_value(src0)
        src1_v = self._as_value(src1)
        return _amdgcn_vop2(
            opcode_str, dest_v, src0_v, src1_v, loc=self._loc, ip=self._kip
        )

    # ---------------------------------------------------------------------------
    # Register allocation
    # ---------------------------------------------------------------------------

    def alloca_vgpr(self, reg: Optional[int] = None) -> ir.Value:
        """Allocate a VGPR register."""
        return AllocaOp(
            VGPRType.get(self._ctx, reg), loc=self._loc, ip=self._kip
        ).result

    def alloca_sgpr(self, reg: Optional[int] = None) -> ir.Value:
        """Allocate an SGPR register."""
        return AllocaOp(
            SGPRType.get(self._ctx, reg), loc=self._loc, ip=self._kip
        ).result

    def alloca_agpr(self, reg: Optional[int] = None) -> ir.Value:
        """Allocate an AGPR register."""
        return AllocaOp(
            AGPRType.get(self._ctx, reg), loc=self._loc, ip=self._kip
        ).result

    def _make_register_range(self, inputs: List[ir.Value]) -> ir.Value:
        return MakeRegisterRangeOp(inputs=inputs, loc=self._loc, ip=self._kip).result

    def alloc_vgprx2(self) -> ir.Value:
        """Allocate a 2-VGPR register range."""
        return self._make_register_range([self.alloca_vgpr() for _ in range(2)])

    def alloc_vgprx4(self) -> ir.Value:
        """Allocate a 4-VGPR register range."""
        return self._make_register_range([self.alloca_vgpr() for _ in range(4)])

    def init_agprx4(self, init_val: ir.Value) -> ir.Value:
        """Allocate and initialize a 4-AGPR register range."""
        inited = [
            _inst.v_accvgpr_write_b32(
                self.alloca_agpr(), init_val, loc=self._loc, ip=self._kip
            )
            for _ in range(4)
        ]
        return self._make_register_range(inited)

    # ---------------------------------------------------------------------------
    # Constants and scalar ops
    # ---------------------------------------------------------------------------

    def constant_i32(self, value: int) -> ir.Value:
        """Emit an i32 constant."""
        return arith.constant(
            ir.IntegerType.get_signless(32, self._ctx),
            value,
            loc=self._loc,
            ip=self._kip,
        )

    def s_mov_b32(self, value: int) -> ir.Value:
        """Move an i32 immediate into an SGPR via s_mov_b32."""
        dest = self.alloca_sgpr()
        c = self.constant_i32(value)
        return _inst.s_mov_b32(dest, c, loc=self._loc, ip=self._kip)

    def sop2(self, opcode: str, src0: ir.Value, src1: ir.Value) -> ir.Value:
        """Scalar ALU 2-operand operation (SOP2)."""
        dest = self.alloca_sgpr()
        from aster.dialects._amdgcn_ops_gen import SOP2Op

        return SOP2Op(
            result=SGPRType.get(self._ctx),
            opcode=ir.Attribute.parse(f"#amdgcn.inst<{opcode}>"),
            outs=dest,
            src0=src0,
            src1=src1,
            loc=self._loc,
            ip=self._kip,
        ).result

    # ---------------------------------------------------------------------------
    # Vector ALU
    # ---------------------------------------------------------------------------

    def vop2(self, opcode: str, src0: ir.Value, src1: ir.Value) -> ir.Value:
        """Vector ALU 2-operand operation."""
        dest = self.alloca_vgpr()
        return self._emit_vop2(opcode, dest, src0, src1)

    def v_add_u32(self, src0: ir.Value, src1: ir.Value) -> ir.Value:
        """VOP2 v_add_u32: src0 + src1 -> VGPR."""
        return self.vop2("v_add_u32", src0, src1)

    def byte_offset(self, tid: ir.Value, elem_bytes: int = 4) -> ir.Value:
        """Compute byte offset = tid * elem_bytes.

        Accepts both index type (from thread_id_x/block_id_x) and VGPR type.
        For index inputs: uses affine.apply (scale) + arith.index_cast + lsir.to_reg.
        For VGPR inputs: uses v_lshlrev_b32_e32 (shift).
        """
        if isinstance(tid.type, ir.IndexType):
            scale_map = ir.Attribute.parse(f"affine_map<(d0) -> (d0 * {elem_bytes})>")
            offset_idx = affined.apply(scale_map, [tid], loc=self._loc, ip=self._kip)
            i32_type = ir.IntegerType.get_signless(32, self._ctx)
            offset_i32 = arith.index_cast(
                i32_type, offset_idx, loc=self._loc, ip=self._kip
            )
            vgpr_type = VGPRType.get(self._ctx, reg=None)
            return lsird.to_reg(vgpr_type, offset_i32, loc=self._loc, ip=self._kip)
        shift = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6}[elem_bytes]
        c_shift = self.constant_i32(shift)
        return self.vop2("v_lshlrev_b32_e32", c_shift, tid)

    def layout_byte_offset(self, tid: ir.Value, layout: "Layout") -> ir.Value:
        """Compute byte offset from thread ID using affine.delinearize + linearize.

        Emits:
          %coords:N = affine.delinearize_index %tid into (shape...)
          %offset   = affine.linearize_index [%coords...] by (stride...)

        Returns an index-typed value. The ASTER pipeline
        (aster-affine-optimize-ptr-add, aster-to-int-arith) lowers these.
        """
        from aster.layout.codegen import Delinearize, Linearize, layout_to_ops

        assert isinstance(
            tid.type, ir.IndexType
        ), "layout_byte_offset expects index-typed tid (from thread_id_x())"

        ops = layout_to_ops(layout)
        idx_type = ir.IndexType.get(self._ctx)
        val: ir.Value = tid
        coords: list[ir.Value] = []

        for op in ops:
            if isinstance(op, Delinearize):
                shape_attr = ir.DenseI64ArrayAttr.get(list(op.basis), self._ctx)
                result = affined.delinearize_index(
                    [idx_type] * len(op.basis),
                    val,
                    [],
                    shape_attr,
                    loc=self._loc,
                    ip=self._kip,
                )
                coords = list(result) if len(op.basis) > 1 else [result]
            elif isinstance(op, Linearize):
                # affine.linearize_index treats basis as sizes (suffix-product
                # strides), NOT as explicit strides. Use affine.apply with an
                # explicit affine_map to get the correct dot product:
                #   offset = c0 * s0 + c1 * s1 + ...
                strides = op.basis
                n = len(strides)
                dims = ", ".join(f"d{i}" for i in range(n))
                terms = " + ".join(f"d{i} * {strides[i]}" for i in range(n))
                map_str = f"affine_map<({dims}) -> ({terms})>"
                amap = ir.Attribute.parse(map_str)
                val = affined.apply(amap, coords, loc=self._loc, ip=self._kip)
            else:
                raise TypeError(f"Unknown layout op: {type(op)}")

        return val

    def index_cast_i32(self, index_val: ir.Value) -> ir.Value:
        """Cast an index value to i32."""
        i32_type = ir.IntegerType.get_signless(32, self._ctx)
        return arith.index_cast(i32_type, index_val, loc=self._loc, ip=self._kip)

    # ---------------------------------------------------------------------------
    # Pointer arithmetic (ptr dialect)
    # ---------------------------------------------------------------------------

    def flat_global_addr(self, sgpr_ptr: ir.Value, byte_offset: ir.Value) -> ir.Value:
        """Compute a flat global address from SGPR pointer + index byte offset.

        Uses lsir.from_reg -> ptr.ptr_add -> lsir.to_reg. The ASTER pipeline
        (aster-affine-optimize-ptr-add) decomposes the ptr.ptr_add offset into
        const/uniform/dynamic components automatically.

        Args:
            sgpr_ptr: base pointer as SGPRx2
            byte_offset: byte offset as index type
        Returns:
            VGPRx2 containing the 64-bit flat global address
        """
        gptr_type = ir.Type.parse("!ptr.ptr<#ptr.generic_space>")
        gptr = lsird.from_reg(gptr_type, sgpr_ptr, loc=self._loc, ip=self._kip)

        off_i32 = self.index_cast_i32(byte_offset)
        addr_ptr = ptrd.ptr_add(
            gptr,
            off_i32,
            ptrd.PtrAddFlags.none,
            loc=self._loc,
            ip=self._kip,
        )

        vx2_type = VGPRRangeType.get(self._ctx, size=2)
        return lsird.to_reg(vx2_type, addr_ptr, loc=self._loc, ip=self._kip)

    # ---------------------------------------------------------------------------
    # MFMA
    # ---------------------------------------------------------------------------

    def mfma(
        self,
        opcode: str,
        acc: ir.Value,
        a: ir.Value,
        b: ir.Value,
    ) -> ir.Value:
        """Matrix fused multiply-add: acc = A * B + acc.

        opcode: e.g. "v_mfma_f32_16x16x16_f16"
        """
        fn = getattr(_inst, opcode, None)
        if fn is not None:
            return fn(acc, a, b, acc, loc=self._loc, ip=self._kip)
        raise ValueError(f"Unknown MFMA opcode: {opcode}")

    # ---------------------------------------------------------------------------
    # Buffer resource descriptor
    # ---------------------------------------------------------------------------

    def make_buffer_rsrc(
        self,
        ptr: ir.Value,
        num_records: ir.Value,
        stride: ir.Value,
        cache_swizzle: bool = False,
        swizzle_enable: bool = False,
        flags: int = 131072,
    ) -> ir.Value:
        """Build a 4-SGPR buffer resource descriptor."""
        return MakeBufferRsrcOp(
            result=SGPRRangeType.get(self._ctx, size=4),
            base_addr=ptr,
            num_records=num_records,
            stride=stride,
            cache_swizzle=ir.BoolAttr.get(cache_swizzle),
            swizzle_enable=ir.BoolAttr.get(swizzle_enable),
            flags=_i32(flags, self._ctx),
            loc=self._loc,
            ip=self._kip,
        ).result

    # ---------------------------------------------------------------------------
    # Buffer memory operations
    # ---------------------------------------------------------------------------

    def _buffer_load_with_dest(
        self,
        opcode: str,
        dest: ir.Value,
        rsrc: ir.Value,
        soffset: ir.Value,
        voffset: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """Emit a buffer load using a pre-allocated dest register (or range)."""
        if const_offset is None:
            const_offset = self.constant_i32(0)
        read_tok_type = ir.Type.parse("!amdgcn.read_token<flat>")
        op = LoadOp(
            opcode=ir.Attribute.parse(f"#amdgcn.inst<{opcode}>"),
            dest=dest,
            addr=rsrc,
            uniform_offset=soffset,
            dynamic_offset=voffset,
            constant_offset=const_offset,
            results=[dest.type, read_tok_type],
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    def buffer_load(
        self,
        rsrc: ir.Value,
        soffset: ir.Value,
        voffset: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """Buffer load (buffer_load_dword) -> single VGPR."""
        dest = AllocaOp(VGPRType.get(self._ctx), loc=self._loc, ip=self._kip).result
        return self._buffer_load_with_dest(
            "buffer_load_dword", dest, rsrc, soffset, voffset, const_offset
        )

    def buffer_load_dwordx2(
        self,
        rsrc: ir.Value,
        soffset: ir.Value,
        voffset: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """Buffer load of 2 dwords -> VGPRRangeType(size=2)."""
        dest = self.alloc_vgprx2()
        return self._buffer_load_with_dest(
            "buffer_load_dwordx2", dest, rsrc, soffset, voffset, const_offset
        )

    def buffer_load_dwordx4(
        self,
        rsrc: ir.Value,
        soffset: ir.Value,
        voffset: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """Buffer load of 4 dwords -> VGPRRangeType(size=4)."""
        dest = self.alloc_vgprx4()
        return self._buffer_load_with_dest(
            "buffer_load_dwordx4", dest, rsrc, soffset, voffset, const_offset
        )

    def _buffer_store(
        self,
        opcode: str,
        data: ir.Value,
        rsrc: ir.Value,
        soffset: ir.Value,
        voffset: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        if const_offset is None:
            const_offset = self.constant_i32(0)
        write_tok_type = ir.Type.parse("!amdgcn.write_token<flat>")
        op = StoreOp(
            opcode=ir.Attribute.parse(f"#amdgcn.inst<{opcode}>"),
            data=data,
            addr=rsrc,
            uniform_offset=soffset,
            dynamic_offset=voffset,
            constant_offset=const_offset,
            results=[write_tok_type],
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    def buffer_store_dword(
        self, data, rsrc, soffset, voffset, const_offset=None
    ) -> ir.Value:
        """Buffer store of a single dword."""
        return self._buffer_store(
            "buffer_store_dword", data, rsrc, soffset, voffset, const_offset
        )

    def buffer_store_dwordx2(
        self, data, rsrc, soffset, voffset, const_offset=None
    ) -> ir.Value:
        """Buffer store of 2 dwords."""
        return self._buffer_store(
            "buffer_store_dwordx2", data, rsrc, soffset, voffset, const_offset
        )

    def buffer_store_dwordx4(
        self, data, rsrc, soffset, voffset, const_offset=None
    ) -> ir.Value:
        """Buffer store of 4 dwords."""
        return self._buffer_store(
            "buffer_store_dwordx4", data, rsrc, soffset, voffset, const_offset
        )

    # ---------------------------------------------------------------------------
    # Global memory operations
    # ---------------------------------------------------------------------------

    def global_load(
        self,
        ptr: ir.Value,
        dyn_offset: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """Global memory load (global_load_dword) -> single VGPR."""
        if const_offset is None:
            const_offset = self.constant_i32(0)
        dest = AllocaOp(VGPRType.get(self._ctx), loc=self._loc, ip=self._kip).result
        read_tok_type = ir.Type.parse("!amdgcn.read_token<flat>")
        op = LoadOp(
            opcode=ir.Attribute.parse("#amdgcn.inst<global_load_dword>"),
            dest=dest,
            addr=ptr,
            dynamic_offset=dyn_offset,
            constant_offset=const_offset,
            results=[dest.type, read_tok_type],
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    def global_store(
        self,
        data: ir.Value,
        ptr: ir.Value,
        dyn_offset: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """Global memory store (global_store_dword)."""
        if const_offset is None:
            const_offset = self.constant_i32(0)
        write_tok_type = ir.Type.parse("!amdgcn.write_token<flat>")
        op = StoreOp(
            opcode=ir.Attribute.parse("#amdgcn.inst<global_store_dword>"),
            data=data,
            addr=ptr,
            dynamic_offset=dyn_offset,
            constant_offset=const_offset,
            results=[write_tok_type],
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    # ---------------------------------------------------------------------------
    # Flat global memory operations (ptr dialect path)
    # ---------------------------------------------------------------------------

    def _flat_global_load(
        self,
        opcode: str,
        dest: ir.Value,
        addr: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """Flat global load using a pre-computed 64-bit address (VGPRx2)."""
        if const_offset is None:
            const_offset = self.constant_i32(0)
        # Global loads need a dynamic_offset operand (zero for flat addr path).
        c0 = self.constant_i32(0)
        zero_v = lsird.to_reg(VGPRType.get(self._ctx), c0, loc=self._loc, ip=self._kip)
        read_tok_type = ir.Type.parse("!amdgcn.read_token<flat>")
        op = LoadOp(
            opcode=ir.Attribute.parse(f"#amdgcn.inst<{opcode}>"),
            dest=dest,
            addr=addr,
            dynamic_offset=zero_v,
            constant_offset=const_offset,
            results=[dest.type, read_tok_type],
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    def flat_global_load_dwordx4(
        self,
        addr: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """Flat global load of 4 dwords from a 64-bit address (VGPRx2)."""
        dest = self.alloc_vgprx4()
        return self._flat_global_load("global_load_dwordx4", dest, addr, const_offset)

    def _flat_global_store(
        self,
        opcode: str,
        data: ir.Value,
        addr: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """Flat global store using a pre-computed 64-bit address (VGPRx2)."""
        if const_offset is None:
            const_offset = self.constant_i32(0)
        c0 = self.constant_i32(0)
        zero_v = lsird.to_reg(VGPRType.get(self._ctx), c0, loc=self._loc, ip=self._kip)
        write_tok_type = ir.Type.parse("!amdgcn.write_token<flat>")
        op = StoreOp(
            opcode=ir.Attribute.parse(f"#amdgcn.inst<{opcode}>"),
            data=data,
            addr=addr,
            dynamic_offset=zero_v,
            constant_offset=const_offset,
            results=[write_tok_type],
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    def flat_global_store_dwordx4(
        self,
        data: ir.Value,
        addr: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """Flat global store of 4 dwords to a 64-bit address (VGPRx2)."""
        return self._flat_global_store("global_store_dwordx4", data, addr, const_offset)

    # ---------------------------------------------------------------------------
    # Synchronization
    # ---------------------------------------------------------------------------

    def wait_vmcnt(self, count: int = 0):
        """Insert s_waitcnt vmcnt=count."""
        SWaitcntOp(vmcnt=_i8(count, self._ctx), loc=self._loc, ip=self._kip)

    def wait_lgkmcnt(self, count: int = 0):
        """Insert s_waitcnt lgkmcnt=count."""
        SWaitcntOp(lgkmcnt=_i8(count, self._ctx), loc=self._loc, ip=self._kip)

    # ---------------------------------------------------------------------------
    # Build
    # ---------------------------------------------------------------------------

    def set_shared_memory_size(self, size: int):
        """Set LDS (shared memory) size for the kernel."""
        self._kernel_attrs["shared_memory_size"] = _i32(size, self._ctx)

    def build(self) -> ir.Module:
        """Finalize the kernel and return the outer ir.Module."""
        for key, val in self._kernel_attrs.items():
            self._kernel_op.operation.attributes[key] = val

        EndKernelOp(loc=self._loc, ip=self._kip)
        return self._outer
