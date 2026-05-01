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
        b = KernelBuilder("my_mod", "my_kernel", target="gfx942")
        b.add_ptr_arg(AccessKind.ReadOnly)
        b.add_ptr_arg(AccessKind.WriteOnly)
        [a_ptr, b_ptr] = b.load_args()
        tid = b.thread_id("x")
        ...
        module = b.build()

The returned module can be passed directly to aster.compiler compile_kernel_module.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from aster import ir

if TYPE_CHECKING:
    from aster.layout import Layout, Swizzle
from aster.dialects import arith
from aster.dialects import affine as affined
from aster.dialects import func as funcd
from aster.dialects._gpu_ops_gen import (
    ThreadIdOp as _GPUThreadIdOp,
    BlockIdOp as _GPUBlockIdOp,
)
from aster.dialects._amdgcn_ops_gen import (
    AllocaOp,
    AllocLDSOp,
    DeallocLDSOp,
    EndKernelOp,
    GetLDSOffsetOp,
    KernelOp,
    LoadArgOp,
    LoadToLDSOp,
    MakeBufferRsrcOp,
    MakeRegisterRangeOp,
    ModuleOp,
    SplitRegisterRangeOp,
    SWaitcntOp,
    WaitOp,
    LoadOp,
    StoreOp,
    VAddU32 as _VAddU32,
    VReadfirstlaneB32 as _VReadfirstlaneB32,
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
)
from aster.dialects import ptr as ptrd


def _i8(value: int, ctx: ir.Context) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(8, ctx), value)


def _i32(value: int, ctx: ir.Context) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32, ctx), value)


def _i64(value: int, ctx: ir.Context) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64, ctx), value)


@dataclass(frozen=True)
class MfmaConfig:
    """MFMA instruction configuration."""

    opcode: str
    # [M, N, K] MFMA tile shape. Must be set explicitly at construction.
    shape: tuple[int, int, int]
    a_regs: int
    b_regs: int
    c_regs: int

    @property
    def k_per_mfma(self) -> int:
        return self.shape[2]

    @property
    def reads_vx4(self) -> bool:
        return self.a_regs == 4


MFMA_F16_CDNA4 = MfmaConfig(
    opcode="v_mfma_f32_16x16x32_f16",
    shape=(16, 16, 32),
    a_regs=4,
    b_regs=4,
    c_regs=4,
)


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
    ):
        ctx = ir.Context.current
        self._ctx = ctx
        self._loc = ir.Location.unknown(ctx)
        self._module_name = module_name
        self._kernel_name = kernel_name
        self._target = target
        self._ptr_args: List[AccessKind] = []
        self._kernel_attrs = {}

        # Build outer builtin.module container.
        self._outer = ir.Module.create(self._loc)

        # Build amdgcn.module inside the outer module.
        outer_ip = ir.InsertionPoint(self._outer.body)
        target_attr = ir.Attribute.parse(f"#amdgcn.target<{target}>")
        self._amdgcn_mod = ModuleOp(
            target=target_attr,
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

        # Cached types -- constructed once, reused everywhere.
        self.idx_type = ir.IndexType.get(ctx)
        self.i32_type = ir.IntegerType.get_signless(32, ctx)
        self.any_type = ir.Type.parse("!aster_utils.any")
        self.m0_type = ir.Type.parse("!amdgcn.m0<0>")
        self.vx2_type = VGPRRangeType.get(ctx, size=2)
        self.vx4_type = VGPRRangeType.get(ctx, size=4)
        self.ax4_type = AGPRRangeType.get(ctx, size=4)
        self.sgpr1_type = ir.Type.parse("!amdgcn.sgpr<[? + 1]>")
        self.sgpr2_type = ir.Type.parse("!amdgcn.sgpr<[? + 2]>")
        self.sgpr4_type = ir.Type.parse("!amdgcn.sgpr<[? + 4]>")
        self.flat_read_tok = ir.Type.parse("!amdgcn.read_token<flat>")
        self.flat_write_tok = ir.Type.parse("!amdgcn.write_token<flat>")
        self.lds_read_tok = ir.Type.parse("!amdgcn.read_token<shared>")
        self.lds_write_tok = ir.Type.parse("!amdgcn.write_token<shared>")

    # ---------------------------------------------------------------------------
    # Pipeline stage annotation
    # ---------------------------------------------------------------------------

    @contextmanager
    def stage(self, stage_id: int):
        """Context manager: tag all ops emitted within with sched.stage."""
        block = self._kip.block
        n_before = sum(1 for _ in block.operations)
        yield
        stage_attr = _i32(stage_id, self._ctx)
        all_ops = list(block.operations)
        for op in all_ops[n_before:]:
            self._tag_stage_recursive(op, stage_attr)

    @staticmethod
    def _tag_stage_recursive(op, stage_attr):
        if "sched.stage" not in op.attributes:
            op.attributes["sched.stage"] = stage_attr
        for region in op.regions:
            for block in region:
                for nested in block.operations:
                    KernelBuilder._tag_stage_recursive(nested, stage_attr)

    # ---------------------------------------------------------------------------
    # Kernel arguments
    # ---------------------------------------------------------------------------

    def add_ptr_arg(
        self,
        access: AccessKind = AccessKind.ReadWrite,
    ) -> int:
        """Record a pointer argument. Returns the argument index.

        Call load_args() after all args are registered to get the MLIR
        Values.
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
    # Scheduled helper functions (for pipelining type erasure)
    # ---------------------------------------------------------------------------

    def define_helper(self, name: str, arg_types: list, ret_types: list, body_fn=None):
        """Define a private func.func in the module body.

        Usable as decorator or direct call::

            @b.define_helper("_load", [ptr_ty, idx_ty], [any_ty, tok_ty])
            def load_fn(bb, ptr, off):
                ...
                return [bb.to_any(data), tok]

            # or: load_fn = b.define_helper("_load", [...], [...], body)

        body_fn(builder, *block_args) emits ops and returns a list of
        values matching ret_types. Returns the function name string.
        """

        def _emit(fn):
            func_type = ir.FunctionType.get(arg_types, ret_types)
            before_ip = ir.InsertionPoint.at_block_begin(self._mod_block)
            fn_op = funcd.FuncOp(
                name, func_type, visibility="private", loc=self._loc, ip=before_ip
            )
            arg_locs = [self._loc] * len(arg_types)
            func_block = ir.Block.create_at_start(fn_op.body, arg_types, arg_locs)
            saved_ip = self._kip
            self._kip = ir.InsertionPoint(func_block)
            ret_vals = fn(self, *list(func_block.arguments))
            if ret_vals is None:
                ret_vals = []
            funcd.ReturnOp(ret_vals, loc=self._loc, ip=self._kip)
            self._kip = saved_ip
            return name

        if body_fn is not None:
            return _emit(body_fn)
        return _emit

    def call_helper(self, name: str, args: list, ret_types: list) -> list:
        """Emit a func.call to a helper function, tagged with current sched.stage."""
        call_op = funcd.CallOp(ret_types, name, args, loc=self._loc, ip=self._kip)
        return list(call_op.results)

    # ---------------------------------------------------------------------------
    # Thread / workgroup / grid IDs (gpu dialect wrappers)
    # ---------------------------------------------------------------------------

    def thread_id(self, dim: str = "x") -> ir.Value:
        """Thread ID within the workgroup (gpu.thread_id)."""
        d = ir.Attribute.parse(f"#gpu<dim {dim}>")
        return _GPUThreadIdOp(d, loc=self._loc, ip=self._kip).result

    def block_id(self, dim: str = "x") -> ir.Value:
        """Workgroup ID within the grid (gpu.block_id)."""
        d = ir.Attribute.parse(f"#gpu<dim {dim}>")
        return _GPUBlockIdOp(d, loc=self._loc, ip=self._kip).result

    def block_dim(self, dim: str = "x") -> ir.Value:
        """Workgroup size (gpu.block_dim)."""
        from aster.dialects._gpu_ops_gen import BlockDimOp as _GPUBlockDimOp

        d = ir.Attribute.parse(f"#gpu<dim {dim}>")
        return _GPUBlockDimOp(d, loc=self._loc, ip=self._kip).result

    def grid_dim(self, dim: str = "x") -> ir.Value:
        """Grid size in workgroups (gpu.grid_dim)."""
        from aster.dialects._gpu_ops_gen import GridDimOp as _GPUGridDimOp

        d = ir.Attribute.parse(f"#gpu<dim {dim}>")
        return _GPUGridDimOp(d, loc=self._loc, ip=self._kip).result

    def lane_id(self, wave_size: int = 64) -> ir.Value:
        """Lane ID within the wave: linear_thread_id % wave_size."""
        ltid = ir.AffineExpr.get_dim(0)
        return self.affine_apply(ltid % wave_size, [self.linear_thread_id()])

    def wave_id(self, wave_size: int = 64) -> ir.Value:
        """Wave ID within the workgroup: linear_thread_id // wave_size."""
        ltid = ir.AffineExpr.get_dim(0)
        return self.affine_apply(
            ir.AffineExpr.get_floor_div(ltid, wave_size), [self.linear_thread_id()]
        )

    def delinearize_index(self, linear_id, sizes):
        """Delinearize a linear index into multi-dimensional coordinates.

        Wraps affine.delinearize_index with static basis.

        Args:
            linear_id: 1-D ir.Value index to decompose.
            sizes: tuple of ints, the basis sizes (first-mode-slowest).

        Returns:
            tuple of ir.Value, one per dimension.
        """
        from aster.dialects._affine_ops_gen import AffineDelinearizeIndexOp

        idx = self.idx_type
        n = len(sizes)
        delin = AffineDelinearizeIndexOp(
            [idx] * n,
            linear_id,
            [],
            list(sizes),
            loc=self._loc,
            ip=self._kip,
        )
        return tuple(delin.results[i] for i in range(n))

    def linearize_index(self, coords, sizes):
        """Linearize multi-dimensional coordinates into a linear index.

        Inverse of delinearize_index. Emits affine.linearize_index with a full
        static basis (hasOuterBound=true) and disjoint=true.
        Note: keeping the outer bound at the type level is important so later
        analyses (value bounds, fold patterns) can reason about the range of the
        result.
        """
        from aster.dialects._affine_ops_gen import AffineLinearizeIndexOp

        sizes = sizes if isinstance(sizes, tuple) else (sizes,)
        return AffineLinearizeIndexOp(
            ir.IndexType.get(),
            list(coords),
            [],
            list(sizes),
            True,
            loc=self._loc,
            ip=self._kip,
        ).result

    def affine_apply(self, expr, dims, symbols=None) -> ir.Value:
        """Emit affine.apply from an ir.AffineExpr, dims and symbols."""
        n_dims = len(dims)
        n_syms = len(symbols) if symbols else 0
        amap = ir.AffineMap.get(n_dims, n_syms, [expr])
        operands = list(dims) + (symbols or [])
        return affined.apply(amap, operands, loc=self._loc, ip=self._kip)

    def linear_thread_id(self) -> ir.Value:
        """Linearized thread ID within the workgroup: tx + bdx * (ty + bdy * tz)."""
        tx_v, ty_v, tz_v = self.thread_id("x"), self.thread_id("y"), self.thread_id("z")
        bdx_v, bdy_v = self.block_dim("x"), self.block_dim("y")
        tx, ty, tz = (ir.AffineExpr.get_dim(i) for i in range(3))
        bdx, bdy = (ir.AffineExpr.get_symbol(i) for i in range(2))
        return self.affine_apply(
            tx + bdx * (ty + bdy * tz), [tx_v, ty_v, tz_v], [bdx_v, bdy_v]
        )

    def linear_block_id(self) -> ir.Value:
        """Linearized workgroup ID across the grid: bx + gdx * (by + gdy * bz)."""
        bx_v, by_v, bz_v = self.block_id("x"), self.block_id("y"), self.block_id("z")
        gdx_v, gdy_v = self.grid_dim("x"), self.grid_dim("y")
        bx, by, bz = (ir.AffineExpr.get_dim(i) for i in range(3))
        gdx, gdy = (ir.AffineExpr.get_symbol(i) for i in range(2))
        return self.affine_apply(
            bx + gdx * (by + gdy * bz), [bx_v, by_v, bz_v], [gdx_v, gdy_v]
        )

    def global_thread_id(self) -> ir.Value:
        """Linearized global thread ID: linear_workgroup_id * threads_per_workgroup + linear_thread_id."""
        ltid_v = self.linear_thread_id()
        lbid_v = self.linear_block_id()
        bdx_v, bdy_v, bdz_v = (
            self.block_dim("x"),
            self.block_dim("y"),
            self.block_dim("z"),
        )
        ltid, lbid = (ir.AffineExpr.get_dim(i) for i in range(2))
        bdx, bdy, bdz = (ir.AffineExpr.get_symbol(i) for i in range(3))
        return self.affine_apply(
            ltid + bdx * bdy * bdz * lbid,
            [ltid_v, lbid_v],
            [bdx_v, bdy_v, bdz_v],
        )

    @staticmethod
    def _as_value(v) -> ir.Value:
        """Extract an ir.Value from either a Value or an OpView."""
        if isinstance(v, ir.Value):
            return v
        return v.results[0]

    def _emit_vop2(self, opcode_str: str, dest: ir.Value, src0, src1) -> ir.Value:
        """Emit amdgcn.vop2."""
        from aster.dialects.amdgcn import vop2

        return vop2(
            opcode=opcode_str,
            dest=self._as_value(dest),
            src0=self._as_value(src0),
            src1=self._as_value(src1),
            loc=self._loc,
            ip=self._kip,
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

    def init_agprx16(self, init_val: ir.Value) -> ir.Value:
        """Allocate and initialize a 16-AGPR register range (32x32 accumulator)."""
        inited = [
            _inst.v_accvgpr_write_b32(
                self.alloca_agpr(), init_val, loc=self._loc, ip=self._kip
            )
            for _ in range(16)
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
        dest = self.alloca_vgpr()
        op = _VAddU32(
            dst0=dest,
            src0=self._as_value(src0),
            src1=self._as_value(src1),
            results=[dest.type],
            loc=self._loc,
            ip=self._kip,
        )
        return op.dst0_res

    def _emit_vop3(
        self, opcode_str: str, dest: ir.Value, src0, src1, *, src2=None
    ) -> ir.Value:
        """Emit amdgcn.vop3."""
        from aster.dialects.amdgcn import vop3

        return vop3(
            opcode=opcode_str,
            dest=self._as_value(dest),
            src0=self._as_value(src0),
            src1=self._as_value(src1),
            src2=self._as_value(src2) if src2 is not None else None,
            loc=self._loc,
            ip=self._kip,
        )

    def vop3(
        self, opcode: str, src0: ir.Value, src1: ir.Value, *, src2: ir.Value = None
    ) -> ir.Value:
        """Vector ALU 3-operand operation."""
        dest = self.alloca_vgpr()
        return self._emit_vop3(opcode, dest, src0, src1, src2=src2)

    def _linearize_layout(self, layout, coord, swizzle=None):
        """Emit layout.linearize (+ optional layout.swizzle) for a linear coord."""
        from aster.dialects import layout as layout_d

        sizes = layout.sizes if isinstance(layout.sizes, tuple) else (layout.sizes,)
        strides = (
            layout.strides if isinstance(layout.strides, tuple) else (layout.strides,)
        )
        attr = layout_d.strided_layout(list(sizes), list(strides), ctx=self._ctx)
        off = layout_d.linearize(coord, attr, loc=self._loc, ip=self._kip)
        if swizzle is None:
            return off
        return layout_d.swizzle(
            off,
            bits=swizzle.bits,
            base=swizzle.base,
            shift=swizzle.shift,
            loc=self._loc,
            ip=self._kip,
        )

    def linearize_layout(
        self,
        coord: ir.Value,
        layout: "Layout",
        swizzle: Optional["Swizzle"] = None,
    ) -> ir.Value:
        """Emit layout.linearize: linear coord -> byte offset via layout strides.

        Optionally applies layout.swizzle for LDS bank conflict
        avoidance.
        """
        return self._linearize_layout(layout, coord, swizzle)

    def apply_swizzle(self, offset: ir.Value, swizzle: "Swizzle") -> ir.Value:
        """Apply XOR swizzle to a byte offset (standalone, without linearize)."""
        from aster.dialects import layout as layout_d

        return layout_d.swizzle(
            offset,
            bits=swizzle.bits,
            base=swizzle.base,
            shift=swizzle.shift,
            loc=self._loc,
            ip=self._kip,
        )

    def index_cast_i32(self, index_val: ir.Value) -> ir.Value:
        """Cast an index value to i32."""
        i32_type = ir.IntegerType.get_signless(32, self._ctx)
        return arith.index_cast(i32_type, index_val, loc=self._loc, ip=self._kip)

    def index_to_vgpr(self, index_val: ir.Value) -> ir.Value:
        """Convert an index value to a VGPR i32 (for buffer_load/store voffset)."""
        i32_val = self.index_cast_i32(index_val)
        vgpr_type = VGPRType.get(self._ctx, reg=None)
        return lsird.to_reg(vgpr_type, i32_val, loc=self._loc, ip=self._kip)

    def index_to_sgpr(self, index_val: ir.Value) -> ir.Value:
        """Convert an index value to an SGPR via v_readfirstlane_b32.

        The index is first materialized as a VGPR i32, then the first
        active lane's value is read into a freshly allocated SGPR.  Use
        for wave-uniform values that must be scalar (e.g. M0 for G2S).
        """
        vgpr_val = self.index_to_vgpr(index_val)
        sgpr = self.alloca_sgpr()
        op = _VReadfirstlaneB32(
            dst0=sgpr,
            src0=vgpr_val,
            results=[sgpr.type],
            loc=self._loc,
            ip=self._kip,
        )
        return op.dst0_res

    # ---------------------------------------------------------------------------
    # Pointer arithmetic (ptr dialect)
    # ---------------------------------------------------------------------------

    def global_addr(self, sgpr_ptr: ir.Value, byte_offset: ir.Value) -> ir.Value:
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

        vx2_type = self.vx2_type
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
            result = fn(acc, a, b, acc, loc=self._loc, ip=self._kip)
            return result
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
        read_tok_type = self.flat_read_tok
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
        write_tok_type = self.flat_write_tok
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
    # Global memory operations (global_load/global_store via 64-bit addr)
    # ---------------------------------------------------------------------------

    def _global_load_op(
        self,
        opcode: str,
        dest: ir.Value,
        addr: ir.Value,
        const_offset: Optional[ir.Value] = None,
        dynamic_offset: Optional[ir.Value] = None,
    ):
        """Global load returning the full LoadOp (for data + token access).

        addr can be SGPRx2 (base pointer) or VGPRx2 (full flat address).
        When addr is SGPRx2, pass the per-thread byte offset as
        dynamic_offset (VGPR) to get saddr+vaddr addressing.
        """
        read_tok_type = self.flat_read_tok
        op = LoadOp(
            opcode=ir.Attribute.parse(f"#amdgcn.inst<{opcode}>"),
            dest=dest,
            addr=addr,
            dynamic_offset=dynamic_offset,
            constant_offset=const_offset,
            results=[dest.type, read_tok_type],
            loc=self._loc,
            ip=self._kip,
        )
        return op

    def global_load_dword(
        self,
        addr: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> tuple[ir.Value, ir.Value]:
        """Global load of 1 dword."""
        dest = self._make_register_range([self.alloca_vgpr()])
        op = self._global_load_op("global_load_dword", dest, addr, const_offset)
        return op.results[0], op.results[1]

    def global_load_dwordx2(
        self,
        addr: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> tuple[ir.Value, ir.Value]:
        """Global load of 2 dwords.

        Warning: dwordx2 limits bandwidth to ~half peak. Prefer dwordx4.
        """
        dest = self.alloc_vgprx2()
        op = self._global_load_op("global_load_dwordx2", dest, addr, const_offset)
        return op.results[0], op.results[1]

    def global_load_dwordx4(
        self,
        addr: ir.Value,
        const_offset: Optional[ir.Value] = None,
        dynamic_offset: Optional[ir.Value] = None,
    ) -> tuple[ir.Value, ir.Value]:
        """Global load of 4 dwords."""
        dest = self.alloc_vgprx4()
        op = self._global_load_op(
            "global_load_dwordx4", dest, addr, const_offset, dynamic_offset
        )
        return op.results[0], op.results[1]

    def _global_store(
        self,
        opcode: str,
        data: ir.Value,
        addr: ir.Value,
        const_offset: Optional[ir.Value] = None,
        dynamic_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """Global store with optional dynamic_offset for saddr+vaddr."""
        write_tok_type = self.flat_write_tok
        op = StoreOp(
            opcode=ir.Attribute.parse(f"#amdgcn.inst<{opcode}>"),
            data=data,
            addr=addr,
            dynamic_offset=dynamic_offset,
            constant_offset=const_offset,
            results=[write_tok_type],
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    def global_store_dword(
        self,
        data: ir.Value,
        addr: ir.Value,
        const_offset: Optional[ir.Value] = None,
        dynamic_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """Global store of 1 dword."""
        return self._global_store(
            "global_store_dword", data, addr, const_offset, dynamic_offset
        )

    def global_store_dwordx4(
        self,
        data: ir.Value,
        addr: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """Global store of 4 dwords to a 64-bit address (VGPRx2)."""
        return self._global_store("global_store_dwordx4", data, addr, const_offset)

    # ---------------------------------------------------------------------------
    # DS (LDS) operations
    # ---------------------------------------------------------------------------

    def _ds_addr(self, addr):
        """Ensure addr is a VGPR. Accepts index, i32, or VGPR transparently.

        This mirrors the .mlir pattern (arith.index_cast + lsir.to_reg)
        so callers can pass index-typed addresses and the conversion to
        VGPR happens at the point of use, not earlier.
        """
        if addr.type == self.idx_type:
            return self.index_to_vgpr(addr)
        i32_type = ir.IntegerType.get_signless(32, self._ctx)
        if addr.type == i32_type:
            vgpr_type = VGPRType.get(self._ctx, reg=None)
            return lsird.to_reg(vgpr_type, addr, loc=self._loc, ip=self._kip)
        return addr

    def _ds_write(self, opcode, data, addr, const_offset=None):
        if const_offset is None:
            const_offset = self.constant_i32(0)
        addr = self._ds_addr(addr)
        write_tok_type = self.lds_write_tok
        op = StoreOp(
            opcode=ir.Attribute.parse(f"#amdgcn.inst<{opcode}>"),
            data=data,
            addr=addr,
            constant_offset=const_offset,
            results=[write_tok_type],
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    def _ds_read(self, opcode, dest, addr, const_offset=None):
        if const_offset is None:
            const_offset = self.constant_i32(0)
        addr = self._ds_addr(addr)
        read_tok_type = self.lds_read_tok
        op = LoadOp(
            opcode=ir.Attribute.parse(f"#amdgcn.inst<{opcode}>"),
            dest=dest,
            addr=addr,
            constant_offset=const_offset,
            results=[dest.type, read_tok_type],
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0], op.results[1]

    def ds_write_b32(self, data, addr, const_offset=None):
        """DS write 32-bit (1 dword) to LDS."""
        return self._ds_write("ds_write_b32", data, addr, const_offset)

    def ds_write_b64(self, data, addr, const_offset=None):
        """DS write 64-bit (2 dwords) to LDS."""
        return self._ds_write("ds_write_b64", data, addr, const_offset)

    def ds_write_b128(self, data, addr, const_offset=None):
        """DS write 128-bit (4 dwords) to LDS."""
        return self._ds_write("ds_write_b128", data, addr, const_offset)

    def ds_read_b32(self, addr, const_offset=None) -> tuple[ir.Value, ir.Value]:
        """DS read 32-bit (1 dword) from LDS."""
        return self._ds_read("ds_read_b32", self.alloca_vgpr(), addr, const_offset)

    def ds_read_b64(self, addr, const_offset=None) -> tuple[ir.Value, ir.Value]:
        """DS read 64-bit (2 dwords) from LDS."""
        return self._ds_read("ds_read_b64", self.alloc_vgprx2(), addr, const_offset)

    def ds_read_b128(self, addr, const_offset=None) -> tuple[ir.Value, ir.Value]:
        """DS read 128-bit (4 dwords) from LDS."""
        return self._ds_read("ds_read_b128", self.alloc_vgprx4(), addr, const_offset)

    # ---------------------------------------------------------------------------
    # G2S: Global-to-LDS direct loads (CDNA4)
    # ---------------------------------------------------------------------------

    def alloc_m0(self) -> ir.Value:
        """Allocate the M0 register (used for G2S LDS destination offset)."""
        m0_type = self.m0_type
        op = AllocaOp(result=m0_type, loc=self._loc, ip=self._kip)
        return op.result

    def set_m0(self, m0: ir.Value, value: ir.Value) -> ir.Value:
        """Set M0 register via s_mov_b32.

        Returns the written M0 value.
        """
        return _inst.s_mov_b32(m0, value, loc=self._loc, ip=self._kip)

    def g2s_buffer_load_dwordx4(
        self,
        m0: ir.Value,
        rsrc: ir.Value,
        soffset: ir.Value,
        voffset: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """G2S: buffer_load_dwordx4 with LDS flag (128-bit per lane).

        Loads 128 bits per lane directly from global memory into LDS,
        bypassing VGPRs. Completion tracked by vmcnt (not lgkmcnt).

        Args:
            m0: M0 register holding LDS destination base offset.
            rsrc: 4-SGPR buffer resource descriptor.
            soffset: Scalar offset (SGPR).
            voffset: Per-lane dynamic offset (VGPR).
            const_offset: Optional constant offset (i32).
        Returns:
            Write token for vmcnt tracking.
        """
        if const_offset is None:
            const_offset = self.constant_i32(0)
        write_tok_type = self.flat_write_tok
        op = LoadToLDSOp(
            opcode=ir.Attribute.parse("#amdgcn.inst<buffer_load_dwordx4_lds>"),
            m0=m0,
            addr=rsrc,
            uniform_offset=soffset,
            dynamic_offset=voffset,
            constant_offset=const_offset,
            results=[write_tok_type],
            loc=self._loc,
            ip=self._kip,
        )
        return op.token

    def g2s_buffer_load_dword(
        self,
        m0: ir.Value,
        rsrc: ir.Value,
        soffset: ir.Value,
        voffset: ir.Value,
        const_offset: Optional[ir.Value] = None,
    ) -> ir.Value:
        """G2S: buffer_load_dword with LDS flag (32-bit per lane).

        Same as g2s_buffer_load_dwordx4 but loads only 32 bits per lane.
        """
        if const_offset is None:
            const_offset = self.constant_i32(0)
        write_tok_type = self.flat_write_tok
        op = LoadToLDSOp(
            opcode=ir.Attribute.parse("#amdgcn.inst<buffer_load_dword_lds>"),
            m0=m0,
            addr=rsrc,
            uniform_offset=soffset,
            dynamic_offset=voffset,
            constant_offset=const_offset,
            results=[write_tok_type],
            loc=self._loc,
            ip=self._kip,
        )
        return op.token

    def s_nop(self, count: int = 0):
        """Insert s_nop with given wait count."""
        _inst.s_nop(imm=count, loc=self._loc, ip=self._kip)

    # ---------------------------------------------------------------------------
    # Synchronization
    # ---------------------------------------------------------------------------

    def wait_deps(self, *tokens):
        """Wait for dependency tokens (from global_load or ds_write)."""
        WaitOp(dependencies=list(tokens), loc=self._loc, ip=self._kip)

    def wait_vmcnt(self, count: int = 0):
        """Insert s_waitcnt vmcnt=count."""
        SWaitcntOp(vmcnt=_i8(count, self._ctx), loc=self._loc, ip=self._kip)

    def wait_lgkmcnt(self, count: int = 0):
        """Insert s_waitcnt lgkmcnt=count."""
        SWaitcntOp(lgkmcnt=_i8(count, self._ctx), loc=self._loc, ip=self._kip)

    # ---------------------------------------------------------------------------
    # LDS allocation
    # ---------------------------------------------------------------------------

    def alloc_lds(self, size_bytes: int) -> tuple[ir.Value, ir.Value]:
        """Allocate LDS buffer."""
        handle = AllocLDSOp(
            static_size=_i64(size_bytes, self._ctx),
            loc=self._loc,
            ip=self._kip,
        ).result
        i32_type = ir.IntegerType.get_signless(32, self._ctx)
        offset_i32 = GetLDSOffsetOp(
            result=i32_type, buffer=handle, loc=self._loc, ip=self._kip
        ).result
        idx_type = self.idx_type
        offset_idx = arith.index_cast(idx_type, offset_i32, loc=self._loc, ip=self._kip)
        return handle, offset_idx

    def dealloc_lds(self, handle):
        """Deallocate an LDS buffer."""
        DeallocLDSOp(buffer=handle, loc=self._loc, ip=self._kip)

    # ---------------------------------------------------------------------------
    # Register range splitting
    # ---------------------------------------------------------------------------

    def join_vx2_to_vx4(self, lo: ir.Value, hi: ir.Value) -> ir.Value:
        """Join two VGPRx2 into one VGPRx4."""
        v_type = VGPRType.get(self._ctx, reg=None)
        lo_split = SplitRegisterRangeOp(
            input=lo, results=[v_type, v_type], loc=self._loc, ip=self._kip
        )
        hi_split = SplitRegisterRangeOp(
            input=hi, results=[v_type, v_type], loc=self._loc, ip=self._kip
        )
        return self._make_register_range(
            [
                lo_split.results_[0],
                lo_split.results_[1],
                hi_split.results_[0],
                hi_split.results_[1],
            ]
        )

    def split_vx4(self, vx4_val) -> tuple[ir.Value, ir.Value]:
        """Split a VGPRx4 into two VGPRx2 values."""
        v_type = VGPRType.get(self._ctx, reg=None)
        op = SplitRegisterRangeOp(
            input=vx4_val,
            results=[v_type, v_type, v_type, v_type],
            loc=self._loc,
            ip=self._kip,
        )
        r = op.results_
        lo = self._make_register_range([r[0], r[1]])
        hi = self._make_register_range([r[2], r[3]])
        return lo, hi

    def split_ax4(self, ax4_val) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
        """Split an AGPRx4 into four individual AGPRs."""
        a_type = AGPRType.get(self._ctx, reg=None)
        op = SplitRegisterRangeOp(
            input=ax4_val,
            results=[a_type, a_type, a_type, a_type],
            loc=self._loc,
            ip=self._kip,
        )
        r = op.results_
        return r[0], r[1], r[2], r[3]

    # ---------------------------------------------------------------------------
    # Constants
    # ---------------------------------------------------------------------------

    def constant_index(self, value: int) -> ir.Value:
        """Create an index-typed constant."""
        idx_type = self.idx_type
        attr = ir.IntegerAttr.get(idx_type, value)
        return arith.ConstantOp(idx_type, attr, loc=self._loc, ip=self._kip).result

    def constant_f32(self, value: float) -> ir.Value:
        """Create an f32 constant."""
        f32_type = ir.F32Type.get(self._ctx)
        attr = ir.FloatAttr.get(f32_type, value)
        return arith.ConstantOp(f32_type, attr, loc=self._loc, ip=self._kip).result

    # ---------------------------------------------------------------------------
    # Structured control flow (scf.for)
    # ---------------------------------------------------------------------------

    def _emit_for(
        self,
        lb: ir.Value,
        ub: ir.Value,
        step: ir.Value,
        iter_args: Optional[list[ir.Value]],
        results: Optional[list],
        constexpr: bool,
        body_fn,
    ) -> None:
        args = list(iter_args) if iter_args else []
        result_types = [a.type for a in args]

        for_op = ir.Operation.create(
            "scf.for",
            results=result_types,
            operands=[lb, ub, step] + args,
            regions=1,
            loc=self._loc,
            ip=self._kip,
        )
        block_arg_types = [lb.type] + result_types
        block_arg_locs = [self._loc] * len(block_arg_types)
        body_block = ir.Block.create_at_start(
            for_op.regions[0], block_arg_types, block_arg_locs
        )

        iv = body_block.arguments[0]
        inner_args = list(body_block.arguments[1:])

        saved_ip = self._kip
        self._kip = ir.InsertionPoint(body_block)

        ret = body_fn(iv, *inner_args) if args else body_fn(iv)
        yield_operands = []
        if ret:
            yield_operands = [
                v.result if hasattr(v, "result") else self._as_value(v) for v in ret
            ]

        ir.Operation.create(
            "scf.yield", operands=yield_operands, loc=self._loc, ip=self._kip
        )
        self._kip = saved_ip

        if constexpr:
            for_op.attributes["aster.constexpr"] = ir.UnitAttr.get(self._ctx)

        if results is not None:
            results.extend(for_op.results)

    def loop(
        self,
        lb: ir.Value,
        ub: ir.Value,
        step: ir.Value,
        iter_args: Optional[list[ir.Value]] = None,
        results: Optional[list] = None,
    ):
        """Decorator: emit scf.for at definition time.

        Void loop::

            @b.loop(c0, k_tiles, c1)
            def _(k_iv):
                ...

        With iter_args::

            results = []
            @b.loop(c0, k_tiles, c1, iter_args=[acc_init], results=results)
            def _(k_iv, acc):
                ...
                return [new_acc]
            acc_final = results[0]
        """

        def decorator(body_fn):
            self._emit_for(lb, ub, step, iter_args, results, False, body_fn)

        return decorator

    def static_loop(
        self,
        lb: ir.Value,
        ub: ir.Value,
        step: ir.Value,
        iter_args: Optional[list[ir.Value]] = None,
        results: Optional[list] = None,
    ):
        """Decorator: emit constexpr scf.for (unrolled) at definition time.

        Void loop::

            @b.static_loop(c0, n_tiles, c1)
            def _(idx):
                ...

        With iter_args::

            results = []
            @b.static_loop(c0, n, c1, iter_args=[init], results=results)
            def _(idx, val):
                ...
                return [new_val]
            final = results[0]
        """

        def decorator(body_fn):
            self._emit_for(lb, ub, step, iter_args, results, True, body_fn)

        return decorator

    def scf_if(self, condition: ir.Value):
        """Decorator: emit scf.if (no results, no else) at definition time.

        Usage::

            @b.scf_if(cond)
            def _():
                b.s_barrier()
        """

        def decorator(body_fn):
            if_op = ir.Operation.create(
                "scf.if",
                results=[],
                operands=[condition],
                regions=2,
                loc=self._loc,
                ip=self._kip,
            )
            then_block = ir.Block.create_at_start(if_op.regions[0], [], [])
            saved_ip = self._kip
            self._kip = ir.InsertionPoint(then_block)
            body_fn()
            ir.Operation.create("scf.yield", operands=[], loc=self._loc, ip=self._kip)
            self._kip = saved_ip
            # Empty else region (required by scf.if).
            else_block = ir.Block.create_at_start(if_op.regions[1], [], [])
            ir.Operation.create(
                "scf.yield",
                operands=[],
                loc=self._loc,
                ip=ir.InsertionPoint(else_block),
            )

        return decorator

    def thread_uniform_if(self, predicate: str, lhs: ir.Value, rhs: ir.Value):
        """Decorator: emit scf.if guarded by a wavefront-uniform condition.

        The condition (arith.cmpi) is emitted right before the scf.if so they
        stay in the same block after CF lowering.  Both operands must be
        wavefront-uniform (e.g. wave_id, constants, block_id).

        Usage::

            @b.thread_uniform_if("ult", wave_id, b.constant_index(4))
            def _():
                b.s_barrier()
        """

        def decorator(body_fn):
            cond = self.assume_uniform(self.arith_cmpi(predicate, lhs, rhs))
            # Emit the scf.if immediately after the compare -- same block.
            if_op = ir.Operation.create(
                "scf.if",
                results=[],
                operands=[cond],
                regions=2,
                loc=self._loc,
                ip=self._kip,
            )
            then_block = ir.Block.create_at_start(if_op.regions[0], [], [])
            saved_ip = self._kip
            self._kip = ir.InsertionPoint(then_block)
            body_fn()
            ir.Operation.create("scf.yield", operands=[], loc=self._loc, ip=self._kip)
            self._kip = saved_ip
            else_block = ir.Block.create_at_start(if_op.regions[1], [], [])
            ir.Operation.create(
                "scf.yield",
                operands=[],
                loc=self._loc,
                ip=ir.InsertionPoint(else_block),
            )

        return decorator

    # ---------------------------------------------------------------------------
    # Memref operations (for multi-tile state buffers)
    # ---------------------------------------------------------------------------

    def memref_alloca(self, size: ir.Value, element_type: ir.Type) -> ir.Value:
        """Allocate memref<?xT> on stack."""
        dyn = ir.ShapedType.get_dynamic_size()
        with self._loc:
            memref_type = ir.MemRefType.get([dyn], element_type)
        op = ir.Operation.create(
            "memref.alloca",
            results=[memref_type],
            operands=[size],
            attributes={
                "operandSegmentSizes": ir.DenseI32ArrayAttr.get([1, 0], self._ctx),
            },
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    def memref_store(self, value, memref: ir.Value, *indices: ir.Value):
        """Store value into memref at indices."""
        v = value.result if hasattr(value, "result") else self._as_value(value)
        ir.Operation.create(
            "memref.store",
            operands=[v, memref, *indices],
            loc=self._loc,
            ip=self._kip,
        )

    def memref_load(self, memref: ir.Value, *indices: ir.Value) -> ir.Value:
        """Load value from memref at indices."""
        memref_type = ir.MemRefType(memref.type)
        element_type = memref_type.element_type
        op = ir.Operation.create(
            "memref.load",
            results=[element_type],
            operands=[memref, *indices],
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    def foreach_tile(self, n, body_fn=None, *, types=None):
        """Static loop over n tiles with optional memref output.

        types: list of (ir.Type, count) pairs.
        body_fn(idx) returns a flat list of values grouped by type.
        Returns memrefs (tuple if >1, single if 1).

        Usable as @decorator or called directly.
        """
        d0 = ir.AffineExpr.get_dim(0)
        c0, c1 = self.constant_index(0), self.constant_index(1)
        bufs, counts = [], []
        if types:
            for ty, count in types:
                bufs.append(self.memref_alloca(self.constant_index(n * count), ty))
                counts.append(count)

        def _run(fn):
            @self.static_loop(c0, self.constant_index(n), c1)
            def _(idx):
                ret = fn(idx)
                if bufs and ret is not None:
                    vi = 0
                    for buf, count in zip(bufs, counts):
                        for t in range(count):
                            si = (
                                idx
                                if count == 1
                                else self.affine_apply(d0 * count + t, [idx])
                            )
                            self.memref_store(ret[vi], buf, si)
                            vi += 1

            if not bufs:
                return None
            return bufs[0] if len(bufs) == 1 else tuple(bufs)

        if body_fn is not None:
            return _run(body_fn)
        return _run

    # ---------------------------------------------------------------------------
    # Arithmetic helpers
    # ---------------------------------------------------------------------------

    def arith_minui(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        """Unsigned minimum of two index/integer values."""
        return arith.minui(lhs, rhs, loc=self._loc, ip=self._kip)

    def arith_cmpi(self, predicate: str, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        """Integer comparison returning i1.

        Predicate is one of: eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge.
        """
        _PRED_MAP = {
            "eq": 0,
            "ne": 1,
            "slt": 2,
            "sle": 3,
            "sgt": 4,
            "sge": 5,
            "ult": 6,
            "ule": 7,
            "ugt": 8,
            "uge": 9,
        }
        i1 = ir.IntegerType.get_signless(1, self._ctx)
        pred_attr = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64, self._ctx), _PRED_MAP[predicate]
        )
        op = ir.Operation.create(
            "arith.cmpi",
            results=[i1],
            operands=[lhs, rhs],
            attributes={"predicate": pred_attr},
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    def assume_uniform(self, value: ir.Value) -> ir.Value:
        """Mark a value as thread-uniform for the uniformity analysis."""
        op = ir.Operation.create(
            "aster_utils.assume_uniform",
            results=[value.type],
            operands=[value],
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    # ---------------------------------------------------------------------------
    # Type erasure (to_any / from_any)
    # ---------------------------------------------------------------------------

    def to_any(self, value: ir.Value) -> ir.Value:
        """Wrap a typed value in !aster_utils.any for pipeline stage crossing."""
        any_type = self.any_type
        op = ir.Operation.create(
            "aster_utils.to_any",
            results=[any_type],
            operands=[value],
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    def from_any(self, any_value: ir.Value, target_type: ir.Type) -> ir.Value:
        """Recover a concrete type from !aster_utils.any."""
        op = ir.Operation.create(
            "aster_utils.from_any",
            results=[target_type],
            operands=[any_value],
            loc=self._loc,
            ip=self._kip,
        )
        return op.results[0]

    # ---------------------------------------------------------------------------
    # Barrier
    # ---------------------------------------------------------------------------

    def s_barrier(self):
        """Insert s_barrier for workgroup synchronization."""
        _inst.s_barrier(loc=self._loc, ip=self._kip)

    # ---------------------------------------------------------------------------
    # Build
    # ---------------------------------------------------------------------------

    def set_shared_memory_size(self, size: int):
        """Set LDS (shared memory) size for the kernel."""
        self._kernel_attrs["shared_memory_size"] = _i32(size, self._ctx)

    def set_block_dims(self, x: int, y: int = 1, z: int = 1):
        """Set workgroup dimensions."""
        self._kernel_attrs["block_dims"] = ir.DenseI32ArrayAttr.get(
            [x, y, z], self._ctx
        )

    def set_grid_dims(self, x: int, y: int = 1, z: int = 1):
        """Set grid dimensions."""
        self._kernel_attrs["grid_dims"] = ir.DenseI32ArrayAttr.get([x, y, z], self._ctx)

    # ---------------------------------------------------------------------------
    # Module finalization
    # ---------------------------------------------------------------------------

    def build(self) -> ir.Module:
        """Finalize the kernel and return the outer ir.Module."""
        for key, val in self._kernel_attrs.items():
            self._kernel_op.operation.attributes[key] = val

        EndKernelOp(loc=self._loc, ip=self._kip)
        return self._outer
