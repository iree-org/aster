from abc import abstractmethod
from math import prod
from typing import Any, Optional, TYPE_CHECKING

from aster import ir
from aster._mlir_libs._amdgcn import SGPRType, VGPRType
from aster.dialects import arith
from aster.dialects.api import (
    alloca_agpr,
    ds_read_b64,
    global_store_dwordx2,
    make_register_range,
    ds_read_b32,
    ds_read_b128,
    global_store_dword,
    global_store_dwordx4,
    s_load_dwordx2,
    s_waitcnt,
    v_mov_b32_e32,
)
from aster.dialects._amdgcn_ops_gen import SplitRegisterRangeOp

if TYPE_CHECKING:
    from typing import Protocol

    Value = Any
    Context = Any
else:
    Value = ir.Value  # type: ignore
    Context = ir.Context  # type: ignore


def _init_k_range(
    ctx: Context,
    vgprs: list[Value],
    k: int,
) -> tuple[Value, int]:
    """Initialize k VGPRs to zero."""
    int_type = ir.IntegerType.get_signless(32, ctx)  # type: ignore
    zero_const = arith.constant(int_type, 0)
    for i in range(k):
        v_mov_b32_e32(vgprs[i], zero_const)
    vgpr_range = make_register_range(vgprs[:k])
    return vgpr_range, 0


def _init_agpr_k_range(
    ctx: Context,
    agprs: list[Value],
    k: int,
) -> tuple[Value, int]:
    """Initialize k AGPRs to zero by creating a range."""
    # Note: For now assume AGPRs are initialized implicitly when creating the
    # range and they don't need explicit zero initialization like VGPRs.
    # TODO: may need to actually connect to e.g. `v_accvgpr_write_b32 a0, 0`
    agpr_range = make_register_range(agprs[:k])
    return agpr_range, 0


def _load_k_range(
    ctx: Context,
    lds_addr: Value,
    vgprs: list[Value],
    k: int,
) -> tuple[Value, int]:
    """Load k VGPRs from LDS memory."""
    assert k in [2, 4], "k must be 2 or 4"

    if k == 4:
        # Create register range and load b128
        vgpr_range = make_register_range(vgprs[:4])
        result = ds_read_b128(vgpr_range, lds_addr, offset=0)
        return result, 1

    if k == 2:
        # Create register range and load b64
        vgpr_range = make_register_range(vgprs[:2])
        result = ds_read_b64(vgpr_range, lds_addr, offset=0)
        return result, 1

    return Value, 0


def get_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(prod(shape[i + 1 :]) for i in range(len(shape)))


def linearize(index: tuple[int, ...], strides: tuple[int, ...]) -> int:
    return sum(strides[i] * index[i] for i in range(len(index)))


def delinearize(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    res = []
    for s in reversed(shape):
        res.append(index % s)
        index = index // s
    return tuple(reversed(res))


def delinearize_with_permutation(
    index: int, shape: tuple[int, ...], permutation: tuple[int, ...]
) -> tuple[int, ...]:
    permuted_shape = tuple(shape[permutation[i]] for i in range(len(shape)))
    res = delinearize(index, permuted_shape)
    res = tuple(res[permutation[i]] for i in range(len(res)))
    return res


class LinearizingArray:
    """Helper class to hold a multi-dimensional array of generic objects.

    Linearizes n-D coordinates into a 1-D index and delinearize it back.
    """

    def __init__(self, shape: tuple[int, ...]):
        self.objects: list[Any] = [None] * prod(shape)
        self.shape = shape
        self.strides = get_strides(shape)

    def len(self) -> int:
        return len(self.objects)

    @abstractmethod
    def get(self, index: tuple[int, ...]) -> Any:
        """Get an element using multi-dimensional indexing."""
        pass

    def __getitem__(self, index: tuple[int, ...]) -> Any:
        linear_idx = linearize(index, self.strides)
        return self.objects[linear_idx]

    def __setitem__(self, index: tuple[int, ...], value: Any) -> None:
        linear_idx = linearize(index, self.strides)
        self.objects[linear_idx] = value


class VRegRangeArray(LinearizingArray):
    """VGPR loading helper class.

    Initiates loads of data into vgprs using _load_k_range / ds_read_b32 and does not
    wait.
    """

    def __init__(
        self,
        ctx: Context,
        vgprs: list[Value],
        shape: tuple[int, ...],
        rs: int,
        lds_addr: Optional[Value] = None,
        global_addr: Optional[tuple[Value, Value]] = None,
        zero_init: bool = False,
    ):
        self.ctx = ctx
        self.vgprs = vgprs
        self.num_ranges = prod(s for s in shape)
        self.vgpr_ranges: list[None | Value] = [None] * self.num_ranges
        self.rs = rs
        self.lds_addr = lds_addr
        self.global_addr = global_addr
        self.zero_init = zero_init
        super().__init__(shape)

    def get(self, index: tuple[int, ...]) -> tuple[Value, int]:
        """Return the count of the operations to wait on."""
        # Linearize the multi-dimensional index
        linear_idx = linearize(index, self.strides)
        res = self.objects[linear_idx]
        if res is not None:
            return res, 0
        if self.zero_init:
            res, count = _init_k_range(
                self.ctx, self.vgprs[linear_idx * self.rs :], self.rs
            )
        else:
            assert self.lds_addr is not None, "lds_addr is required for loading"
            res, count = _load_k_range(
                self.ctx,
                self.lds_addr,
                self.vgprs[linear_idx * self.rs :],
                self.rs,
            )
        self.objects[linear_idx] = res
        return res, count


class ARegRangeArray(LinearizingArray):
    """AGPR accumulator helper class.

    Manages AGPR accumulator registers. AGPRs are used for accumulation and don't
    support loading from LDS or global memory directly.
    """

    def __init__(
        self,
        ctx: Context,
        agprs: list[Value],
        shape: tuple[int, ...],
        rs: int,
        zero_init: bool = False,
    ):
        self.ctx = ctx
        self.agprs = agprs
        self.num_ranges = prod(s for s in shape)
        self.rs = rs
        self.zero_init = zero_init
        super().__init__(shape)

    def get(self, index: tuple[int, ...]) -> tuple[Value, int]:
        """Return the AGPR range for the given index."""
        # Linearize the multi-dimensional index
        linear_idx = linearize(index, self.strides)
        res = self.objects[linear_idx]
        if res is not None:
            return res, 0
        if self.zero_init:
            res, count = _init_agpr_k_range(
                self.ctx, self.agprs[linear_idx * self.rs :], self.rs
            )
        else:
            # For non-zero init, just create the range
            res, count = _init_agpr_k_range(
                self.ctx, self.agprs[linear_idx * self.rs :], self.rs
            )
        self.objects[linear_idx] = res
        return res, count


def allocate_lds_addrs(
    ctx: Context,
    arrays: list[VRegRangeArray],
    offsets: list[int],
    unused_vgprs: list[Value],
) -> list[Value]:
    """Allocate VGPRs for LDS addresses with specified offsets.

    Args:
        ctx: MLIR context
        arrays: List of VRegRangeArray objects to set lds_addr for
        offsets: List of offsets for each array's lds_addr
        unused_vgprs: List of unused VGPRs to allocate from

    Returns:
        List of remaining unused VGPRs after allocation
    """
    assert len(arrays) == len(offsets), "number of arrays must match number of offsets"
    assert len(unused_vgprs) >= len(
        arrays
    ), f"need at least {len(arrays)} more unused VGPRs for lds_addr got: {len(unused_vgprs)} left"

    int_type = ir.IntegerType.get_signless(32, ctx)  # type: ignore
    for i, (array, offset) in enumerate(zip(arrays, offsets)):
        offset_const = arith.constant(int_type, offset)
        lds_addr_op = v_mov_b32_e32(unused_vgprs[i], offset_const)
        array.lds_addr = (
            lds_addr_op.result if hasattr(lds_addr_op, "result") else unused_vgprs[i]
        )

    # Advance unused VGPRs
    return unused_vgprs[len(arrays) :]


def setup_indexing_registers() -> int:
    """Return the number of VGPRs that should be initialized to zero for indexing.

    Returns:
        Number of VGPRs needed for indexing (currently 1 for vaddr_offset)
    """
    return 1


def allocate_indexing_vgprs(
    ctx: Context,
    unused_vgprs: list[Value],
    num_vgprs: int,
) -> tuple[list[Value], list[Value]]:
    """Allocate VGPRs for indexing and initialize them to zero.

    Args:
        ctx: MLIR context
        unused_vgprs: List of unused VGPRs to allocate from
        num_vgprs: Number of VGPRs to allocate

    Returns:
        Tuple of:
        - List of allocated indexing VGPRs
        - Updated list of unused VGPRs (with consumed ones removed)
    """
    assert (
        len(unused_vgprs) >= num_vgprs
    ), f"need at least {num_vgprs} unused VGPRs for indexing"
    int_type = ir.IntegerType.get_signless(32, ctx)  # type: ignore
    zero_const = arith.constant(int_type, 0)
    indexing_vgprs = []
    for i in range(num_vgprs):
        indexing_op = v_mov_b32_e32(unused_vgprs[i], zero_const)
        indexing_vgpr = (
            indexing_op.result if hasattr(indexing_op, "result") else unused_vgprs[i]
        )
        indexing_vgprs.append(indexing_vgpr)
    return indexing_vgprs, unused_vgprs[num_vgprs:]


def allocate_vgprs_for_shapes(
    ctx: Context,
    vgpr_pool: list[Value],
    shapes: list[tuple[int, ...]],
    register_range_sizes: int | list[int],
) -> tuple[list[VRegRangeArray], list[Value]]:
    """Allocate vgprs from contiguous vgpr_pool and create VRegRangeArray objects.

    This is a naive minimal implementation to support allocating vgprs for tensors
    of shapes passed in `shapes`.

    Only allocates VGPRs for shape data. Addresses (lds_addr, global_addr)
    should be provided separately.

    Args:
        ctx: MLIR context
        vgpr_pool: Contiguous register allocation, aligned to register_range_size
        shapes: List of shapes, one per VRegRangeArray
        register_range_sizes: Register range size for each shape element (int or list of ints, one per shape)

    Returns:
        Tuple of:
        - List of VRegRangeArray objects with allocated vgprs for shape data
        - List of unused VGPRs after allocation
    """
    # TODO: maybe a sanity check for unallocated registers when we don't ask for
    # them explicitly?
    # # Verify allocated registers
    # for i, vgpr in enumerate(vgpr_pool):
    #     reg_type = VGPRType(vgpr.type)
    #     if reg_type.is_relocatable():
    #         raise ValueError(f"vgpr at index {i} must be allocated (non-relocatable)")

    # Convert single int to list
    if isinstance(register_range_sizes, int):
        register_range_sizes = [register_range_sizes] * len(shapes)

    if len(register_range_sizes) != len(shapes):
        raise ValueError(
            f"register_range_sizes must have same length as shapes: {len(register_range_sizes)} != {len(shapes)}"
        )

    # Calculate total VGPRs needed for shapes
    total_shape_vgprs = sum(
        prod(shape) * reg_size for shape, reg_size in zip(shapes, register_range_sizes)
    )

    # Verify we have enough VGPRs
    if total_shape_vgprs > len(vgpr_pool):
        raise ValueError(
            f"matmul requires vgpr_pool of length >= {total_shape_vgprs} for shapes, got {len(vgpr_pool)}"
        )

    # Allocate VGPRs for each shape
    arrays = []
    shape_idx = 0
    for i, (shape, register_range_size) in enumerate(zip(shapes, register_range_sizes)):
        num_vgprs_needed = prod(shape) * register_range_size

        # increment shape_idx to be a multiple of register_range_size
        shape_idx = (
            (shape_idx + register_range_size - 1) // register_range_size
        ) * register_range_size
        print(f"{num_vgprs_needed} starting from shape_idx {shape_idx}")

        vgprs_for_shape = vgpr_pool[shape_idx : shape_idx + num_vgprs_needed]
        shape_idx += num_vgprs_needed

        array = VRegRangeArray(ctx, vgprs_for_shape, shape, register_range_size)
        arrays.append(array)

    # Return unused VGPRs
    unused_vgprs = vgpr_pool[total_shape_vgprs:]

    return arrays, unused_vgprs


def allocate_agprs_for_shape(
    ctx: Context,
    agpr_pool: list[Value],
    shape: tuple[int, ...],
    register_range_size: int,
) -> tuple[ARegRangeArray, list[Value]]:
    """Allocate AGPRs from contiguous agpr_pool and create ARegRangeArray object.

    Args:
        ctx: MLIR context
        agpr_pool: Contiguous AGPR allocation, aligned to register_range_size
        shape: Shape for the ARegRangeArray
        register_range_size: Register range size for each shape element

    Returns:
        Tuple of:
        - ARegRangeArray object with allocated agprs for shape data
        - List of unused AGPRs after allocation
    """
    # Calculate total AGPRs needed for shape
    total_shape_agprs = prod(shape) * register_range_size

    # Verify we have enough AGPRs
    if total_shape_agprs > len(agpr_pool):
        raise ValueError(
            f"matmul requires agpr_pool of length >= {total_shape_agprs} for shape, got {len(agpr_pool)}"
        )

    # Allocate AGPRs for shape
    agprs_for_shape = agpr_pool[:total_shape_agprs]
    array = ARegRangeArray(ctx, agprs_for_shape, shape, register_range_size)

    # Return unused AGPRs
    unused_agprs = agpr_pool[total_shape_agprs:]

    return array, unused_agprs


def setup_matmul_arrays(
    ctx: Context,
    sgprs: list[Value],
    vgprs: list[Value],
    operand_register_size: int,
    accum_register_size: int,
    m_regs: int,
    n_regs: int,
    k_regs: int,
    lds_sizes: list[int],
    agprs: Optional[list[Value]] = None,
    reserved_sgprs: int = 0,
    reserved_vgprs: int = 0,
    reserved_agprs: int = 0,
) -> tuple[
    list[Value],
    list[Value],
    list[Value],
    VRegRangeArray,
    VRegRangeArray,
    VRegRangeArray | ARegRangeArray,
]:
    """Setup matmul arrays with kernel arguments, VGPR/AGPR allocation, and LDS addresses.

    Args:
        ctx: MLIR context
        sgprs: SGPRs to allocate from (reserved registers already removed by build_module)
        vgprs: VGPRs to allocate from (reserved registers already removed by build_module)
        operand_register_size: Register range size for A and B operands
        accum_register_size: Register range size for C accumulator
        m_regs: Matrix dimension M registers
        n_regs: Matrix dimension N registers
        k_regs: Matrix dimension K registers
        lds_sizes: LDS sizes in bytes for [A, B, C] arrays
        agprs: Optional AGPRs to allocate from. If provided and non-empty, C will use AGPRs.
            Reserved registers already removed by build_module
        reserved_sgprs: Deprecated - reserved registers are handled in build_module (default: 0)
        reserved_vgprs: Deprecated - reserved registers are handled in build_module (default: 0)
        reserved_agprs: Deprecated - reserved registers are handled in build_module (default: 0)

    Returns:
        Tuple of:
        - Unused SGPRs after setup
        - Unused VGPRs after setup
        - Unused AGPRs after setup
        - A array (VRegRangeArray)
        - B array (VRegRangeArray)
        - C array (VRegRangeArray or ARegRangeArray)
    """
    # TODO: load kernel arguments.
    # Reserved registers are already removed by build_module, so we use all provided registers
    unused_sgprs = sgprs

    if agprs is not None and len(agprs) > 0:
        # Allocate VGPRs for A and B, AGPRs for C
        shapes_ab = [(m_regs, k_regs), (k_regs, n_regs)]
        register_sizes_ab = [operand_register_size, operand_register_size]
        arrays_ab, unused_vgprs = allocate_vgprs_for_shapes(
            ctx, vgprs, shapes_ab, register_sizes_ab
        )
        A, B = arrays_ab

        # Allocate AGPRs for C (reserved registers already removed by build_module)
        C_shape = (m_regs, n_regs)
        C, unused_agprs = allocate_agprs_for_shape(
            ctx, agprs, C_shape, accum_register_size
        )
        C.zero_init = True

        # Compute LDS offsets as cumulative sum of sizes
        lds_offsets = [0]
        for size in lds_sizes[:-1]:
            lds_offsets.append(lds_offsets[-1] + size)

        # Allocate LDS addresses (only for A and B, C uses AGPRs)
        unused_vgprs = allocate_lds_addrs(ctx, [A, B], lds_offsets[:2], unused_vgprs)
    else:
        # Allocate VGPRs for shapes: A and B use operand_register_size, C uses accum_register_size
        shapes = [(m_regs, k_regs), (k_regs, n_regs), (m_regs, n_regs)]
        register_sizes = [
            operand_register_size,
            operand_register_size,
            accum_register_size,
        ]
        arrays, unused_vgprs = allocate_vgprs_for_shapes(
            ctx, vgprs, shapes, register_sizes
        )
        A, B, C = arrays

        # Compute LDS offsets as cumulative sum of sizes
        lds_offsets = [0]
        for size in lds_sizes[:-1]:
            lds_offsets.append(lds_offsets[-1] + size)

        # Allocate LDS addresses
        unused_vgprs = allocate_lds_addrs(ctx, [A, B, C], lds_offsets, unused_vgprs)

        # Initialize C to zero
        C.zero_init = True

        # If agprs were provided but not used, return them as unused
        unused_agprs = agprs if agprs is not None else []

    # TODO: set global addresses from loaded kernel arguments.
    return unused_sgprs, unused_vgprs, unused_agprs, A, B, C
