"""MLIR/LLVM-free HIP execution utilities.

This module merges capsule wrapping, GPU memory management, and kernel
execution. It does NOT import any MLIR or LLVM libraries, making it safe
to use under rocprofv3 (which crashes when LLVM.so is loaded alongside its
own LLVM).

Usage:
    from aster.execution.core import execute_hsaco, InputArray, OutputArray

    if not system_has_gpu("gfx942"):
        pytest.skip("no GPU")

    times_ns = execute_hsaco(
        hsaco_path="kernel.hsaco",
        kernel_name="my_kernel",
        arguments=[InputArray(A), InputArray(B), OutputArray(C)],
        grid_dim=(304, 1, 1),
        block_dim=(256, 1, 1),
        num_iterations=5,
    )
"""

import ctypes
import dataclasses
from typing import Any, Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# PyCapsule helpers (private + public aliases)
# ---------------------------------------------------------------------------


def _capsule(ptr):
    """Wrap a raw pointer in a PyCapsule (nb_handle)."""
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return PyCapsule_New(ptr, b"nb_handle", None)


def _uncapsule(capsule):
    """Extract a raw pointer from a PyCapsule (nb_handle)."""
    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.restype = ctypes.c_void_p
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return PyCapsule_GetPointer(ctypes.py_object(capsule), b"nb_handle")


# Public aliases used by aster.utils and external callers.
wrap_pointer_in_capsule = _capsule
unwrap_pointer_from_capsule = _uncapsule


# ---------------------------------------------------------------------------
# RAII GPU resource wrappers
# ---------------------------------------------------------------------------


class GpuBuffer:
    """RAII wrapper for a GPU memory buffer.

    Allocates device memory on construction and frees it on destruction.
    """

    def __init__(self, size_bytes: int) -> None:
        from aster._mlir_libs._runtime_module import hip_malloc

        self._freed = True  # Guard against partial construction.
        self._size_bytes = size_bytes
        self._handle = hip_malloc(size_bytes)
        if self._handle is None:
            raise RuntimeError(f"Failed to allocate GPU memory of size {size_bytes}")
        self._ptr_value = _uncapsule(self._handle)
        self._freed = False

    def __del__(self) -> None:
        self.free()

    def free(self) -> None:
        """Release the GPU buffer.

        Safe to call multiple times.
        """
        if not getattr(self, "_freed", True):
            from aster._mlir_libs._runtime_module import hip_free

            hip_free(self._handle)
            self._freed = True

    @property
    def ptr(self):
        """PyCapsule wrapping the raw device pointer (for hip_memcpy_* calls)."""
        return self._handle

    @property
    def ptr_value(self) -> int:
        """Raw integer value of the device pointer (for kernel args)."""
        return self._ptr_value

    @property
    def size_bytes(self) -> int:
        """Allocated size in bytes."""
        return self._size_bytes

    def copy_from_host(self, array) -> None:
        """Copy data from a numpy array on the host into this GPU buffer."""
        import numpy as np
        from aster._mlir_libs._runtime_module import hip_memcpy_host_to_device

        if not isinstance(array, np.ndarray):
            raise TypeError(f"expected numpy array, got {type(array)}")
        if array.nbytes > self._size_bytes:
            raise ValueError(
                f"array ({array.nbytes} bytes) exceeds buffer ({self._size_bytes} bytes)"
            )
        hip_memcpy_host_to_device(
            self._handle, _capsule(array.ctypes.data), array.nbytes
        )

    def copy_to_host(self, array) -> None:
        """Copy data from this GPU buffer into a numpy array on the host."""
        import numpy as np
        from aster._mlir_libs._runtime_module import hip_memcpy_device_to_host

        if not isinstance(array, np.ndarray):
            raise TypeError(f"expected numpy array, got {type(array)}")
        if array.nbytes > self._size_bytes:
            raise ValueError(
                f"array ({array.nbytes} bytes) exceeds buffer ({self._size_bytes} bytes)"
            )
        hip_memcpy_device_to_host(
            _capsule(array.ctypes.data), self._handle, array.nbytes
        )


class GpuStream:
    """RAII wrapper for a HIP stream.

    Creates a new stream on construction and destroys it on destruction.
    """

    def __init__(self) -> None:
        from aster._mlir_libs._runtime_module import hip_stream_create

        self._destroyed = True  # Guard against partial construction.
        self._handle = hip_stream_create()
        self._destroyed = False

    def __del__(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        """Destroy the stream.

        Safe to call multiple times.
        """
        if not getattr(self, "_destroyed", True):
            from aster._mlir_libs._runtime_module import hip_stream_destroy

            hip_stream_destroy(self._handle)
            self._destroyed = True

    def synchronize(self) -> None:
        """Block until all work in this stream is complete."""
        from aster._mlir_libs._runtime_module import hip_stream_synchronize

        hip_stream_synchronize(self._handle)

    @property
    def handle(self):
        """Raw handle for use in lower-level HIP calls."""
        return self._handle


class GpuEvent:
    """RAII wrapper for a HIP event.

    Creates a new event on construction and destroys it on destruction.
    """

    def __init__(self) -> None:
        from aster._mlir_libs._runtime_module import hip_event_create

        self._destroyed = True  # Guard against partial construction.
        self._handle = hip_event_create()
        self._destroyed = False

    def __del__(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        """Destroy the event.

        Safe to call multiple times.
        """
        if not getattr(self, "_destroyed", True):
            from aster._mlir_libs._runtime_module import hip_event_destroy

            hip_event_destroy(self._handle)
            self._destroyed = True

    def record(self) -> None:
        """Record this event in the default stream."""
        from aster._mlir_libs._runtime_module import hip_event_record

        hip_event_record(self._handle)

    def synchronize(self) -> None:
        """Block until this event has been recorded."""
        from aster._mlir_libs._runtime_module import hip_event_synchronize

        hip_event_synchronize(self._handle)

    def elapsed_ms(self, start: "GpuEvent") -> float:
        """Return elapsed time in milliseconds between *start* and this event."""
        from aster._mlir_libs._runtime_module import hip_event_elapsed_time

        return hip_event_elapsed_time(start._handle, self._handle)

    def elapsed_ns(self, start: "GpuEvent") -> int:
        """Return elapsed time in nanoseconds between *start* and this event."""
        return int(self.elapsed_ms(start) * 1_000_000)

    @property
    def handle(self):
        """Raw handle for use in lower-level HIP calls."""
        return self._handle


class GpuFunction:
    """Thin wrapper around a HIP kernel function handle.

    The lifetime is tied to the owning :class:`GpuModule`; do not call
    :meth:`GpuModule.unload` while a ``GpuFunction`` from that module is in
    use.
    """

    def __init__(self, handle, module: "GpuModule") -> None:
        self._handle = handle
        # Keep a reference so the module is not unloaded while we exist.
        self._module = module

    def launch(
        self,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        args: list,
    ) -> None:
        """Launch the kernel with the given grid/block dimensions.

        Args are converted to a capsule internally. HIP copies the
        argument values into its own buffer during the launch call, so
        the local ctypes structures can be freed as soon as this method
        returns.
        """
        from aster._mlir_libs._runtime_module import hip_module_launch_kernel

        capsule, kernel_args, kernel_ptr_arr = create_kernel_args_capsule(*args)
        hip_module_launch_kernel(
            self._handle,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            capsule,
        )
        # kernel_args and kernel_ptr_arr kept alive until here.
        del kernel_args, kernel_ptr_arr

    @property
    def handle(self):
        """Raw handle for use in lower-level HIP calls."""
        return self._handle


class GpuModule:
    """RAII wrapper for a loaded HIP module.

    Loads an HSACO binary on construction and unloads it on destruction.
    """

    def __init__(self, hsaco_path: str) -> None:
        from aster._mlir_libs._runtime_module import (
            hip_module_load_data,
            hip_module_get_function,
        )

        self._unloaded = True  # Guard against partial construction.
        if hsaco_path is None:
            raise RuntimeError("hsaco_path is None — assembly failed")
        with open(hsaco_path, "rb") as f:
            hsaco_binary = f.read()
        self._module = hip_module_load_data(hsaco_binary)
        self._unloaded = False
        # Cache hip_module_get_function to avoid re-importing in get_function.
        self._get_function_fn = hip_module_get_function

    def __del__(self) -> None:
        self.unload()

    def unload(self) -> None:
        """Unload the module.

        Safe to call multiple times.
        """
        if not getattr(self, "_unloaded", True):
            from aster._mlir_libs._runtime_module import hip_module_unload

            hip_module_unload(self._module)
            self._unloaded = True

    def get_function(self, name: str) -> GpuFunction:
        """Return a :class:`GpuFunction` for the named kernel entry point."""
        handle = self._get_function_fn(self._module, name.encode())
        return GpuFunction(handle, self)


# ---------------------------------------------------------------------------
# Kernel argument array wrappers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class InputArray:
    """Numpy array passed as a read-only kernel input.

    Data is copied from host to GPU before the first launch. No copy-
    back is performed after the kernel runs.
    """

    array: Any


@dataclasses.dataclass
class OutputArray:
    """Numpy array used as a kernel output.

    No host-to-device copy is performed before the launch. After the
    first iteration the GPU buffer is copied back to the host array.
    """

    array: Any


@dataclasses.dataclass
class InOutArray:
    """Numpy array that is both read and written by the kernel.

    Data is copied host-to-device before the first launch and device-to-
    host after the first iteration.
    """

    array: Any


# ---------------------------------------------------------------------------
# Memory manager
# ---------------------------------------------------------------------------


class MemoryManager:
    """Pairs numpy arrays with :class:`GpuBuffer` objects for easy H<->D sync.

    Use :meth:`register` to create the GPU-side buffer for an array. The
    manager tracks the pair by the array's identity (``id``), so the same
    numpy object must be passed to subsequent calls.

    Example::

        mm = MemoryManager()
        buf_a = mm.register(a)
        buf_b = mm.register(b, upload=False)
        # ... launch kernel using buf_a.ptr_value, buf_b.ptr_value ...
        mm.sync_from_gpu(b)   # copy result back to host
        mm.release_all()
    """

    def __init__(self) -> None:
        # Maps id(array) -> (GpuBuffer, array) pairs. The array reference
        # prevents the object from being GC'd and its id reused.
        self._buffers: dict = {}

    def register(self, array, upload: bool = True) -> GpuBuffer:
        """Allocate a GPU buffer and optionally upload *array* data.

        If the array is already registered, the existing buffer is returned
        without re-uploading.

        Args:
            array: Numpy array to pair with the new GPU buffer.
            upload: If ``True`` (default), copy the host data to GPU immediately.
                    If ``False``, only allocate the buffer without copying.
        """
        import numpy as np

        if not isinstance(array, np.ndarray):
            raise TypeError(f"expected numpy array, got {type(array)}")
        key = id(array)
        if key in self._buffers:
            buf, _ = self._buffers[key]
            return buf
        buf = GpuBuffer(array.nbytes)
        if upload:
            buf.copy_from_host(array)
        self._buffers[key] = (buf, array)
        return buf

    def sync_to_gpu(self, array) -> None:
        """Copy current host data to the paired GPU buffer."""
        buf, _ = self._get_entry(array)
        buf.copy_from_host(array)

    def sync_from_gpu(self, array) -> None:
        """Copy GPU buffer data back to the paired host array."""
        buf, _ = self._get_entry(array)
        buf.copy_to_host(array)

    def get_buffer(self, array) -> GpuBuffer:
        """Return the :class:`GpuBuffer` paired with *array*."""
        buf, _ = self._get_entry(array)
        return buf

    def release(self, array) -> None:
        """Free the GPU buffer for *array* and remove the tracking entry."""
        key = id(array)
        if key not in self._buffers:
            raise KeyError("array is not registered with this MemoryManager")
        buf, _ = self._buffers.pop(key)
        buf.free()

    def release_all(self) -> None:
        """Free all tracked GPU buffers."""
        for buf, _ in self._buffers.values():
            buf.free()
        self._buffers.clear()

    def _get_entry(self, array):
        key = id(array)
        if key not in self._buffers:
            raise KeyError("array is not registered with this MemoryManager")
        return self._buffers[key]


# ---------------------------------------------------------------------------
# Kernel argument capsule construction
# ---------------------------------------------------------------------------


def create_kernel_args_capsule(*args: Any):
    """Build a kernel arguments capsule from a mix of types.

    Supported argument types:

    * :class:`GpuBuffer` — passes the device pointer as ``c_void_p``.
    * ``numpy.ndarray`` — passes the host data pointer as ``c_void_p``.
    * ``int`` — packed as ``c_int32``.
    * ``float`` — packed as ``c_float``.
    * ``numpy.integer`` — packed as ``c_int32``.
    * ``numpy.floating`` — packed as ``c_float``.
    * ``ctypes`` scalar instances (``c_int8``, ``c_int16``, ``c_int32``,
      ``c_int64``, ``c_float``, ``c_double``, ``c_void_p``) — passed as-is.

    Returns:
        Tuple of ``(args_capsule, kernel_args, kernel_ptr_arr)``:

        * ``args_capsule``: PyCapsule to pass to ``hip_module_launch_kernel``.
        * ``kernel_args``: ctypes structure (must remain alive until after
          the kernel launch).
        * ``kernel_ptr_arr``: Array of pointers into the structure fields
          (must remain alive until after the kernel launch).
    """
    import numpy as np

    _ctypes_scalar_types = (
        ctypes.c_int8,
        ctypes.c_int16,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_double,
        ctypes.c_void_p,
    )

    c_args = []
    c_struct_fields = []
    for i, arg in enumerate(args):
        if isinstance(arg, GpuBuffer):
            c_args.append(ctypes.c_void_p(arg.ptr_value))
            c_struct_fields.append((f"_field{i}", ctypes.c_void_p))
        elif isinstance(arg, np.ndarray):
            c_args.append(ctypes.c_void_p(arg.ctypes.data))
            c_struct_fields.append((f"_field{i}", ctypes.c_void_p))
        elif isinstance(arg, np.integer):
            c_args.append(ctypes.c_int32(int(arg)))
            c_struct_fields.append((f"_field{i}", ctypes.c_int32))
        elif isinstance(arg, np.floating):
            c_args.append(ctypes.c_float(float(arg)))
            c_struct_fields.append((f"_field{i}", ctypes.c_float))
        elif isinstance(arg, int):
            c_args.append(ctypes.c_int32(arg))
            c_struct_fields.append((f"_field{i}", ctypes.c_int32))
        elif isinstance(arg, float):
            c_args.append(ctypes.c_float(arg))
            c_struct_fields.append((f"_field{i}", ctypes.c_float))
        elif isinstance(arg, _ctypes_scalar_types):
            c_args.append(arg)
            c_struct_fields.append((f"_field{i}", type(arg)))
        else:
            raise TypeError(f"unsupported argument type: {type(arg)}")

    if not c_args:
        return None, None, None

    class _Args(ctypes.Structure):
        _fields_ = c_struct_fields

    kernel_args = _Args()
    for i, arg in enumerate(c_args):
        setattr(kernel_args, f"_field{i}", arg)

    ptr_arr_t = ctypes.c_void_p * len(c_args)
    kernel_args_addr = ctypes.addressof(kernel_args)
    kernel_ptr_arr = ptr_arr_t(
        *[
            kernel_args_addr + getattr(_Args, f"_field{i}").offset
            for i in range(len(c_args))
        ]
    )

    args_capsule = _capsule(ctypes.addressof(kernel_ptr_arr))
    return args_capsule, kernel_args, kernel_ptr_arr


# ---------------------------------------------------------------------------
# Internal kernel launch helpers
# ---------------------------------------------------------------------------


class _TimedLaunch:
    """Context manager that brackets a GPU kernel launch with HIP event timing.

    Records start/stop events around the ``with`` body, synchronizes on exit,
    and exposes the wall time as ``elapsed_ns``.

    Args:
        start_event: :class:`GpuEvent` used as the start marker.
        stop_event: :class:`GpuEvent` used as the stop marker.
        flush_llc: Optional LLC-flush object. ``flush_llc()`` is called before
            the start event is recorded if provided.

    Example::

        with _TimedLaunch(start_event, stop_event, flush_llc) as t:
            function.launch(grid, block, args)
        times_ns.append(t.elapsed_ns)
    """

    elapsed_ns: int

    def __init__(self, start_event: GpuEvent, stop_event: GpuEvent, flush_llc=None):
        self._start = start_event
        self._stop = stop_event
        self._flush_llc = flush_llc

    def __enter__(self):
        if self._flush_llc is not None:
            self._flush_llc.flush_llc()
        self._start.record()
        return self

    def __exit__(self, *_):
        self._stop.record()
        self._stop.synchronize()
        self.elapsed_ns = self._stop.elapsed_ns(self._start)


# ---------------------------------------------------------------------------
# Main execution entry point
# ---------------------------------------------------------------------------


def execute_hsaco(
    hsaco_path: str,
    kernel_name: str,
    arguments: list,
    grid_dim: Tuple[int, int, int] = (1, 1, 1),
    block_dim: Tuple[int, int, int] = (64, 1, 1),
    num_iterations: int = 1,
    device_id: Optional[int] = None,
    flush_llc: Optional[Any] = None,
    verify_fn: Optional[Callable] = None,
    memory_manager: Optional[MemoryManager] = None,
) -> List[int]:
    """Execute a pre-compiled HSACO kernel on GPU.

    Args:
        hsaco_path: Path to the .hsaco file.
        kernel_name: Name of the kernel entry point.
        arguments: List of kernel arguments. Each item may be:
            * :class:`InputArray` — numpy array copied H→D before launch only.
            * :class:`OutputArray` — numpy array allocated on GPU; copied D→H
              after the first iteration.
            * :class:`InOutArray` — numpy array copied H→D before launch and
              D→H after the first iteration.
            * A raw ``numpy.ndarray`` — treated as :class:`InOutArray`.
            * A scalar (``int``, ``float``, numpy scalar, or ctypes scalar).
        grid_dim: (gridX, gridY, gridZ).
        block_dim: (blockX, blockY, blockZ).
        num_iterations: Number of kernel launches (for timing).
        device_id: GPU device to use. None keeps the current device.
        flush_llc: Optional object with initialize()/flush_llc()/cleanup()
            methods called around each iteration for LLC flushing.
        verify_fn: Optional callable invoked as ``verify_fn(arguments)`` after
            the first iteration to validate results. Receives the normalised
            arguments list (raw ndarrays already wrapped).
        memory_manager: Optional :class:`MemoryManager` to use for GPU buffer
            allocation and transfers. If ``None``, a temporary one is created
            and released on return.

    Returns:
        List of execution times in nanoseconds, one per iteration.
    """
    import numpy as np
    from aster._mlir_libs._runtime_module import hip_init, hip_set_device

    hip_init()
    if device_id is not None:
        hip_set_device(device_id)

    # Clear any sticky HIP error from a previous failed call in this process.
    try:
        from aster._mlir_libs._runtime_module import hip_clear_last_error

        hip_clear_last_error()
    except ImportError:
        pass

    # Normalise raw ndarray → InOutArray.
    arguments = [InOutArray(a) if isinstance(a, np.ndarray) else a for a in arguments]

    owns_mm = memory_manager is None
    mm = MemoryManager() if owns_mm else memory_manager

    # Register arrays with the memory manager.
    for arg in arguments:
        if isinstance(arg, (InputArray, InOutArray)):
            mm.register(arg.array, upload=True)
        elif isinstance(arg, OutputArray):
            mm.register(arg.array, upload=False)

    # Build the flat argument list passed to GpuFunction.launch.
    launch_args = [
        (
            mm.get_buffer(arg.array)
            if isinstance(arg, (InputArray, OutputArray, InOutArray))
            else arg
        )
        for arg in arguments
    ]

    module = GpuModule(hsaco_path)
    function = module.get_function(kernel_name)

    start_event = GpuEvent()
    stop_event = GpuEvent()
    times_ns = []

    if flush_llc is not None:
        flush_llc.initialize()

    try:
        for it in range(num_iterations):
            with _TimedLaunch(start_event, stop_event, flush_llc) as t:
                function.launch(grid_dim, block_dim, launch_args)
            times_ns.append(t.elapsed_ns)

            if it == 0:
                for arg in arguments:
                    if isinstance(arg, (OutputArray, InOutArray)):
                        mm.sync_from_gpu(arg.array)
                if verify_fn is not None:
                    verify_fn(arguments)
    finally:
        if flush_llc is not None:
            flush_llc.cleanup()
        if owns_mm:
            mm.release_all()

    return times_ns
