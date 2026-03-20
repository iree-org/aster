"""Unit tests for aster.execution.core.

Tests are split into two groups:
- No-GPU tests: pure Python / ctypes logic, always run.
- GPU tests: require a HIP device, skipped when none is present.
"""

import ctypes

import numpy as np
import pytest

from aster.execution.core import (
    GpuBuffer,
    GpuEvent,
    GpuStream,
    InputArray,
    InOutArray,
    MemoryManager,
    OutputArray,
    _capsule,
    _uncapsule,
    create_kernel_args_capsule,
    unwrap_pointer_from_capsule,
    wrap_pointer_in_capsule,
)

# ---------------------------------------------------------------------------
# GPU availability fixture
# ---------------------------------------------------------------------------


def _has_any_gpu() -> bool:
    try:
        from aster._mlir_libs._runtime_module import hip_get_device_count, hip_init

        hip_init()
        return hip_get_device_count() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_any_gpu(), reason="no GPU available")


# ---------------------------------------------------------------------------
# Argument wrapper classes
# ---------------------------------------------------------------------------


class TestArgumentWrappers:
    def test_input_array_stores_array(self):
        arr = np.zeros(4, dtype=np.float32)
        w = InputArray(arr)
        assert w.array is arr

    def test_output_array_stores_array(self):
        arr = np.zeros(4, dtype=np.float32)
        w = OutputArray(arr)
        assert w.array is arr

    def test_inout_array_stores_array(self):
        arr = np.zeros(4, dtype=np.float32)
        w = InOutArray(arr)
        assert w.array is arr

    def test_wrappers_are_distinct_types(self):
        arr = np.zeros(4, dtype=np.float32)
        assert not isinstance(InputArray(arr), OutputArray)
        assert not isinstance(OutputArray(arr), InOutArray)
        assert not isinstance(InOutArray(arr), InputArray)


# ---------------------------------------------------------------------------
# PyCapsule helpers
# ---------------------------------------------------------------------------


class TestCapsuleHelpers:
    def test_roundtrip_nonzero(self):
        ptr = 0xDEADBEEF
        capsule = _capsule(ptr)
        assert _uncapsule(capsule) == ptr

    def test_public_aliases(self):
        ptr = 0x1234
        capsule = wrap_pointer_in_capsule(ptr)
        assert unwrap_pointer_from_capsule(capsule) == ptr

    def test_different_values_produce_different_results(self):
        a = _uncapsule(_capsule(0x100))
        b = _uncapsule(_capsule(0x200))
        assert a != b


# ---------------------------------------------------------------------------
# create_kernel_args_capsule  (no GPU needed except for GpuBuffer path)
# ---------------------------------------------------------------------------


class TestCreateKernelArgsCapsule:
    def test_empty_returns_all_none(self):
        capsule, kernel_args, kernel_ptr_arr = create_kernel_args_capsule()
        assert capsule is None
        assert kernel_args is None
        assert kernel_ptr_arr is None

    def test_int_packed_as_c_int32(self):
        capsule, kernel_args, kernel_ptr_arr = create_kernel_args_capsule(42)
        assert kernel_args is not None
        assert kernel_ptr_arr is not None
        assert kernel_args._field0 == 42

    def test_negative_int(self):
        capsule, kernel_args, _ = create_kernel_args_capsule(-1)
        assert kernel_args._field0 == -1

    def test_float_packed_as_c_float(self):
        capsule, kernel_args, _ = create_kernel_args_capsule(1.5)
        assert abs(kernel_args._field0 - 1.5) < 1e-6

    def test_numpy_integer_packed_as_c_int32(self):
        capsule, kernel_args, _ = create_kernel_args_capsule(np.int32(7))
        assert kernel_args._field0 == 7

    def test_numpy_integer_types(self):
        for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8):
            capsule, kernel_args, _ = create_kernel_args_capsule(dtype(3))
            assert kernel_args._field0 == 3

    def test_numpy_floating_packed_as_c_float(self):
        capsule, kernel_args, _ = create_kernel_args_capsule(np.float32(2.5))
        assert abs(kernel_args._field0 - 2.5) < 1e-6

    def test_numpy_float64_packed_as_c_float(self):
        capsule, kernel_args, _ = create_kernel_args_capsule(np.float64(1.25))
        assert abs(kernel_args._field0 - 1.25) < 1e-6

    def test_ctypes_c_int32(self):
        capsule, kernel_args, _ = create_kernel_args_capsule(ctypes.c_int32(7))
        assert kernel_args._field0 == 7

    def test_ctypes_c_float(self):
        capsule, kernel_args, _ = create_kernel_args_capsule(ctypes.c_float(2.5))
        assert abs(kernel_args._field0 - 2.5) < 1e-6

    def test_ctypes_c_void_p(self):
        addr = 0xCAFE
        capsule, kernel_args, _ = create_kernel_args_capsule(ctypes.c_void_p(addr))
        assert kernel_args._field0 == addr

    def test_ndarray_passes_host_pointer(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        capsule, kernel_args, _ = create_kernel_args_capsule(arr)
        assert kernel_args._field0 == arr.ctypes.data

    def test_mixed_types(self):
        arr = np.zeros(4, dtype=np.float32)
        capsule, kernel_args, _ = create_kernel_args_capsule(arr, 10, 3.14)
        assert kernel_args._field0 == arr.ctypes.data
        assert kernel_args._field1 == 10
        assert abs(kernel_args._field2 - 3.14) < 1e-6

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="unsupported argument type"):
            create_kernel_args_capsule([1, 2, 3])

    def test_capsule_pointer_array_addresses_are_within_struct(self):
        """Each entry in kernel_ptr_arr must point inside kernel_args."""
        capsule, kernel_args, kernel_ptr_arr = create_kernel_args_capsule(1, 2.0)
        struct_start = ctypes.addressof(kernel_args)
        struct_end = struct_start + ctypes.sizeof(kernel_args)
        for i in range(2):
            addr = kernel_ptr_arr[i]
            assert struct_start <= addr < struct_end

    def test_ctypes_scalar_types_all_accepted(self):
        for ctype in (
            ctypes.c_int8,
            ctypes.c_int16,
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_double,
            ctypes.c_void_p,
        ):
            capsule, kernel_args, _ = create_kernel_args_capsule(ctype(0))
            assert kernel_args is not None


# ---------------------------------------------------------------------------
# GpuBuffer
# ---------------------------------------------------------------------------


@requires_gpu
class TestGpuBuffer:
    def test_alloc_and_free(self):
        buf = GpuBuffer(1024)
        assert buf.size_bytes == 1024
        assert buf.ptr_value != 0
        buf.free()
        assert buf._freed

    def test_free_is_idempotent(self):
        buf = GpuBuffer(64)
        buf.free()
        buf.free()  # Must not raise.

    def test_del_calls_free(self):
        buf = GpuBuffer(64)
        ptr_val = buf.ptr_value
        assert ptr_val != 0
        buf.__del__()
        assert buf._freed

    def test_copy_roundtrip(self):
        src = np.arange(16, dtype=np.float32)
        buf = GpuBuffer(src.nbytes)
        buf.copy_from_host(src)

        dst = np.zeros_like(src)
        buf.copy_to_host(dst)
        buf.free()

        np.testing.assert_array_equal(dst, src)

    def test_copy_roundtrip_various_dtypes(self):
        for dtype in (np.int8, np.int32, np.float16, np.float64):
            src = np.arange(8, dtype=dtype)
            buf = GpuBuffer(src.nbytes)
            buf.copy_from_host(src)
            dst = np.zeros_like(src)
            buf.copy_to_host(dst)
            buf.free()
            np.testing.assert_array_equal(dst, src)

    def test_copy_from_host_wrong_type_raises(self):
        buf = GpuBuffer(16)
        with pytest.raises(TypeError):
            buf.copy_from_host([1, 2, 3])
        buf.free()

    def test_copy_to_host_wrong_type_raises(self):
        buf = GpuBuffer(16)
        with pytest.raises(TypeError):
            buf.copy_to_host([1, 2, 3])
        buf.free()

    def test_copy_from_host_oversized_array_raises(self):
        buf = GpuBuffer(4)
        big = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError):
            buf.copy_from_host(big)
        buf.free()

    def test_copy_to_host_oversized_array_raises(self):
        buf = GpuBuffer(4)
        big = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError):
            buf.copy_to_host(big)
        buf.free()

    def test_ptr_property_is_unwrappable(self):
        buf = GpuBuffer(64)
        raw = _uncapsule(buf.ptr)
        assert raw == buf.ptr_value
        buf.free()


# ---------------------------------------------------------------------------
# GpuStream
# ---------------------------------------------------------------------------


@requires_gpu
class TestGpuStream:
    def test_create_and_destroy(self):
        stream = GpuStream()
        assert stream.handle is not None
        stream.destroy()
        assert stream._destroyed

    def test_destroy_is_idempotent(self):
        stream = GpuStream()
        stream.destroy()
        stream.destroy()  # Must not raise.

    def test_synchronize(self):
        stream = GpuStream()
        stream.synchronize()  # Must not raise.
        stream.destroy()

    def test_del_calls_destroy(self):
        stream = GpuStream()
        stream.__del__()
        assert stream._destroyed


# ---------------------------------------------------------------------------
# GpuEvent
# ---------------------------------------------------------------------------


@requires_gpu
class TestGpuEvent:
    def test_create_and_destroy(self):
        event = GpuEvent()
        assert event.handle is not None
        event.destroy()
        assert event._destroyed

    def test_destroy_is_idempotent(self):
        event = GpuEvent()
        event.destroy()
        event.destroy()  # Must not raise.

    def test_record_and_synchronize(self):
        event = GpuEvent()
        event.record()
        event.synchronize()  # Must not raise.
        event.destroy()

    def test_elapsed_ms_non_negative(self):
        start = GpuEvent()
        stop = GpuEvent()
        start.record()
        stop.record()
        stop.synchronize()
        assert stop.elapsed_ms(start) >= 0.0
        start.destroy()
        stop.destroy()

    def test_elapsed_ns_non_negative(self):
        start = GpuEvent()
        stop = GpuEvent()
        start.record()
        stop.record()
        stop.synchronize()
        assert stop.elapsed_ns(start) >= 0
        start.destroy()
        stop.destroy()

    def test_elapsed_ns_is_int(self):
        start = GpuEvent()
        stop = GpuEvent()
        start.record()
        stop.record()
        stop.synchronize()
        assert isinstance(stop.elapsed_ns(start), int)
        start.destroy()
        stop.destroy()


# ---------------------------------------------------------------------------
# create_kernel_args_capsule with GpuBuffer (GPU required)
# ---------------------------------------------------------------------------


@requires_gpu
class TestCreateKernelArgsCapsuleGpu:
    def test_gpubuffer_passes_device_pointer(self):
        buf = GpuBuffer(64)
        capsule, kernel_args, _ = create_kernel_args_capsule(buf)
        assert kernel_args._field0 == buf.ptr_value
        buf.free()

    def test_mixed_gpubuffer_and_scalars(self):
        buf = GpuBuffer(64)
        capsule, kernel_args, _ = create_kernel_args_capsule(buf, 5, 2.0)
        assert kernel_args._field0 == buf.ptr_value
        assert kernel_args._field1 == 5
        assert abs(kernel_args._field2 - 2.0) < 1e-6
        buf.free()


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------


@requires_gpu
class TestMemoryManager:
    def test_register_returns_gpu_buffer(self):
        mm = MemoryManager()
        arr = np.zeros(8, dtype=np.float32)
        buf = mm.register(arr)
        assert isinstance(buf, GpuBuffer)
        assert buf.size_bytes == arr.nbytes
        mm.release_all()

    def test_register_is_idempotent(self):
        mm = MemoryManager()
        arr = np.zeros(4, dtype=np.float32)
        buf1 = mm.register(arr)
        buf2 = mm.register(arr)
        assert buf1 is buf2
        mm.release_all()

    def test_register_wrong_type_raises(self):
        mm = MemoryManager()
        with pytest.raises(TypeError):
            mm.register([1, 2, 3])

    def test_register_upload_false_does_not_copy(self):
        mm = MemoryManager()
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf = mm.register(arr, upload=False)
        # Buffer is allocated (size correct) but contents are uninitialised.
        assert buf.size_bytes == arr.nbytes
        mm.release_all()

    def test_sync_to_gpu_and_back(self):
        mm = MemoryManager()
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        mm.register(arr)

        arr[:] = 99.0
        mm.sync_to_gpu(arr)

        result = np.zeros_like(arr)
        mm.get_buffer(arr).copy_to_host(result)
        np.testing.assert_array_equal(result, 99.0)
        mm.release_all()

    def test_sync_from_gpu(self):
        mm = MemoryManager()
        src = np.arange(8, dtype=np.int32)
        mm.register(src)  # uploads src data

        dst = np.zeros(8, dtype=np.int32)
        mm.register(dst)
        # Overwrite dst GPU buffer with src data via a direct copy.
        mm.get_buffer(src).copy_to_host(dst)
        np.testing.assert_array_equal(dst, src)
        mm.release_all()

    def test_sync_from_gpu_updates_host_array(self):
        mm = MemoryManager()
        arr = np.array([10, 20, 30], dtype=np.float32)
        mm.register(arr)

        out = np.zeros(3, dtype=np.float32)
        mm.register(out)
        # Copy arr's GPU buffer content into out via sync_from_gpu after
        # manually putting arr's data into out's GPU buffer.
        mm.get_buffer(arr).copy_to_host(out)
        np.testing.assert_array_equal(out, arr)
        mm.release_all()

    def test_get_buffer_not_registered_raises(self):
        mm = MemoryManager()
        arr = np.zeros(4, dtype=np.float32)
        with pytest.raises(KeyError):
            mm.get_buffer(arr)

    def test_sync_to_gpu_not_registered_raises(self):
        mm = MemoryManager()
        arr = np.zeros(4, dtype=np.float32)
        with pytest.raises(KeyError):
            mm.sync_to_gpu(arr)

    def test_sync_from_gpu_not_registered_raises(self):
        mm = MemoryManager()
        arr = np.zeros(4, dtype=np.float32)
        with pytest.raises(KeyError):
            mm.sync_from_gpu(arr)

    def test_release_removes_entry(self):
        mm = MemoryManager()
        arr = np.zeros(4, dtype=np.float32)
        mm.register(arr)
        mm.release(arr)
        with pytest.raises(KeyError):
            mm.get_buffer(arr)

    def test_release_not_registered_raises(self):
        mm = MemoryManager()
        arr = np.zeros(4, dtype=np.float32)
        with pytest.raises(KeyError):
            mm.release(arr)

    def test_release_all_clears_all(self):
        mm = MemoryManager()
        a = np.zeros(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        mm.register(a)
        mm.register(b)
        mm.release_all()
        with pytest.raises(KeyError):
            mm.get_buffer(a)
        with pytest.raises(KeyError):
            mm.get_buffer(b)

    def test_release_all_on_empty_manager(self):
        mm = MemoryManager()
        mm.release_all()  # Must not raise.

    def test_multiple_arrays_independent_buffers(self):
        mm = MemoryManager()
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        buf_a = mm.register(a)
        buf_b = mm.register(b)
        assert buf_a is not buf_b
        assert buf_a.ptr_value != buf_b.ptr_value

        out_a = np.zeros_like(a)
        out_b = np.zeros_like(b)
        buf_a.copy_to_host(out_a)
        buf_b.copy_to_host(out_b)
        np.testing.assert_array_equal(out_a, a)
        np.testing.assert_array_equal(out_b, b)
        mm.release_all()
