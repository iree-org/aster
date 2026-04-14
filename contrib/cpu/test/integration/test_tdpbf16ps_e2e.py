import os

import ml_dtypes
import numpy as np

from aster_cpu.execution.helpers import compile_and_run

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MLIR = os.path.join(_THIS_DIR, "tdpbf16ps.mlir")


def _vnni_pack(b: np.ndarray) -> np.ndarray:
    """Pack a row-major (K, N) BF16 tile into (K/2, N, 2) for `tdpbf16ps` B."""
    k, n = b.shape
    return b.reshape(k // 2, 2, n).transpose(0, 2, 1).copy()


def test_tdpbf16ps_e2e(tmp_path, m=16, n=16, k=32):
    """Test AMX tdpbf16ps end-to-end kernel execution from parsed MLIR file."""
    a_data = np.full((m, k), 1.0, dtype=ml_dtypes.bfloat16)
    b_data = np.full((k, n), 2.0, dtype=ml_dtypes.bfloat16)
    b_data_packed = _vnni_pack(b_data)
    c_data = np.zeros((m, n), dtype=np.float32)

    def verify_fn(input_args, output_args):
        c = np.array(output_args[0])
        ref = np.matmul(a_data, b_data).astype(np.float32)
        assert np.array_equal(c, ref), f"AMX kernel failed! c: {c}, ref: {ref}"

    compile_and_run(
        _MLIR,
        "test_tdpbf16ps",
        input_data=[a_data, b_data_packed],
        output_data=[c_data],
        verify_fn=verify_fn,
        tmp_path=tmp_path,
        print_asm=True,
    )
