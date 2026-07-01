"""3D block / thread id probe for gfx1250+ with a PLAIN (non-cluster) launch.

Test the block_id and thread_id are properly mapped on GFX12.5+ in the absence
of cluster dimensions.

Launch: grid=(2,3,4), block=(3,5,7).

Each thread writes 8 4B values:

    out[row] = [flat_block_id, block_id x/y/z, flat_thread_id, thread_id x/y/z]

where row = flat_block_id * threads_per_wg + flat_thread_id.
"""

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_GFX1250_CLUSTER_ASM_PASS_PIPELINE

GRID_DIM = (2, 3, 4)  # grid extent per axis (no clustering)
BLOCK_DIM = (3, 5, 7)  # prime thread counts per axis

N_WG = GRID_DIM[0] * GRID_DIM[1] * GRID_DIM[2]
THREADS_PER_WG = BLOCK_DIM[0] * BLOCK_DIM[1] * BLOCK_DIM[2]
N_ROWS = N_WG * THREADS_PER_WG

COL_NAMES = (
    "flat_block_id",
    "block_id_x",
    "block_id_y",
    "block_id_z",
    "flat_thread_id",
    "thread_id_x",
    "thread_id_y",
    "thread_id_z",
)
COLS = len(COL_NAMES)

SENTINEL = -7


def _flat_block(bx: int, by: int, bz: int) -> int:
    return bx + GRID_DIM[0] * (by + GRID_DIM[1] * bz)


def _flat_thread(tx: int, ty: int, tz: int) -> int:
    return tx + BLOCK_DIM[0] * (ty + BLOCK_DIM[1] * tz)


def _expected_row(bx: int, by: int, bz: int, tx: int, ty: int, tz: int) -> np.ndarray:
    return np.array(
        [
            _flat_block(bx, by, bz),
            bx,
            by,
            bz,
            _flat_thread(tx, ty, tz),
            tx,
            ty,
            tz,
        ],
        dtype=np.int32,
    )


def _expected() -> np.ndarray:
    exp = np.empty((N_ROWS, COLS), dtype=np.int32)
    for bz in range(GRID_DIM[2]):
        for by in range(GRID_DIM[1]):
            for bx in range(GRID_DIM[0]):
                for tz in range(BLOCK_DIM[2]):
                    for ty in range(BLOCK_DIM[1]):
                        for tx in range(BLOCK_DIM[0]):
                            flat_b = _flat_block(bx, by, bz)
                            flat_t = _flat_thread(tx, ty, tz)
                            row = flat_b * THREADS_PER_WG + flat_t
                            exp[row] = _expected_row(bx, by, bz, tx, ty, tz)
    return exp


def _format_row(row: np.ndarray) -> str:
    return ", ".join(f"{name}={row[i]}" for i, name in enumerate(COL_NAMES))


@pytest.mark.parametrize("target", ["gfx1250", "gfx1251"])
def test_gfx1250_block_id_e2e(target):
    out = np.full((N_ROWS, COLS), SENTINEL, dtype=np.int32)

    def preprocess(x):
        return x.replace("<gfx1250>", f"<{target}>")

    def verify(inputs, outputs):
        got = outputs[0].reshape(N_ROWS, COLS)
        exp = _expected()

        mismatches = [row for row in range(N_ROWS) if not np.array_equal(got[row], exp[row])]
        wrote = [row for row in range(N_ROWS) if np.any(got[row] != SENTINEL)]
        print(f"rows written: {len(wrote)}/{N_ROWS} (expected {N_ROWS})")
        for row in mismatches[:8]:
            print(f"  row {row}: got [{_format_row(got[row])}]")
            print(f"           exp [{_format_row(exp[row])}]")
        if len(mismatches) > 8:
            print(f"  ... and {len(mismatches) - 8} more mismatches")

        np.testing.assert_array_equal(
            got,
            exp,
            err_msg=(
                "block/thread id mapping mismatch (plain launch); each row must "
                f"be [{', '.join(COL_NAMES)}]"
            ),
        )

    compile_and_run(
        "../Target/ASM/gfx1250-cluster-id-asm.mlir",
        "block_id_probe",
        output_data=[out],
        pass_pipeline=TEST_GFX1250_CLUSTER_ASM_PASS_PIPELINE,
        mcpu=target,
        preprocess=preprocess,
        library_paths=[],
        wavefront_size=32,
        grid_dim=GRID_DIM,
        block_dim=BLOCK_DIM,
        verify_fn=verify,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
