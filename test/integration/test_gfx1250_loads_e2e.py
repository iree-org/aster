import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_GFX1250_CLUSTER_ASM_PASS_PIPELINE

WAVEFRONT_SIZE = 32
N_LANES = WAVEFRONT_SIZE
DWORDS_PER_LANE = 4
N_DWORDS = N_LANES * DWORDS_PER_LANE


@pytest.mark.parametrize("target", ["gfx1250", "gfx1251"])
def test_gfx1250_loads_e2e(target):
    src = np.arange(N_DWORDS, dtype=np.int32)
    dst = np.zeros(N_DWORDS, dtype=np.int32)

    def preprocess(x):
        return x.replace("<gfx1250>", f"<{target}>")

    def verify(inputs, outputs):
        np.testing.assert_array_equal(
            outputs[0],
            inputs[0],
            err_msg="ds_load_b128 LDS round-trip mismatch",
        )

    compile_and_run(
        "../Target/ASM/gfx1250-loads-e2e.mlir",
        "ds_load_b128",
        input_data=[src],
        output_data=[dst],
        pass_pipeline=TEST_GFX1250_CLUSTER_ASM_PASS_PIPELINE,
        mcpu=target,
        preprocess=preprocess,
        library_paths=[],
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=(N_LANES, 1, 1),
        grid_dim=(1, 1, 1),
        verify_fn=verify,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
