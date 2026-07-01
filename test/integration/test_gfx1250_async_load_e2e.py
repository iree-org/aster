import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_GFX1250_CLUSTER_ASM_PASS_PIPELINE

WAVEFRONT_SIZE = 32
N_LANES = WAVEFRONT_SIZE
# Dwords moved per lane: b32 kernels move 1, the b128 kernel moves 4.
DWORDS_PER_LANE = {
    "async_load_to_lds": 1,
    "async_load_wait": 1,
    "async_load_b128_saddr": 4,
}


@pytest.mark.parametrize("target", ["gfx1250", "gfx1251"])
@pytest.mark.parametrize(
    "kernel",
    [
        "async_load_to_lds",
        "async_load_wait",
        "async_load_b128_saddr",
    ],
)
def test_gfx1250_async_load_e2e(kernel, target):
    n_dwords = N_LANES * DWORDS_PER_LANE[kernel]
    src = np.arange(n_dwords, dtype=np.int32)
    dst = np.zeros(n_dwords, dtype=np.int32)

    def preprocess(x):
        return x.replace("<gfx1250>", f"<{target}>")

    def verify(inputs, outputs):
        np.testing.assert_array_equal(
            outputs[0],
            inputs[0],
            err_msg=f"{kernel} multi-lane roundtrip failed",
        )

    input_data = [src]
    output_data = []
    verify_fn = None
    # Note: async_load_to_lds is fire-and-forget, no out buffer to verify.
    if kernel != "async_load_to_lds":
        output_data = [dst]
        verify_fn = verify

    compile_and_run(
        "../Target/ASM/gfx1250-async-load-e2e.mlir",
        kernel,
        input_data=input_data,
        output_data=output_data,
        pass_pipeline=TEST_GFX1250_CLUSTER_ASM_PASS_PIPELINE,
        mcpu=target,
        preprocess=preprocess,
        library_paths=[],
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=(N_LANES, 1, 1),
        grid_dim=(1, 1, 1),
        verify_fn=verify_fn,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
