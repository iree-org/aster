import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_GFX1250_CLUSTER_ASM_PASS_PIPELINE

N_CLUSTERS = 2  # number of clusters
N_WG_PER_CLUSTER = 4  # workgroups per cluster (== cluster_dims[0], M0 = 15)
N_WG = N_CLUSTERS * N_WG_PER_CLUSTER  # 8 total workgroups
N_WAVES = 4  # waves per WG
N_LANES = 32  # lanes per wave (wave32)
THREADS_PER_WG = N_WAVES * N_LANES  # 128 threads
SENTINEL = -1  # host pre-fill; phase 1 must overwrite it


@pytest.mark.parametrize("target", ["gfx1250", "gfx1251"])
@pytest.mark.parametrize(
    ("mlir_file", "kernel"),
    [
        # Sync multicast: cluster_load_b32 -> VGPR.
        ("../Target/ASM/gfx1250-cluster-loads-asm.mlir", "cload"),
        # Async multicast: cluster_load_async_to_lds_b32 -> LDS, s_wait_asynccnt,
        # ds_load_b32 back. Same numeric result as the sync variant.
        ("../Target/ASM/gfx1250-cluster-async-loads-asm.mlir", "cload_async"),
    ],
)
def test_gfx1250_cluster_loads_e2e(mlir_file, kernel, target):
    # scratch[wg_id][tid] pre-filled with the -1 sentinel.
    # with the global id; an un-drained read (no barrier) would still see -1.
    scratch_buf = np.full((N_WG, THREADS_PER_WG), SENTINEL, dtype=np.int32)
    output_buf = np.zeros((N_WG, THREADS_PER_WG), dtype=np.int32)

    def preprocess(x):
        return x.replace("<gfx1250>", f"<{target}>")

    def verify(inputs, outputs):
        out = outputs[0].reshape(N_WG, THREADS_PER_WG)
        for c in range(N_CLUSTERS):
            base_wg = c * N_WG_PER_CLUSTER  # cluster's lowest WG
            rows = out[base_wg : base_wg + N_WG_PER_CLUSTER]
            # intra-cluster equality: all 4 WGs cluster_load the same row.
            for w in range(1, N_WG_PER_CLUSTER):
                np.testing.assert_array_equal(
                    rows[w],
                    rows[0],
                    err_msg=(
                        f"cluster {c}: WG {base_wg + w} and WG {base_wg} must "
                        "receive the same cluster_load broadcast"
                    ),
                )

    compile_and_run(
        mlir_file,
        kernel,
        input_data=[scratch_buf.reshape(-1)],
        output_data=[output_buf.reshape(-1)],
        pass_pipeline=TEST_GFX1250_CLUSTER_ASM_PASS_PIPELINE,
        mcpu=target,
        wavefront_size=32,
        grid_dim=(N_WG, 1, 1),
        block_dim=(THREADS_PER_WG, 1, 1),
        preprocess=preprocess,
        verify_fn=verify,
        library_paths=[],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
