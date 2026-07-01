import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_GFX1250_CLUSTER_ASM_PASS_PIPELINE


@pytest.mark.parametrize("target", ["gfx1250", "gfx1251"])
def test_gfx1250_cluster_e2e(target):
    def preprocess(x):
        return x.replace("<gfx1250>", f"<{target}>")

    compile_and_run(
        "../Target/ASM/gfx1250-cluster-id-asm.mlir",
        "cluster_ids",
        pass_pipeline=TEST_GFX1250_CLUSTER_ASM_PASS_PIPELINE,
        mcpu=target,
        preprocess=preprocess,
        library_paths=[],
        wavefront_size=32,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
