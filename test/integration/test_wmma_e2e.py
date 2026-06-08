import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_LOWER_MINIMAL_PASS_PIPELINE


@pytest.mark.parametrize("target", ["gfx1250", "gfx1251"])
@pytest.mark.parametrize(
    "kernel", ["wmma_f16", "wmma_bf16", "wmma_f16_reuse", "wmma_f16_neg"]
)
def test_wmma_e2e(kernel, target):
    def preprocess(x):
        return x.replace("<gfx1250>", f"<{target}>")

    compile_and_run(
        "wmma-e2e.mlir",
        kernel,
        pass_pipeline=TEST_LOWER_MINIMAL_PASS_PIPELINE,
        mcpu=target,
        preprocess=preprocess,
        library_paths=[],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
