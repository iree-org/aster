import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_LOWER_WAITS_MINIMAL_PASS_PIPELINE


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
    def preprocess(x):
        return x.replace("<gfx1250>", f"<{target}>")

    compile_and_run(
        "../Target/ASM/gfx1250-loads-asm.mlir",
        kernel,
        pass_pipeline=TEST_LOWER_WAITS_MINIMAL_PASS_PIPELINE,
        mcpu=target,
        preprocess=preprocess,
        library_paths=[],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
