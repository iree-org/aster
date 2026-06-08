import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE


@pytest.mark.parametrize("target", ["gfx1250", "gfx1251"])
def test_setprio_inc_wg_e2e(target):
    def preprocess(x):
        return x.replace("<gfx1250>", f"<{target}>")

    compile_and_run(
        "setprio-inc-wg-e2e.mlir",
        "setprio_inc_wg",
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        mcpu=target,
        preprocess=preprocess,
        library_paths=[],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
