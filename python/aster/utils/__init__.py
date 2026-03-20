"""Backward-compatible re-export surface for aster.utils.

All names that were previously importable from the flat aster/utils.py module are still
accessible here, delegating to their new canonical locations in aster.utils.capsule,
aster.execution, and aster.compilation.
"""

from typing import Optional
from pathlib import Path

from aster.utils.capsule import unwrap_pointer_from_capsule, wrap_pointer_in_capsule
from aster.execution import (
    copy_array_to_gpu,
    copy_from_gpu_buffer,
    create_kernel_args_capsule,
    create_kernel_args_capsule_from_numpy,
)
from aster.compiler import (
    assemble_to_hsaco,
    compile_to_hsaco,
    translate_module,
)


def system_has_mcpu(mcpu: str, rocm_path: Optional[Path] = None) -> bool:
    """Delegate to aster.execution.hip.system_has_gpu (the single canonical impl)."""
    from aster.execution.hip import system_has_gpu

    return system_has_gpu(mcpu)
