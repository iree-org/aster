"""GPU execution utilities: memory management, kernel args, and HIP runtime."""

from aster.execution.execution import (
    copy_array_to_gpu,
    copy_from_gpu_buffer,
    create_kernel_args_capsule,
    create_kernel_args_capsule_from_numpy,
)
from aster.execution.hip import (
    KernelResources,
    compute_register_budget,
    execute_hsaco,
    parse_asm_kernel_resources,
    system_has_gpu,
)
