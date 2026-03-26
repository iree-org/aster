"""PyTorch-profiler-based timing utilities for GEMM benchmarks."""

from typing import Any, Callable

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile


def profile_fn(
    fn: Callable[[], Any],
    *,
    num_its: int = 10,
    warmup: int = 5,
    print_profile: bool = True,
    row_limit: int = 20,
) -> float:
    """Profile *fn* with the PyTorch profiler and return the average kernel time.

    The profiler schedule skips *warmup* steps, warms up for *warmup* steps,
    then records *num_its* active steps.  Returns the average GPU kernel time
    in milliseconds across the active steps.

    Args:
        fn: Zero-argument callable to profile (called once per step).
        num_its: Number of active profiler steps whose times are averaged.
        warmup: Number of warm-up steps (also used for skip_first).
        print_profile: Whether to print the profiler table to stdout.
        row_limit: Maximum rows shown in the profiler table.

    Returns:
        Average kernel time in milliseconds.
    """
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    # repeat=1 so the cycle fires exactly once; without it, key_averages()
    # can be empty because the profiler clears events between cycles.
    schedule = torch.profiler.schedule(
        wait=warmup, warmup=warmup, active=num_its, skip_first=warmup
    )
    total_steps = warmup * 3 + num_its  # skip_first + warmup + active

    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()

    with torch_profile(
        activities=activities,
        schedule=schedule,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(total_steps):
            fn()
            torch.cuda.synchronize()
            prof.step()

    events = prof.key_averages(group_by_input_shape=True)

    if print_profile:
        print(events.table(sort_by="cuda_time_total", row_limit=row_limit), flush=True)

    total_us = sum(
        e.self_device_time_total
        for e in events
        if e.device_type == DeviceType.CUDA
        and not e.key.startswith("ProfilerStep")
        and "memcpy" not in e.key.lower()
    )

    return total_us / (num_its * 1_000.0)  # microseconds → milliseconds
