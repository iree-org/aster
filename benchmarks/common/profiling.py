"""PyTorch-profiler-based timing utilities for GEMM benchmarks."""

import time
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
    """Profile *fn* and return the average kernel time in milliseconds.

    Tries the PyTorch profiler first. If no CUDA events are captured
    (e.g. IREE dispatches that bypass the HIP tracer), falls back to
    synchronize-bracketed wall-clock timing.
    """
    ms = _profile_pytorch(
        fn,
        num_its=num_its,
        warmup=warmup,
        print_profile=print_profile,
        row_limit=row_limit,
    )
    if ms > 0:
        return ms

    # Fallback: wall-clock timing with cuda.synchronize barriers.
    if print_profile:
        print(
            "  (profiler captured 0 CUDA time, falling back to wall-clock)", flush=True
        )
    return _profile_wallclock(fn, num_its=num_its, warmup=warmup)


def _profile_pytorch(
    fn: Callable[[], Any],
    *,
    num_its: int,
    warmup: int,
    print_profile: bool,
    row_limit: int,
) -> float:
    """Profile with PyTorch profiler.

    Returns 0.0 if no CUDA events captured.
    """
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    schedule = torch.profiler.schedule(
        wait=warmup, warmup=warmup, active=num_its, skip_first=warmup
    )
    total_steps = warmup * 3 + num_its

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

    return total_us / (num_its * 1_000.0)


def _profile_wallclock(
    fn: Callable[[], Any],
    *,
    num_its: int,
    warmup: int,
) -> float:
    """Wall-clock timing with cuda.synchronize barriers."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_its):
        fn()
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / num_its * 1_000.0
