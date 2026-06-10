"""Pass pipelines for ASTER -> StinkyTofu handoff."""

from aster.pass_pipelines import (
    PHASE_CONSTEXPR_EXPANSION,
    PHASE_LOWER_TO_AMDGCN,
    PHASE_PRE_SCHEDULING_CLEANUP,
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PipelineConfigProtocol,
    amdgcn_kernel,
    amdgcn_module,
    builtin_module,
    phase_scf_pipelining,
)


def phase_amdgcn_handoff_backend(
    num_vgprs=256,
    num_agprs=256,
    ll_sched=0,
    hoist_iter_arg_waits=False,
):
    opts = []
    if num_vgprs != 256:
        opts.append(f"num-vgprs={num_vgprs}")
    if num_agprs != 256:
        opts.append(f"num-agprs={num_agprs}")
    if hoist_iter_arg_waits:
        opts.append("hoist-iter-arg-waits=true")
    ll_sched = int(ll_sched)
    if ll_sched > 0:
        opts.append(f"ll-sched={ll_sched}")
    if opts:
        return f"amdgcn-handoff-backend{{{' '.join(opts)}}}"
    return "amdgcn-handoff-backend"


def make_stinkytofu_handoff_pipeline(
    mapping: PipelineConfigProtocol,
    *,
    num_vgprs: int = 256,
    num_agprs: int = 256,
) -> str:
    """Build the StinkyTofu handoff pipeline.

    Common make_default_pass_pipeline until register allocation, then
    forks. Waits are stripped instead of lowered, no
    scheduler/hazards/priority handling. Output is register-allocated,
    wait-free asm for a downstream consumption by StinkyTofu.
    """
    return builtin_module(
        PHASE_PRE_SCHEDULING_CLEANUP,
        PHASE_CONSTEXPR_EXPANSION,
        phase_scf_pipelining(
            lcm_unroll=mapping.lcm_unroll,
            unroll_factor_multiplier=mapping.unroll_factor_multiplier,
            epilogue_peeling=mapping.epilogue_peeling,
            prologue_peeling=mapping.prologue_peeling,
            rotate_stage=mapping.rotate_stage,
        ),
        "aster-destructure-struct-iter-args",
        "canonicalize",
        "cse",
        PHASE_SROA,
        POST_SROA_CLEANUPS,
        # LDS resource allocation is deferred to the handoff backend (after the
        # low-level scheduler) so the scheduler sees live alloc_lds/get_lds_offset
        # handles and the materialized get_lds_offset alloca is colored. See the
        # LDS-alloc block in buildAMDGCNHandoffBackendPassPipeline.
        PHASE_LOWER_TO_AMDGCN,
        amdgcn_module(amdgcn_kernel("aster-hoist-ops")),
        phase_amdgcn_handoff_backend(
            num_vgprs=num_vgprs,
            num_agprs=num_agprs,
            ll_sched=mapping.ll_sched,
            hoist_iter_arg_waits=mapping.hoist_wait,
        ),
    )
