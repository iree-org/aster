"""Search for GEMM configs where double-buffer software pipelining helps.

Pipelining (double-buffer LDS) helps when BOTH conditions hold:
  1. Without pipeline: global load latency is on the critical path.
     (context-switch to other WGs cannot fully hide it)
  2. With pipeline: issuing loads at the start of each iteration gives the
     single-wave instruction stream enough time to hide the latency.

Analytically:
  hide_nopipe  = wg_occ_nopipe  × num_waves × compute_cyc
  hide_pipe_ilp = DS_READ_CYCLES + compute_cyc   (single-WG ILP coverage)

  Pipelining helps iff:
    hide_nopipe  < MEMORY_LATENCY   (memory-bound without pipeline)
    hide_pipe_ilp >= MEMORY_LATENCY  (pipeline covers the gap)
    2 × lds_per_wg <= LDS_PER_CU    (double-buffer fits)

Hardware constants (gfx942 / MI300X):
  LDS per CU:          64 KB
  Max WGs per CU:       8
  Global load latency: ~400 cycles   (HBM, approximation)
  MFMA throughput:       8 cycles    (v_mfma_f32_16x16x16_f16 per wavefront)
  DS read latency:      ~40 cycles   (ds_read_b64 to waitcnt)
"""

import itertools
import os
import sys
from dataclasses import dataclass

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, ".."))
sys.path.insert(0, os.path.join(_DIR, "..", "..", "..", ".."))

# Import only the config class; avoid pulling in aster.execution which requires
# a GPU runtime (this is a static-analysis script).
try:
    from gemm_fp16_lds import GEMMConfig
except (ImportError, ModuleNotFoundError):
    # Fallback: inline the pure-Python GEMMConfig dataclass.
    @dataclass
    class GEMMConfig:
        m: int
        n: int
        k: int
        m_tile: int
        n_tile: int
        k_tile: int
        num_waves: int
        swizzle: int = 1

        def __post_init__(self):
            assert self.m % self.m_tile == 0
            assert self.n % self.n_tile == 0
            assert self.k % self.k_tile == 0
            assert self.k_tile % 32 == 0
            assert (self.m_t * self.k_t) % self.num_waves == 0
            assert (self.n_t * self.k_t) % self.num_waves == 0
            _q = self.m_t * self.n_t * self.k_inner // self.num_waves
            assert _q > 0 and (_q & (_q - 1)) == 0
            assert self.m_wg % self.swizzle == 0

        @property
        def m_wg(self):
            return self.m // self.m_tile

        @property
        def n_wg(self):
            return self.n // self.n_tile

        @property
        def k_t(self):
            return self.k_tile // 32

        @property
        def m_t(self):
            return self.m_tile // 16

        @property
        def n_t(self):
            return self.n_tile // 16

        @property
        def k_inner(self):
            return self.k_tile // 16

        @property
        def tiles_per_wave(self):
            return self.m_t * self.n_t // self.num_waves

        @property
        def a_per_wave(self):
            return self.m_t * self.k_t // self.num_waves

        @property
        def b_per_wave(self):
            return self.n_t * self.k_t // self.num_waves

        @property
        def a_lds_bytes(self):
            return self.k_t * self.m_t * 1024

        @property
        def b_lds_bytes(self):
            return self.k_t * self.n_t * 1024


# ── Hardware constants ──────────────────────────────────────────────────────
LDS_PER_CU = 65536  # bytes, gfx942
MAX_WGS_PER_CU = 8  # hard cap
VGPR_PER_LANE = 256  # max VGPRs per wavefront
NUM_SIMDS_PER_CU = 4
MEMORY_LATENCY = 400  # cycles, global load (HBM)
DS_READ_CYCLES = 40  # cycles, ds_read_b64 latency
MFMA_CYCLES = 8  # cycles, throughput per wavefront


# ── Per-config analysis ─────────────────────────────────────────────────────


def _lds_occupancy(lds_per_wg: int) -> int:
    if lds_per_wg == 0 or lds_per_wg > LDS_PER_CU:
        return 0
    return min(LDS_PER_CU // lds_per_wg, MAX_WGS_PER_CU)


def _vgpr_occupancy(cfg: GEMMConfig) -> int:
    """Max WGs/CU limited by VGPR budget (coarse estimate)."""
    a_load = cfg.a_per_wave * 4
    b_load = cfg.b_per_wave * 4
    a_lds = cfg.tiles_per_wave * cfg.k_inner * 2
    b_lds = cfg.tiles_per_wave * cfg.k_inner * 2
    vgprs_per_wave = a_load + b_load + a_lds + b_lds + 20
    # Each SIMD has VGPR_PER_LANE VGPRs; 4 SIMDs per CU.
    waves_per_cu = (VGPR_PER_LANE * NUM_SIMDS_PER_CU) // max(vgprs_per_wave, 1)
    return min(waves_per_cu // cfg.num_waves, MAX_WGS_PER_CU)


def analyze(cfg: GEMMConfig) -> dict:
    lds_nopipe = cfg.a_lds_bytes + cfg.b_lds_bytes
    lds_pipe = 2 * lds_nopipe

    occ_nopipe = min(_lds_occupancy(lds_nopipe), _vgpr_occupancy(cfg))
    occ_pipe = min(_lds_occupancy(lds_pipe), _vgpr_occupancy(cfg))

    # MFMAs per wave per K-outer-iteration.
    compute_cyc = cfg.tiles_per_wave * cfg.k_inner * MFMA_CYCLES

    # Latency hiding via context-switch (all waves on CU while one waits).
    hide_nopipe = occ_nopipe * cfg.num_waves * compute_cyc
    # Latency hiding via single-WG ILP (pipeline issues loads compute_cyc
    # + DS_READ_CYCLES before they are needed).
    hide_pipe_ilp = DS_READ_CYCLES + compute_cyc

    mem_bound_nopipe = hide_nopipe < MEMORY_LATENCY
    covered_by_pipe = hide_pipe_ilp >= MEMORY_LATENCY
    fits_pipe = occ_pipe >= 1

    return {
        "lds_nopipe": lds_nopipe,
        "lds_pipe": lds_pipe,
        "occ_nopipe": occ_nopipe,
        "occ_pipe": occ_pipe,
        "compute_cyc": compute_cyc,
        "hide_nopipe": hide_nopipe,
        "hide_pipe_ilp": hide_pipe_ilp,
        "mem_bound_nopipe": mem_bound_nopipe,
        "covered_by_pipe": covered_by_pipe,
        "fits_pipe": fits_pipe,
        "helps": mem_bound_nopipe and covered_by_pipe and fits_pipe,
    }


# ── Config sweep ────────────────────────────────────────────────────────────


def _divisors(n: int):
    return [d for d in range(1, n + 1) if n % d == 0]


def sweep():
    candidates = []
    counts = dict(total=0, valid=0, mem_bound=0, helps=0)

    for num_waves in [1, 2, 4, 8]:
        for m_mult, n_mult in itertools.product(range(1, 9), range(1, 9)):
            for k_t in [1, 2, 4, 8]:
                m_tile = m_mult * num_waves * 16
                n_tile = n_mult * num_waves * 16
                k_tile = k_t * 32
                # Use dimensions large enough to be representative.
                m = m_tile * 32
                n = n_tile * 32
                k = k_tile * 64
                counts["total"] += 1
                try:
                    cfg = GEMMConfig(
                        m,
                        n,
                        k,
                        m_tile,
                        n_tile,
                        k_tile,
                        num_waves,
                        swizzle=1,
                    )
                except AssertionError:
                    continue

                r = analyze(cfg)
                if r["occ_nopipe"] < 1:
                    continue  # LDS doesn't fit at all — skip.
                counts["valid"] += 1
                if r["mem_bound_nopipe"]:
                    counts["mem_bound"] += 1
                if r["helps"]:
                    counts["helps"] += 1
                    candidates.append((cfg, r))

    return candidates, counts


# ── Output ──────────────────────────────────────────────────────────────────


def _print_analysis(counts: dict):
    print(f"  Configs enumerated:             {counts['total']}")
    print(f"  Configs valid (≥1 WG/CU):       {counts['valid']}")
    print(f"  Memory-bound without pipeline:  {counts['mem_bound']}")
    print(f"  Pipeline predicted to help:     {counts['helps']}")


def _print_why_not():
    print("""
Why pipelining cannot help for any config in this kernel
─────────────────────────────────────────────────────────
Two conditions must hold simultaneously:

  (A) Without pipeline, occupancy-based context switching does NOT fully hide
      global load latency:
          wg_occ × num_waves × compute_cyc  <  MEMORY_LATENCY (400 cycles)

  (B) With pipeline, the single-WG instruction stream DOES hide it:
          DS_READ_CYCLES + compute_cyc  ≥  MEMORY_LATENCY (400 cycles)
      →   compute_cyc  ≥  360 cycles

Substituting (B) into (A):
          wg_occ × num_waves  <  400 / 360  ≈  1.11
          → wg_occ × num_waves  =  1  (single wavefront on the whole CU)

That means wg_occ = 1 AND num_waves = 1, which is impractical.

The reason wg_occ × num_waves is never that small can be shown analytically.
For LDS-limited occupancy with square tiles (m_t = n_t = T):

  wg_occ_LDS  =  64 KB / (k_t × 2T × 1 KB)  =  32 / (k_t × T)

  compute_cyc  =  tiles_per_wave × k_inner × 8
               =  (T² / num_waves) × (k_t × 2) × 8
               =  (T² / num_waves) × k_t × 16

  hide_nopipe  =  [32 / (k_t × T)] × num_waves × [(T² / num_waves) × k_t × 16]
               =  32 × T × 16
               =  512 × T

The k_t and num_waves factors cancel exactly.  For the minimum feasible tile
T = 1 (m_tile = n_tile = 16 × num_waves):

  hide_nopipe  ≥  512 × 1  =  512 cycles  >  400 cycles  (ALWAYS)

Occupancy-based latency hiding is always sufficient.  The pipeline's benefit
(hiding within a single WG via ILP) requires compute_cyc ≥ 360, but at that
compute density the many WGs / waves already hide the latency without any
pipeline at all.  Doubling the LDS only reduces occupancy and adds barriers.

What GEMMPerf.md assumes
─────────────────────────
The note "1.5–2× throughput improvement" applies to kernels where the tile
size is fixed externally (e.g. large fused ops) and occupancy is already
pinned at 1 WG/CU by register pressure, not LDS.  In that case:
  - Double-buffer adds no LDS cost to occupancy (already at 1 WG).
  - Single-WG ILP can hide global loads if compute_cyc ≥ 360 (≈45 MFMAs).
For this kernel the tile size and register pressure scale together, so you
never reach the "1 WG but enough compute" regime.
""")


def main():
    print(f"Hardware model: gfx942 / MI300X")
    print(f"  LDS per CU:          {LDS_PER_CU // 1024} KB")
    print(f"  Global load latency: {MEMORY_LATENCY} cycles")
    print(f"  MFMA throughput:     {MFMA_CYCLES} cycles / wavefront")
    print(f"  DS read latency:     {DS_READ_CYCLES} cycles")
    print()
    print(
        "Sweeping configs (num_waves ∈ {1,2,4,8}, m/n mult ∈ 1..8, k_t ∈ {1,2,4,8}) …"
    )
    candidates, counts = sweep()
    print()
    _print_analysis(counts)

    if candidates:
        print()
        print("Configs where pipelining is predicted to help:")
        hdr = f"  {'config':<52} {'LDS':>6} {'LDS×2':>6} {'occ':>4} {'occ_p':>5} "
        hdr += f"{'comp':>5} {'hide_np':>8} {'hide_p':>7}"
        print(hdr)
        by_gap = sorted(
            candidates,
            key=lambda x: x[1]["hide_pipe_ilp"] - x[1]["hide_nopipe"],
            reverse=True,
        )
        for cfg, r in by_gap[:20]:
            label = (
                f"mt{cfg.m_tile}×nt{cfg.n_tile}×kt{cfg.k_tile}" f"_nw{cfg.num_waves}"
            )
            print(
                f"  {label:<52}"
                f" {r['lds_nopipe']//1024:>4}KB"
                f" {r['lds_pipe']//1024:>4}KB"
                f" {r['occ_nopipe']:>4}"
                f" {r['occ_pipe']:>5}"
                f" {r['compute_cyc']:>5}"
                f" {r['hide_nopipe']:>8}"
                f" {r['hide_pipe_ilp']:>7}"
            )
    else:
        _print_why_not()


if __name__ == "__main__":
    main()
