import argparse
import os
import subprocess
import sys

ARCHS = {
    "gfx942": dict(
        bench="bench_perf_102_gemm_python_multitile_directb_cdna3",
        test="test_102_gemm_python_multitile_directb_cdna3",
        run_fn="_run_multitile",
        base=(
            "m2432xn4096xk4096_wg19x16x1_w4x1x1_twg8x16x2_pipestrat1_wgcu1_lcm1"
            "_um3_epeel1_llilpsched2_mfmagap4_lgkmgap0_hoistwait0_rotc1_lt_flat"
        ),
    ),
    "gfx950": dict(
        bench="bench_perf_102_gemm_python_multitile_g2s_directb_cdna4",
        test="test_102_gemm_python_multitile_g2s_directb_cdna4",
        run_fn="_run_cdna4_gemm",
        base=(
            "m4096xn4096xk4096_wg16x16x1_w2x2x1_twg16x16x1_pipestrat1_wgcu1_lcm1"
            "_um3_epeel1_llilpsched2_mfmagap4_lgkmgap0_hoistwait0_rotc1_lt_flat_cdna4"
        ),
    ),
}


def find_root():
    sentinel = "contrib/kittens/test/bench"
    d = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.isdir(os.path.join(d, sentinel)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            sys.exit("ERROR: run from inside the aster checkout (cannot find %s)" % sentinel)
        d = parent


def window_label(base, n):
    # n == 32 is the default window (no token); else insert _ilpwin<n> after lgkmgap.
    return base if n == 32 else base.replace("_lgkmgap0_", "_lgkmgap0_ilpwin%d_" % n)


def classify(p, out):
    if p is None:
        return "TIMEOUT"
    if "RUN_OK" in p.stdout:
        return "PASS"
    if "illegal memory access" in out:
        return "illegal memory access"
    if any(
        s in out
        for s in (
            "assert_allclose",
            "Not equal to tolerance",
            "Mismatched elements",
            "Max absolute difference",
            "AssertionError",
        )
    ):
        return "numeric mismatch"
    if p.returncode is not None and p.returncode < 0:
        return "crash (signal %d)" % (-p.returncode)
    if "Skipped" in out or ("skip" in out.lower() and "gpu" in out.lower()):
        return "SKIPPED (no GPU)"
    return "fail (rc=%s)" % p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS), default="gfx942")
    ap.add_argument("--reps", type=int, default=5, help="runs per window (intermittent fault)")
    ap.add_argument("--windows", default="4,8,16,32", help="comma-separated ilp window sizes")
    ap.add_argument(
        "--fault-detail",
        action="store_true",
        help="run only the failing window with HIP fault reporting (blocking launch + AMD_LOG_LEVEL) "
        "so the illegal access surfaces at the faulting kernel with the page-fault address; prints full output",
    )
    args = ap.parse_args()

    arch = args.arch
    cfg = ARCHS[arch]
    root = find_root()
    contrib = os.path.join(root, "contrib")
    test = os.path.join(root, "contrib/kittens/test")
    bench = os.path.join(root, "contrib/kittens/test/bench")
    windows = [int(w) for w in args.windows.split(",")]

    # In fault-detail mode, probe only the failing window and surface the GPU
    # fault at its source (not deferred to event destroy) with the faulting VA.
    env = dict(os.environ)
    if args.fault_detail:
        windows = [8]
        env["HIP_LAUNCH_BLOCKING"] = "1"
        env["AMD_LOG_LEVEL"] = "3"
        env["HSA_ENABLE_INTERRUPT"] = "1"

    driver = (
        "import sys\n"
        "sys.path[:0] = [%r, %r, %r]\n"
        % (contrib, test, bench)
        + "from %s import _from_label\n" % cfg["bench"]
        + "from %s import %s\n" % (cfg["test"], cfg["run_fn"])
        + "%s(_from_label(sys.argv[1], %r))\n" % (cfg["run_fn"], arch)
        + "print('RUN_OK', flush=True)\n"
    )

    print("=== um3 schedule-dependence test ===")
    print("arch:", arch, " reps:", args.reps, " windows:", windows, " root:", root)
    print("python:", sys.executable)

    results = []
    for n in windows:
        lbl = window_label(cfg["base"], n)
        outcomes = []
        for r in range(args.reps):
            print("  win%-3d rep %d/%d ..." % (n, r + 1, args.reps), flush=True)
            p, out = None, ""
            try:
                p = subprocess.run(
                    [sys.executable, "-c", driver, lbl], capture_output=True, text=True, timeout=900, env=env
                )
                out = (p.stdout or "") + "\n" + (p.stderr or "")
            except subprocess.TimeoutExpired:
                p = None
            if args.fault_detail and p is not None:
                print("----- FULL OUTPUT (fault-detail, paste back) -----")
                print(out[-6000:])
                print("----- end -----")
            outcomes.append((classify(p, out), [x for x in out.splitlines() if x.strip()][-3:]))
        nfail = sum(1 for v, _ in outcomes if v != "PASS" and not v.startswith("SKIPPED"))
        nskip = sum(1 for v, _ in outcomes if v.startswith("SKIPPED"))
        kinds = sorted({v for v, _ in outcomes if v != "PASS" and not v.startswith("SKIPPED")})
        tail = next((t for v, t in outcomes if v != "PASS" and not v.startswith("SKIPPED")), [])
        results.append((n, lbl, args.reps, nfail, nskip, kinds, tail))

    print("\n================= RESULTS (paste back) =================")
    print("arch=%s reps=%d" % (arch, args.reps))
    for n, lbl, reps, nfail, nskip, kinds, tail in results:
        note = "  [%s]" % ", ".join(kinds) if kinds else ""
        if nskip == reps:
            note = "  [SKIPPED: no GPU]"
        print("  win%-3d : %d/%d FAIL%s" % (n, nfail, reps, note))

    print("\n================= DETAIL (first failure per window) =================")
    for n, lbl, reps, nfail, nskip, kinds, tail in results:
        if nfail:
            print("win%d : %d/%d failed %s" % (n, nfail, reps, kinds))
            print("    label: %s" % lbl)
            for ln in tail:
                print("    | %s" % ln[:170])
            print()


if __name__ == "__main__":
    main()
