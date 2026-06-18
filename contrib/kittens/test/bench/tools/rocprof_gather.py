"""
Usage:
  python contrib/kittens/test/bench/tools/rocprof_gather.py \
      /path/to/campaign_top20_<bench>_<ts>.txt
  python .../rocprof_gather.py CAMPAIGN.txt --top-n 5 --iterations 100 --dry-run
"""

import argparse
import concurrent.futures
import glob
import json
import os
import re
import subprocess
import sys
import time

# Knob tokens encoded in the config label (best-effort; missing -> omitted).
_LABEL_KNOB_RE = {
    "ll_ilp_sched": r"llilpsched(-?\d+)",
    "mfma_gap": r"mfmagap(\d+)",
    "lgkm_gap": r"lgkmgap(\d+)",
    "vmem_gap": r"vmemgap(\d+)",
    "max_load_distance": r"maxload(?:dist)?(\d+)",
    "min_lgkm_distance": r"minlgkm(\d+)",
    "hoist_wait": r"hoistwait(\d+)",
    "rotate_compute_stage": r"rotc(\d+)",
    "epilogue_peeling": r"epeel(\d+)",
    "ps": r"pipestrat(\d+)",
    "unroll_mult": r"\bum(\d+)",
    "wg_per_cu": r"wgcu(\d+)",
}

# tools -> bench -> test -> kittens -> contrib -> repo root (5 levels up).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
_ROCPROF_SH = os.path.join(_REPO_ROOT, "contrib", "kittens", "rocprof-kittens-bench.sh")


def _parse_label_knobs(label):
    """Extract ILP-relevant knobs from the config label."""
    knobs = {}
    for name, pat in _LABEL_KNOB_RE.items():
        m = re.search(pat, label)
        if m:
            knobs[name] = int(m.group(1))
    return knobs


def parse_campaign(path):
    """Parse a campaign_top20 file -> (meta, [config dicts])."""
    with open(path) as f:
        lines = f.read().splitlines()

    meta = {"campaign_file": os.path.abspath(path)}
    for ln in lines[:6]:
        m = re.search(r"Bench:\s*(\S+)\s+size:\s*m=(\d+)\s*n=(\d+)\s*k=(\d+)\s*mcpu:\s*(\S+)", ln)
        if m:
            meta.update(
                bench_label=m.group(1),
                target_m=int(m.group(2)),
                target_n=int(m.group(3)),
                target_k=int(m.group(4)),
                mcpu=m.group(5),
            )
            break

    configs = []
    i = 0
    perf_re = re.compile(r"^#\s*(\d+)\s*\[(T\d+)\]\s*([\d.]+)\s*TF/s\s*p50\s*\(\s*([\d.]+)%\s*peak\)")
    while i < len(lines):
        pm = perf_re.match(lines[i].strip())
        if not pm:
            i += 1
            continue
        rank, tier, tflops, pct_peak = pm.groups()
        axes = lines[i + 1].strip() if i + 1 < len(lines) else ""
        repro = ""
        for j in (i + 2, i + 3):
            if j < len(lines) and "repro:" in lines[j]:
                repro = lines[j].split("repro:", 1)[1].strip()
                break
        # repro: python <script_path> <label>
        script_path, label = None, None
        rm = re.search(r"(\S+\.py)\s+(\S+)\s*$", repro)
        if rm:
            script_path, label = rm.group(1), rm.group(2)
        configs.append(
            dict(
                rank=int(rank),
                tier=tier,
                tflops=float(tflops),
                pct_peak=float(pct_peak),
                axes=axes,
                repro_cmd=repro,
                script_path=script_path,
                label=label,
                ilp_knobs=_parse_label_knobs(label or ""),
            )
        )
        i += 3
    return meta, configs


def _script_rel_to_test(script_path):
    """rocprof-kittens-bench.sh wants the script relative to contrib/kittens/test/."""
    marker = "contrib/kittens/test/"
    if marker in script_path:
        return script_path.split(marker, 1)[1]
    # Already relative (e.g. bench/foo.py) or a bare name.
    return script_path if script_path.startswith("bench/") else os.path.join("bench", os.path.basename(script_path))


def _run_rocprof(script_rel, label, mcpu, iterations, hsaco, dry_run):
    """Run the rocprof shell script (counters = its own default); return
    (trace_dir, hsaco, stdout, rc, pretty).

    With a pre-compiled hsaco the shell script skips Phase 1 (compile)
    and only runs Phase 2 (rocprofv3 ATT).
    """
    env = dict(os.environ)
    env["ITERATIONS"] = str(iterations)
    # The bench single-config path requires --mcpu (forwarded to the bench
    # script by rocprof-kittens-bench.sh ahead of --compile-only/--hsaco).
    cmd = [_ROCPROF_SH, script_rel, label]
    if mcpu:
        cmd += ["--mcpu", mcpu]
    if hsaco:
        cmd += ["--hsaco", hsaco]  # reuse pre-compiled HSACO -> Phase 2 only
    pretty = f"ITERATIONS={iterations} {' '.join(cmd)}"
    if dry_run:
        return None, None, pretty, 0, pretty
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    trace_dir, hsaco = None, None
    for ln in out.splitlines():
        m = re.search(r"Trace:\s*(\S+)", ln)
        if m:
            trace_dir = m.group(1)
        m = re.search(r"HSACO:\s*(\S+)", ln)
        if m:
            hsaco = m.group(1)
    # trace dir is printed relative to cwd by the shell script
    if trace_dir and not os.path.isabs(trace_dir):
        trace_dir = os.path.join(_REPO_ROOT, trace_dir)
    return trace_dir, hsaco, out, proc.returncode, pretty


def _find_code_jsons(trace_dir):
    if not trace_dir or not os.path.isdir(trace_dir):
        return []
    return sorted(glob.glob(os.path.join(trace_dir, "**", "code.json"), recursive=True))


def _derive_bubbles(code_json_obj):
    """Best-effort bubble metrics from a code.json instruction array.

    code.json is an array of per-PC records {pc, asm, hit_count,
    stall_count, latency, counters}. Field names vary by rocprofv3
    version, so every access is guarded; raw content is inlined
    regardless.
    """
    recs = code_json_obj if isinstance(code_json_obj, list) else code_json_obj.get("instructions", [])
    if not isinstance(recs, list) or not recs:
        return {"note": "code.json not an instruction array; see raw"}

    def is_mfma(r):
        return "mfma" in str(r.get("asm", "")).lower()

    def hit(r):
        return float(r.get("hit_count", r.get("hits", 0)) or 0)

    def stall(r):
        return float(r.get("stall_count", r.get("stalls", 0)) or 0)

    mfma = [r for r in recs if is_mfma(r)]
    mfma_hits = sum(hit(r) for r in mfma)
    mfma_stalls = sum(stall(r) for r in mfma)

    # max consecutive-MFMA run and mean inter-MFMA gap, in program (PC) order.
    runs, cur = [], 0
    gaps, since = [], None
    for r in recs:
        if is_mfma(r):
            cur += 1
            if since is not None:
                gaps.append(since)
            since = 0
        else:
            if cur:
                runs.append(cur)
            cur = 0
            if since is not None:
                since += 1
    if cur:
        runs.append(cur)

    # Sum any per-PC counters into PMC totals.
    pmc = {}
    for r in recs:
        for k, v in (r.get("counters", {}) or {}).items():
            try:
                pmc[k] = pmc.get(k, 0) + float(v)
            except (TypeError, ValueError):
                pass

    def frac(a, b):
        return round(a / b, 4) if b else None

    derived = {
        "num_instructions": len(recs),
        "num_mfma": len(mfma),
        "mfma_hits": mfma_hits,
        "mfma_stalls": mfma_stalls,
        "bubble_pct": round(100.0 * mfma_stalls / mfma_hits, 3) if mfma_hits else None,
        "max_mfma_run": max(runs) if runs else 0,
        "mean_inter_mfma_gap": round(sum(gaps) / len(gaps), 3) if gaps else None,
        "pmc_from_code_json": pmc or None,
    }
    iv = pmc.get("SQ_INSTS_VALU")
    if iv:
        derived["mfma_issue_frac"] = frac(pmc.get("SQ_INSTS_VALU_MFMA", 0), iv)
        wv = pmc.get("SQ_WAIT_INST_VALU", 0)
        derived["mfma_stall_frac"] = frac(wv, wv + iv)
    return derived


def _hsaco_path(label):
    """Per-config HSACO path under /tmp, named by the config label."""
    return os.path.join("/tmp", f"rocprof_{label}.hsaco")


def _compile_hsaco(script_full, label, mcpu, hsaco_path, dry_run):
    """Compile one config to hsaco_path via the bench --compile-only path (no GPU).

    Returns (rc, output, cmd_str).
    """
    cmd = [sys.executable, script_full, label, "--mcpu", mcpu, "--compile-only", "--hsaco", hsaco_path]
    cmd_str = " ".join(cmd)
    if dry_run:
        return 0, cmd_str, cmd_str
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
    return proc.returncode, proc.stdout + "\n" + proc.stderr, cmd_str


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign_file", help="campaign_top20_*.txt from a perf-explore run")
    ap.add_argument("--top-n", type=int, default=20, help="profile the top N configs (default 20)")
    ap.add_argument("--iterations", type=int, default=100, help="kernel launches per rocprof run (default 100)")
    ap.add_argument(
        "--bench-script", default=None, help="override the bench script path (relative to contrib/kittens/test/)"
    )
    ap.add_argument("--out", default=None, help="aggregate JSON output path (default: alongside the campaign file)")
    ap.add_argument("--dry-run", action="store_true", help="parse + print the commands without profiling")
    args = ap.parse_args()

    meta, configs = parse_campaign(args.campaign_file)
    configs = [c for c in configs[: args.top_n] if c["label"]]
    if not configs:
        print("No configs parsed from campaign file. Check the format.", file=sys.stderr)
        return 2

    mcpu = meta.get("mcpu")
    ts = time.strftime("%Y%m%d_%H%M%S")
    bench = meta.get("bench_label", "bench")
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.campaign_file)),
        f"rocprof_gather_{bench}_{ts}.json",
    )

    print("=" * 78)
    print(
        f"rocprof-gather: {bench}  m={meta.get('target_m')} n={meta.get('target_n')} "
        f"k={meta.get('target_k')} mcpu={mcpu}"
    )
    print(f"campaign: {meta['campaign_file']}")
    print(f"configs:  top {len(configs)}   rocprof iterations: {args.iterations}")
    print(f"output:   {out_path}")
    print("=" * 78)

    def _script_full(cfg):
        if args.bench_script:
            return os.path.join(_REPO_ROOT, "contrib", "kittens", "test", args.bench_script)
        return os.path.join(_REPO_ROOT, cfg["script_path"])

    # Phase 1: compile every HSACO in parallel (CPU-bound, no GPU). Each config
    # gets a stable /tmp/rocprof_<label>.hsaco that Phase 2 reuses.
    print(f"\nPhase 1: compiling {len(configs)} HSACOs in parallel ...")
    compiled = {}  # rank -> (hsaco_path or None, rc, output)

    def _do_compile(cfg):
        hp = _hsaco_path(cfg["label"])
        rc, out, cmd_str = _compile_hsaco(_script_full(cfg), cfg["label"], mcpu, hp, args.dry_run)
        return cfg, hp, rc, out, cmd_str

    workers = min(len(configs), os.cpu_count() or 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for cfg, hp, rc, out, cmd_str in ex.map(_do_compile, configs):
            ok = args.dry_run or (rc == 0 and os.path.exists(hp))
            compiled[cfg["rank"]] = (hp if ok else None, rc, out)
            if args.dry_run:
                print(f"  compile #{cfg['rank']:>2}: {cmd_str}")
            else:
                print(f"  compile #{cfg['rank']:>2} [{'ok' if ok else 'FAILED'}] {os.path.basename(hp)}")
                if not ok:
                    print(out, file=sys.stderr)

    # Phase 2: rocprof sequentially (single GPU), reusing the pre-compiled HSACOs,
    # ITERATIONS launches each.
    print(f"\nPhase 2: rocprof (reusing HSACOs, {args.iterations} iters) ...")
    results = []
    for idx, cfg in enumerate(configs, 1):
        label = cfg["label"]
        hp, crc, cout = compiled[cfg["rank"]]
        script_rel = args.bench_script or _script_rel_to_test(cfg["script_path"] or "")
        entry = dict(
            rank=cfg["rank"],
            tier=cfg["tier"],
            label=label,
            perf_tf_s=cfg["tflops"],
            perf_pct_peak=cfg["pct_peak"],
            axes=cfg["axes"],
            ilp_knobs=cfg["ilp_knobs"],
            mcpu=mcpu,
            repro_cmd=cfg["repro_cmd"],
            script_rel=script_rel,
            hsaco=hp,
        )
        if hp is None:
            print(f"\n[{idx}/{len(configs)}] #{cfg['rank']} COMPILE FAILED  {label}", file=sys.stderr)
            entry["returncode"] = crc
            entry["note"] = "compile failed; not profiled (see stdout_tail)"
            entry["stdout_tail"] = "\n".join(cout.splitlines()[-40:])
            results.append(entry)
            continue
        print(f"\n[{idx}/{len(configs)}] #{cfg['rank']} {cfg['tflops']:.1f} TF/s  {label}")
        trace_dir, _hs, stdout, rc, pretty = _run_rocprof(script_rel, label, mcpu, args.iterations, hp, args.dry_run)
        print(f"    cmd: {pretty}")
        entry["rocprof_cmd"] = pretty
        entry["trace_dir"] = trace_dir
        entry["returncode"] = rc
        if args.dry_run:
            results.append(entry)
            continue
        if rc != 0:
            print(f"    rocprof FAILED (rc={rc}); captured output:", file=sys.stderr)
            print(stdout, file=sys.stderr)
            entry["stdout_tail"] = "\n".join(stdout.splitlines()[-40:])
            results.append(entry)
            continue
        code_jsons = _find_code_jsons(trace_dir)
        entry["code_json_paths"] = code_jsons
        if not code_jsons:
            entry["note"] = (
                "no code.json -- ATT trace decoder library likely not on "
                "LD_LIBRARY_PATH (rocprof-kittens-bench.sh does not pass "
                "--att-library-path). Raw .att files remain in trace_dir."
            )
            print(f"    WARN: no code.json under {trace_dir}", file=sys.stderr)
            results.append(entry)
            continue
        with open(code_jsons[0]) as f:
            code_obj = json.load(f)
        entry["code_json"] = code_obj
        entry["derived"] = _derive_bubbles(code_obj)
        d = entry["derived"]
        print(f"    code.json: {code_jsons[0]}")
        print(
            f"    bubble_pct={d.get('bubble_pct')} max_mfma_run={d.get('max_mfma_run')} "
            f"mean_inter_mfma_gap={d.get('mean_inter_mfma_gap')}"
        )
        results.append(entry)

    aggregate = dict(meta=meta, generated=ts, iterations=args.iterations, configs=results)
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    if args.dry_run:
        print(f"\ndry-run: {len(results)} configs (commands above). Would write: {out_path}")
        return 0
    nfail = sum(1 for r in results if r.get("returncode", 0) != 0)
    nnocode = sum(1 for r in results if r.get("returncode", 0) == 0 and not r.get("code_json"))
    print(f"\nSummary: {len(results)} configs, {nfail} failed, {nnocode} no-code.json")
    for r in results:
        st = "FAILED" if r.get("returncode", 0) != 0 else ("ok" if r.get("code_json") else "no-code.json")
        print(f"  #{r['rank']:>2} [{st:<12}] {r['label']}")
    print(f"\nDetails + errors (per-config stdout_tail): {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
