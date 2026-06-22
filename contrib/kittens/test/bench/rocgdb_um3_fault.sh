#!/usr/bin/env bash
# Catch the um3 ilpwin8 GPU memory fault precisely under rocgdb and print the
# faulting wave PC, the disassembly around it, the address-base SGPRs, and the
# backtrace. The kernel runs IN-PROCESS (rocgdb must trace the process that owns
# the GPU queue), so this does NOT go through run_um3_schedule_test.py (which
# spawns subprocesses rocgdb would not follow).
#
# Run from inside the aster checkout with the GPU venv active:
#   bash contrib/kittens/test/bench/rocgdb_um3_fault.sh              # gfx942, window 8
#   bash contrib/kittens/test/bench/rocgdb_um3_fault.sh gfx950       # gfx950, window 8
#   bash contrib/kittens/test/bench/rocgdb_um3_fault.sh gfx942 16    # a passing control
set -u
ARCH="${1:-gfx942}"
WIN="${2:-8}"

# Find the repo root (the dir that contains contrib/kittens/test/bench).
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$DIR"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/contrib/kittens/test/bench" ]; do
  ROOT="$(dirname "$ROOT")"
done
if [ ! -d "$ROOT/contrib/kittens/test/bench" ]; then
  echo "ERROR: run from inside the aster checkout (cannot find contrib/kittens/test/bench)"
  exit 1
fi
cd "$ROOT"

DRIVER="$(mktemp /tmp/um3_rocgdb_driver_XXXXXX.py)"
trap 'rm -f "$DRIVER"' EXIT

# The in-process driver. Written to a real file so there is no inline-indent issue.
cat > "$DRIVER" <<'PYEOF'
import os
import sys

sys.path[:0] = ["contrib", "contrib/kittens/test", "contrib/kittens/test/bench"]

ARCHS = {
    "gfx942": (
        "bench_perf_102_gemm_python_multitile_directb_cdna3",
        "test_102_gemm_python_multitile_directb_cdna3",
        "_run_multitile",
        "m2432xn4096xk4096_wg19x16x1_w4x1x1_twg8x16x2_pipestrat1_wgcu1_lcm1_um3"
        "_epeel1_llilpsched2_mfmagap4_lgkmgap0_{W}hoistwait0_rotc1_lt_flat",
    ),
    "gfx950": (
        "bench_perf_102_gemm_python_multitile_g2s_directb_cdna4",
        "test_102_gemm_python_multitile_g2s_directb_cdna4",
        "_run_cdna4_gemm",
        "m4096xn4096xk4096_wg16x16x1_w2x2x1_twg16x16x1_pipestrat1_wgcu1_lcm1_um3"
        "_epeel1_llilpsched2_mfmagap4_lgkmgap0_{W}hoistwait0_rotc1_lt_flat_cdna4",
    ),
}

arch = os.environ.get("UM3_ARCH", "gfx942")
win = int(os.environ.get("UM3_WIN", "8"))
benchmod, testmod, runfn, base = ARCHS[arch]
wins = "" if win == 32 else "ilpwin%d_" % win
label = base.format(W=wins)
bench = __import__(benchmod)
test = __import__(testmod)
print("[driver] arch=%s win=%d label=%s" % (arch, win, label), flush=True)
getattr(test, runfn)(bench._from_label(label, arch))
print("[driver] RUN_OK (no fault this run)", flush=True)
PYEOF

echo "=== rocgdb precise-memory fault probe: arch=$ARCH window=$WIN root=$ROOT ==="
echo "    (window 8 should fault; 16/32 are passing controls)"
UM3_ARCH="$ARCH" UM3_WIN="$WIN" rocgdb -batch \
  -ex 'set pagination off' \
  -ex 'set confirm off' \
  -ex 'set amdgpu precise-memory on' \
  -ex 'run' \
  -ex 'echo \n==== FAULTING WAVE: PC + EXEC ====\n' \
  -ex 'info registers pc exec' \
  -ex 'echo \n==== ADDRESS-BASE SGPRs (s[0:1]=C out, s[4:5]=A, s[6:7]=B) ====\n' \
  -ex 'info registers s0 s1 s4 s5 s6 s7' \
  -ex 'echo \n==== DISASM AROUND FAULTING PC ====\n' \
  -ex 'x/10i $pc-28' \
  -ex 'echo \n==== OFFSET CHAIN (scalar): s8=s10+2  s12=s8<<11  s10=loop ctr ====\n' \
  -ex 'info registers s8 s10 s12' \
  -ex 'echo \n==== FAULTING OFFSET VGPRs (per-lane): v2 is the faulting load offset; v1/v3 siblings ====\n' \
  -ex 'info registers v1 v2 v3' \
  -ex 'echo \n==== LANE-OFFSET BASES feeding the offsets: v185 v186 (= 1966080/1967104 + v155) ====\n' \
  -ex 'info registers v185 v186 v155' \
  -ex 'echo \n==== ALL THREAD / WAVE BACKTRACES ====\n' \
  -ex 'thread apply all backtrace' \
  --args python "$DRIVER"
