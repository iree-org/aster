#!/usr/bin/env bash
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Quick smoke test for the perf_*.py tools.
#
# Exercises CLI parsing, JSONL parsing, registry update, and dashboard
# comparison on synthetic fixtures. Pure Python -- no GPU, no compilation,
# no built worktree required. Exits non-zero on the first failure.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

step() { printf '\n=== %s ===\n' "$*"; }

step "--help on each tool (verifies argparse composition)"
"$PY" "$HERE/perf_best_known_update.py" --help >/dev/null
"$PY" "$HERE/perf_dashboard.py"          --help >/dev/null
"$PY" "$HERE/perf_evaluate.py"           --help >/dev/null
"$PY" "$HERE/perf_explore.py"            --help >/dev/null

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

BK="$TMP/best_known.json"
SWEEP="$TMP/sweep.jsonl"
DASH="$TMP/dash.jsonl"

step "Synthesize fixtures in $TMP"
cat > "$BK" <<'EOF'
{
  "gfx942": {
    "bench_perf_001_gemm_fp16_weak_scaled": {
      "4096x4096x4096": [
        {"label": "m4096xn4096xk4096_old_winner", "expected_tflops": 900.0}
      ]
    }
  }
}
EOF

cat > "$SWEEP" <<'EOF'
BENCH_RESULT_JSON:{"mcpu":"gfx942","bench":"bench_perf_001_gemm_fp16_weak_scaled","M":4096,"N":4096,"K":4096,"label":"m4096xn4096xk4096_new_top","tflops_median":950.0}
BENCH_RESULT_JSON:{"mcpu":"gfx942","bench":"bench_perf_001_gemm_fp16_weak_scaled","M":4096,"N":4096,"K":4096,"label":"m4096xn4096xk4096_runner","tflops_median":920.0}
EOF

step "perf_best_known_update --apply (merges 2 new labels into 1-entry slot)"
"$PY" "$HERE/perf_best_known_update.py" --input "$SWEEP" --best-known-file "$BK" --apply
"$PY" - <<EOF
import json
d = json.load(open("$BK"))
slot = d["gfx942"]["bench_perf_001_gemm_fp16_weak_scaled"]["4096x4096x4096"]
assert len(slot) == 3, slot
assert slot[0]["label"] == "m4096xn4096xk4096_new_top"   and slot[0]["expected_tflops"] == 950.0, slot[0]
assert slot[1]["label"] == "m4096xn4096xk4096_runner"    and slot[1]["expected_tflops"] == 920.0, slot[1]
assert slot[2]["label"] == "m4096xn4096xk4096_old_winner" and slot[2]["expected_tflops"] == 900.0, slot[2]
print(f"  registry slot now has {len(slot)} entries, top-1={slot[0]['label']}: OK")
EOF

step "perf_dashboard --dry-run (lists stored entries)"
"$PY" "$HERE/perf_dashboard.py" --dry-run --best-known-file "$BK" | tee "$TMP/dry.out" | tail -3
grep -q "stored entries listed" "$TMP/dry.out"

step "Build dashboard measurements JSONL covering every stored label"
"$PY" - <<EOF > "$DASH"
import json
d = json.load(open("$BK"))
for mcpu, by_bench in d.items():
    for bench, by_size in by_bench.items():
        for size_key, slot in by_size.items():
            M, N, K = (int(x) for x in size_key.split("x"))
            for e in slot:
                print("BENCH_RESULT_JSON:" + json.dumps({
                    "mcpu": mcpu, "bench": bench, "M": M, "N": N, "K": K,
                    "label": e["label"], "tflops_median": e["expected_tflops"],
                }))
EOF

step "perf_dashboard --measurements (every stored label matches -> all OK)"
"$PY" "$HERE/perf_dashboard.py" --measurements "$DASH" --best-known-file "$BK" | tee "$TMP/dash.out" | tail -3
grep -q "3 OK"     "$TMP/dash.out"
grep -q "0 REGRESS" "$TMP/dash.out"

step "perf_dashboard with one REGRESS measurement (exit code 1)"
sed 's/950.0/700.0/' "$DASH" > "$TMP/dash_regress.jsonl"
set +e
"$PY" "$HERE/perf_dashboard.py" --measurements "$TMP/dash_regress.jsonl" --best-known-file "$BK" >/dev/null
rc=$?
set -e
if [ "$rc" -ne 1 ]; then
  echo "expected exit 1 (REGRESS), got $rc" >&2
  exit 1
fi
echo "  exit code = 1: OK"

echo
echo "SMOKE CHECK OK"
