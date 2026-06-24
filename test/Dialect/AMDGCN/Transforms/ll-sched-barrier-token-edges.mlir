// RUN: aster-opt %s --pass-pipeline="builtin.module(test-amdgcn-sched-graph)" 2>&1 | FileCheck %s

!v = !amdgcn.vgpr
!s = !amdgcn.sgpr

amdgcn.module @token_barrier_deps target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @token_barrier_deps
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "ds_write_b32 -> token_barrier"
  // CHECK-NOT:     label = "token_barrier -> ds_read_b32"
  // CHECK-NOT:     label = "token_barrier -> s_mov_b32"
  // CHECK-NOT:     label = "s_mov_b32 -> token_barrier"
  // CHECK:       }
  amdgcn.kernel @token_barrier_deps {
    %addr = amdgcn.alloca : !v
    %data = amdgcn.alloca : !v
    %rd = amdgcn.alloca : !v
    %s0 = amdgcn.alloca : !s
    %c0 = arith.constant 0 : i32
    %c7 = arith.constant 7 : i32
    %tok = amdgcn.ds_write_b32 data %data addr %addr offset c(%c0) : ins(!v, !v) mods(i32) -> !amdgcn.write_token<shared>
    %bar = amdgcn.token_barrier scope(<workgroup>) deps %tok : !amdgcn.write_token<shared>
    %r, %t = amdgcn.ds_read_b32 dest %rd addr %addr offset c(%c0) : outs(!v) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %m = amdgcn.s_mov_b32 outs(%s0) ins(%c7) : outs(!s) ins(i32)
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @token_barrier_after target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @token_barrier_after
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "ds_write_b32 -> token_barrier"
  // CHECK-DAG:     label = "token_barrier -> ds_read_b32"
  // CHECK-NOT:     label = "token_barrier -> s_mov_b32"
  // CHECK:       }
  amdgcn.kernel @token_barrier_after {
    %addr = amdgcn.alloca : !v
    %data = amdgcn.alloca : !v
    %rd = amdgcn.alloca : !v
    %s0 = amdgcn.alloca : !s
    %c0 = arith.constant 0 : i32
    %c7 = arith.constant 7 : i32
    %tok = amdgcn.ds_write_b32 data %data addr %addr offset c(%c0) : ins(!v, !v) mods(i32) -> !amdgcn.write_token<shared>
    %bar = amdgcn.token_barrier scope(<workgroup>) deps %tok : !amdgcn.write_token<shared>
    %r, %t = amdgcn.ds_read_b32 dest %rd addr %addr offset c(%c0) : outs(!v) ins(!v) mods(i32) -> !amdgcn.read_token<shared> fence_token %bar : !amdgcn.fence_token
    %m = amdgcn.s_mov_b32 outs(%s0) ins(%c7) : outs(!s) ins(i32)
    amdgcn.end_kernel
  }
}

// -----

// Plain barrier conservatively pins cross-thread memory ops.
amdgcn.module @plain_barrier target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @plain_barrier
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "ds_write_b32 -> barrier"
  // CHECK-DAG:     label = "barrier -> ds_read_b32"
  // CHECK-NOT:     label = "barrier -> s_mov_b32"
  // CHECK:       }
  amdgcn.kernel @plain_barrier {
    %addr = amdgcn.alloca : !v
    %data = amdgcn.alloca : !v
    %rd = amdgcn.alloca : !v
    %s0 = amdgcn.alloca : !s
    %c0 = arith.constant 0 : i32
    %c7 = arith.constant 7 : i32
    %tok = amdgcn.ds_write_b32 data %data addr %addr offset c(%c0) : ins(!v, !v) mods(i32) -> !amdgcn.write_token<shared>
    amdgcn.barrier scope(<workgroup>)
    %r, %t = amdgcn.ds_read_b32 dest %rd addr %addr offset c(%c0) : outs(!v) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %m = amdgcn.s_mov_b32 outs(%s0) ins(%c7) : outs(!s) ins(i32)
    amdgcn.end_kernel
  }
}

// -----

// BarrierOpInterface ops are serialized in program order.
amdgcn.module @barrier_serialization target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @barrier_serialization
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "barrier -> barrier"
  // CHECK-DAG:     label = "barrier -> token_barrier"
  // CHECK:       }
  amdgcn.kernel @barrier_serialization {
    %addr = amdgcn.alloca : !v
    %data = amdgcn.alloca : !v
    %c0 = arith.constant 0 : i32
    %tok = amdgcn.ds_write_b32 data %data addr %addr offset c(%c0) : ins(!v, !v) mods(i32) -> !amdgcn.write_token<shared>
    amdgcn.barrier scope(<workgroup>)
    amdgcn.barrier scope(<workgroup>)
    %bar = amdgcn.token_barrier scope(<workgroup>) deps %tok : !amdgcn.write_token<shared>
    amdgcn.end_kernel
  }
}

// -----

// SSA-managed successors: only the ds_read that takes the fence token via
// `fence_token` is pinned after the barrier.
amdgcn.module @token_barrier_acquire_precise target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @token_barrier_acquire_precise
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "token_barrier -> ds_read_b32"
  // CHECK-NOT:     label = "token_barrier -> ds_read_b64"
  // CHECK:       }
  amdgcn.kernel @token_barrier_acquire_precise {
    %addr = amdgcn.alloca : !v
    %addr2 = amdgcn.alloca : !v
    %data = amdgcn.alloca : !v
    %rd32 = amdgcn.alloca : !v
    %rd0 = amdgcn.alloca : !v
    %rd1 = amdgcn.alloca : !v
    %rd64 = amdgcn.make_register_range %rd0, %rd1 : !v, !v
    %c0 = arith.constant 0 : i32
    %wtok = amdgcn.ds_write_b32 data %data addr %addr offset c(%c0) : ins(!v, !v) mods(i32) -> !amdgcn.write_token<shared>
    %bar = amdgcn.token_barrier scope(<workgroup>) deps %wtok : !amdgcn.write_token<shared>
    %r32, %t32 = amdgcn.ds_read_b32 dest %rd32 addr %addr offset c(%c0) : outs(!v) ins(!v) mods(i32) -> !amdgcn.read_token<shared> fence_token %bar : !amdgcn.fence_token
    %r64, %t64 = amdgcn.ds_read_b64 dest %rd64 addr %addr2 offset c(%c0) : outs(!amdgcn.vgpr<[? + 2]>) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }
}
