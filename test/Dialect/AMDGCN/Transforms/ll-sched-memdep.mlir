// RUN: aster-opt %s --pass-pipeline="builtin.module(test-amdgcn-sched-graph)" 2>&1 | FileCheck %s

// Memory-dependence edges in the per-block SchedGraph: proper tracking of
// RAW/WAR/WAW.

amdgcn.module @same_buffer_mod target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @same_buffer
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "ds_write_b32 -> ds_read_b32"
  // CHECK:       }
  amdgcn.kernel @same_buffer {
    %data = amdgcn.alloca : !amdgcn.vgpr
    %dst = amdgcn.alloca : !amdgcn.vgpr
    %perthread = amdgcn.alloca : !amdgcn.vgpr
    %bufA = amdgcn.alloc_lds 256 alignment 16
    %offA = amdgcn.get_lds_offset %bufA : i32
    %tokA = amdgcn.ds_write_b32 data %data addr %perthread offset c(%offA)
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    %valA, %tokB = amdgcn.ds_read_b32 dest %dst addr %perthread offset c(%offA)
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @two_buffers_mod target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @two_buffers
  // CHECK:       digraph SchedGraph
  // CHECK-NOT:     label = "ds_write_b32 -> ds_read_b32"
  // CHECK:       }
  amdgcn.kernel @two_buffers {
    %data = amdgcn.alloca : !amdgcn.vgpr
    %dst = amdgcn.alloca : !amdgcn.vgpr
    %perthread = amdgcn.alloca : !amdgcn.vgpr
    %bufA = amdgcn.alloc_lds 256 alignment 16
    %bufB = amdgcn.alloc_lds 256 alignment 16
    %offA = amdgcn.get_lds_offset %bufA : i32
    %offB = amdgcn.get_lds_offset %bufB : i32
    %tokA = amdgcn.ds_write_b32 data %data addr %perthread offset c(%offA)
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    %valB, %tokB = amdgcn.ds_read_b32 dest %dst addr %perthread offset c(%offB)
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @war_mod target = #amdgcn.target<gfx942> {
  // WAR: a read then a write to the same buffer must stay ordered.
  // CHECK-LABEL: Kernel: @war_same_buffer
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "ds_read_b32 -> ds_write_b32"
  // CHECK:       }
  amdgcn.kernel @war_same_buffer {
    %data = amdgcn.alloca : !amdgcn.vgpr
    %dst = amdgcn.alloca : !amdgcn.vgpr
    %perthread = amdgcn.alloca : !amdgcn.vgpr
    %bufA = amdgcn.alloc_lds 256 alignment 16
    %offA = amdgcn.get_lds_offset %bufA : i32
    %valA, %tokR = amdgcn.ds_read_b32 dest %dst addr %perthread offset c(%offA)
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    %tokW = amdgcn.ds_write_b32 data %data addr %perthread offset c(%offA)
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @waw_mod target = #amdgcn.target<gfx942> {
  // WAW: two writes to the same buffer must stay ordered.
  // CHECK-LABEL: Kernel: @waw_same_buffer
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "ds_write_b32 -> ds_write_b32"
  // CHECK:       }
  amdgcn.kernel @waw_same_buffer {
    %data0 = amdgcn.alloca : !amdgcn.vgpr
    %data1 = amdgcn.alloca : !amdgcn.vgpr
    %perthread = amdgcn.alloca : !amdgcn.vgpr
    %bufA = amdgcn.alloc_lds 256 alignment 16
    %offA = amdgcn.get_lds_offset %bufA : i32
    %tokW0 = amdgcn.ds_write_b32 data %data0 addr %perthread offset c(%offA)
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    %tokW1 = amdgcn.ds_write_b32 data %data1 addr %perthread offset c(%offA)
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    amdgcn.end_kernel
  }
}

// -----

// Non-rotating 4-buffer GEMM double-buffer (A0,A1,B0,B1).

amdgcn.module @gemm_buffers_no_rotate_mod target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @gemm_buffers_no_rotate Block: 1
  // CHECK:       digraph SchedGraph
  // wA feeds both A sub-tile reads; wB feeds both B sub-tile reads (RAW).
  // CHECK-DAG:     label = "ds_write_b32/test.wA -> ds_read_b32/test.rA0"
  // CHECK-DAG:     label = "ds_write_b32/test.wA -> ds_read_b32/test.rA1"
  // CHECK-DAG:     label = "ds_write_b32/test.wB -> ds_read_b32/test.rB0"
  // CHECK-DAG:     label = "ds_write_b32/test.wB -> ds_read_b32/test.rB1"
  // Distinct A/B alloc_lds: no cross-operand RAW and no wA/wB WAW.
  // CHECK-NOT:     label = "ds_write_b32/test.wA -> ds_read_b32/test.rB0"
  // CHECK-NOT:     label = "ds_write_b32/test.wA -> ds_read_b32/test.rB1"
  // CHECK-NOT:     label = "ds_write_b32/test.wB -> ds_read_b32/test.rA0"
  // CHECK-NOT:     label = "ds_write_b32/test.wB -> ds_read_b32/test.rA1"
  // CHECK-NOT:     label = "ds_write_b32/test.wA -> ds_write_b32/test.wB"
  // CHECK:       }
  amdgcn.kernel @gemm_buffers_no_rotate {
    %dataA  = amdgcn.alloca : !amdgcn.vgpr
    %dataB  = amdgcn.alloca : !amdgcn.vgpr
    %dstA0  = amdgcn.alloca : !amdgcn.vgpr
    %dstA1  = amdgcn.alloca : !amdgcn.vgpr
    %dstB0  = amdgcn.alloca : !amdgcn.vgpr
    %dstB1  = amdgcn.alloca : !amdgcn.vgpr
    %pt     = amdgcn.alloca : !amdgcn.vgpr
    %cond   = arith.constant 1 : i1
    // Four distinct LDS buffers: A tile double-buffer, B tile double-buffer.
    %ldsA0  = amdgcn.alloc_lds 1024 offset 0
    %ldsA1  = amdgcn.alloc_lds 1024 offset 1024
    %ldsB0  = amdgcn.alloc_lds 1024 offset 2048
    %ldsB1  = amdgcn.alloc_lds 1024 offset 3072
    %offA0  = amdgcn.get_lds_offset %ldsA0 : i32
    %offA1  = amdgcn.get_lds_offset %ldsA1 : i32
    %offB0  = amdgcn.get_lds_offset %ldsB0 : i32
    %offB1  = amdgcn.get_lds_offset %ldsB1 : i32
    // Non-rotating: pass offsets as iter_args and yield them unchanged.
    cf.br ^body(%offA0, %offA1, %offB0, %offB1 : i32, i32, i32, i32)

  ^body(%curA: i32, %prevA: i32, %curB: i32, %prevB: i32):
    // Write A tile to the current A buffer (traces precisely to ldsA0).
    %tokWA = amdgcn.ds_write_b32 data %dataA addr %pt offset c(%curA)
      {test.wA} : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    // Write B tile to the current B buffer (traces precisely to ldsB0).
    %tokWB = amdgcn.ds_write_b32 data %dataB addr %pt offset c(%curB)
      {test.wB} : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    // Read A sub-tile 0 from the same current A buffer.
    %valA0, %tokRA0 = amdgcn.ds_read_b32 dest %dstA0 addr %pt offset c(%curA)
      {test.rA0} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    // Read A sub-tile 1 from the same current A buffer.
    %valA1, %tokRA1 = amdgcn.ds_read_b32 dest %dstA1 addr %pt offset c(%curA)
      {test.rA1} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    // Read B sub-tile 0 from the same current B buffer.
    %valB0, %tokRB0 = amdgcn.ds_read_b32 dest %dstB0 addr %pt offset c(%curB)
      {test.rB0} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    // Read B sub-tile 1 from the same current B buffer.
    %valB1, %tokRB1 = amdgcn.ds_read_b32 dest %dstB1 addr %pt offset c(%curB)
      {test.rB1} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    // Yield unchanged: non-rotating double-buffer.
    cf.cond_br %cond, ^body(%curA, %prevA, %curB, %prevB : i32, i32, i32, i32), ^exit

  ^exit:
    amdgcn.end_kernel
  }
}

// -----

// Rotating 4-buffer GEMM double-buffer (A0,A1,B0,B1) defeats the current
// disambiguation logic.
// TODO: improve when needed.

amdgcn.module @gemm_buffers_rotate_mod target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @gemm_buffers_rotate Block: 1
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "ds_write_b32/test.wA -> ds_write_b32/test.wB"
  // CHECK-DAG:     label = "ds_write_b32/test.wA -> ds_read_b32/test.rA0"
  // CHECK-DAG:     label = "ds_write_b32/test.wA -> ds_read_b32/test.rA1"
  // CHECK-DAG:     label = "ds_write_b32/test.wA -> ds_read_b32/test.rB0"
  // CHECK-DAG:     label = "ds_write_b32/test.wA -> ds_read_b32/test.rB1"
  // CHECK-DAG:     label = "ds_write_b32/test.wB -> ds_read_b32/test.rA0"
  // CHECK-DAG:     label = "ds_write_b32/test.wB -> ds_read_b32/test.rA1"
  // CHECK-DAG:     label = "ds_write_b32/test.wB -> ds_read_b32/test.rB0"
  // CHECK-DAG:     label = "ds_write_b32/test.wB -> ds_read_b32/test.rB1"
  // CHECK:       }
  amdgcn.kernel @gemm_buffers_rotate {
    %dataA  = amdgcn.alloca : !amdgcn.vgpr
    %dataB  = amdgcn.alloca : !amdgcn.vgpr
    %dstA0  = amdgcn.alloca : !amdgcn.vgpr
    %dstA1  = amdgcn.alloca : !amdgcn.vgpr
    %dstB0  = amdgcn.alloca : !amdgcn.vgpr
    %dstB1  = amdgcn.alloca : !amdgcn.vgpr
    %pt     = amdgcn.alloca : !amdgcn.vgpr
    %cond   = arith.constant 1 : i1
    // Four distinct LDS buffers: A tile double-buffer, B tile double-buffer.
    %ldsA0  = amdgcn.alloc_lds 1024 offset 0
    %ldsA1  = amdgcn.alloc_lds 1024 offset 1024
    %ldsB0  = amdgcn.alloc_lds 1024 offset 2048
    %ldsB1  = amdgcn.alloc_lds 1024 offset 3072
    %offA0  = amdgcn.get_lds_offset %ldsA0 : i32
    %offA1  = amdgcn.get_lds_offset %ldsA1 : i32
    %offB0  = amdgcn.get_lds_offset %ldsB0 : i32
    %offB1  = amdgcn.get_lds_offset %ldsB1 : i32
    // Rotating: initial cur = A1/B1, prev = A0/B0.
    // On the back edge %prevA<->%curA and %prevB<->%curB swap each cycle.
    cf.br ^body(%offA1, %offA0, %offB1, %offB0 : i32, i32, i32, i32)

  ^body(%curA: i32, %prevA: i32, %curB: i32, %prevB: i32):
    // Write A tile to the current A buffer (rotates between ldsA0 and ldsA1).
    %tokWA = amdgcn.ds_write_b32 data %dataA addr %pt offset c(%curA)
      {test.wA} : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    // Write B tile to the current B buffer (rotates between ldsB0 and ldsB1).
    %tokWB = amdgcn.ds_write_b32 data %dataB addr %pt offset c(%curB)
      {test.wB} : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    // Read A sub-tile 0 from the previous A buffer (written last iteration).
    %valA0, %tokRA0 = amdgcn.ds_read_b32 dest %dstA0 addr %pt offset c(%prevA)
      {test.rA0} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    // Read A sub-tile 1 from the previous A buffer.
    %valA1, %tokRA1 = amdgcn.ds_read_b32 dest %dstA1 addr %pt offset c(%prevA)
      {test.rA1} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    // Read B sub-tile 0 from the previous B buffer (written last iteration).
    %valB0, %tokRB0 = amdgcn.ds_read_b32 dest %dstB0 addr %pt offset c(%prevB)
      {test.rB0} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    // Read B sub-tile 1 from the previous B buffer.
    %valB1, %tokRB1 = amdgcn.ds_read_b32 dest %dstB1 addr %pt offset c(%prevB)
      {test.rB1} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    // Rotate: swap cur<->prev for both A and B on the back edge.
    cf.cond_br %cond, ^body(%prevA, %curA, %prevB, %curB : i32, i32, i32, i32), ^exit

  ^exit:
    amdgcn.end_kernel
  }
}
