// RUN: aster-opt %s --pass-pipeline="builtin.module(test-amdgcn-sched-graph)" 2>&1 | FileCheck %s

!v = !amdgcn.vgpr

amdgcn.module @same_buffer target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @same_buffer
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "ds_write_b32 -> ds_read_b32"
  // CHECK:       }
  amdgcn.kernel @same_buffer {
    %data      = amdgcn.alloca : !v
    %dst       = amdgcn.alloca : !v
    %perthread = amdgcn.alloca : !v
    %bufA      = amdgcn.alloc_lds 256 alignment 16
    %offA      = amdgcn.get_lds_offset %bufA : i32
    %tokA = amdgcn.ds_write_b32 data %data addr %perthread offset c(%offA)
      : ins(!v, !v) mods(i32) -> !amdgcn.write_token<shared>
    %valA, %tokB = amdgcn.ds_read_b32 dest %dst addr %perthread offset c(%offA)
      : outs(!v) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @two_buffers target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @two_buffers
  // CHECK:       digraph SchedGraph
  // CHECK-NOT:     label = "ds_write_b32 -> ds_read_b32"
  // CHECK:       }
  amdgcn.kernel @two_buffers {
    %data      = amdgcn.alloca : !v
    %dst       = amdgcn.alloca : !v
    %perthread = amdgcn.alloca : !v
    %bufA      = amdgcn.alloc_lds 256 alignment 16
    %bufB      = amdgcn.alloc_lds 256 alignment 16
    %offA      = amdgcn.get_lds_offset %bufA : i32
    %offB      = amdgcn.get_lds_offset %bufB : i32
    %tokA = amdgcn.ds_write_b32 data %data addr %perthread offset c(%offA)
      : ins(!v, !v) mods(i32) -> !amdgcn.write_token<shared>
    %valB, %tokB = amdgcn.ds_read_b32 dest %dst addr %perthread offset c(%offB)
      : outs(!v) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @selective_flush target = #amdgcn.target<gfx942> {
  // Exactly one edge: write_a -> read_a; write_b (distinct buffer) gets none.
  // CHECK-LABEL: Kernel: @selective_flush
  // CHECK:         digraph SchedGraph
  // CHECK-COUNT-1: label = "ds_write_b32 -> ds_read_b32"
  // CHECK-NOT:     label = "ds_write_b32 -> ds_read_b32"
  // CHECK:         }
  amdgcn.kernel @selective_flush {
    %dataA     = amdgcn.alloca : !v
    %dataB     = amdgcn.alloca : !v
    %dst       = amdgcn.alloca : !v
    %perthread = amdgcn.alloca : !v
    %bufA      = amdgcn.alloc_lds 256 alignment 16
    %bufB      = amdgcn.alloc_lds 256 alignment 16
    %offA      = amdgcn.get_lds_offset %bufA : i32
    %offB      = amdgcn.get_lds_offset %bufB : i32
    %tokA = amdgcn.ds_write_b32 data %dataA addr %perthread offset c(%offA)
      : ins(!v, !v) mods(i32) -> !amdgcn.write_token<shared>
    %tokB = amdgcn.ds_write_b32 data %dataB addr %perthread offset c(%offB)
      : ins(!v, !v) mods(i32) -> !amdgcn.write_token<shared>
    %valA, %tokR = amdgcn.ds_read_b32 dest %dst addr %perthread offset c(%offA)
      : outs(!v) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @transitive_address target = #amdgcn.target<gfx942> {
  // read addr reaches bufB through arith.addi; still distinct from bufA: no edge.
  // CHECK-LABEL: Kernel: @transitive_address
  // CHECK:       digraph SchedGraph
  // CHECK-NOT:     label = "ds_write_b32 -> ds_read_b32"
  // CHECK:       }
  amdgcn.kernel @transitive_address {
    %dataA     = amdgcn.alloca : !v
    %dst       = amdgcn.alloca : !v
    %perthread = amdgcn.alloca : !v
    %bufA      = amdgcn.alloc_lds 256 alignment 16
    %bufB      = amdgcn.alloc_lds 256 alignment 16
    %offA      = amdgcn.get_lds_offset %bufA : i32
    %offB      = amdgcn.get_lds_offset %bufB : i32
    %c0        = arith.constant 0 : i32
    %adjB      = arith.addi %offB, %c0 : i32
    %tokA = amdgcn.ds_write_b32 data %dataA addr %perthread offset c(%offA)
      : ins(!v, !v) mods(i32) -> !amdgcn.write_token<shared>
    %valB, %tokR = amdgcn.ds_read_b32 dest %dst addr %perthread offset c(%adjB)
      : outs(!v) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }
}
