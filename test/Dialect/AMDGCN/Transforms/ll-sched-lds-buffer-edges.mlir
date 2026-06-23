// RUN: aster-opt %s --pass-pipeline="builtin.module(test-amdgcn-sched-graph)" 2>&1 | FileCheck %s

!v = !amdgcn.vgpr

amdgcn.module @lds_buffer_lifetime target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @lds_buffer_lifetime
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "get_lds_offset -> dealloc_lds"
  // CHECK-DAG:     label = "ds_write_b32 -> dealloc_lds"
  // CHECK:       }
  amdgcn.kernel @lds_buffer_lifetime {
    %data      = amdgcn.alloca : !v
    %perthread = amdgcn.alloca : !v
    %buf       = amdgcn.alloc_lds 256 alignment 16
    %off       = amdgcn.get_lds_offset %buf : i32
    %tok = amdgcn.ds_write_b32 data %data addr %perthread offset c(%off)
      : ins(!v, !v) mods(i32) -> !amdgcn.write_token<shared>
    amdgcn.dealloc_lds %buf
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @two_lds_buffers target = #amdgcn.target<gfx942> {
  // Each buffer only pins its own offset users before its dealloc.
  // CHECK-LABEL: Kernel: @two_lds_buffers
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "get_lds_offset/test.bufA -> dealloc_lds/test.bufA"
  // CHECK-DAG:     label = "ds_write_b32/test.writeA -> dealloc_lds/test.bufA"
  // CHECK-DAG:     label = "get_lds_offset/test.bufB -> dealloc_lds/test.bufB"
  // CHECK-DAG:     label = "ds_read_b32/test.readB -> dealloc_lds/test.bufB"
  // CHECK-NOT:     label = "ds_write_b32/test.writeA -> dealloc_lds/test.bufB"
  // CHECK-NOT:     label = "ds_read_b32/test.readB -> dealloc_lds/test.bufA"
  // CHECK:       }
  amdgcn.kernel @two_lds_buffers {
    %dataA     = amdgcn.alloca : !v
    %dataB     = amdgcn.alloca : !v
    %dst       = amdgcn.alloca : !v
    %perthread = amdgcn.alloca : !v
    %bufA      = amdgcn.alloc_lds 256 alignment 16 {test.bufA}
    %bufB      = amdgcn.alloc_lds 256 alignment 16 {test.bufB}
    %offA      = amdgcn.get_lds_offset %bufA {test.bufA} : i32
    %offB      = amdgcn.get_lds_offset %bufB {test.bufB} : i32
    %tokA = amdgcn.ds_write_b32 data %dataA addr %perthread offset c(%offA)
      {test.writeA} : ins(!v, !v) mods(i32) -> !amdgcn.write_token<shared>
    %valB, %tokB = amdgcn.ds_read_b32 dest %dst addr %perthread offset c(%offB)
      {test.readB} : outs(!v) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.dealloc_lds %bufA {test.bufA}
    amdgcn.dealloc_lds %bufB {test.bufB}
    amdgcn.end_kernel
  }
}
