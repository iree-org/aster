// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-low-level-scheduler{preset=1},amdgcn-lds-alloc,amdgcn-convert-lds-buffers)))" | FileCheck %s

// LDS resource allocation runs after the low-level scheduler: the scheduler sees
// live alloc_lds/get_lds_offset handles, then amdgcn-lds-alloc assigns byte
// offsets and amdgcn-convert-lds-buffers folds get_lds_offset to constants.

!v = !amdgcn.vgpr

amdgcn.module @lds_alloc_after_sched target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: kernel @lds_alloc_after_sched
  // CHECK-SAME:    shared_memory_size = 512
  // CHECK-DAG:     arith.constant 0 : i32
  // CHECK-DAG:     arith.constant 256 : i32
  // CHECK-NOT:     alloc_lds
  // CHECK-NOT:     get_lds_offset
  amdgcn.kernel @lds_alloc_after_sched {
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
