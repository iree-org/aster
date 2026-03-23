// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-low-level-scheduler{dump-dag=true})))" --split-input-file 2>&1 | FileCheck %s --check-prefix=CHECK --dump-input=fail

// Tests for the low-level scheduler's dependency DAG construction.
// Each test verifies specific edge types independently of the scheduling
// heuristic. The dump-dag option prints the DAG without reordering.

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>

// SSA def-use: vop1 result used by vop2 creates an edge.
// CHECK-LABEL: DAG for kernel @ssa_def_use
// CHECK: node: %{{.*}} (amdgcn.vop1.vop1) [queue=valu
// CHECK:   -> %{{.*}} (amdgcn.vop2)
amdgcn.module @ssa_def_use target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  kernel @ssa_def_use {
    %v0 = amdgcn.alloca : !v
    %v1 = amdgcn.alloca : !v
    %v2 = amdgcn.alloca : !v
    %r0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %v0, %v1 : (!v, !v) -> !v
    %r1 = amdgcn.vop2 v_add_u32 outs %v2 ins %r0, %v1 : !v, !v, !v
    amdgcn.end_kernel
  }
}

// -----

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>

// Token wait -> data consumer: wait has edge to mfma that uses loaded data.
// CHECK-LABEL: DAG for kernel @token_wait_consumer
// CHECK: node: %dest_res (amdgcn.load) [queue=lgkm
// CHECK:   -> <<amdgcn.wait>> (amdgcn.wait)
// CHECK: node: <<amdgcn.wait>> (amdgcn.wait)
// CHECK:   -> %{{.*}} (amdgcn.vop3p.vop3p_mai)
amdgcn.module @token_wait_consumer target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  kernel @token_wait_consumer {
    %lds_addr = amdgcn.alloca : !v
    %d0 = amdgcn.alloca : !v
    %d1 = amdgcn.alloca : !v
    %lds_dest = amdgcn.make_register_range %d0, %d1 : !v, !v
    %a0 = amdgcn.alloca : !v
    %a1 = amdgcn.alloca : !v
    %a_range = amdgcn.make_register_range %a0, %a1 : !v, !v
    %c0 = amdgcn.alloca : !v
    %c1 = amdgcn.alloca : !v
    %c2 = amdgcn.alloca : !v
    %c3 = amdgcn.alloca : !v
    %acc = amdgcn.make_register_range %c0, %c1, %c2, %c3 : !v, !v, !v, !v
    %e0 = amdgcn.alloca : !v
    %e1 = amdgcn.alloca : !v
    %e2 = amdgcn.alloca : !v
    %e3 = amdgcn.alloca : !v
    %dst = amdgcn.make_register_range %e0, %e1, %e2, %e3 : !v, !v, !v, !v
    %off = arith.constant 0 : i32
    %data, %tok = amdgcn.load ds_read_b64 dest %lds_dest addr %lds_addr offset c(%off)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %tok : !amdgcn.read_token<shared>
    %mfma = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst, %data, %a_range, %acc
        : !vx2, !vx2, !vx4 -> !vx4
    amdgcn.end_kernel
  }
}

// -----

!v = !amdgcn.vgpr

// Independent VALU: wait must NOT have edge to unrelated vop2.
// CHECK-LABEL: DAG for kernel @valu_independent_of_wait
// CHECK: node: <<amdgcn.wait>> (amdgcn.wait)
// CHECK-NOT:   -> %{{.*}} (amdgcn.vop2)
// CHECK: node: %vdst0_res (amdgcn.vop2) [queue=valu
amdgcn.module @valu_independent_of_wait target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  kernel @valu_independent_of_wait {
    %va = amdgcn.alloca : !v
    %vb = amdgcn.alloca : !v
    %lds_addr = amdgcn.alloca : !v
    %lds_d = amdgcn.alloca : !v
    %c0 = arith.constant 0 : i32
    %c42 = arith.constant 42 : i32
    %data, %tok = amdgcn.load ds_read_b32 dest %lds_d addr %lds_addr offset c(%c0)
        : dps(!v) ins(!v, i32) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %tok : !amdgcn.read_token<shared>
    %r0 = amdgcn.vop2 v_add_u32 outs %va ins %c42, %vb : !v, i32, !v
    amdgcn.end_kernel
  }
}

// -----

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>

// Memory-only barrier: ds_write -> s_barrier -> ds_read edges.
// CHECK-LABEL: DAG for kernel @barrier_mem_only
// CHECK: node: %{{.*}} (amdgcn.store) [queue=lgkm
// CHECK:   -> <<amdgcn.sopp.sopp>> (amdgcn.sopp.sopp)
// CHECK: node: <<amdgcn.sopp.sopp>> (amdgcn.sopp.sopp) [queue=unknown
// CHECK:   -> %{{.*}} (amdgcn.load)
amdgcn.module @barrier_mem_only target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  kernel @barrier_mem_only {
    %a0 = amdgcn.alloca : !v
    %a1 = amdgcn.alloca : !v
    %d0 = amdgcn.alloca : !v
    %d1 = amdgcn.alloca : !v
    %data0 = amdgcn.make_register_range %d0, %d1 : !v, !v
    %r0 = amdgcn.alloca : !v
    %r1 = amdgcn.alloca : !v
    %rd0 = amdgcn.make_register_range %r0, %r1 : !v, !v
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %wt0 = amdgcn.store ds_write_b64 data %data0 addr %a0 offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    amdgcn.sopp.sopp #amdgcn.inst<s_barrier>
    %rd0_res, %rt0 = amdgcn.load ds_read_b64 dest %rd0 addr %a1 offset c(%c8)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }
}

// -----

!v = !amdgcn.vgpr

// i1 serialization: select (consumer of cmpi_1) must precede cmpi_2.
// CHECK-LABEL: DAG for kernel @i1_serialization
// CHECK: node: %{{.*}} (lsir.select)
// CHECK:   -> %{{.*}} (lsir.cmpi)
amdgcn.module @i1_serialization target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  kernel @i1_serialization {
    %v0 = amdgcn.alloca : !v
    %v1 = amdgcn.alloca : !v
    %v2 = amdgcn.alloca : !v
    %v3 = amdgcn.alloca : !v
    %c0 = arith.constant 0 : i32
    %cmp1 = lsir.cmpi i32 slt %v0, %c0 : !v, i32
    %sel1 = lsir.select %v1, %cmp1, %v2, %v0 : !v, i1, !v, !v
    %cmp2 = lsir.cmpi i32 slt %v2, %c0 : !v, i32
    %sel2 = lsir.select %v3, %cmp2, %v0, %v2 : !v, i1, !v, !v
    amdgcn.end_kernel
  }
}

// -----

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>

// No per-domain chain: independent LDS ops can reorder freely.
// Two ds_writes to different addresses with no token dependency
// should NOT have an edge between them.
// CHECK-LABEL: DAG for kernel @no_domain_chain
// CHECK: node: %[[W1:.*]] (amdgcn.store) [queue=lgkm
// CHECK-NOT:   -> %{{.*}} (amdgcn.store)
// CHECK: node: %[[W2:.*]] (amdgcn.store) [queue=lgkm
amdgcn.module @no_domain_chain target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  kernel @no_domain_chain {
    %a0 = amdgcn.alloca : !v
    %a1 = amdgcn.alloca : !v
    %d0 = amdgcn.alloca : !v
    %d1 = amdgcn.alloca : !v
    %data0 = amdgcn.make_register_range %d0, %d1 : !v, !v
    %d2 = amdgcn.alloca : !v
    %d3 = amdgcn.alloca : !v
    %data1 = amdgcn.make_register_range %d2, %d3 : !v, !v
    %c0 = arith.constant 0 : i32
    %wt0 = amdgcn.store ds_write_b64 data %data0 addr %a0 offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    %wt1 = amdgcn.store ds_write_b64 data %data1 addr %a1 offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    amdgcn.end_kernel
  }
}
