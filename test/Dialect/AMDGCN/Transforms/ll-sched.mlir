// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-low-level-scheduler{debug-stalls=false})))" | FileCheck %s

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>

amdgcn.module @test target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  // Two independent cmpi+select chains must NOT be interleaved.
  // All i1 producers write to VCC/SCC, so overlapping lifetimes = clobber.
  // CHECK-LABEL: kernel @i1_serialize_cmpi_select
  // CHECK:         lsir.cmpi
  // CHECK-NEXT:    lsir.select
  // CHECK:         lsir.cmpi
  // CHECK-NEXT:    lsir.select
  // CHECK:         end_kernel
  amdgcn.kernel @i1_serialize_cmpi_select {
    %v0 = amdgcn.alloca : !v
    %v1 = amdgcn.alloca : !v
    %v2 = amdgcn.alloca : !v
    %v3 = amdgcn.alloca : !v
    %c0 = arith.constant 0 : i32
    %cmp_a = lsir.cmpi i32 slt %v0, %c0 : !v, i32
    lsir.select %v1, %cmp_a, %v2, %v0 : !v, i1, !v, !v
    %cmp_b = lsir.cmpi i32 slt %v2, %c0 : !v, i32
    lsir.select %v3, %cmp_b, %v0, %v2 : !v, i1, !v, !v
    amdgcn.end_kernel
  }

  // Three independent cmpi+select chains: verify complete serialization.
  // CHECK-LABEL: kernel @i1_serialize_three_chains
  // CHECK:         lsir.cmpi
  // CHECK-NEXT:    lsir.select
  // CHECK:         lsir.cmpi
  // CHECK-NEXT:    lsir.select
  // CHECK:         lsir.cmpi
  // CHECK-NEXT:    lsir.select
  // CHECK:         end_kernel
  amdgcn.kernel @i1_serialize_three_chains {
    %v0 = amdgcn.alloca : !v
    %v1 = amdgcn.alloca : !v
    %v2 = amdgcn.alloca : !v
    %v3 = amdgcn.alloca : !v
    %v4 = amdgcn.alloca : !v
    %v5 = amdgcn.alloca : !v
    %c0 = arith.constant 0 : i32
    %cmp_a = lsir.cmpi i32 slt %v0, %c0 : !v, i32
    lsir.select %v1, %cmp_a, %v2, %v0 : !v, i1, !v, !v
    %cmp_b = lsir.cmpi i32 slt %v2, %c0 : !v, i32
    lsir.select %v3, %cmp_b, %v0, %v2 : !v, i1, !v, !v
    %cmp_c = lsir.cmpi i32 slt %v4, %c0 : !v, i32
    lsir.select %v5, %cmp_c, %v0, %v4 : !v, i1, !v, !v
    amdgcn.end_kernel
  }

  // cmpi with fan-out: one cmpi has TWO select consumers.
  // CHECK-LABEL: kernel @i1_serialize_fanout
  // CHECK:         %[[CMP_A:.*]] = lsir.cmpi
  // CHECK:         lsir.select {{.*}}, %[[CMP_A]],
  // CHECK:         lsir.select {{.*}}, %[[CMP_A]],
  // CHECK:         %[[CMP_B:.*]] = lsir.cmpi
  // CHECK:         lsir.select {{.*}}, %[[CMP_B]],
  // CHECK:         end_kernel
  amdgcn.kernel @i1_serialize_fanout {
    %v0 = amdgcn.alloca : !v
    %v1 = amdgcn.alloca : !v
    %v2 = amdgcn.alloca : !v
    %v3 = amdgcn.alloca : !v
    %v4 = amdgcn.alloca : !v
    %v5 = amdgcn.alloca : !v
    %c0 = arith.constant 0 : i32
    %cmp_a = lsir.cmpi i32 slt %v0, %c0 : !v, i32
    lsir.select %v1, %cmp_a, %v2, %v0 : !v, i1, !v, !v
    lsir.select %v3, %cmp_a, %v4, %v0 : !v, i1, !v, !v
    %cmp_b = lsir.cmpi i32 slt %v2, %c0 : !v, i32
    lsir.select %v5, %cmp_b, %v0, %v2 : !v, i1, !v, !v
    amdgcn.end_kernel
  }

  // Independent VALU and SALU: GraphBuilder allows free reordering.
  // Same-queue ops group together (VALU first, then SALU).
  // CHECK-LABEL: kernel @group_valu_salu
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         sop1 s_mov_b32
  // CHECK:         sop1 s_mov_b32
  // CHECK:         end_kernel
  amdgcn.kernel @group_valu_salu {
    %v0 = amdgcn.alloca : !v
    %v1 = amdgcn.alloca : !v
    %v2 = amdgcn.alloca : !v
    %v3 = amdgcn.alloca : !v
    %s0 = amdgcn.alloca : !s
    %s1 = amdgcn.alloca : !s
    %r0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %v0, %v1 : (!v, !v) -> !v
    %c0 = arith.constant 0 : i32
    %rs0 = amdgcn.sop1 s_mov_b32 outs %s0 ins %c0 : !s, i32
    %r1 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %v2, %v3 : (!v, !v) -> !v
    %c1 = arith.constant 1 : i32
    %rs1 = amdgcn.sop1 s_mov_b32 outs %s1 ins %c1 : !s, i32
    amdgcn.end_kernel
  }

  // Data dependency: vop2 depends on vop1 result. SALU is independent.
  // CHECK-LABEL: kernel @respect_data_deps
  // CHECK:         %[[R0:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         vop2 v_add_u32 outs %{{.*}} ins %[[R0]],
  // CHECK:         sop1 s_mov_b32
  // CHECK:         end_kernel
  amdgcn.kernel @respect_data_deps {
    %v0 = amdgcn.alloca : !v
    %v1 = amdgcn.alloca : !v
    %v2 = amdgcn.alloca : !v
    %s0 = amdgcn.alloca : !s
    %r0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %v0, %v2 : (!v, !v) -> !v
    %c0 = arith.constant 42 : i32
    %rs0 = amdgcn.sop1 s_mov_b32 outs %s0 ins %c0 : !s, i32
    %r1 = amdgcn.vop2 v_add_u32 outs %v1 ins %r0, %v2 : !v, !v, !v
    amdgcn.end_kernel
  }

  // VALU addr computations batch before VMEM loads (SSA deps).
  // CHECK-LABEL: kernel @vmem_addr_load_interleave
  // CHECK:         vop2 v_add_u32
  // CHECK:         vop2 v_add_u32
  // CHECK:         vop2 v_add_u32
  // CHECK:         vop2 v_add_u32
  // CHECK:         load global_load_dwordx4
  // CHECK:         load global_load_dwordx4
  // CHECK:         load global_load_dwordx4
  // CHECK:         load global_load_dwordx4
  // CHECK:         end_kernel
  amdgcn.kernel @vmem_addr_load_interleave {
    %base = amdgcn.alloca : !v
    %sa0 = amdgcn.alloca : !s
    %sa1 = amdgcn.alloca : !s
    %addr = amdgcn.make_register_range %sa0, %sa1 : !s, !s
    %d0_0 = amdgcn.alloca : !v
    %d0_1 = amdgcn.alloca : !v
    %d0_2 = amdgcn.alloca : !v
    %d0_3 = amdgcn.alloca : !v
    %d0 = amdgcn.make_register_range %d0_0, %d0_1, %d0_2, %d0_3 : !v, !v, !v, !v
    %d1_0 = amdgcn.alloca : !v
    %d1_1 = amdgcn.alloca : !v
    %d1_2 = amdgcn.alloca : !v
    %d1_3 = amdgcn.alloca : !v
    %d1 = amdgcn.make_register_range %d1_0, %d1_1, %d1_2, %d1_3 : !v, !v, !v, !v
    %d2_0 = amdgcn.alloca : !v
    %d2_1 = amdgcn.alloca : !v
    %d2_2 = amdgcn.alloca : !v
    %d2_3 = amdgcn.alloca : !v
    %d2 = amdgcn.make_register_range %d2_0, %d2_1, %d2_2, %d2_3 : !v, !v, !v, !v
    %d3_0 = amdgcn.alloca : !v
    %d3_1 = amdgcn.alloca : !v
    %d3_2 = amdgcn.alloca : !v
    %d3_3 = amdgcn.alloca : !v
    %d3 = amdgcn.make_register_range %d3_0, %d3_1, %d3_2, %d3_3 : !v, !v, !v, !v
    %off0 = amdgcn.alloca : !v
    %off1 = amdgcn.alloca : !v
    %off2 = amdgcn.alloca : !v
    %off3 = amdgcn.alloca : !v
    %c0 = arith.constant 0 : i32
    %c1024 = arith.constant 1024 : i32
    %c2048 = arith.constant 2048 : i32
    %c3072 = arith.constant 3072 : i32
    %a0 = amdgcn.vop2 v_add_u32 outs %off0 ins %c0, %base : !v, i32, !v
    %a1 = amdgcn.vop2 v_add_u32 outs %off1 ins %c1024, %base : !v, i32, !v
    %a2 = amdgcn.vop2 v_add_u32 outs %off2 ins %c2048, %base : !v, i32, !v
    %a3 = amdgcn.vop2 v_add_u32 outs %off3 ins %c3072, %base : !v, i32, !v
    %r0, %t0 = amdgcn.load global_load_dwordx4 dest %d0 addr %addr offset d(%a0)
        : dps(!vx4) ins(!sx2, !v) -> !amdgcn.read_token<flat>
    %r1, %t1 = amdgcn.load global_load_dwordx4 dest %d1 addr %addr offset d(%a1)
        : dps(!vx4) ins(!sx2, !v) -> !amdgcn.read_token<flat>
    %r2, %t2 = amdgcn.load global_load_dwordx4 dest %d2 addr %addr offset d(%a2)
        : dps(!vx4) ins(!sx2, !v) -> !amdgcn.read_token<flat>
    %r3, %t3 = amdgcn.load global_load_dwordx4 dest %d3 addr %addr offset d(%a3)
        : dps(!vx4) ins(!sx2, !v) -> !amdgcn.read_token<flat>
    amdgcn.end_kernel
  }

  // s_barrier is a workgroup sync point. GraphBuilder treats it as
  // a sync point but doesn't force LDS ordering within a wavefront.
  // CHECK-LABEL: kernel @barrier_separates_lds
  // CHECK:         store ds_write_b64
  // CHECK:         store ds_write_b64
  // CHECK:         load ds_read_b64
  // CHECK:         load ds_read_b64
  // CHECK:         sopp <s_barrier>
  // CHECK:         end_kernel
  amdgcn.kernel @barrier_separates_lds {
    %addr0 = amdgcn.alloca : !v
    %addr1 = amdgcn.alloca : !v
    %wd0 = amdgcn.alloca : !v
    %wd1 = amdgcn.alloca : !v
    %data0 = amdgcn.make_register_range %wd0, %wd1 : !v, !v
    %wd2 = amdgcn.alloca : !v
    %wd3 = amdgcn.alloca : !v
    %data1 = amdgcn.make_register_range %wd2, %wd3 : !v, !v
    %rd0 = amdgcn.alloca : !v
    %rd1 = amdgcn.alloca : !v
    %dst0 = amdgcn.make_register_range %rd0, %rd1 : !v, !v
    %rd2 = amdgcn.alloca : !v
    %rd3 = amdgcn.alloca : !v
    %dst1 = amdgcn.make_register_range %rd2, %rd3 : !v, !v
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %wt0 = amdgcn.store ds_write_b64 data %data0 addr %addr0 offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    %wt1 = amdgcn.store ds_write_b64 data %data1 addr %addr1 offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    amdgcn.sopp.sopp #amdgcn.inst<s_barrier>
    %rr0, %rt0 = amdgcn.load ds_read_b64 dest %dst0 addr %addr0 offset c(%c8)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %rr1, %rt1 = amdgcn.load ds_read_b64 dest %dst1 addr %addr1 offset c(%c8)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }

  // LDS ops: same-queue ties broken by block position.
  // CHECK-LABEL: kernel @lds_ops_ordered
  // CHECK:         store ds_write_b64
  // CHECK:         store ds_write_b64
  // CHECK:         load ds_read_b64
  // CHECK:         end_kernel
  amdgcn.kernel @lds_ops_ordered {
    %addr = amdgcn.alloca : !v
    %wd0 = amdgcn.alloca : !v
    %wd1 = amdgcn.alloca : !v
    %data0 = amdgcn.make_register_range %wd0, %wd1 : !v, !v
    %wd2 = amdgcn.alloca : !v
    %wd3 = amdgcn.alloca : !v
    %data1 = amdgcn.make_register_range %wd2, %wd3 : !v, !v
    %rd0 = amdgcn.alloca : !v
    %rd1 = amdgcn.alloca : !v
    %dst = amdgcn.make_register_range %rd0, %rd1 : !v, !v
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %wt0 = amdgcn.store ds_write_b64 data %data0 addr %addr offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    %rr0, %rt0 = amdgcn.load ds_read_b64 dest %dst addr %addr offset c(%c8)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %wt1 = amdgcn.store ds_write_b64 data %data1 addr %addr offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    amdgcn.end_kernel
  }

  // Regression test, ensuring that the read cannot be scheduled before the write.
  // CHECK-LABEL: kernel @lgkm_wait_gates_ds_read
  // CHECK:         store ds_write_b64
  // CHECK:         wait deps
  // CHECK:         load ds_read_b64
  // CHECK:         end_kernel
  amdgcn.kernel @lgkm_wait_gates_ds_read {
    // Read-side: only alloca/constant deps → ds_read is SSA-ready early.
    %rd0 = amdgcn.alloca : !v
    %rd1 = amdgcn.alloca : !v
    %raddr = amdgcn.alloca : !v
    // Write-side: data computed via a VALU chain → ds_write (and wait) is late.
    %va = amdgcn.alloca : !v
    %vb = amdgcn.alloca : !v
    %waddr = amdgcn.alloca : !v
    %c0 = arith.constant 0 : i32
    // VALU chain: %wd0 and %wd1 can only be scheduled after %r0.
    %r0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %va, %vb : (!v, !v) -> !v
    %wd0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %va, %r0 : (!v, !v) -> !v
    %wd1 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vb, %r0 : (!v, !v) -> !v
    // %dst becomes ready as soon as rd0/rd1 are scheduled (early), making
    // ds_read SSA-ready while the VALU chain (%wd0, %wd1) is still in flight.
    %dst = amdgcn.make_register_range %rd0, %rd1 : !v, !v
    %data = amdgcn.make_register_range %wd0, %wd1 : !v, !v
    %wt = amdgcn.store ds_write_b64 data %data addr %waddr offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    amdgcn.wait deps %wt : !amdgcn.write_token<shared>
    %rr, %rt = amdgcn.load ds_read_b64 dest %dst addr %raddr offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }

  // VMEM ops: same-queue ties broken by block position.
  // CHECK-LABEL: kernel @vmem_ops_ordered
  // CHECK:         store global_store_dword
  // CHECK:         store global_store_dword
  // CHECK:         load global_load_dwordx4
  // CHECK:         end_kernel
  amdgcn.kernel @vmem_ops_ordered {
    %sa0 = amdgcn.alloca : !s
    %sa1 = amdgcn.alloca : !s
    %addr = amdgcn.make_register_range %sa0, %sa1 : !s, !s
    %data0 = amdgcn.alloca : !v
    %data1 = amdgcn.alloca : !v
    %dd0 = amdgcn.alloca : !v
    %dd1 = amdgcn.alloca : !v
    %dd2 = amdgcn.alloca : !v
    %dd3 = amdgcn.alloca : !v
    %dst = amdgcn.make_register_range %dd0, %dd1, %dd2, %dd3 : !v, !v, !v, !v
    %off = amdgcn.alloca : !v
    %dr0 = amdgcn.make_register_range %data0 : !v
    %dr1 = amdgcn.make_register_range %data1 : !v
    %wt0 = amdgcn.store global_store_dword data %dr0 addr %addr
        : ins(!v, !sx2) -> !amdgcn.write_token<flat>
    %rr0, %rt0 = amdgcn.load global_load_dwordx4 dest %dst addr %addr offset d(%off)
        : dps(!vx4) ins(!sx2, !v) -> !amdgcn.read_token<flat>
    %wt1 = amdgcn.store global_store_dword data %dr1 addr %addr
        : ins(!v, !sx2) -> !amdgcn.write_token<flat>
    amdgcn.end_kernel
  }

}
