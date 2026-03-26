// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-low-level-scheduler{debug-stalls=false skip-precondition=true})))" | FileCheck %s

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>

amdgcn.module @test target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  func.func private @alloc_vgpr() -> !v {
    %r = amdgcn.alloca : !v
    return %r : !v
  }
  func.func private @alloc_vgprx2() -> !vx2 {
    %r0 = amdgcn.alloca : !v
    %r1 = amdgcn.alloca : !v
    %range = amdgcn.make_register_range %r0, %r1 : !v, !v
    return %range : !vx2
  }
  func.func private @alloc_vgprx4() -> !vx4 {
    %r0 = amdgcn.alloca : !v
    %r1 = amdgcn.alloca : !v
    %r2 = amdgcn.alloca : !v
    %r3 = amdgcn.alloca : !v
    %range = amdgcn.make_register_range %r0, %r1, %r2, %r3 : !v, !v, !v, !v
    return %range : !vx4
  }
  func.func private @alloc_sgpr() -> !s {
    %r = amdgcn.alloca : !s
    return %r : !s
  }
  func.func private @alloc_sgprx2() -> !sx2 {
    %r0 = amdgcn.alloca : !s
    %r1 = amdgcn.alloca : !s
    %range = amdgcn.make_register_range %r0, %r1 : !s, !s
    return %range : !sx2
  }

  // Two independent cmpi+select chains must NOT be interleaved.
  // All i1 producers write to VCC/SCC, so overlapping lifetimes = clobber.
  // The scheduler must serialize: all consumers of cmpi_a before cmpi_b.
  // CHECK-LABEL: kernel @i1_serialize_cmpi_select
  // CHECK:         lsir.cmpi
  // CHECK-NEXT:    lsir.select
  // CHECK:         lsir.cmpi
  // CHECK-NEXT:    lsir.select
  // CHECK:         end_kernel
  amdgcn.kernel @i1_serialize_cmpi_select {
    %v0 = func.call @alloc_vgpr() : () -> !v
    %v1 = func.call @alloc_vgpr() : () -> !v
    %v2 = func.call @alloc_vgpr() : () -> !v
    %v3 = func.call @alloc_vgpr() : () -> !v
    %c0 = arith.constant 0 : i32
    // Two independent signed-compare + conditional-select chains.
    // Without i1 serialization, the scheduler would interleave cmpi ops
    // to avoid back-to-back VALU stalls.
    %cmp_a = lsir.cmpi i32 slt %v0, %c0 : !v, i32
    lsir.select %v1, %cmp_a, %v2, %v0 : !v, i1, !v, !v
    %cmp_b = lsir.cmpi i32 slt %v2, %c0 : !v, i32
    lsir.select %v3, %cmp_b, %v0, %v2 : !v, i1, !v, !v
    amdgcn.end_kernel
  }

  // Three independent cmpi+select chains: verify complete serialization.
  // Each cmpi's select must complete before the next cmpi.
  // CHECK-LABEL: kernel @i1_serialize_three_chains
  // CHECK:         lsir.cmpi
  // CHECK-NEXT:    lsir.select
  // CHECK:         lsir.cmpi
  // CHECK-NEXT:    lsir.select
  // CHECK:         lsir.cmpi
  // CHECK-NEXT:    lsir.select
  // CHECK:         end_kernel
  amdgcn.kernel @i1_serialize_three_chains {
    %v0 = func.call @alloc_vgpr() : () -> !v
    %v1 = func.call @alloc_vgpr() : () -> !v
    %v2 = func.call @alloc_vgpr() : () -> !v
    %v3 = func.call @alloc_vgpr() : () -> !v
    %v4 = func.call @alloc_vgpr() : () -> !v
    %v5 = func.call @alloc_vgpr() : () -> !v
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
  // Both selects must appear before the next cmpi.
  // CHECK-LABEL: kernel @i1_serialize_fanout
  // CHECK:         %[[CMP_A:.*]] = lsir.cmpi
  // CHECK:         lsir.select {{.*}}, %[[CMP_A]],
  // CHECK:         lsir.select {{.*}}, %[[CMP_A]],
  // CHECK:         %[[CMP_B:.*]] = lsir.cmpi
  // CHECK:         lsir.select {{.*}}, %[[CMP_B]],
  // CHECK:         end_kernel
  amdgcn.kernel @i1_serialize_fanout {
    %v0 = func.call @alloc_vgpr() : () -> !v
    %v1 = func.call @alloc_vgpr() : () -> !v
    %v2 = func.call @alloc_vgpr() : () -> !v
    %v3 = func.call @alloc_vgpr() : () -> !v
    %v4 = func.call @alloc_vgpr() : () -> !v
    %v5 = func.call @alloc_vgpr() : () -> !v
    %c0 = arith.constant 0 : i32
    // cmpi_a has two consumers (fan-out = 2).
    %cmp_a = lsir.cmpi i32 slt %v0, %c0 : !v, i32
    lsir.select %v1, %cmp_a, %v2, %v0 : !v, i1, !v, !v
    lsir.select %v3, %cmp_a, %v4, %v0 : !v, i1, !v, !v
    %cmp_b = lsir.cmpi i32 slt %v2, %c0 : !v, i32
    lsir.select %v5, %cmp_b, %v0, %v2 : !v, i1, !v, !v
    amdgcn.end_kernel
  }

  // Total-order: program order preserved.
  // CHECK-LABEL: kernel @group_valu_salu
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         sop1 s_mov_b32
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         sop1 s_mov_b32
  // CHECK:         end_kernel
  amdgcn.kernel @group_valu_salu {
    %v0 = func.call @alloc_vgpr() : () -> !v
    %v1 = func.call @alloc_vgpr() : () -> !v
    %v2 = func.call @alloc_vgpr() : () -> !v
    %v3 = func.call @alloc_vgpr() : () -> !v
    %s0 = func.call @alloc_sgpr() : () -> !s
    %s1 = func.call @alloc_sgpr() : () -> !s
    // interleaved: valu, salu, valu, salu (no data deps)
    %r0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %v0, %v1 : (!v, !v) -> !v
    %c0 = arith.constant 0 : i32
    %rs0 = amdgcn.sop1 s_mov_b32 outs %s0 ins %c0 : !s, i32
    %r1 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %v2, %v3 : (!v, !v) -> !v
    %c1 = arith.constant 1 : i32
    %rs1 = amdgcn.sop1 s_mov_b32 outs %s1 ins %c1 : !s, i32
    amdgcn.end_kernel
  }

  // Total-order: program order preserved.
  // CHECK-LABEL: kernel @respect_data_deps
  // CHECK:         %[[R0:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         sop1 s_mov_b32
  // CHECK:         vop2 v_add_u32 outs %{{.*}} ins %[[R0]],
  // CHECK:         end_kernel
  amdgcn.kernel @respect_data_deps {
    %v0 = func.call @alloc_vgpr() : () -> !v
    %v1 = func.call @alloc_vgpr() : () -> !v
    %v2 = func.call @alloc_vgpr() : () -> !v
    %s0 = func.call @alloc_sgpr() : () -> !s
    %r0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %v0, %v2 : (!v, !v) -> !v
    %c0 = arith.constant 42 : i32
    %rs0 = amdgcn.sop1 s_mov_b32 outs %s0 ins %c0 : !s, i32
    // vop2 depends on %r0
    %r1 = amdgcn.vop2 v_add_u32 outs %v1 ins %r0, %v2 : !v, !v, !v
    amdgcn.end_kernel
  }

  // Total-order: program order preserved.
  // CHECK-LABEL: kernel @mfma_before_valu
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         vop3p_mai <v_mfma_f32_16x16x16_f16>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         vop3p_mai <v_mfma_f32_16x16x16_f16>
  // CHECK:         end_kernel
  amdgcn.kernel @mfma_before_valu {
    %a = func.call @alloc_vgprx2() : () -> !vx2
    %b = func.call @alloc_vgprx2() : () -> !vx2
    %c0 = func.call @alloc_vgprx4() : () -> !vx4
    %c1 = func.call @alloc_vgprx4() : () -> !vx4
    %dst0 = func.call @alloc_vgprx4() : () -> !vx4
    %dst1 = func.call @alloc_vgprx4() : () -> !vx4
    %va = func.call @alloc_vgpr() : () -> !v
    %vb = func.call @alloc_vgpr() : () -> !v
    %vc = func.call @alloc_vgpr() : () -> !v
    %vd = func.call @alloc_vgpr() : () -> !v
    %ve = func.call @alloc_vgpr() : () -> !v
    %vf = func.call @alloc_vgpr() : () -> !v
    %vg = func.call @alloc_vgpr() : () -> !v
    %vh = func.call @alloc_vgpr() : () -> !v
    // input: valu, valu, mfma, valu, valu, mfma (all independent)
    %r0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %va, %vb : (!v, !v) -> !v
    %r1 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vc, %vd : (!v, !v) -> !v
    %m0 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst0, %a, %b, %c0
        : !vx2, !vx2, !vx4 -> !vx4
    %r2 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %ve, %vf : (!v, !v) -> !v
    %r3 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vg, %vh : (!v, !v) -> !v
    %m1 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst1, %a, %b, %c1
        : !vx2, !vx2, !vx4 -> !vx4
    amdgcn.end_kernel
  }

  // VALU addr computations batch before VMEM loads (SSA deps).
  // All addrs compute first, then all loads fire back-to-back.
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
    %base = func.call @alloc_vgpr() : () -> !v
    %addr = func.call @alloc_sgprx2() : () -> !sx2
    %d0 = func.call @alloc_vgprx4() : () -> !vx4
    %d1 = func.call @alloc_vgprx4() : () -> !vx4
    %d2 = func.call @alloc_vgprx4() : () -> !vx4
    %d3 = func.call @alloc_vgprx4() : () -> !vx4
    %off0 = func.call @alloc_vgpr() : () -> !v
    %off1 = func.call @alloc_vgpr() : () -> !v
    %off2 = func.call @alloc_vgpr() : () -> !v
    %off3 = func.call @alloc_vgpr() : () -> !v
    %c0 = arith.constant 0 : i32
    %c1024 = arith.constant 1024 : i32
    %c2048 = arith.constant 2048 : i32
    %c3072 = arith.constant 3072 : i32
    // input: all addrs then all loads (already the barrier-preserved order)
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

  // Total-order: program order preserved.
  // CHECK-LABEL: kernel @waitcnt_is_barrier
  // CHECK:         vop2 v_add_u32
  // CHECK:         sopp.s_waitcnt
  // CHECK:         vop2 v_add_u32
  // CHECK:         end_kernel
  amdgcn.kernel @waitcnt_is_barrier {
    %v0 = func.call @alloc_vgpr() : () -> !v
    %v1 = func.call @alloc_vgpr() : () -> !v
    %v2 = func.call @alloc_vgpr() : () -> !v
    %v3 = func.call @alloc_vgpr() : () -> !v
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // valu before waitcnt
    %r0 = amdgcn.vop2 v_add_u32 outs %v0 ins %c0, %v1 : !v, i32, !v
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    // valu after waitcnt -- must stay after
    %r1 = amdgcn.vop2 v_add_u32 outs %v2 ins %c1, %v3 : !v, i32, !v
    amdgcn.end_kernel
  }

  // s_barrier is a barrier: ds_writes before it, ds_reads after it.
  // CHECK-LABEL: kernel @barrier_separates_lds
  // CHECK:         store ds_write_b64
  // CHECK:         store ds_write_b64
  // CHECK:         sopp <s_barrier>
  // CHECK:         load ds_read_b64
  // CHECK:         load ds_read_b64
  // CHECK:         end_kernel
  amdgcn.kernel @barrier_separates_lds {
    %addr0 = func.call @alloc_vgpr() : () -> !v
    %addr1 = func.call @alloc_vgpr() : () -> !v
    %data0 = func.call @alloc_vgprx2() : () -> !vx2
    %data1 = func.call @alloc_vgprx2() : () -> !vx2
    %dst0 = func.call @alloc_vgprx2() : () -> !vx2
    %dst1 = func.call @alloc_vgprx2() : () -> !vx2
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    // ds_writes, then barrier, then ds_reads
    %wt0 = amdgcn.store ds_write_b64 data %data0 addr %addr0 offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    %wt1 = amdgcn.store ds_write_b64 data %data1 addr %addr1 offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    amdgcn.sopp.sopp #amdgcn.inst<s_barrier>
    %rd0, %rt0 = amdgcn.load ds_read_b64 dest %dst0 addr %addr0 offset c(%c8)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %rd1, %rt1 = amdgcn.load ds_read_b64 dest %dst1 addr %addr1 offset c(%c8)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }

  // LDS ops: conservative ordering preserves program order.
  // CHECK-LABEL: kernel @lds_ops_ordered
  // CHECK:         store ds_write_b64
  // CHECK:         load ds_read_b64
  // CHECK:         store ds_write_b64
  // CHECK:         end_kernel
  amdgcn.kernel @lds_ops_ordered {
    %addr = func.call @alloc_vgpr() : () -> !v
    %data0 = func.call @alloc_vgprx2() : () -> !vx2
    %data1 = func.call @alloc_vgprx2() : () -> !vx2
    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %wt0 = amdgcn.store ds_write_b64 data %data0 addr %addr offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    %rd0, %rt0 = amdgcn.load ds_read_b64 dest %dst addr %addr offset c(%c8)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %wt1 = amdgcn.store ds_write_b64 data %data1 addr %addr offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    amdgcn.end_kernel
  }

  // VMEM store-load-store: memory chain preserves program order.
  // No RAW (load doesn't wait for store0's token).
  // WAW: store0 -> store1 (writes complete in issued order on AMDGPU).
  // WAR: load -> store1 (load must read before store1 overwrites).
  // The memory chain captures all three by chaining all mem ops.
  // CHECK-LABEL: kernel @vmem_ops_ordered
  // CHECK:         store global_store_dword
  // CHECK:         load global_load_dwordx4
  // CHECK:         store global_store_dword
  // CHECK:         end_kernel
  amdgcn.kernel @vmem_ops_ordered {
    %addr = func.call @alloc_sgprx2() : () -> !sx2
    %data0 = func.call @alloc_vgpr() : () -> !v
    %data1 = func.call @alloc_vgpr() : () -> !v
    %dst = func.call @alloc_vgprx4() : () -> !vx4
    %off = func.call @alloc_vgpr() : () -> !v
    %dr0 = amdgcn.make_register_range %data0 : !v
    %dr1 = amdgcn.make_register_range %data1 : !v
    %wt0 = amdgcn.store global_store_dword data %dr0 addr %addr
        : ins(!v, !sx2) -> !amdgcn.write_token<flat>
    %rd0, %rt0 = amdgcn.load global_load_dwordx4 dest %dst addr %addr offset d(%off)
        : dps(!vx4) ins(!sx2, !v) -> !amdgcn.read_token<flat>
    %wt1 = amdgcn.store global_store_dword data %dr1 addr %addr
        : ins(!v, !sx2) -> !amdgcn.write_token<flat>
    amdgcn.end_kernel
  }

}
