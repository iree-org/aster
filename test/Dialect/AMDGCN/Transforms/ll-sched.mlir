// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-low-level-scheduler{debug-stalls=false})))" | FileCheck %s
// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-low-level-scheduler)))" | FileCheck %s --check-prefix=STALL

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

  // CHECK-LABEL: kernel @group_valu_salu
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK-NEXT:    amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         sop1 s_mov_b32
  // CHECK-NEXT:    sop1 s_mov_b32
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

  // CHECK-LABEL: kernel @respect_data_deps
  // CHECK:         %[[R0:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         vop2 v_add_u32 outs %{{.*}} ins %[[R0]],
  // CHECK:         sop1 s_mov_b32
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

  // CHECK-LABEL: kernel @mfma_before_valu
  // CHECK:         vop3p_mai <v_mfma_f32_16x16x16_f16>
  // CHECK:         vop3p_mai <v_mfma_f32_16x16x16_f16>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
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

  // CHECK-LABEL: kernel @lds_before_valu
  // CHECK:         load ds_read_b32
  // CHECK:         load ds_read_b32
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         end_kernel
  amdgcn.kernel @lds_before_valu {
    %lds_addr = func.call @alloc_vgpr() : () -> !v
    %lds_d0 = func.call @alloc_vgpr() : () -> !v
    %lds_d1 = func.call @alloc_vgpr() : () -> !v
    %va = func.call @alloc_vgpr() : () -> !v
    %vb = func.call @alloc_vgpr() : () -> !v
    %vc = func.call @alloc_vgpr() : () -> !v
    %vd = func.call @alloc_vgpr() : () -> !v
    %ve = func.call @alloc_vgpr() : () -> !v
    %vf = func.call @alloc_vgpr() : () -> !v
    %vg = func.call @alloc_vgpr() : () -> !v
    %vh = func.call @alloc_vgpr() : () -> !v
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    // input: valu, valu, lds, valu, valu, lds (all independent)
    %r0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %va, %vb : (!v, !v) -> !v
    %r1 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vc, %vd : (!v, !v) -> !v
    %lds0, %tok0 = amdgcn.load ds_read_b32 dest %lds_d0 addr %lds_addr offset c(%c0)
        : dps(!v) ins(!v, i32) -> !amdgcn.read_token<shared>
    %r2 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %ve, %vf : (!v, !v) -> !v
    %r3 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vg, %vh : (!v, !v) -> !v
    %lds1, %tok1 = amdgcn.load ds_read_b32 dest %lds_d1 addr %lds_addr offset c(%c4)
        : dps(!v) ins(!v, i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }

  // CHECK-LABEL: kernel @mixed_latency_hiding
  // CHECK-DAG:     store global_store_dword
  // CHECK-DAG:     load ds_read_b32
  // CHECK-DAG:     vop3p_mai <v_mfma_f32_16x16x16_f16>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK-NEXT:    amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         end_kernel
  amdgcn.kernel @mixed_latency_hiding {
    %addr = func.call @alloc_sgprx2() : () -> !sx2
    %vdata = func.call @alloc_vgpr() : () -> !v
    %data_range = amdgcn.make_register_range %vdata : !v
    %lds_addr = func.call @alloc_vgpr() : () -> !v
    %lds_dst = func.call @alloc_vgpr() : () -> !v
    %c0 = arith.constant 0 : i32
    %a = func.call @alloc_vgprx2() : () -> !vx2
    %b = func.call @alloc_vgprx2() : () -> !vx2
    %c = func.call @alloc_vgprx4() : () -> !vx4
    %dst = func.call @alloc_vgprx4() : () -> !vx4
    %va = func.call @alloc_vgpr() : () -> !v
    %vb = func.call @alloc_vgpr() : () -> !v
    %vc = func.call @alloc_vgpr() : () -> !v
    %vd = func.call @alloc_vgpr() : () -> !v
    // input: valu, valu, mfma, lds, vmem (worst order)
    %r0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %va, %vb : (!v, !v) -> !v
    %r1 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vc, %vd : (!v, !v) -> !v
    %mfma = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst, %a, %b, %c
        : !vx2, !vx2, !vx4 -> !vx4
    %lds, %lds_tok = amdgcn.load ds_read_b32 dest %lds_dst addr %lds_addr offset c(%c0)
        : dps(!v) ins(!v, i32) -> !amdgcn.read_token<shared>
    %vmem_tok = amdgcn.store global_store_dword data %data_range addr %addr
        : ins(!v, !sx2) -> !amdgcn.write_token<flat>
    amdgcn.end_kernel
  }

  // CHECK-LABEL: kernel @vmem_interleave_valu
  // CHECK:         store global_store_dword
  // CHECK:         store global_store_dword
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32>
  // CHECK:         store global_store_dword
  // CHECK:         end_kernel
  amdgcn.kernel @vmem_interleave_valu {
    %addr = func.call @alloc_sgprx2() : () -> !sx2
    %d0 = func.call @alloc_vgpr() : () -> !v
    %d1 = func.call @alloc_vgpr() : () -> !v
    %d2 = func.call @alloc_vgpr() : () -> !v
    %dr0 = amdgcn.make_register_range %d0 : !v
    %dr1 = amdgcn.make_register_range %d1 : !v
    %dr2 = amdgcn.make_register_range %d2 : !v
    %va = func.call @alloc_vgpr() : () -> !v
    %vb = func.call @alloc_vgpr() : () -> !v
    %vc = func.call @alloc_vgpr() : () -> !v
    %vd = func.call @alloc_vgpr() : () -> !v
    %ve = func.call @alloc_vgpr() : () -> !v
    %vf = func.call @alloc_vgpr() : () -> !v
    %vg = func.call @alloc_vgpr() : () -> !v
    %vh = func.call @alloc_vgpr() : () -> !v
    // input: valu, vmem, valu, vmem, valu, vmem, valu (bad order)
    %r0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %va, %vb : (!v, !v) -> !v
    %t0 = amdgcn.store global_store_dword data %dr0 addr %addr
        : ins(!v, !sx2) -> !amdgcn.write_token<flat>
    %r1 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vc, %vd : (!v, !v) -> !v
    %t1 = amdgcn.store global_store_dword data %dr1 addr %addr
        : ins(!v, !sx2) -> !amdgcn.write_token<flat>
    %r2 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %ve, %vf : (!v, !v) -> !v
    %t2 = amdgcn.store global_store_dword data %dr2 addr %addr
        : ins(!v, !sx2) -> !amdgcn.write_token<flat>
    %r3 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vg, %vh : (!v, !v) -> !v
    amdgcn.end_kernel
  }

  // STALL-LABEL: kernel @vmem_stall_annotation
  // STALL:         store {{.*}} {sched.stall_cycles = 0 : i64}
  // STALL:         store {{.*}} {sched.stall_cycles = 0 : i64}
  // STALL:         store {{.*}} {sched.stall_cycles = 120 : i64, sched.stall_reason = "vmem full"}
  // STALL:         end_kernel
  amdgcn.kernel @vmem_stall_annotation {
    %addr = func.call @alloc_sgprx2() : () -> !sx2
    %d0 = func.call @alloc_vgpr() : () -> !v
    %d1 = func.call @alloc_vgpr() : () -> !v
    %d2 = func.call @alloc_vgpr() : () -> !v
    %dr0 = amdgcn.make_register_range %d0 : !v
    %dr1 = amdgcn.make_register_range %d1 : !v
    %dr2 = amdgcn.make_register_range %d2 : !v
    // 3 VMEM stores -- 3rd stalls (VMEM queue full, depth=2)
    %t0 = amdgcn.store global_store_dword data %dr0 addr %addr
        : ins(!v, !sx2) -> !amdgcn.write_token<flat>
    %t1 = amdgcn.store global_store_dword data %dr1 addr %addr
        : ins(!v, !sx2) -> !amdgcn.write_token<flat>
    %t2 = amdgcn.store global_store_dword data %dr2 addr %addr
        : ins(!v, !sx2) -> !amdgcn.write_token<flat>
    amdgcn.end_kernel
  }

  // CHECK-LABEL: kernel @vmem_addr_load_interleave
  // first 2 pairs are tight: addr immediately before its load
  // CHECK:         vop2 v_add_u32
  // CHECK-NEXT:    load global_load_dwordx4
  // CHECK:         vop2 v_add_u32
  // CHECK-NEXT:    load global_load_dwordx4
  // remaining addrs and loads interleaved (not all addrs before all loads)
  // CHECK:         vop2 v_add_u32
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
    // input: all addrs then all loads (worst order)
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

  // Stress test with test_inst ops carrying sched.queue and sched.exec_latency.
  // Scenario: 4 VMEM loads (512cy), 2 LDS reads (128cy), 2 MFMAs (16cy),
  // 8 VALU fillers (4cy), plus addr deps for each load.
  // Expected: loads interleaved with other work, not clustered.
  // CHECK-LABEL: kernel @stress_mixed_queues
  // high-latency ops (lgkm, vmem, xdl) interleaved with valu, not clustered
  // CHECK:         test_inst {{.*}}sched.queue = "lgkm"
  // CHECK:         test_inst {{.*}}sched.queue = "lgkm"
  // CHECK:         test_inst {{.*}}sched.queue = "xdl"
  // CHECK:         test_inst {{.*}}sched.queue = "vmem"
  // CHECK:         test_inst {{.*}}sched.queue = "xdl"
  // CHECK:         test_inst {{.*}}sched.queue = "vmem"
  // CHECK:         test_inst {{.*}}sched.queue = "vmem"
  // CHECK:         test_inst {{.*}}sched.queue = "vmem"
  // CHECK:         end_kernel
  amdgcn.kernel @stress_mixed_queues {
    %v0 = func.call @alloc_vgpr() : () -> !v
    %v1 = func.call @alloc_vgpr() : () -> !v
    %v2 = func.call @alloc_vgpr() : () -> !v
    %v3 = func.call @alloc_vgpr() : () -> !v
    %v4 = func.call @alloc_vgpr() : () -> !v
    %v5 = func.call @alloc_vgpr() : () -> !v
    %v6 = func.call @alloc_vgpr() : () -> !v
    %v7 = func.call @alloc_vgpr() : () -> !v
    // VALU filler (no deps on anything interesting)
    %f0 = amdgcn.test_inst outs %v0 {sched.queue = "valu", sched.exec_latency = 4 : i64}
        : (!v) -> !v
    %f1 = amdgcn.test_inst outs %v1 {sched.queue = "valu", sched.exec_latency = 4 : i64}
        : (!v) -> !v
    %f2 = amdgcn.test_inst outs %v2 {sched.queue = "valu", sched.exec_latency = 4 : i64}
        : (!v) -> !v
    %f3 = amdgcn.test_inst outs %v3 {sched.queue = "valu", sched.exec_latency = 4 : i64}
        : (!v) -> !v
    %f4 = amdgcn.test_inst outs %v4 {sched.queue = "valu", sched.exec_latency = 4 : i64}
        : (!v) -> !v
    %f5 = amdgcn.test_inst outs %v5 {sched.queue = "valu", sched.exec_latency = 4 : i64}
        : (!v) -> !v
    %f6 = amdgcn.test_inst outs %v6 {sched.queue = "valu", sched.exec_latency = 4 : i64}
        : (!v) -> !v
    %f7 = amdgcn.test_inst outs %v7 {sched.queue = "valu", sched.exec_latency = 4 : i64}
        : (!v) -> !v
    // addr computations (VALU, each produces an address for a VMEM load)
    %a0 = amdgcn.test_inst outs %v0 ins %f0 {sched.queue = "valu", sched.exec_latency = 4 : i64}
        : (!v, !v) -> !v
    %a1 = amdgcn.test_inst outs %v1 ins %f1 {sched.queue = "valu", sched.exec_latency = 4 : i64}
        : (!v, !v) -> !v
    %a2 = amdgcn.test_inst outs %v2 ins %f2 {sched.queue = "valu", sched.exec_latency = 4 : i64}
        : (!v, !v) -> !v
    %a3 = amdgcn.test_inst outs %v3 ins %f3 {sched.queue = "valu", sched.exec_latency = 4 : i64}
        : (!v, !v) -> !v
    // VMEM loads (each depends on its addr computation)
    %ld0 = amdgcn.test_inst outs %v0 ins %a0 {sched.queue = "vmem", sched.exec_latency = 512 : i64}
        : (!v, !v) -> !v
    %ld1 = amdgcn.test_inst outs %v1 ins %a1 {sched.queue = "vmem", sched.exec_latency = 512 : i64}
        : (!v, !v) -> !v
    %ld2 = amdgcn.test_inst outs %v2 ins %a2 {sched.queue = "vmem", sched.exec_latency = 512 : i64}
        : (!v, !v) -> !v
    %ld3 = amdgcn.test_inst outs %v3 ins %a3 {sched.queue = "vmem", sched.exec_latency = 512 : i64}
        : (!v, !v) -> !v
    // LDS reads (independent of loads)
    %ds0 = amdgcn.test_inst outs %v4 ins %f4 {sched.queue = "lgkm", sched.exec_latency = 128 : i64}
        : (!v, !v) -> !v
    %ds1 = amdgcn.test_inst outs %v5 ins %f5 {sched.queue = "lgkm", sched.exec_latency = 128 : i64}
        : (!v, !v) -> !v
    // MFMAs (depend on LDS results)
    %mf0 = amdgcn.test_inst outs %v6 ins %ds0, %f6 {sched.queue = "xdl", sched.exec_latency = 16 : i64}
        : (!v, !v, !v) -> !v
    %mf1 = amdgcn.test_inst outs %v7 ins %ds1, %f7 {sched.queue = "xdl", sched.exec_latency = 16 : i64}
        : (!v, !v, !v) -> !v
    amdgcn.end_kernel
  }

  // s_waitcnt is a barrier: VALU ops must not move past it.
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

  // LDS ops must not bypass each other (conservative memory chain).
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
    // write, read, write -- must preserve this order (no SSA dep between them)
    %wt0 = amdgcn.store ds_write_b64 data %data0 addr %addr offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    %rd0, %rt0 = amdgcn.load ds_read_b64 dest %dst addr %addr offset c(%c8)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %wt1 = amdgcn.store ds_write_b64 data %data1 addr %addr offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    amdgcn.end_kernel
  }

  // VMEM ops must not bypass each other (conservative memory chain).
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
    // store, load, store -- must preserve this order
    %wt0 = amdgcn.store global_store_dword data %dr0 addr %addr
        : ins(!v, !sx2) -> !amdgcn.write_token<flat>
    %rd0, %rt0 = amdgcn.load global_load_dwordx4 dest %dst addr %addr offset d(%off)
        : dps(!vx4) ins(!sx2, !v) -> !amdgcn.read_token<flat>
    %wt1 = amdgcn.store global_store_dword data %dr1 addr %addr
        : ins(!v, !sx2) -> !amdgcn.write_token<flat>
    amdgcn.end_kernel
  }

  // Independent VALU ops schedule across a wait.
  // The wait only blocks ops that use the waited-on load's data.
  // CHECK-LABEL: kernel @valu_across_wait
  // CHECK:         load ds_read_b64
  // wait stays before its data consumer (the MFMA)
  // CHECK:         wait deps
  // CHECK:         vop3p_mai
  // independent VALU scheduled past the wait (no dep on waited data)
  // CHECK:         vop2 v_add_u32
  // CHECK:         end_kernel
  amdgcn.kernel @valu_across_wait {
    %va = func.call @alloc_vgpr() : () -> !v
    %vb = func.call @alloc_vgpr() : () -> !v
    %vc = func.call @alloc_vgpr() : () -> !v
    %lds_addr = func.call @alloc_vgpr() : () -> !v
    %lds_d = func.call @alloc_vgprx2() : () -> !vx2
    %a = func.call @alloc_vgprx2() : () -> !vx2
    %b = func.call @alloc_vgprx2() : () -> !vx2
    %acc0 = func.call @alloc_vgprx4() : () -> !vx4
    %dst = func.call @alloc_vgprx4() : () -> !vx4
    %c0 = arith.constant 0 : i32
    %c42 = arith.constant 42 : i32
    // lds load produces data + token
    %lds_data, %tok = amdgcn.load ds_read_b64 dest %lds_d addr %lds_addr offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    // independent valu (no dep on lds data)
    %r0 = amdgcn.vop2 v_add_u32 outs %va ins %c42, %vb : !v, i32, !v
    // wait for lds
    amdgcn.wait deps %tok : !amdgcn.read_token<shared>
    // mfma uses lds data (must come after wait)
    %mfma = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst, %lds_data, %b, %acc0
        : !vx2, !vx2, !vx4 -> !vx4
    amdgcn.end_kernel
  }
}
