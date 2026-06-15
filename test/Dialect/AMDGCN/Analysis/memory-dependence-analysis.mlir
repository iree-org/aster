// RUN: aster-opt %s --test-memory-dependence-analysis 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Normal form: create() verifies flat CFG (#amdgcn.no_scf_ops semantics).
//===----------------------------------------------------------------------===//

// CHECK: normal form violation: SCF dialect operations are disallowed
amdgcn.module @test_normal_form target = #amdgcn.target<gfx942> {
  amdgcn.kernel @has_scf {
    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %st = arith.constant 1 : index
    scf.for %i = %lb to %ub step %st {
      scf.yield
    }
    amdgcn.end_kernel
  }
}

//===----------------------------------------------------------------------===//
// Single-block edge kinds: RAW, WAR, WAW, and RAR (no edge).
//===----------------------------------------------------------------------===//

//   CHECK-LABEL: Kernel: raw_war_waw
amdgcn.module @test_edge_kinds target = #amdgcn.target<gfx942> {
  amdgcn.kernel @raw_war_waw {
    %data0 = amdgcn.alloca : !amdgcn.vgpr
    %data1 = amdgcn.alloca : !amdgcn.vgpr
    %dst0  = amdgcn.alloca : !amdgcn.vgpr
    %dst1  = amdgcn.alloca : !amdgcn.vgpr
    %addr  = amdgcn.alloca : !amdgcn.vgpr
    %bufA  = amdgcn.alloc_lds 64 alignment 16
    %offA  = amdgcn.get_lds_offset %bufA : i32

    // CHECK: Operation: {{.*}}ds_write{{.*}}test.W0
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %tokW0 = amdgcn.ds_write_b32 data %data0 addr %addr offset c(%offA) { test.W0 }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // CHECK: Operation: {{.*}}ds_read{{.*}}test.R0
    // CHECK:   RAW deps ending here: 1: test.W0
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %val0, %tokR0 = amdgcn.ds_read_b32 dest %dst0 addr %addr offset c(%offA) { test.R0 }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>

    // CHECK: Operation: {{.*}}ds_read{{.*}}test.R1
    // CHECK:   RAW deps ending here: 1: test.W0
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %val1, %tokR1 = amdgcn.ds_read_b32 dest %dst1 addr %addr offset c(%offA) { test.R1 }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>

    // CHECK: Operation: {{.*}}ds_write{{.*}}test.W1
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 2: test.R0, test.R1
    // CHECK:   WAW deps ending here: 1: test.W0
    %tokW1 = amdgcn.ds_write_b32 data %data1 addr %addr offset c(%offA) { test.W1 }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    amdgcn.end_kernel
  }
}

//===----------------------------------------------------------------------===//
// G2S: buffer_load_lds cross-resource RAW to a following ds_read.
//===----------------------------------------------------------------------===//

//   CHECK-LABEL: Kernel: g2s_then_read
amdgcn.module @test_g2s target = #amdgcn.target<gfx950> {
  amdgcn.kernel @g2s_then_read {
    %m0 = amdgcn.alloca : !amdgcn.m0<0>
    %c0 = arith.constant 0 : i32
    amdgcn.s_mov_b32 outs(%m0) ins(%c0) : outs(!amdgcn.m0<0>) ins(i32)
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %s2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %s3 = amdgcn.alloca : !amdgcn.sgpr<3>
    %rsrc = amdgcn.make_register_range %s0, %s1, %s2, %s3
      : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>, !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
    %soff = amdgcn.alloca : !amdgcn.sgpr<4>
    %voff = amdgcn.alloca : !amdgcn.vgpr<0>
    %dst  = amdgcn.alloca : !amdgcn.vgpr
    %addr = amdgcn.alloca : !amdgcn.vgpr
    %bufA = amdgcn.alloc_lds 64 alignment 16
    %offA = amdgcn.get_lds_offset %bufA : i32

    // CHECK: Operation: {{.*}}buffer_load_lds_dword{{.*}}test.g2s
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %tokG = amdgcn.buffer_load_lds_dword addr %rsrc m0 %m0 offset u(%soff) + off_idx(%voff) + c(%c0) {offen, test.g2s}
      : ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.m0<0>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>) mods(i32) -> !amdgcn.read_token<flat>

    // CHECK: Operation: {{.*}}ds_read{{.*}}test.read
    // CHECK:   RAW deps ending here: 1: test.g2s
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %val, %tokR = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%offA) { test.read }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }
}

//===----------------------------------------------------------------------===//
// Global: assume_noalias disambiguation and loop iter_arg tracing.
//===----------------------------------------------------------------------===//

//   CHECK-LABEL: Kernel: global_noalias
amdgcn.module @test_global target = #amdgcn.target<gfx942> {
  amdgcn.kernel @global_noalias {
    %data0 = lsir.alloca : !amdgcn.vgpr<[? + 4]>
    %data1 = lsir.alloca : !amdgcn.vgpr<[? + 4]>
    %voff  = amdgcn.alloca : !amdgcn.vgpr
    %c0    = arith.constant 0 : i32
    %a_s = lsir.alloca : !amdgcn.sgpr<[? + 2]>
    %b_s = lsir.alloca : !amdgcn.sgpr<[? + 2]>
    %ptr:2 = lsir.assume_noalias %a_s, %b_s
      : (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>) -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>)

    // CHECK: Operation: {{.*}}global_store{{.*}}test.wa
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %tokA = amdgcn.global_store_dwordx4 data %data0 addr %ptr#0 offset d(%voff) + c(%c0) { test.wa }
      : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>

    // CHECK: Operation: {{.*}}global_store{{.*}}test.wb
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %tokB = amdgcn.global_store_dwordx4 data %data1 addr %ptr#1 offset d(%voff) + c(%c0) { test.wb }
      : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>

    // CHECK: Operation: {{.*}}global_store{{.*}}test.wa2
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 1: test.wa
    %tokC = amdgcn.global_store_dwordx4 data %data1 addr %ptr#0 offset d(%voff) + c(%c0) { test.wa2 }
      : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.end_kernel
  }
}

//   CHECK-LABEL: Kernel: global_noalias_loop_arg
amdgcn.module @test_global_loop target = #amdgcn.target<gfx942> {
  amdgcn.kernel @global_noalias_loop_arg {
    %data0 = lsir.alloca : !amdgcn.vgpr<[? + 4]>
    %data1 = lsir.alloca : !amdgcn.vgpr<[? + 4]>
    %voff  = amdgcn.alloca : !amdgcn.vgpr
    %c0    = arith.constant 0 : i32
    %cond  = arith.constant 1 : i1
    %a_s = lsir.alloca : !amdgcn.sgpr<[? + 2]>
    %b_s = lsir.alloca : !amdgcn.sgpr<[? + 2]>
    %ptr:2 = lsir.assume_noalias %a_s, %b_s
      : (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>) -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>)
    cf.br ^body(%ptr#0, %ptr#1 : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>)

  ^body(%pa: !amdgcn.sgpr<[? + 2]>, %pb: !amdgcn.sgpr<[? + 2]>):
    // CHECK: Operation: {{.*}}global_store{{.*}}test.g_wa
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 1: test.g_wa
    %tokA = amdgcn.global_store_dwordx4 data %data0 addr %pa offset d(%voff) + c(%c0) { test.g_wa }
      : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>

    // CHECK: Operation: {{.*}}global_store{{.*}}test.g_wb
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 1: test.g_wb
    %tokB = amdgcn.global_store_dwordx4 data %data1 addr %pb offset d(%voff) + c(%c0) { test.g_wb }
      : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    cf.cond_br %cond, ^body(%pa, %pb : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>), ^exit

  ^exit:
    amdgcn.end_kernel
  }
}

//===----------------------------------------------------------------------===//
// LDS buffer disambiguation: distinct alloc_lds vs same buffer.
//===----------------------------------------------------------------------===//

//   CHECK-LABEL: Kernel: two_buffers
amdgcn.module @test_lds_buffer_disambig target = #amdgcn.target<gfx942> {
  amdgcn.kernel @two_buffers {
    %data = amdgcn.alloca : !amdgcn.vgpr
    %dst  = amdgcn.alloca : !amdgcn.vgpr
    %addr = amdgcn.alloca : !amdgcn.vgpr
    %bufA = amdgcn.alloc_lds 256 alignment 16
    %bufB = amdgcn.alloc_lds 256 alignment 16
    %offA = amdgcn.get_lds_offset %bufA : i32
    %offB = amdgcn.get_lds_offset %bufB : i32

    // CHECK: Operation: {{.*}}ds_write{{.*}}test.write_a
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %tokA = amdgcn.ds_write_b32 data %data addr %addr offset c(%offA) { test.write_a }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // CHECK: Operation: {{.*}}ds_read{{.*}}test.read_b
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %valB, %tokB = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%offB) { test.read_b }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>

    amdgcn.end_kernel
  }

  //   CHECK-LABEL: Kernel: same_buffer
  amdgcn.kernel @same_buffer {
    %data = amdgcn.alloca : !amdgcn.vgpr
    %dst  = amdgcn.alloca : !amdgcn.vgpr
    %addr = amdgcn.alloca : !amdgcn.vgpr
    %bufA = amdgcn.alloc_lds 256 alignment 16
    %offA = amdgcn.get_lds_offset %bufA : i32

    // CHECK: Operation: {{.*}}ds_write{{.*}}test.write_a
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %tokA = amdgcn.ds_write_b32 data %data addr %addr offset c(%offA) { test.write_a }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // CHECK: Operation: {{.*}}ds_read{{.*}}test.read_a
    // CHECK:   RAW deps ending here: 1: test.write_a
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %valA, %tokR = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%offA) { test.read_a }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>

    amdgcn.end_kernel
  }

  //   CHECK-LABEL: Kernel: selective_flush
  amdgcn.kernel @selective_flush {
    %dataA = amdgcn.alloca : !amdgcn.vgpr
    %dataB = amdgcn.alloca : !amdgcn.vgpr
    %dst   = amdgcn.alloca : !amdgcn.vgpr
    %addr  = amdgcn.alloca : !amdgcn.vgpr
    %bufA  = amdgcn.alloc_lds 256 alignment 16
    %bufB  = amdgcn.alloc_lds 256 alignment 16
    %offA  = amdgcn.get_lds_offset %bufA : i32
    %offB  = amdgcn.get_lds_offset %bufB : i32

    // CHECK: Operation: {{.*}}ds_write{{.*}}test.write_a
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %tokA = amdgcn.ds_write_b32 data %dataA addr %addr offset c(%offA) { test.write_a }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // CHECK: Operation: {{.*}}ds_write{{.*}}test.write_b
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %tokB = amdgcn.ds_write_b32 data %dataB addr %addr offset c(%offB) { test.write_b }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // CHECK: Operation: {{.*}}ds_read{{.*}}test.read_a
    // CHECK:   RAW deps ending here: 1: test.write_a
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %valA, %tokR = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%offA) { test.read_a }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>

    amdgcn.end_kernel
  }

  //   CHECK-LABEL: Kernel: transitive_address
  amdgcn.kernel @transitive_address {
    %dataA = amdgcn.alloca : !amdgcn.vgpr
    %dst   = amdgcn.alloca : !amdgcn.vgpr
    %addr  = amdgcn.alloca : !amdgcn.vgpr
    %bufA  = amdgcn.alloc_lds 256 alignment 16
    %bufB  = amdgcn.alloc_lds 256 alignment 16
    %offA  = amdgcn.get_lds_offset %bufA : i32
    %offB  = amdgcn.get_lds_offset %bufB : i32
    %c0    = arith.constant 0 : i32
    %adjB  = arith.addi %offB, %c0 : i32

    // CHECK: Operation: {{.*}}ds_write{{.*}}test.write_a
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %tokA = amdgcn.ds_write_b32 data %dataA addr %addr offset c(%offA) { test.write_a }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // CHECK: Operation: {{.*}}ds_read{{.*}}test.read_b_adj
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %valB, %tokR = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%adjB) { test.read_b_adj }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>

    amdgcn.end_kernel
  }
}

//===----------------------------------------------------------------------===//
// Loop-carried dependences via the back edge.
//===----------------------------------------------------------------------===//

amdgcn.module @test_loop target = #amdgcn.target<gfx942> {
  //   CHECK-LABEL: Kernel: loop_rw
  amdgcn.kernel @loop_rw {
    %data = amdgcn.alloca : !amdgcn.vgpr
    %dst  = amdgcn.alloca : !amdgcn.vgpr
    %addr = amdgcn.alloca : !amdgcn.vgpr
    %bufA = amdgcn.alloc_lds 64 alignment 16
    %offA = amdgcn.get_lds_offset %bufA : i32
    %cond = arith.constant 1 : i1
    cf.br ^body

  ^body:
    // CHECK: Operation: {{.*}}ds_read{{.*}}test.r
    // CHECK:   RAW deps ending here: 1: test.w
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %v, %tokR = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%offA) { test.r }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>

    // CHECK: Operation: {{.*}}ds_write{{.*}}test.w
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 1: test.r
    // CHECK:   WAW deps ending here: 1: test.w
    %tokW = amdgcn.ds_write_b32 data %data addr %addr offset c(%offA) { test.w }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    cf.cond_br %cond, ^body, ^exit

  ^exit:
    amdgcn.end_kernel
  }
}

//===----------------------------------------------------------------------===//
// Cross-block dependences via backward predecessor walk.
//===----------------------------------------------------------------------===//

amdgcn.module @test_multiblock target = #amdgcn.target<gfx942> {
  //   CHECK-LABEL: Kernel: cross_block_raw
  amdgcn.kernel @cross_block_raw {
    %data = amdgcn.alloca : !amdgcn.vgpr
    %dst  = amdgcn.alloca : !amdgcn.vgpr
    %addr = amdgcn.alloca : !amdgcn.vgpr
    %bufA = amdgcn.alloc_lds 64 alignment 16
    %offA = amdgcn.get_lds_offset %bufA : i32

    // CHECK: Operation: {{.*}}ds_write{{.*}}test.bb0_write
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %tokW = amdgcn.ds_write_b32 data %data addr %addr offset c(%offA) { test.bb0_write }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    cf.br ^bb1

  ^bb1:
    // CHECK: Operation: {{.*}}ds_read{{.*}}test.bb1_read
    // CHECK:   RAW deps ending here: 1: test.bb0_write
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %v, %tokR = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%offA) { test.bb1_read }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }

  //   CHECK-LABEL: Kernel: diamond
  amdgcn.kernel @diamond {
    %data0 = amdgcn.alloca : !amdgcn.vgpr
    %data1 = amdgcn.alloca : !amdgcn.vgpr
    %dst   = amdgcn.alloca : !amdgcn.vgpr
    %addr  = amdgcn.alloca : !amdgcn.vgpr
    %bufA  = amdgcn.alloc_lds 64 alignment 16
    %offA  = amdgcn.get_lds_offset %bufA : i32
    %cond  = arith.constant 1 : i1
    cf.cond_br %cond, ^bb1, ^bb2

  ^bb1:
    %tokA = amdgcn.ds_write_b32 data %data0 addr %addr offset c(%offA) { test.arm1 }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    cf.br ^bb3

  ^bb2:
    %tokB = amdgcn.ds_write_b32 data %data1 addr %addr offset c(%offA) { test.arm2 }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    cf.br ^bb3

  ^bb3:
    // CHECK: Operation: {{.*}}ds_read{{.*}}test.join
    // CHECK:   RAW deps ending here: 2: test.arm1, test.arm2
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %v, %tokR = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%offA) { test.join }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }
}

//===----------------------------------------------------------------------===//
// Multi-buffer LDS forms: single, non-rotating, and rotating iter_args.
//===----------------------------------------------------------------------===//

//   CHECK-LABEL: Kernel: single_buf_straight
amdgcn.module @test_single_buf target = #amdgcn.target<gfx942> {
  amdgcn.kernel @single_buf_straight {
    %data = amdgcn.alloca : !amdgcn.vgpr
    %dst  = amdgcn.alloca : !amdgcn.vgpr
    %addr = amdgcn.alloca : !amdgcn.vgpr
    %lds  = amdgcn.alloc_lds 256 offset 0
    %off  = amdgcn.get_lds_offset %lds : i32

    // CHECK: Operation: {{.*}}ds_write{{.*}}test.write_single
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %wtok = amdgcn.ds_write_b32 data %data addr %addr offset c(%off) { test.write_single }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // CHECK: Operation: {{.*}}ds_read{{.*}}test.read_single
    // CHECK:   RAW deps ending here: 1: test.write_single
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %v, %rtok = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%off) { test.read_single }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }

  //   CHECK-LABEL: Kernel: single_buf_loop_invariant_arg
  amdgcn.kernel @single_buf_loop_invariant_arg {
    %data = amdgcn.alloca : !amdgcn.vgpr
    %dst  = amdgcn.alloca : !amdgcn.vgpr
    %addr = amdgcn.alloca : !amdgcn.vgpr
    %cond = arith.constant 1 : i1
    %lds  = amdgcn.alloc_lds 256 offset 0
    %off  = amdgcn.get_lds_offset %lds : i32
    cf.br ^body(%off : i32)

  ^body(%loop_off: i32):
    // CHECK: Operation: {{.*}}ds_write{{.*}}test.write_loop
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 1: test.read_loop
    // CHECK:   WAW deps ending here: 1: test.write_loop
    %wtok = amdgcn.ds_write_b32 data %data addr %addr offset c(%loop_off) { test.write_loop }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // CHECK: Operation: {{.*}}ds_read{{.*}}test.read_loop
    // CHECK:   RAW deps ending here: 1: test.write_loop
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %v, %rtok = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%loop_off) { test.read_loop }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    cf.cond_br %cond, ^body(%loop_off : i32), ^exit

  ^exit:
    amdgcn.end_kernel
  }
}

//   CHECK-LABEL: Kernel: two_buf_no_rotate
amdgcn.module @test_two_buf_no_rotate target = #amdgcn.target<gfx942> {
  amdgcn.kernel @two_buf_no_rotate {
    %data = amdgcn.alloca : !amdgcn.vgpr
    %dst  = amdgcn.alloca : !amdgcn.vgpr
    %addr = amdgcn.alloca : !amdgcn.vgpr
    %cond = arith.constant 1 : i1
    %lds0 = amdgcn.alloc_lds 256 offset 0
    %lds1 = amdgcn.alloc_lds 256 offset 256
    %off0 = amdgcn.get_lds_offset %lds0 : i32
    %off1 = amdgcn.get_lds_offset %lds1 : i32
    cf.br ^body(%off0, %off1 : i32, i32)

  ^body(%cur0: i32, %cur1: i32):
    // CHECK: Operation: {{.*}}ds_write{{.*}}test.write_buf0
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 1: test.write_buf0
    %wtok = amdgcn.ds_write_b32 data %data addr %addr offset c(%cur0) { test.write_buf0 }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // CHECK: Operation: {{.*}}ds_read{{.*}}test.read_buf1
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %v, %rtok = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%cur1) { test.read_buf1 }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    cf.cond_br %cond, ^body(%cur0, %cur1 : i32, i32), ^exit

  ^exit:
    amdgcn.end_kernel
  }

  //   CHECK-LABEL: Kernel: two_buf_invariant_refs
  amdgcn.kernel @two_buf_invariant_refs {
    %data = amdgcn.alloca : !amdgcn.vgpr
    %dst  = amdgcn.alloca : !amdgcn.vgpr
    %addr = amdgcn.alloca : !amdgcn.vgpr
    %cond = arith.constant 1 : i1
    %lds0 = amdgcn.alloc_lds 256 offset 0
    %lds1 = amdgcn.alloc_lds 256 offset 256
    %off0 = amdgcn.get_lds_offset %lds0 : i32
    %off1 = amdgcn.get_lds_offset %lds1 : i32
    cf.br ^body

  ^body:
    // CHECK: Operation: {{.*}}ds_write{{.*}}test.wI_buf0
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 1: test.wI_buf0
    %wtok = amdgcn.ds_write_b32 data %data addr %addr offset c(%off0) { test.wI_buf0 }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // CHECK: Operation: {{.*}}ds_read{{.*}}test.rI_buf1
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %v, %rtok = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%off1) { test.rI_buf1 }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    cf.cond_br %cond, ^body, ^exit

  ^exit:
    amdgcn.end_kernel
  }
}

//   CHECK-LABEL: Kernel: double_buffer_rotate
amdgcn.module @test_rotating target = #amdgcn.target<gfx942> {
  amdgcn.kernel @double_buffer_rotate {
    %data = amdgcn.alloca : !amdgcn.vgpr
    %dst  = amdgcn.alloca : !amdgcn.vgpr
    %addr = amdgcn.alloca : !amdgcn.vgpr
    %cond = arith.constant 1 : i1
    %lds0 = amdgcn.alloc_lds 256 offset 0
    %off0 = amdgcn.get_lds_offset %lds0 : i32
    %lds1 = amdgcn.alloc_lds 256 offset 256
    %off1 = amdgcn.get_lds_offset %lds1 : i32
    cf.br ^body(%off1, %off0 : i32, i32)

  ^body(%cur: i32, %prev: i32):
    // CHECK: Operation: {{.*}}ds_write{{.*}}test.rot_write_cur
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 1: test.rot_read_prev
    // CHECK:   WAW deps ending here: 1: test.rot_write_cur
    %wtok = amdgcn.ds_write_b32 data %data addr %addr offset c(%cur) { test.rot_write_cur }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // CHECK: Operation: {{.*}}ds_read{{.*}}test.rot_read_prev
    // CHECK:   RAW deps ending here: 1: test.rot_write_cur
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %v, %rtok = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%prev) { test.rot_read_prev }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    cf.cond_br %cond, ^body(%prev, %cur : i32, i32), ^exit

  ^exit:
    amdgcn.end_kernel
  }

  //   CHECK-LABEL: Kernel: triple_buffer_rotate
  amdgcn.kernel @triple_buffer_rotate {
    %data = amdgcn.alloca : !amdgcn.vgpr
    %dst  = amdgcn.alloca : !amdgcn.vgpr
    %addr = amdgcn.alloca : !amdgcn.vgpr
    %cond = arith.constant 1 : i1
    %lds0 = amdgcn.alloc_lds 256 offset 0
    %off0 = amdgcn.get_lds_offset %lds0 : i32
    %lds1 = amdgcn.alloc_lds 256 offset 256
    %off1 = amdgcn.get_lds_offset %lds1 : i32
    %lds2 = amdgcn.alloc_lds 256 offset 512
    %off2 = amdgcn.get_lds_offset %lds2 : i32
    cf.br ^body(%off2, %off1, %off0 : i32, i32, i32)

  ^body(%cur: i32, %mid: i32, %old: i32):
    // CHECK: Operation: {{.*}}ds_write{{.*}}test.tri_write_cur
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 1: test.tri_read_old
    // CHECK:   WAW deps ending here: 2: test.tri_write_cur, test.tri_write_mid
    %wtok = amdgcn.ds_write_b32 data %data addr %addr offset c(%cur) { test.tri_write_cur }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // CHECK: Operation: {{.*}}ds_write{{.*}}test.tri_write_mid
    // CHECK:   RAW deps ending here: 0:
    // CHECK:   WAR deps ending here: 1: test.tri_read_old
    // CHECK:   WAW deps ending here: 2: test.tri_write_cur, test.tri_write_mid
    %wtok2 = amdgcn.ds_write_b32 data %data addr %addr offset c(%mid) { test.tri_write_mid }
      : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // CHECK: Operation: {{.*}}ds_read{{.*}}test.tri_read_old
    // CHECK:   RAW deps ending here: 2: test.tri_write_cur, test.tri_write_mid
    // CHECK:   WAR deps ending here: 0:
    // CHECK:   WAW deps ending here: 0:
    %v, %rtok = amdgcn.ds_read_b32 dest %dst addr %addr offset c(%old) { test.tri_read_old }
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    cf.cond_br %cond, ^body(%old, %cur, %mid : i32, i32, i32), ^exit

  ^exit:
    amdgcn.end_kernel
  }
}
