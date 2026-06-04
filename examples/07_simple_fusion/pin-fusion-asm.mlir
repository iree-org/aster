// RUN: aster-opt %s \
// RUN:     --aster-selective-inlining --symbol-dce --amdgcn-backend \
// RUN:   | aster-translate --mlir-to-asm | FileCheck %s

// Cross-function fusion with a mixed-allocation handoff.

amdgcn.module @pin_fusion target = #amdgcn.target<gfx942> {

  // CHECK-LABEL: fused
  //       CHECK: s_load_dwordx2 s[2:3], s[0:1], 0
  //       CHECK: s_load_dwordx2 s[4:5], s[0:1], 8
  //       CHECK: s_load_dwordx2 s[0:1], s[0:1], 16
  //       CHECK: v_mov_b32 v0, 0
  //       CHECK: v_mov_b32 v1, 0
  //       CHECK: s_waitcnt lgkmcnt(0)
  //       CHECK: global_load_dwordx4 v[12:15], v0, s[2:3]
  //       CHECK: global_load_dwordx4 v[16:19], v1, s[4:5]
  //       CHECK: s_waitcnt vmcnt(0)
  //       CHECK: v_add_f32 v40, v12, v16
  //       CHECK: v_add_f32 v41, v13, v17
  //       CHECK: v_add_f32 v42, v14, v18
  //       CHECK: v_add_f32 v43, v15, v19
  //       CHECK: v_mov_b32 v[[ZERO:[0-9]+]], 0
  //   CHECK-NOT: v_mov_b32 {{.*}}, v40
  //   CHECK-NOT: v_mov_b32 {{.*}}, v41
  //   CHECK-NOT: v_mov_b32 {{.*}}, v42
  //   CHECK-NOT: v_mov_b32 {{.*}}, v43
  //       CHECK: v_max_f32 v0, v[[ZERO]], v40
  //       CHECK: v_max_f32 v1, v[[ZERO]], v41
  //       CHECK: v_max_f32 v2, v[[ZERO]], v42
  //       CHECK: v_max_f32 v3, v[[ZERO]], v43
  //       CHECK: global_store_dwordx4 off, v[0:3], s[0:1]
  //       CHECK: s_endpgm
  amdgcn.kernel @fused arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {grid_dims = array<i32: 1, 1, 1>} {
    // 1. Get pointers from kernel arguments.
    %a_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %b_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    %out_ptr = amdgcn.load_arg 2 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    // 2. Call add4 (or any other ASTERized super-optimized chunk of asm)
    %s:4 = func.call @add4(%a_ptr, %b_ptr)
        : (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>)
        -> (!amdgcn.vgpr<40>, !amdgcn.vgpr<41>, !amdgcn.vgpr<42>, !amdgcn.vgpr<43>)

    // 3. Call bridge that copies hardcoded allocated registers to the virtual
    // inputs of relu4 (could also be noop aliases).
    %x:4 = func.call @bridge_add_to_relu4(%s#0, %s#1, %s#2, %s#3)
        : (!amdgcn.vgpr<40>, !amdgcn.vgpr<41>, !amdgcn.vgpr<42>, !amdgcn.vgpr<43>)
        -> (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr)

    // 4. Call relu4 (a cheap function for which super high-perf asm is not worth
    // manual optimization). Check moves are folded away automatically.
    func.call @relu4(%x#0, %x#1, %x#2, %x#3, %out_ptr)
        : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>) -> ()

    amdgcn.end_kernel
  }

  // This is 1-1 with asm but in ASTERized form and wrapped into a func.func.
  // In the future this could be a slice of a larger function where we e.g. drop
  // global_store ops.
  func.func private @add4(
      %a_ptr: !amdgcn.sgpr<[? + 2]>, %b_ptr: !amdgcn.sgpr<[? + 2]>)
      -> (!amdgcn.vgpr<40>, !amdgcn.vgpr<41>, !amdgcn.vgpr<42>, !amdgcn.vgpr<43>) {
    %c0 = arith.constant 0 : i32
    %a0 = amdgcn.alloca : !amdgcn.vgpr<12>
    %a1 = amdgcn.alloca : !amdgcn.vgpr<13>
    %a2 = amdgcn.alloca : !amdgcn.vgpr<14>
    %a3 = amdgcn.alloca : !amdgcn.vgpr<15>
    %b0 = amdgcn.alloca : !amdgcn.vgpr<16>
    %b1 = amdgcn.alloca : !amdgcn.vgpr<17>
    %b2 = amdgcn.alloca : !amdgcn.vgpr<18>
    %b3 = amdgcn.alloca : !amdgcn.vgpr<19>
    %a = amdgcn.make_register_range %a0, %a1, %a2, %a3
        : !amdgcn.vgpr<12>, !amdgcn.vgpr<13>, !amdgcn.vgpr<14>, !amdgcn.vgpr<15>
    %b = amdgcn.make_register_range %b0, %b1, %b2, %b3
        : !amdgcn.vgpr<16>, !amdgcn.vgpr<17>, !amdgcn.vgpr<18>, !amdgcn.vgpr<19>
    %off_a = amdgcn.alloca : !amdgcn.vgpr
    %off_b = amdgcn.alloca : !amdgcn.vgpr
    %off0_a = amdgcn.v_mov_b32 outs(%off_a) ins(%c0) : outs(!amdgcn.vgpr) ins(i32)
    %off0_b = amdgcn.v_mov_b32 outs(%off_b) ins(%c0) : outs(!amdgcn.vgpr) ins(i32)
    %ta = amdgcn.global_load_dwordx4 dest %a addr %a_ptr offset d(%off0_a) + c(%c0)
        : outs(!amdgcn.vgpr<[12 : 16]>) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
    %tb = amdgcn.global_load_dwordx4 dest %b addr %b_ptr offset d(%off0_b) + c(%c0)
        : outs(!amdgcn.vgpr<[16 : 20]>) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    %s0 = amdgcn.alloca : !amdgcn.vgpr<40>
    %s1 = amdgcn.alloca : !amdgcn.vgpr<41>
    %s2 = amdgcn.alloca : !amdgcn.vgpr<42>
    %s3 = amdgcn.alloca : !amdgcn.vgpr<43>
    amdgcn.v_add_f32 outs(%s0) ins(%a0, %b0) : outs(!amdgcn.vgpr<40>) ins(!amdgcn.vgpr<12>, !amdgcn.vgpr<16>)
    amdgcn.v_add_f32 outs(%s1) ins(%a1, %b1) : outs(!amdgcn.vgpr<41>) ins(!amdgcn.vgpr<13>, !amdgcn.vgpr<17>)
    amdgcn.v_add_f32 outs(%s2) ins(%a2, %b2) : outs(!amdgcn.vgpr<42>) ins(!amdgcn.vgpr<14>, !amdgcn.vgpr<18>)
    amdgcn.v_add_f32 outs(%s3) ins(%a3, %b3) : outs(!amdgcn.vgpr<43>) ins(!amdgcn.vgpr<15>, !amdgcn.vgpr<19>)
    return %s0, %s1, %s2, %s3 : !amdgcn.vgpr<40>, !amdgcn.vgpr<41>, !amdgcn.vgpr<42>, !amdgcn.vgpr<43>
  }

  // This is a reusable function whose registers will be allocated in the context
  // of the caller constraints.
  func.func private @relu4(
      %x0: !amdgcn.vgpr, %x1: !amdgcn.vgpr, %x2: !amdgcn.vgpr, %x3: !amdgcn.vgpr,
      %out_ptr: !amdgcn.sgpr<[? + 2]>) {
    %c0 = arith.constant 0 : i32
    %z = amdgcn.alloca : !amdgcn.vgpr
    %zr = amdgcn.v_mov_b32 outs(%z) ins(%c0) : outs(!amdgcn.vgpr) ins(i32)
    %o0 = amdgcn.alloca : !amdgcn.vgpr
    %o1 = amdgcn.alloca : !amdgcn.vgpr
    %o2 = amdgcn.alloca : !amdgcn.vgpr
    %o3 = amdgcn.alloca : !amdgcn.vgpr
    %m0 = amdgcn.v_max_f32 outs(%o0) ins(%zr, %x0) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
    %m1 = amdgcn.v_max_f32 outs(%o1) ins(%zr, %x1) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
    %m2 = amdgcn.v_max_f32 outs(%o2) ins(%zr, %x2) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
    %m3 = amdgcn.v_max_f32 outs(%o3) ins(%zr, %x3) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
    %data = amdgcn.make_register_range %m0, %m1, %m2, %m3
        : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %tok = amdgcn.global_store_dwordx4 data %data addr %out_ptr offset c(%c0)
        : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    return
  }

  // Noop bridge from pinned add outputs to generic unallocated relu inputs.
  // Uses explicit v_mov_b32 that get folded away automatically.
  func.func private @bridge_add_to_relu4(
      %s0: !amdgcn.vgpr<40>, %s1: !amdgcn.vgpr<41>, %s2: !amdgcn.vgpr<42>, %s3: !amdgcn.vgpr<43>)
      -> (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) {
    %x0a = amdgcn.alloca : !amdgcn.vgpr
    %x1a = amdgcn.alloca : !amdgcn.vgpr
    %x2a = amdgcn.alloca : !amdgcn.vgpr
    %x3a = amdgcn.alloca : !amdgcn.vgpr
    %x0 = amdgcn.v_mov_b32 outs(%x0a) ins(%s0) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<40>)
    %x1 = amdgcn.v_mov_b32 outs(%x1a) ins(%s1) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<41>)
    %x2 = amdgcn.v_mov_b32 outs(%x2a) ins(%s2) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<42>)
    %x3 = amdgcn.v_mov_b32 outs(%x3a) ins(%s3) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<43>)
    return %x0, %x1, %x2, %x3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  }
}
