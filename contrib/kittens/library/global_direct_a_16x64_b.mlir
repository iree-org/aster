// Direct global-to-register A + permutation for 16x16 MFMA (LDS bypass).
//
// Split into issue (bpermutes) and resolve (wait + cndmask) so that
// bpermute latency can be hidden behind MFMAs at the top-level k_loop.
//
// Issue: @issue_A_bpermutes  -> returns future(vx4 of bpermuted regs + token)
// Resolve: @resolve_A_fragment -> waits on token, does cndmask, returns vx2

// Register types
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!sx2 = !amdgcn.sgpr<[? + 2]>

// Future types
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

// Bpermute future: 4 bpermuted VGPRs (as vx4) + shared token.
// The vx4 data is NOT valid until the token is waited on.
!future_bpermute = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

amdgcn.library @kittens_global_direct_a_16x64_b isa = [#amdgcn.isa<cdna3>] {
  // From indexing.mlir
  func.func private @bpermute_addr_A_16x16_f16(index) -> index
  func.func private @vbase_select_A_16x16_f16() -> index
  // From indexing_ptr.mlir
  func.func private @index_to_vgpr_i32(index) -> !v

  // Issue phase: 4 ds_bpermute_b32 ops, NO wait.
  // Returns future(vx4 + token). The vx4 holds the 4 bpermuted VGPR handles;
  // actual data arrives when the token is resolved.
  //
  // %loaded: vx4 from global_load_dwordx4 in 16x64_b cooperative fill layout.
  // %k_sel: 0 for K0 (columns 0-15), 1 for K1 (columns 16-31).
  func.func private @issue_A_bpermutes(
      %loaded: !vx4, %k_sel: index
  ) -> !future_bpermute {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32

    // Split vx4 into 4 individual VGPRs.
    %v0, %v1, %v2, %v3 = amdgcn.split_register_range %loaded : !vx4
    %src_buf = memref.alloca(%c4) : memref<?x!v>
    memref.store %v0, %src_buf[%c0] : memref<?x!v>
    memref.store %v1, %src_buf[%c1] : memref<?x!v>
    memref.store %v2, %src_buf[%c2] : memref<?x!v>
    memref.store %v3, %src_buf[%c3] : memref<?x!v>

    // Compute ds_bpermute byte address.
    %addr_idx = func.call @bpermute_addr_A_16x16_f16(%k_sel) : (index) -> index
    %addr_v = func.call @index_to_vgpr_i32(%addr_idx) : (index) -> !v

    // Issue all 4 ds_bpermute_b32 without waiting.
    %bp_buf = memref.alloca(%c4) : memref<?x!v>
    scf.for %i = %c0 to %c3 step %c1 {
      %src = memref.load %src_buf[%i] : memref<?x!v>
      %bp_d = lsir.alloca : !v
      %bp_r, %_tok = amdgcn.load ds_bpermute_b32 dest %bp_d addr %addr_v
          offset d(%src) + c(%c0_i32)
          : dps(!v) ins(!v, !v, i32) -> !amdgcn.read_token<shared>
      memref.store %bp_r, %bp_buf[%i] : memref<?x!v>
    } {aster.constexpr}

    // 4th bpermute: capture token (covers all 4 via lgkmcnt).
    %src3 = memref.load %src_buf[%c3] : memref<?x!v>
    %bp3_d = lsir.alloca : !v
    %bp3_r, %last_tok = amdgcn.load ds_bpermute_b32 dest %bp3_d addr %addr_v
        offset d(%src3) + c(%c0_i32)
        : dps(!v) ins(!v, !v, i32) -> !amdgcn.read_token<shared>
    memref.store %bp3_r, %bp_buf[%c3] : memref<?x!v>

    // Pack 4 bpermuted VGPRs as vx4 + token into a future.
    %bp0 = memref.load %bp_buf[%c0] : memref<?x!v>
    %bp1 = memref.load %bp_buf[%c1] : memref<?x!v>
    %bp2 = memref.load %bp_buf[%c2] : memref<?x!v>
    %bp3 = memref.load %bp_buf[%c3] : memref<?x!v>
    %bp_vx4 = amdgcn.make_register_range %bp0, %bp1, %bp2, %bp3 : !v, !v, !v, !v
    %bp_any = aster_utils.to_any %bp_vx4 : !vx4
    %future = aster_utils.struct_create(%bp_any, %last_tok)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_bpermute
    return %future : !future_bpermute
  }

  // Resolve phase: wait on bpermute token, then cndmask to select final vx2.
  // Call this right before the MFMA that consumes the A fragment.
  func.func private @resolve_A_fragment(
      %bp_future: !future_bpermute
  ) -> !vx2 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32

    // Wait on bpermute token and unpack the 4 VGPRs.
    %bp_any, %tok = aster_utils.struct_extract %bp_future ["value", "token"]
        : !future_bpermute -> !aster_utils.any, !amdgcn.read_token<shared>
    amdgcn.wait deps %tok : !amdgcn.read_token<shared>
    %bp_vx4 = aster_utils.from_any %bp_any : !vx4
    %bp0, %bp1, %bp2, %bp3 = amdgcn.split_register_range %bp_vx4 : !vx4

    // vbase select + cndmask to pick the right VGPR pair.
    %vbase_idx = func.call @vbase_select_A_16x16_f16() : () -> index
    %vbase_v = func.call @index_to_vgpr_i32(%vbase_idx) : (index) -> !v
    %mask_d = lsir.alloca : !sx2
    %mask = amdgcn.cmpi v_cmp_eq_i32_e64 outs %mask_d ins %c0_i32, %vbase_v
        : dps(!sx2) ins(i32, !v)

    %out_buf = memref.alloca(%c2) : memref<?x!v>
    // bp_lo = bp[i], bp_hi = bp[i+2]
    %out0_d = lsir.alloca : !v
    %out0 = amdgcn.vop3 v_cndmask_b32_e64 outs %out0_d ins %bp2, %bp0 src2 = %mask
        : !v, !v, !v, !sx2
    %out1_d = lsir.alloca : !v
    %out1 = amdgcn.vop3 v_cndmask_b32_e64 outs %out1_d ins %bp3, %bp1 src2 = %mask
        : !v, !v, !v, !sx2

    %result = amdgcn.make_register_range %out0, %out1 : !v, !v
    return %result : !vx2
  }
}
