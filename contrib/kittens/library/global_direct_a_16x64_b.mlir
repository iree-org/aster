// Direct global-to-register A + permutation for 16x16 MFMA (LDS bypass).
//
// Given a vx4 from a cooperative dwordx4 global load (16x64_b layout),
// uses ds_bpermute_b32 to rearrange into MFMA A fragment layout.
// No LDS allocation needed -- ds_bpermute uses LDS hardware without memory.

// Register types
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!sx2 = !amdgcn.sgpr<[? + 2]>

amdgcn.library @kittens_global_direct_a_16x64_b isa = [#amdgcn.isa<cdna3>] {
  // From indexing.mlir
  func.func private @bpermute_addr_A_16x16_f16(index) -> index
  func.func private @vbase_select_A_16x16_f16() -> index
  // From indexing_ptr.mlir
  func.func private @index_to_vgpr_i32(index) -> !v

  // Permute a cooperative-fill vx4 into one MFMA A fragment (vx2).
  //
  // %loaded: vx4 from global_load_dwordx4 in 16x64_b cooperative fill layout.
  // %k_sel: 0 for K0 (columns 0-15), 1 for K1 (columns 16-31).
  // Returns !vx2 ready for v_mfma_f32_16x16x16_f16.
  func.func private @permute_A_fragment(
      %loaded: !vx4, %k_sel: index
  ) -> !vx2 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32

    // Step 1: Split vx4 into buffer of 4 individual VGPRs.
    // Note: static vs loop + dynamic with a constexpr `extract_register %loaded, %i`
    %v0, %v1, %v2, %v3 = amdgcn.split_register_range %loaded : !vx4
    %src_buf = memref.alloca(%c4) : memref<?x!v>
    memref.store %v0, %src_buf[%c0] : memref<?x!v>
    memref.store %v1, %src_buf[%c1] : memref<?x!v>
    memref.store %v2, %src_buf[%c2] : memref<?x!v>
    memref.store %v3, %src_buf[%c3] : memref<?x!v>

    // Step 2: Compute ds_bpermute byte address from indexing.mlir.
    %addr_idx = func.call @bpermute_addr_A_16x16_f16(%k_sel) : (index) -> index
    %addr_v = func.call @index_to_vgpr_i32(%addr_idx) : (index) -> !v

    // Step 3: ds_bpermute all 4 source VGPRs + wait.
    %bp_buf = memref.alloca(%c4) : memref<?x!v>
    scf.for %i = %c0 to %c4 step %c1 {
      %src = memref.load %src_buf[%i] : memref<?x!v>
      %bp_d = lsir.alloca : !v
      %bp_r, %bp_tok = amdgcn.load ds_bpermute_b32 dest %bp_d addr %addr_v
          offset d(%src) + c(%c0_i32)
          : dps(!v) ins(!v, !v, i32) -> !amdgcn.read_token<shared>
      amdgcn.wait deps %bp_tok : !amdgcn.read_token<shared>
      memref.store %bp_r, %bp_buf[%i] : memref<?x!v>
    } {aster.constexpr}

    // Step 4: Select the right VGPR pair based on vbase from indexing.mlir.
    // mask = 1 where vbase==0 (lanes 0-15, 32-47), cndmask picks src1.
    // mask = 0 where vbase==1 (lanes 16-31, 48-63), cndmask picks src0.
    %vbase_idx = func.call @vbase_select_A_16x16_f16() : () -> index
    %vbase_v = func.call @index_to_vgpr_i32(%vbase_idx) : (index) -> !v
    %mask_d = lsir.alloca : !sx2
    %mask = amdgcn.cmpi v_cmp_eq_i32_e64 outs %mask_d ins %c0_i32, %vbase_v
        : dps(!sx2) ins(i32, !v)

    // Step 5: v_cndmask_b32_e64 to pick between vbase=0 (bp[i]) and vbase=2 (bp[i+2]).
    %out_buf = memref.alloca(%c2) : memref<?x!v>
    scf.for %i = %c0 to %c2 step %c1 {
      %hi_idx = affine.apply affine_map<(i) -> (i + 2)>(%i)
      %bp_lo = memref.load %bp_buf[%i] : memref<?x!v>
      %bp_hi = memref.load %bp_buf[%hi_idx] : memref<?x!v>
      %out_d = lsir.alloca : !v
      // result = mask[lane] ? bp_lo (vbase=0) : bp_hi (vbase=2)
      %out = amdgcn.vop3 v_cndmask_b32_e64 outs %out_d ins %bp_hi, %bp_lo src2 = %mask
          : !v, !v, !v, !sx2
      memref.store %out, %out_buf[%i] : memref<?x!v>
    } {aster.constexpr}

    %out0 = memref.load %out_buf[%c0] : memref<?x!v>
    %out1 = memref.load %out_buf[%c1] : memref<?x!v>
    %result = amdgcn.make_register_range %out0, %out1 : !v, !v
    return %result : !vx2
  }
}
