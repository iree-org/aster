// Conversion and pack operation e2e kernels.
//
// Each kernel: global_load src -> convert/pack -> global_store dst
// Each lane loads/stores one dword at ptr + lane_id * 4.
//
// Two-pointer kernels (VOP1 conversions):
//   arg0: src pointer (8 bytes)  -- input
//   arg1: dst pointer (8 bytes)  -- output
//
// Three-pointer kernels (VOP3 packs):
//   arg0: src0 pointer (8 bytes) -- input
//   arg1: src1 pointer (8 bytes) -- input
//   arg2: dst pointer (8 bytes)  -- output
amdgcn.module @conversion_pack_mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  func.func private @load_two_ptrs()
      -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>) {
    %src_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    %dst_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr_range<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %src_ptr, %dst_ptr
      : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>
  }

  func.func private @load_three_ptrs()
      -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>,
          !amdgcn.sgpr_range<[? + 2]>) {
    %src0_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    %src1_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr_range<[? + 2]>
    %dst_ptr = amdgcn.load_arg 2 : !amdgcn.sgpr_range<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %src0_ptr, %src1_ptr, %dst_ptr
      : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>,
        !amdgcn.sgpr_range<[? + 2]>
  }

  // ---------------------------------------------------------------------------
  // VOP1: v_cvt_f32_f16 -- f16 (in low 16 bits of dword) -> f32
  // ---------------------------------------------------------------------------

  amdgcn.kernel @cvt_f32_f16_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %src_ptr, %dst_ptr = func.call @load_two_ptrs()
      : () -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)

    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_alloc ins %c2, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32

    %load_dest = amdgcn.alloca : !amdgcn.vgpr
    %loaded, %tok_ld = amdgcn.load global_load_dword dest %load_dest addr %src_ptr
      offset d(%voffset) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.read_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    %cvt_dest = amdgcn.alloca : !amdgcn.vgpr
    %converted = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_f32_f16> %cvt_dest, %loaded
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %tok_st = amdgcn.store global_store_dword data %converted addr %dst_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // ---------------------------------------------------------------------------
  // VOP1: v_cvt_f16_f32 -- f32 -> f16 (in low 16 bits of dword)
  // ---------------------------------------------------------------------------

  amdgcn.kernel @cvt_f16_f32_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %src_ptr, %dst_ptr = func.call @load_two_ptrs()
      : () -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)

    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_alloc ins %c2, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32

    %load_dest = amdgcn.alloca : !amdgcn.vgpr
    %loaded, %tok_ld = amdgcn.load global_load_dword dest %load_dest addr %src_ptr
      offset d(%voffset) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.read_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    %cvt_dest = amdgcn.alloca : !amdgcn.vgpr
    %converted = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_f16_f32> %cvt_dest, %loaded
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %tok_st = amdgcn.store global_store_dword data %converted addr %dst_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // ---------------------------------------------------------------------------
  // VOP1: v_cvt_f32_u32 -- u32 -> f32
  // ---------------------------------------------------------------------------

  amdgcn.kernel @cvt_f32_u32_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %src_ptr, %dst_ptr = func.call @load_two_ptrs()
      : () -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)

    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_alloc ins %c2, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32

    %load_dest = amdgcn.alloca : !amdgcn.vgpr
    %loaded, %tok_ld = amdgcn.load global_load_dword dest %load_dest addr %src_ptr
      offset d(%voffset) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.read_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    %cvt_dest = amdgcn.alloca : !amdgcn.vgpr
    %converted = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_f32_u32> %cvt_dest, %loaded
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %tok_st = amdgcn.store global_store_dword data %converted addr %dst_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // ---------------------------------------------------------------------------
  // VOP1: v_cvt_f32_i32 -- i32 -> f32
  // ---------------------------------------------------------------------------

  amdgcn.kernel @cvt_f32_i32_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %src_ptr, %dst_ptr = func.call @load_two_ptrs()
      : () -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)

    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_alloc ins %c2, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32

    %load_dest = amdgcn.alloca : !amdgcn.vgpr
    %loaded, %tok_ld = amdgcn.load global_load_dword dest %load_dest addr %src_ptr
      offset d(%voffset) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.read_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    %cvt_dest = amdgcn.alloca : !amdgcn.vgpr
    %converted = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_f32_i32> %cvt_dest, %loaded
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %tok_st = amdgcn.store global_store_dword data %converted addr %dst_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // ---------------------------------------------------------------------------
  // VOP1: v_cvt_u32_f32 -- f32 -> u32
  // ---------------------------------------------------------------------------

  amdgcn.kernel @cvt_u32_f32_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %src_ptr, %dst_ptr = func.call @load_two_ptrs()
      : () -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)

    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_alloc ins %c2, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32

    %load_dest = amdgcn.alloca : !amdgcn.vgpr
    %loaded, %tok_ld = amdgcn.load global_load_dword dest %load_dest addr %src_ptr
      offset d(%voffset) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.read_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    %cvt_dest = amdgcn.alloca : !amdgcn.vgpr
    %converted = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_u32_f32> %cvt_dest, %loaded
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %tok_st = amdgcn.store global_store_dword data %converted addr %dst_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // ---------------------------------------------------------------------------
  // VOP1: v_cvt_i32_f32 -- f32 -> i32
  // ---------------------------------------------------------------------------

  amdgcn.kernel @cvt_i32_f32_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %src_ptr, %dst_ptr = func.call @load_two_ptrs()
      : () -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)

    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_alloc ins %c2, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32

    %load_dest = amdgcn.alloca : !amdgcn.vgpr
    %loaded, %tok_ld = amdgcn.load global_load_dword dest %load_dest addr %src_ptr
      offset d(%voffset) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.read_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    %cvt_dest = amdgcn.alloca : !amdgcn.vgpr
    %converted = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_i32_f32> %cvt_dest, %loaded
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %tok_st = amdgcn.store global_store_dword data %converted addr %dst_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // ---------------------------------------------------------------------------
  // VOP3: v_pack_b32_f16 -- pack two f16 (low 16 bits of dwords) into one b32
  // ---------------------------------------------------------------------------

  amdgcn.kernel @pack_b32_f16_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %src0_ptr, %src1_ptr, %dst_ptr = func.call @load_three_ptrs()
      : () -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>,
               !amdgcn.sgpr_range<[? + 2]>)

    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_alloc ins %c2, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32

    %load0_dest = amdgcn.alloca : !amdgcn.vgpr
    %src0_val, %tok0 = amdgcn.load global_load_dword dest %load0_dest addr %src0_ptr
      offset d(%voffset) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.read_token<flat>

    %load1_dest = amdgcn.alloca : !amdgcn.vgpr
    %src1_val, %tok1 = amdgcn.load global_load_dword dest %load1_dest addr %src1_ptr
      offset d(%voffset) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.read_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    %pack_dest = amdgcn.alloca : !amdgcn.vgpr
    %packed = amdgcn.vop3 v_pack_b32_f16 outs %pack_dest ins %src0_val, %src1_val
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

    %tok_st = amdgcn.store global_store_dword data %packed addr %dst_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // ---------------------------------------------------------------------------
  // VOP3: v_cvt_pk_fp8_f32 -- pack-convert two f32 to two fp8 in low 16 bits
  // ---------------------------------------------------------------------------

  amdgcn.kernel @cvt_pk_fp8_f32_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %src0_ptr, %src1_ptr, %dst_ptr = func.call @load_three_ptrs()
      : () -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>,
               !amdgcn.sgpr_range<[? + 2]>)

    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_alloc ins %c2, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32

    %load0_dest = amdgcn.alloca : !amdgcn.vgpr
    %src0_val, %tok0 = amdgcn.load global_load_dword dest %load0_dest addr %src0_ptr
      offset d(%voffset) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.read_token<flat>

    %load1_dest = amdgcn.alloca : !amdgcn.vgpr
    %src1_val, %tok1 = amdgcn.load global_load_dword dest %load1_dest addr %src1_ptr
      offset d(%voffset) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.read_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    %cvt_dest = amdgcn.alloca : !amdgcn.vgpr
    %converted = amdgcn.vop3 v_cvt_pk_fp8_f32 outs %cvt_dest ins %src0_val, %src1_val
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

    %tok_st = amdgcn.store global_store_dword data %converted addr %dst_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // ---------------------------------------------------------------------------
  // VOP3: v_cvt_pk_bf8_f32 -- pack-convert two f32 to two bf8 in low 16 bits
  // ---------------------------------------------------------------------------

  amdgcn.kernel @cvt_pk_bf8_f32_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %src0_ptr, %src1_ptr, %dst_ptr = func.call @load_three_ptrs()
      : () -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>,
               !amdgcn.sgpr_range<[? + 2]>)

    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_alloc ins %c2, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32

    %load0_dest = amdgcn.alloca : !amdgcn.vgpr
    %src0_val, %tok0 = amdgcn.load global_load_dword dest %load0_dest addr %src0_ptr
      offset d(%voffset) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.read_token<flat>

    %load1_dest = amdgcn.alloca : !amdgcn.vgpr
    %src1_val, %tok1 = amdgcn.load global_load_dword dest %load1_dest addr %src1_ptr
      offset d(%voffset) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.read_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    %cvt_dest = amdgcn.alloca : !amdgcn.vgpr
    %converted = amdgcn.vop3 v_cvt_pk_bf8_f32 outs %cvt_dest ins %src0_val, %src1_val
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

    %tok_st = amdgcn.store global_store_dword data %converted addr %dst_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
