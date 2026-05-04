// RUN: aster-opt %s --verify-roundtrip
//
// E2E test: ds_permute_b32 and ds_bpermute_b32 cross-lane permutation.
//
// ds_permute_b32:  VDST[(ADDR[i] + offset)/4 % 64] = DATA0[i]  (forward/scatter)
// ds_bpermute_b32: VDST[i] = DATA0[(ADDR[i] + offset)/4 % 64]  (backward/gather)
//
// Two kernels:
//   1. permute_rotate: forward rotate shows ds_permute puts data where addr says.
//      addr[i] = ((i+1) & 63) * 4, data[i] = i.
//      Result: output[(i+1)%64] = i, so output[j] = (j+63)%64.
//   2. permute_bpermute_idempotent: bpermute(permute(data, addr), addr) = data.
//      Same rotate addr for both steps. Result: output[i] = i.

amdgcn.module @perm_e2e_mod target = #amdgcn.target<gfx942> {

  func.func private @load_output_ptr() -> !amdgcn.sgpr<[? + 2]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0
    return %out_ptr : !amdgcn.sgpr<[? + 2]>
  }

  // ----- Test 1: Forward rotate by 1 -----
  // block=(64,1,1), grid=(1,1,1). Output: 64 dwords.
  // addr[i] = ((i+1) & 63) * 4, data[i] = i.
  // permute: VDST[addr[i]/4] = DATA[i], so output[(i+1)%64] = i.
  // Expected: output[j] = (j+63)%64.
  amdgcn.kernel @permute_rotate arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid_x = amdgcn.thread_id x : !amdgcn.vgpr

    // dst_lane = (tid_x + 1) & 63
    %c1 = arith.constant 1 : i32
    %inc_a = amdgcn.alloca : !amdgcn.vgpr
    %inc = amdgcn.v_add_u32 outs(%inc_a) ins(%c1, %tid_x) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %c63 = arith.constant 63 : i32
    %wrap_a = amdgcn.alloca : !amdgcn.vgpr
    %wrapped = amdgcn.v_and_b32 outs(%wrap_a) ins(%c63, %inc) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %c2 = arith.constant 2 : i32
    %addr_a = amdgcn.alloca : !amdgcn.vgpr
    %addr = amdgcn.v_lshlrev_b32 outs(%addr_a) ins(%c2, %wrapped) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    // permute: result[(i+1)%64] = tid_x[i] = i
    %perm_a = amdgcn.alloca : !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %result, %tok_perm = amdgcn.ds_permute_b32 outs(%perm_a) ins(%addr, %tid_x) args(%c0)
        : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr) args(i32) -> !amdgcn.read_token<shared>
    amdgcn.s_waitcnt lgkmcnt = 0

    // Store result[i] to output[i]
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid_x) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %tok_st = amdgcn.global_store_dword data %result addr %out_ptr offset d(%voffset) + c(%c0) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }

  // ----- Test 2: Idempotency: bpermute(permute(data, addr), addr) = data -----
  // block=(64,1,1), grid=(1,1,1). Output: 64 dwords.
  // Same rotate addr for both steps.
  //   permute: temp[(i+1)%64] = i
  //   bpermute: result[i] = temp[(i+1)%64] = i
  // Expected: output[i] = i.
  amdgcn.kernel @permute_bpermute_idempotent arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid_x = amdgcn.thread_id x : !amdgcn.vgpr

    // addr = ((tid_x + 1) & 63) * 4
    %c1 = arith.constant 1 : i32
    %inc_a = amdgcn.alloca : !amdgcn.vgpr
    %inc = amdgcn.v_add_u32 outs(%inc_a) ins(%c1, %tid_x) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %c63 = arith.constant 63 : i32
    %wrap_a = amdgcn.alloca : !amdgcn.vgpr
    %wrapped = amdgcn.v_and_b32 outs(%wrap_a) ins(%c63, %inc) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %c2 = arith.constant 2 : i32
    %addr_a = amdgcn.alloca : !amdgcn.vgpr
    %addr = amdgcn.v_lshlrev_b32 outs(%addr_a) ins(%c2, %wrapped) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    // Step 1 - permute: temp[(i+1)%64] = i
    %perm_a = amdgcn.alloca : !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %temp, %tok_perm = amdgcn.ds_permute_b32 outs(%perm_a) ins(%addr, %tid_x) args(%c0)
        : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr) args(i32) -> !amdgcn.read_token<shared>
    amdgcn.s_waitcnt lgkmcnt = 0

    // Step 2 - bpermute with same addr: result[i] = temp[(i+1)%64] = i
    %bperm_a = amdgcn.alloca : !amdgcn.vgpr
    %result, %tok_bp = amdgcn.ds_bpermute_b32 outs(%bperm_a) ins(%addr, %temp) args(%c0)
        : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr) args(i32) -> !amdgcn.read_token<shared>
    amdgcn.s_waitcnt lgkmcnt = 0

    // Store result[i] to output[i]; expect output[i] = i
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid_x) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %tok_st = amdgcn.global_store_dword data %result addr %out_ptr offset d(%voffset) + c(%c0) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}
