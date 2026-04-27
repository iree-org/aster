// Vector addition: c[tid] = a[tid] + b[tid]
// First kernel that loads data from memory, computes, and stores back.
//
// The kernarg loading boilerplate is isolated in a func.func and inlined
// by the compiler. This is asm-level composition: reusable functions that
// compile down to zero-overhead inline code.
//
// Register map:
//   s[0:1] = kernarg base (hardware-initialized)
//   s[2:3] = a_ptr, s[4:5] = b_ptr, s[6:7] = c_ptr
//   v0     = thread ID (hardware-initialized)
//   v1     = byte offset = tid * 4
//   v2     = a[tid], then reused for a[tid] + b[tid]
//   v3     = b[tid]

module {
  amdgcn.module @vadd target = <gfx942> {

    // Reusable: load three buffer pointers from kernarg segment.
    func.func private @load_3_ptrs(%kernarg: !amdgcn.sgpr<[0 : 2]>)
        -> (!amdgcn.sgpr<[2 : 4]>, !amdgcn.sgpr<[4 : 6]>, !amdgcn.sgpr<[6 : 8]>) {
      %s2 = amdgcn.alloca : !amdgcn.sgpr<2>
      %s3 = amdgcn.alloca : !amdgcn.sgpr<3>
      %s4 = amdgcn.alloca : !amdgcn.sgpr<4>
      %s5 = amdgcn.alloca : !amdgcn.sgpr<5>
      %s6 = amdgcn.alloca : !amdgcn.sgpr<6>
      %s7 = amdgcn.alloca : !amdgcn.sgpr<7>

      %a = amdgcn.make_register_range %s2, %s3
        : !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
      %b = amdgcn.make_register_range %s4, %s5
        : !amdgcn.sgpr<4>, !amdgcn.sgpr<5>
      %c = amdgcn.make_register_range %s6, %s7
        : !amdgcn.sgpr<6>, !amdgcn.sgpr<7>

      %c0  = arith.constant 0  : i32
      %c8  = arith.constant 8  : i32
      %c16 = arith.constant 16 : i32

      amdgcn.load s_load_dwordx2 dest %a addr %kernarg
        offset c(%c0)
        : dps(!amdgcn.sgpr<[2 : 4]>) ins(!amdgcn.sgpr<[0 : 2]>, i32)
          -> !amdgcn.read_token<constant>
      amdgcn.load s_load_dwordx2 dest %b addr %kernarg
        offset c(%c8)
        : dps(!amdgcn.sgpr<[4 : 6]>) ins(!amdgcn.sgpr<[0 : 2]>, i32)
          -> !amdgcn.read_token<constant>
      amdgcn.load s_load_dwordx2 dest %c addr %kernarg
        offset c(%c16)
        : dps(!amdgcn.sgpr<[6 : 8]>) ins(!amdgcn.sgpr<[0 : 2]>, i32)
          -> !amdgcn.read_token<constant>
      amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

      return %a, %b, %c
        : !amdgcn.sgpr<[2 : 4]>, !amdgcn.sgpr<[4 : 6]>, !amdgcn.sgpr<[6 : 8]>
    }

    kernel @kernel arguments <[
      #amdgcn.buffer_arg<address_space = generic, access = read_only>,
      #amdgcn.buffer_arg<address_space = generic, access = read_only>,
      #amdgcn.buffer_arg<address_space = generic, access = write_only>
    ]> attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
      %s0 = alloca : !amdgcn.sgpr<0>
      %s1 = alloca : !amdgcn.sgpr<1>
      %v0 = alloca : !amdgcn.vgpr<0>
      %v1 = alloca : !amdgcn.vgpr<1>
      %v2 = alloca : !amdgcn.vgpr<2>
      %v3 = alloca : !amdgcn.vgpr<3>

      // Load buffer pointers (encapsulated in a reusable function)
      %kernarg = amdgcn.make_register_range %s0, %s1
        : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
      %a_ptr, %b_ptr, %c_ptr = func.call @load_3_ptrs(%kernarg)
        : (!amdgcn.sgpr<[0 : 2]>)
          -> (!amdgcn.sgpr<[2 : 4]>, !amdgcn.sgpr<[4 : 6]>, !amdgcn.sgpr<[6 : 8]>)

      // Byte offset = tid * 4
      %c2 = arith.constant 2 : i32
      vop2 v_lshlrev_b32_e32 outs %v1 ins %c2, %v0
        : !amdgcn.vgpr<1>, i32, !amdgcn.vgpr<0>

      // Load a[tid] and b[tid]
      %a_data = amdgcn.make_register_range %v2 : !amdgcn.vgpr<2>
      %c0 = arith.constant 0 : i32
      amdgcn.load global_load_dword dest %a_data addr %a_ptr
        offset d(%v1) + c(%c0)
        : dps(!amdgcn.vgpr<[2 : 3]>) ins(!amdgcn.sgpr<[2 : 4]>, !amdgcn.vgpr<1>, i32)
          -> !amdgcn.read_token<flat>

      %b_data = amdgcn.make_register_range %v3 : !amdgcn.vgpr<3>
      amdgcn.load global_load_dword dest %b_data addr %b_ptr
        offset d(%v1) + c(%c0)
        : dps(!amdgcn.vgpr<[3 : 4]>) ins(!amdgcn.sgpr<[4 : 6]>, !amdgcn.vgpr<1>, i32)
          -> !amdgcn.read_token<flat>

      // Wait for all outstanding vector memory operations
      amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

      // c[tid] = a[tid] + b[tid]
      vop2 v_add_u32 outs %v2 ins %v2, %v3
        : !amdgcn.vgpr<2>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>

      // Store result
      %c_data = amdgcn.make_register_range %v2 : !amdgcn.vgpr<2>
      amdgcn.store global_store_dword data %c_data addr %c_ptr
        offset d(%v1) + c(%c0)
        : ins(!amdgcn.vgpr<[2 : 3]>, !amdgcn.sgpr<[6 : 8]>, !amdgcn.vgpr<1>, i32)
          -> !amdgcn.write_token<flat>

      amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
      end_kernel
    }
  }
}
