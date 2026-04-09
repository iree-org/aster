// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --amdgcn-late-waits --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// End-to-end test for G2S (Global-to-LDS) buffer_load_dword with LDS flag.
//
// Each lane loads one dword from global memory directly into LDS via
// buffer_load_dword_lds. The hardware computes the per-lane LDS address as:
//   LDS_ADDR = M0[17:2]*4 + inst_offset + ThreadID*4
//
// With M0=44 and inst_offset=0, lane i writes to LDS byte offset 44 + i*4.
// M0 must be dword-aligned (low 2 bits masked by hardware: M0[17:2]*4).
// After the G2S completes (vmcnt=0), each lane reads its value back from
// LDS via ds_read_b32 and stores to the output buffer.
//
// Arguments:
//   arg0: src buffer pointer (input) - 64 dwords to load
//   arg1: params pointer (input)     - [num_bytes, soffset]
//   arg2: dst buffer pointer (output) - 64 dwords result

// CHECK-LABEL: g2s_roundtrip_kernel:
// CHECK:       s_mov_b32 m0, 44
// CHECK:       buffer_load_dword v{{[0-9]+}}, s[{{.*}}], s{{[0-9]+}} offen lds
// CHECK-NEXT:  s_waitcnt vmcnt(0)
// CHECK:       ds_read_b32
// CHECK:       global_store_dword
// CHECK:       s_endpgm

amdgcn.module @g2s_e2e_mod target = #amdgcn.target<gfx950> isa = #amdgcn.isa<cdna4> {

  func.func private @alloc_vgpr() -> !amdgcn.vgpr

  // Load three 64-bit pointer args from kernarg segment, then dereference
  // params pointer to get scalar values.
  func.func private @load_kernargs()
      -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>,
          !amdgcn.sgpr, !amdgcn.sgpr) {
    %src_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %params_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    %dst_ptr = amdgcn.load_arg 2 : !amdgcn.sgpr<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Load [num_bytes, soffset] from params
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32

    %nbytes_dest = amdgcn.alloca : !amdgcn.sgpr
    %num_bytes, %t0 = amdgcn.load s_load_dword dest %nbytes_dest addr %params_ptr
      offset c(%c0)
      : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>, i32)
        -> !amdgcn.read_token<constant>

    %soff_dest = amdgcn.alloca : !amdgcn.sgpr
    %soffset, %t1 = amdgcn.load s_load_dword dest %soff_dest addr %params_ptr
      offset c(%c4)
      : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>, i32)
        -> !amdgcn.read_token<constant>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    return %src_ptr, %dst_ptr, %num_bytes, %soffset
      : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>,
        !amdgcn.sgpr, !amdgcn.sgpr
  }

  // M0=44 base + 64 lanes * 4 bytes = 300 bytes of LDS (round up to 512)
  amdgcn.kernel @g2s_roundtrip_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 512 : i32} {

    %src_ptr, %dst_ptr, %num_bytes, %soffset =
      func.call @load_kernargs()
        : () -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>,
                 !amdgcn.sgpr, !amdgcn.sgpr)

    // Build buffer descriptor for source (raw mode, stride=0)
    // flags = 0x24924: DST_SEL_XYZW=SEL_X(4), NUM_FORMAT=UINT(4),
    //                  DATA_FORMAT=32(4).
    // CDNA4 requires non-zero DST_SEL for buffer_load_dword_lds;
    // DST_SEL=0 causes all-zeros or garbage on the LDS path.
    %c0_stride = arith.constant 0 : i32
    %src_rsrc = amdgcn.make_buffer_rsrc %src_ptr, %num_bytes, %c0_stride,
      cache_swizzle = false, swizzle_enable = false, flags = 149796
      : (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr<[? + 4]>

    // Thread ID
    %tid = amdgcn.thread_id x : !amdgcn.vgpr

    // Compute byte offset for global memory: tid * 4
    %voff_alloc = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_alloc ins %c2, %tid
      : !amdgcn.vgpr, i32, !amdgcn.vgpr

    // Set M0 = 44 (non-trivial dword-aligned LDS base offset for coverage).
    // Hardware writes to LDS at M0[17:2]*4 + tid*4 = 44 + tid*4.
    // M0 must be dword-aligned (low 2 bits are masked by hardware).
    %m0 = amdgcn.alloca : !amdgcn.m0
    %c44 = arith.constant 44 : i32
    amdgcn.sop1 s_mov_b32 outs %m0 ins %c44 : !amdgcn.m0, i32

    // 1 NOP required after SALU writes M0 before G2S (CDNA4 hazard)
    amdgcn.sopp.sopp #amdgcn.inst<s_nop> , imm = 10

    %c0 = arith.constant 0 : i32

    // G2S: buffer_load_dword with LDS flag
    // Each lane loads src[tid] -> LDS[44 + tid*4]
    %tok_g2s = amdgcn.load_lds buffer_load_dword_lds m0 %m0 addr %src_rsrc
        offset u(%soffset) + d(%voffset) + c(%c0)
        : ins(!amdgcn.m0, !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>

    // Wait for G2S to complete (vmcnt tracks buffer loads).
    // Must use token-based wait so the late-waits pass preserves it.
    amdgcn.wait deps %tok_g2s : !amdgcn.write_token<flat>

    // Read back from LDS: ds_read_b32 at offset 44 + tid*4
    // voffset is tid*4, add M0 offset via constant_offset = 44
    %lds_dest = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %lds_val, %tok_lds = amdgcn.load ds_read_b32 dest %lds_dest addr %voffset
      offset c(%c44)
      : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32)
        -> !amdgcn.read_token<shared>

    // Wait for LDS read to complete (lgkmcnt tracks DS ops)
    amdgcn.wait deps %tok_lds : !amdgcn.read_token<shared>

    // Store result to output: dst[tid] = lds_val
    // Thread offset for output = tid * 4
    %tok_st = amdgcn.store global_store_dword data %lds_val addr %dst_ptr
        offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>

    amdgcn.wait deps %tok_st : !amdgcn.write_token<flat>
    amdgcn.end_kernel
  }
}
