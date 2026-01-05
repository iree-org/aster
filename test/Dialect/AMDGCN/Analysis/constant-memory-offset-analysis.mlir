// RUN: aster-opt %s --test-constant-memory-offset-analysis --allow-unregistered-dialect | FileCheck %s

amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  // CHECK-LABEL: Kernel: test_kernel
  amdgcn.kernel @test_kernel {
    %i, %j, %k = "make_indices"() : () -> (index, index, index)

    %offset_index = affine.apply affine_map<()[s0, s1, s2] -> (16 + s0 * s1 * s2)>()[%i, %j, %k]
    %offset_i32 = index.casts %offset_index : index to i32
    %vgpr_alloc = amdgcn.alloca : !amdgcn.vgpr
    %offset_vgpr = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vgpr_alloc, %offset_i32
      : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr

    %vgpr_load_dst = amdgcn.alloca : !amdgcn.vgpr
    %vgpr_load_dst_range = amdgcn.make_register_range %vgpr_load_dst : !amdgcn.vgpr
    %vgpr_addr_lo = amdgcn.alloca : !amdgcn.vgpr
    %vgpr_addr_hi = amdgcn.alloca : !amdgcn.vgpr
    %vgpr_addr = amdgcn.make_register_range %vgpr_addr_lo, %vgpr_addr_hi : !amdgcn.vgpr, !amdgcn.vgpr
    // CHECK: Operation: {{.*}}amdgcn.flat.global_load
    // CHECK-NEXT: CONSTANT OFFSET: 16 affine_map=()[s0, s1, s2] -> ((s0 * s1) * s2)
    %loaded = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %vgpr_load_dst_range, %vgpr_addr[%offset_vgpr]
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>[!amdgcn.vgpr] -> !amdgcn.vgpr_range<[? + 1]>

    %sgpr_alloc_lo = amdgcn.alloca : !amdgcn.sgpr
    %sgpr_alloc_hi = amdgcn.alloca : !amdgcn.sgpr
    %sgpr_addr = amdgcn.make_register_range %sgpr_alloc_lo, %sgpr_alloc_hi : !amdgcn.sgpr, !amdgcn.sgpr
    // CHECK: Operation: amdgcn.flat.global_store
    // CHECK-NEXT: CONSTANT OFFSET: 16 affine_map=()[s0, s1, s2] -> ((s0 * s1) * s2)
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %loaded, %sgpr_addr[%offset_vgpr]
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>[!amdgcn.vgpr]

    %sgpr_load_dst = amdgcn.alloca : !amdgcn.sgpr
    %sgpr_load_dst_range = amdgcn.make_register_range %sgpr_load_dst : !amdgcn.sgpr
    %sgpr_smem_addr_lo = amdgcn.alloca : !amdgcn.sgpr
    %sgpr_smem_addr_hi = amdgcn.alloca : !amdgcn.sgpr
    %sgpr_smem_addr = amdgcn.make_register_range %sgpr_smem_addr_lo, %sgpr_smem_addr_hi : !amdgcn.sgpr, !amdgcn.sgpr
    // SMEM operations only have static offset attribute, skip
    amdgcn.smem.load #amdgcn.inst<s_load_dword> %sgpr_load_dst_range, %sgpr_smem_addr
      : !amdgcn.sgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.sgpr_range<[? + 1]>

    %sgpr_store_data = amdgcn.alloca : !amdgcn.sgpr
    %sgpr_store_data_range = amdgcn.make_register_range %sgpr_store_data : !amdgcn.sgpr
    // SMEM operations only have static offset attribute, skip
    amdgcn.smem.store #amdgcn.inst<s_store_dword> %sgpr_store_data_range, %sgpr_smem_addr
      : !amdgcn.sgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>

    %vgpr_ds_read_dst = amdgcn.alloca : !amdgcn.vgpr
    %vgpr_ds_read_dst_range = amdgcn.make_register_range %vgpr_ds_read_dst : !amdgcn.vgpr
    %vgpr_ds_addr = amdgcn.alloca : !amdgcn.vgpr
    // CHECK: Operation: {{.*}}amdgcn.ds.read
    // CHECK-NEXT: CONSTANT OFFSET: 16 affine_map=()[s0, s1, s2] -> ((s0 * s1) * s2)
    %ds_read_result = amdgcn.ds.read #amdgcn.inst<ds_read_b32> %vgpr_ds_read_dst_range, %vgpr_ds_addr, offset = %offset_i32
      : !amdgcn.vgpr, i32 -> !amdgcn.vgpr_range<[? + 1]>

    // Test DSWriteOp
    %vgpr_ds_write_data = amdgcn.alloca : !amdgcn.vgpr
    %vgpr_ds_write_data_range = amdgcn.make_register_range %vgpr_ds_write_data : !amdgcn.vgpr
    // CHECK: Operation: amdgcn.ds.write
    // CHECK-NEXT: CONSTANT OFFSET: 16 affine_map=()[s0, s1, s2] -> ((s0 * s1) * s2)
    amdgcn.ds.write #amdgcn.inst<ds_write_b32> %vgpr_ds_write_data_range, %vgpr_ds_addr, offset = %offset_i32
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr, i32

    amdgcn.end_kernel
  }
}
