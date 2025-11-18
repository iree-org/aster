// Test memory dependence analysis
// RUN: aster-opt %s --test-memory-dependence-analysis 2>& 1 | FileCheck %s

// This test will be enabled once we integrate the analysis into a test pass

//   CHECK-LABEL: Kernel: test_kernel
amdgcn.module @test_memory_dependence target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    %0 = amdgcn.alloca : !amdgcn.vgpr
    %1 = amdgcn.alloca : !amdgcn.vgpr
    %2 = amdgcn.alloca : !amdgcn.vgpr
    %3 = amdgcn.alloca : !amdgcn.vgpr
    %4 = amdgcn.alloca : !amdgcn.sgpr
    %5 = amdgcn.alloca : !amdgcn.sgpr
    %11 = amdgcn.alloca : !amdgcn.vgpr
    %12 = amdgcn.alloca : !amdgcn.vgpr
    %13 = amdgcn.alloca : !amdgcn.vgpr
    %14 = amdgcn.alloca : !amdgcn.vgpr
    %addr_range = amdgcn.make_register_range %4, %5 : !amdgcn.sgpr, !amdgcn.sgpr
    %dst_range0 = amdgcn.make_register_range %0 : !amdgcn.vgpr
    %dst_range1 = amdgcn.make_register_range %1 : !amdgcn.vgpr
    %dst_range2 = amdgcn.make_register_range %2 : !amdgcn.vgpr
    %dst_range3 = amdgcn.make_register_range %11, %12 : !amdgcn.vgpr, !amdgcn.vgpr

    // Three global loads
    // CHECK: Operation: {{.*}}global_load{{.*}}test.load_tag_0
    // CHECK-NEXT: PENDING BEFORE: 0:
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %6 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %dst_range0, %addr_range, offset = 0 { test.load_tag_0 }
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>

    // CHECK: Operation: {{.*}}global_load{{.*}}test.load_tag_1
    // CHECK-NEXT: PENDING BEFORE: 1: test.load_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %7 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %dst_range1, %addr_range, offset = 4 { test.load_tag_1 }
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>

    // CHECK: Operation: {{.*}}global_load{{.*}}test.load_tag_2
    // CHECK-NEXT: PENDING BEFORE: 2: test.load_tag_0, test.load_tag_1
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %8 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %dst_range2, %addr_range, offset = 8 { test.load_tag_2 }
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>

    // Use second load_tag_1 - forces to also flush load_tag_0
    // CHECK: Operation: {{.*}}split_register_range{{.*}}test.compute_tag_0
    // CHECK-NEXT: PENDING BEFORE: 3: test.load_tag_0, test.load_tag_1, test.load_tag_2
    // CHECK-NEXT: MUST FLUSH NOW: 2: test.load_tag_0, test.load_tag_1
    %split1 = amdgcn.split_register_range %7 { test.compute_tag_0 } : !amdgcn.vgpr_range<[? + 1]>
    %9 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %3, %split1
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    // add ds_write of %9 here
    // CHECK: Operation: {{.*}}ds.write{{.*}}test.ds_write_tag_0
    // CHECK-NEXT: PENDING BEFORE: 1: test.load_tag_2
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %c12_ds_mem = arith.constant 12 : i32
    %ds_data_range = amdgcn.make_register_range %9 : !amdgcn.vgpr
    amdgcn.ds.write #amdgcn.inst<ds_write_b32> %ds_data_range, %13, offset = %c12_ds_mem { test.ds_write_tag_0 }
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr, i32

    // Store depends on the computation
    // CHECK: Operation: {{.*}}global_store{{.*}}test.store_tag_0
    // CHECK-NEXT: PENDING BEFORE: 2: test.load_tag_2, test.ds_write_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %data_range = amdgcn.make_register_range %9 : !amdgcn.vgpr
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %data_range, %addr_range, offset = 12 { test.store_tag_0 }
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>

    // CHECK: Operation: {{.*}}global_store{{.*}}test.store_tag_1
    // CHECK-NEXT: PENDING BEFORE: 3: test.load_tag_2, test.ds_write_tag_0, test.store_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %data_range, %addr_range, offset = 16 { test.store_tag_1 }
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>

    // add ds_read of %addr_range here with the same offset as the ds_write
    // CHECK: Operation: {{.*}}ds.read{{.*}}test.ds_read_tag_0
    // CHECK-NEXT: PENDING BEFORE: 4: test.load_tag_2, test.ds_write_tag_0, test.store_tag_0, test.store_tag_1
    // CHECK-NEXT: MUST FLUSH NOW: 1: test.ds_write_tag_0
    %ds_dst_range = amdgcn.make_register_range %14 : !amdgcn.vgpr
    %ds_14 = amdgcn.ds.read #amdgcn.inst<ds_read_b32> %ds_dst_range, %13, offset = %c12_ds_mem { test.ds_read_tag_0 }
      : !amdgcn.vgpr, i32 -> !amdgcn.vgpr_range<[? + 1]>

    // global_load of 2 VGPRs at offsets 4 and 5
    // CHECK: Operation: {{.*}}global_load{{.*}}test.load_tag_3
    // CHECK-NEXT: PENDING BEFORE: 4: test.load_tag_2, test.store_tag_0, test.store_tag_1, test.ds_read_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 3: test.load_tag_2, test.store_tag_0, test.store_tag_1
    %10 = amdgcn.flat.global_load #amdgcn.inst<global_load_dwordx2> %dst_range3, %addr_range, offset = 16 { test.load_tag_3 }
      : !amdgcn.vgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 2]>

    // global_store of ds_14 here
    // CHECK: Operation: {{.*}}global_store{{.*}}test.global_store_tag_0
    // CHECK-NEXT: PENDING BEFORE: 2: test.ds_read_tag_0, test.load_tag_3
    // CHECK-NEXT: MUST FLUSH NOW: 1: test.ds_read_tag_0
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %ds_14, %addr_range, offset = 12 { test.global_store_tag_0 }
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>

    // CHECK: Operation: {{.*}}end_kernel{{.*}}
    // CHECK-NEXT: PENDING BEFORE: 2: test.load_tag_3, test.global_store_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 2: test.load_tag_3, test.global_store_tag_0
    amdgcn.end_kernel { test.end_tag }
  }


  //   CHECK-LABEL: Kernel: test_timing_ops
  amdgcn.kernel @test_timing_ops arguments <[#amdgcn.buffer_arg<address_space = generic>]> {
    %4 = amdgcn.alloca : !amdgcn.sgpr
    %5 = amdgcn.alloca : !amdgcn.sgpr
    %addr_range = amdgcn.make_register_range %4, %5 : !amdgcn.sgpr, !amdgcn.sgpr

    // s_memtime ops are universal aliases
    //      CHECK: Operation: {{.*}}test.s_memtime_tag_0
    // CHECK-NEXT: PENDING BEFORE:
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %start_time = amdgcn.smem.load #amdgcn.inst<s_memtime> %addr_range { test.s_memtime_tag_0 }
      : !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.sgpr_range<[? + 2]>

    //       CHECK: Operation: {{.*}}test.s_memtime_tag_1
    // CHECK-NEXT: PENDING BEFORE: 1: test.s_memtime_tag_0,
    // CHECK-NEXT: MUST FLUSH NOW: 1: test.s_memtime_tag_0,
    %end_time = amdgcn.smem.load #amdgcn.inst<s_memtime> %addr_range { test.s_memtime_tag_1 }
      : !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.sgpr_range<[? + 2]>


    //      CHECK: Operation: {{.*}}{test.s_store_dwordx2_tag_0}
    // CHECK-NEXT: PENDING BEFORE: 1: test.s_memtime_tag_1,
    // CHECK-NEXT: MUST FLUSH NOW: 1: test.s_memtime_tag_1,
    // write the end time to the buffer to avoid DCE (dataflow analysis forces this)
    amdgcn.smem.store #amdgcn.inst<s_store_dwordx2> %end_time, %addr_range, offset = 128 { test.s_store_dwordx2_tag_0 }
      : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>

    //      CHECK: Operation: {{.*}}test.end_tag
    // CHECK-NEXT: PENDING BEFORE: 1: test.s_store_dwordx2_tag_0,
    // CHECK-NEXT: MUST FLUSH NOW: 1: test.s_store_dwordx2_tag_0,
    amdgcn.end_kernel { test.end_tag }
  }

  //   CHECK-LABEL: Kernel: test_noalias_pointers
  amdgcn.kernel @test_noalias_pointers {
    %0 = amdgcn.alloca : !amdgcn.vgpr
    %1 = amdgcn.alloca : !amdgcn.vgpr
    %2 = amdgcn.alloca : !amdgcn.sgpr
    %3 = amdgcn.alloca : !amdgcn.sgpr
    %4 = amdgcn.alloca : !amdgcn.sgpr
    %5 = amdgcn.alloca : !amdgcn.sgpr
    %addr_range1 = amdgcn.make_register_range %2, %3 : !amdgcn.sgpr, !amdgcn.sgpr
    %addr_range2 = amdgcn.make_register_range %4, %5 : !amdgcn.sgpr, !amdgcn.sgpr
    %dst_range = amdgcn.make_register_range %0 : !amdgcn.vgpr
    %data_range = amdgcn.make_register_range %1 : !amdgcn.vgpr

    // Mark pointers as non-aliasing using lsir.assume_noalias
    %ptr1_noalias, %ptr2_noalias = lsir.assume_noalias %addr_range1, %addr_range2
      : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)
      -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)

    // Store to ptr1 - this creates a pending memory operation
    // CHECK: Operation: {{.*}}global_store{{.*}}test.store_noalias_tag_0
    // CHECK-NEXT: PENDING BEFORE: 0:
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %data_range, %ptr1_noalias, offset = 0 { test.store_noalias_tag_0 }
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>

    // Load from ptr2 - because of lsir.assume_noalias, this should NOT require
    // synchronization with the store to ptr1 (they don't alias)
    // CHECK: Operation: {{.*}}global_load{{.*}}test.load_noalias_tag_0
    // CHECK-NEXT: PENDING BEFORE: 1: test.store_noalias_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %loaded = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %dst_range, %ptr2_noalias, offset = 0 { test.load_noalias_tag_0 }
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>

    // CHECK: Operation: {{.*}}end_kernel{{.*}}
    // CHECK-NEXT: PENDING BEFORE: 2: test.store_noalias_tag_0, test.load_noalias_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 2: test.store_noalias_tag_0, test.load_noalias_tag_0
    amdgcn.end_kernel { test.end_noalias_tag }
  }
}
