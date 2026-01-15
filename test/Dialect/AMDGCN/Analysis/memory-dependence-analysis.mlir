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

    // Define offset constants
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %c12 = arith.constant 12 : i32
    %c16 = arith.constant 16 : i32
    %c128 = arith.constant 128 : i32

    // Three global loads
    // CHECK: Operation: {{.*}}global_load{{.*}}test.load_tag_0
    // CHECK-NEXT: PENDING BEFORE: 0:
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %6, %tok1 = amdgcn.load global_load_dword dest %dst_range0 addr %addr_range offset c(%c0) { test.load_tag_0 } : dps(!amdgcn.vgpr_range<[? + 1]>) ins(!amdgcn.sgpr_range<[? + 2]>, i32) -> !amdgcn.read_token<flat>

    // CHECK: Operation: {{.*}}global_load{{.*}}test.load_tag_1
    // CHECK-NEXT: PENDING BEFORE: 1: test.load_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %7, %tok2 = amdgcn.load global_load_dword dest %dst_range1 addr %addr_range offset c(%c4) { test.load_tag_1 } : dps(!amdgcn.vgpr_range<[? + 1]>) ins(!amdgcn.sgpr_range<[? + 2]>, i32) -> !amdgcn.read_token<flat>

    // CHECK: Operation: {{.*}}global_load{{.*}}test.load_tag_2
    // CHECK-NEXT: PENDING BEFORE: 2: test.load_tag_0, test.load_tag_1
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %8, %tok3 = amdgcn.load global_load_dword dest %dst_range2 addr %addr_range offset c(%c8) { test.load_tag_2 } : dps(!amdgcn.vgpr_range<[? + 1]>) ins(!amdgcn.sgpr_range<[? + 2]>, i32) -> !amdgcn.read_token<flat>

    // Use second load_tag_1 - forces to also flush load_tag_0
    // CHECK: Operation: {{.*}}split_register_range{{.*}}test.compute_tag_0
    // CHECK-NEXT: PENDING BEFORE: 3: test.load_tag_0, test.load_tag_1, test.load_tag_2
    // CHECK-NEXT: MUST FLUSH NOW: 2: test.load_tag_0, test.load_tag_1
    %split1 = amdgcn.split_register_range %7 { test.compute_tag_0 } : !amdgcn.vgpr_range<[? + 1]>
    %9 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %3, %split1
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    // add ds_write of %9 here
    // CHECK: Operation: {{.*}}ds_write{{.*}}test.ds_write_tag_0
    // CHECK-NEXT: PENDING BEFORE: 1: test.load_tag_2
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %c12_ds_mem = arith.constant 12 : i32
    %ds_data_range = amdgcn.make_register_range %9 : !amdgcn.vgpr
    %tok_ds1 = amdgcn.store ds_write_b32 data %ds_data_range addr %13 offset c(%c12_ds_mem) { test.ds_write_tag_0 }
      : ins(!amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // Store depends on the computation
    // CHECK: Operation: {{.*}}global_store{{.*}}test.store_tag_0
    // CHECK-NEXT: PENDING BEFORE: 2: test.load_tag_2, test.ds_write_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %data_range = amdgcn.make_register_range %9 : !amdgcn.vgpr
    %tok4 = amdgcn.store global_store_dword data %data_range addr %addr_range offset c(%c12) { test.store_tag_0 } : ins(!amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>, i32) -> !amdgcn.write_token<flat>

    // CHECK: Operation: {{.*}}global_store{{.*}}test.store_tag_1
    // CHECK-NEXT: PENDING BEFORE: 3: test.load_tag_2, test.ds_write_tag_0, test.store_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %tok5 = amdgcn.store global_store_dword data %data_range addr %addr_range offset c(%c16) { test.store_tag_1 } : ins(!amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>, i32) -> !amdgcn.write_token<flat>

    // add ds_read of %addr_range here with the same offset as the ds_write
    // CHECK: Operation: {{.*}}ds_read{{.*}}test.ds_read_tag_0
    // CHECK-NEXT: PENDING BEFORE: 4: test.load_tag_2, test.ds_write_tag_0, test.store_tag_0, test.store_tag_1
    // CHECK-NEXT: MUST FLUSH NOW: 1: test.ds_write_tag_0
    %ds_dst_range = amdgcn.make_register_range %14 : !amdgcn.vgpr
    %ds_14, %tok_ds2 = amdgcn.load ds_read_b32 dest %ds_dst_range addr %13 offset c(%c12_ds_mem) { test.ds_read_tag_0 }
      : dps(!amdgcn.vgpr_range<[? + 1]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>

    // global_load of 2 VGPRs at offsets 4 and 5
    // CHECK: Operation: {{.*}}global_load{{.*}}test.load_tag_3
    // CHECK-NEXT: PENDING BEFORE: 4: test.load_tag_2, test.store_tag_0, test.store_tag_1, test.ds_read_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 3: test.load_tag_2, test.store_tag_0, test.store_tag_1
    %10, %tok6 = amdgcn.load global_load_dwordx2 dest %dst_range3 addr %addr_range offset c(%c16) { test.load_tag_3 } : dps(!amdgcn.vgpr_range<[? + 2]>) ins(!amdgcn.sgpr_range<[? + 2]>, i32) -> !amdgcn.read_token<flat>

    // global_store of ds_14 here
    // CHECK: Operation: {{.*}}global_store{{.*}}test.global_store_tag_0
    // CHECK-NEXT: PENDING BEFORE: 2: test.ds_read_tag_0, test.load_tag_3
    // CHECK-NEXT: MUST FLUSH NOW: 1: test.ds_read_tag_0
    %tok7 = amdgcn.store global_store_dword data %ds_14 addr %addr_range offset c(%c12) { test.global_store_tag_0 } : ins(!amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>, i32) -> !amdgcn.write_token<flat>

    // CHECK: Operation: {{.*}}end_kernel{{.*}}
    // CHECK-NEXT: PENDING BEFORE: 2: test.load_tag_3, test.global_store_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 2: test.load_tag_3, test.global_store_tag_0
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

    %c0_na = arith.constant 0 : i32

    // Store to ptr1 - this creates a pending memory operation
    // CHECK: Operation: {{.*}}global_store{{.*}}test.store_noalias_tag_0
    // CHECK-NEXT: PENDING BEFORE: 0:
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %tok_na1 = amdgcn.store global_store_dword data %data_range addr %ptr1_noalias offset c(%c0_na) { test.store_noalias_tag_0 } : ins(!amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>, i32) -> !amdgcn.write_token<flat>

    // Load from ptr2 - because of lsir.assume_noalias, this should NOT require
    // synchronization with the store to ptr1 (they don't alias)
    // CHECK: Operation: {{.*}}global_load{{.*}}test.load_noalias_tag_0
    // CHECK-NEXT: PENDING BEFORE: 1: test.store_noalias_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 0:
    %loaded, %tok_na2 = amdgcn.load global_load_dword dest %dst_range addr %ptr2_noalias offset c(%c0_na) { test.load_noalias_tag_0 } : dps(!amdgcn.vgpr_range<[? + 1]>) ins(!amdgcn.sgpr_range<[? + 2]>, i32) -> !amdgcn.read_token<flat>

    // CHECK: Operation: {{.*}}end_kernel{{.*}}
    // CHECK-NEXT: PENDING BEFORE: 2: test.store_noalias_tag_0, test.load_noalias_tag_0
    // CHECK-NEXT: MUST FLUSH NOW: 2: test.store_noalias_tag_0, test.load_noalias_tag_0
    amdgcn.end_kernel { test.end_noalias_tag }
  }
}
