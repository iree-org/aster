// Test proper counting of multiple global_load operations
// RUN: aster-opt %s --amdgcn-optimize-straight-line-waits --split-input-file | FileCheck %s

//   CHECK-LABEL: amdgcn.module @test_global_load_counting
//     CHECK-NOT:   amdgcn.sopp.s_waitcnt
// CHECK-COUNT-3:   amdgcn.flat.global_load
// Wait for first 2 -> vmcnt reaches 1
//         CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1
//         CHECK:   amdgcn.vop1.vop1
//         CHECK:   amdgcn.flat.global_store
//         CHECK:   amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
//    CHECK-NEXT:   end_kernel
amdgcn.module @test_global_load_counting target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    %0 = amdgcn.alloca : !amdgcn.vgpr
    %1 = amdgcn.alloca : !amdgcn.vgpr
    %2 = amdgcn.alloca : !amdgcn.vgpr
    %3 = amdgcn.alloca : !amdgcn.vgpr
    %4 = amdgcn.alloca : !amdgcn.sgpr
    %5 = amdgcn.alloca : !amdgcn.sgpr
    %addr_range = amdgcn.make_register_range %4, %5 : !amdgcn.sgpr, !amdgcn.sgpr

    // Three global loads
    %dst_range0 = amdgcn.make_register_range %0 : !amdgcn.vgpr
    %6 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %dst_range0, %addr_range, offset = 0
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>
    %dst_range1 = amdgcn.make_register_range %1 : !amdgcn.vgpr
    %7 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %dst_range1, %addr_range, offset = 4
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>
    %dst_range2 = amdgcn.make_register_range %2 : !amdgcn.vgpr
    %8 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %dst_range2, %addr_range, offset = 8
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>

    // Use second load (index 1) - should wait for 2 operations (indices 0 and 1)
    %split1 = amdgcn.split_register_range %7 : !amdgcn.vgpr_range<[? + 1]>
    %9 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %3, %split1
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %data_range = amdgcn.make_register_range %9 : !amdgcn.vgpr
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %data_range, %addr_range, offset = 12
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>

    amdgcn.end_kernel
  }
}

// -----

// CHECK-LABEL: amdgcn.module @test_ds_read_counting
//     CHECK-NOT:   amdgcn.sopp.s_waitcnt
// CHECK-COUNT-3:   amdgcn.ds.read
// Wait for first 2 -> lgkmcnt reaches 1
//         CHECK:   amdgcn.sopp.s_waitcnt {{.*}} lgkmcnt = 1
//         CHECK:   amdgcn.vop1.vop1
//         CHECK:   amdgcn.ds.write
//         CHECK:   amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
//    CHECK-NEXT:   end_kernel
amdgcn.module @test_ds_read_counting target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    %0 = amdgcn.alloca : !amdgcn.vgpr
    %1 = amdgcn.alloca : !amdgcn.vgpr
    %2 = amdgcn.alloca : !amdgcn.vgpr
    %3 = amdgcn.alloca : !amdgcn.vgpr
    %4 = amdgcn.alloca : !amdgcn.vgpr

    // Three ds reads
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %c12 = arith.constant 12 : i32
    %dst_range0 = amdgcn.make_register_range %0 : !amdgcn.vgpr
    %5 = amdgcn.ds.read #amdgcn.inst<ds_read_b32> %dst_range0, %4, offset = %c0
      : !amdgcn.vgpr, i32 -> !amdgcn.vgpr_range<[? + 1]>
    %dst_range1 = amdgcn.make_register_range %1 : !amdgcn.vgpr
    %6 = amdgcn.ds.read #amdgcn.inst<ds_read_b32> %dst_range1, %4, offset = %c4
      : !amdgcn.vgpr, i32 -> !amdgcn.vgpr_range<[? + 1]>
    %dst_range2 = amdgcn.make_register_range %2 : !amdgcn.vgpr
    %7 = amdgcn.ds.read #amdgcn.inst<ds_read_b32> %dst_range2, %4, offset = %c8
      : !amdgcn.vgpr, i32 -> !amdgcn.vgpr_range<[? + 1]>

    // Use second read (index 1) - should wait for 2 operations (indices 0 and 1)
    %split1 = amdgcn.split_register_range %6 : !amdgcn.vgpr_range<[? + 1]>
    %8 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %3, %split1
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %data_range = amdgcn.make_register_range %8 : !amdgcn.vgpr
    amdgcn.ds.write #amdgcn.inst<ds_write_b32> %data_range, %4, offset = %c12
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr, i32

    amdgcn.end_kernel
  }
}

// -----

// CHECK-LABEL: amdgcn.module @test_global_mixed_counting
amdgcn.module @test_global_mixed_counting target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    %0 = amdgcn.alloca : !amdgcn.vgpr
    %1 = amdgcn.alloca : !amdgcn.vgpr
    %2 = amdgcn.alloca : !amdgcn.vgpr
    %3 = amdgcn.alloca : !amdgcn.vgpr
    %4 = amdgcn.alloca : !amdgcn.sgpr
    %5 = amdgcn.alloca : !amdgcn.sgpr
    %addr_range = amdgcn.make_register_range %4, %5 : !amdgcn.sgpr, !amdgcn.sgpr

    // Two global loads
    %dst_range0 = amdgcn.make_register_range %0 : !amdgcn.vgpr
    %6 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %dst_range0, %addr_range, offset = 0
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>
    // Store at non aliasing locatio
    %data_range0 = amdgcn.make_register_range %2 : !amdgcn.vgpr
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %data_range0, %addr_range, offset = 8
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>
    %dst_range1 = amdgcn.make_register_range %1 : !amdgcn.vgpr
    %7 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %dst_range1, %addr_range, offset = 4
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>

    //     CHECK-NOT:   amdgcn.sopp.s_waitcnt
    //         CHECK:   amdgcn.flat.global_load
    //         CHECK:   amdgcn.flat.global_store
    //         CHECK:   amdgcn.flat.global_load
    // Use second load (index 1) - needs to  wait for all 3 memory operations
    //         CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 0
    //         CHECK:   amdgcn.vop1.vop1
    %split1 = amdgcn.split_register_range %7 : !amdgcn.vgpr_range<[? + 1]>
    %8 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %3, %split1
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    //         CHECK:   amdgcn.flat.global_store
    //         CHECK:   amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
    //    CHECK-NEXT:   end_kernel
    %data_range1 = amdgcn.make_register_range %8 : !amdgcn.vgpr
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %data_range1, %addr_range, offset = 12
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>

    amdgcn.end_kernel
  }
}

// -----

// CHECK-LABEL: amdgcn.module @test_ds_mixed_counting
amdgcn.module @test_ds_mixed_counting target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    // Allocate registers for loads
    %g0 = amdgcn.alloca : !amdgcn.vgpr
    %g1 = amdgcn.alloca : !amdgcn.vgpr
    %g2 = amdgcn.alloca : !amdgcn.vgpr
    %g3 = amdgcn.alloca : !amdgcn.vgpr
    %g4 = amdgcn.alloca : !amdgcn.vgpr
    %g5 = amdgcn.alloca : !amdgcn.vgpr

    // Allocate registers for ds operations (need 2 per read for b64)
    %ds0 = amdgcn.alloca : !amdgcn.vgpr
    %ds1 = amdgcn.alloca : !amdgcn.vgpr
    %ds2 = amdgcn.alloca : !amdgcn.vgpr
    %ds3 = amdgcn.alloca : !amdgcn.vgpr
    %ds4 = amdgcn.alloca : !amdgcn.vgpr
    %ds5 = amdgcn.alloca : !amdgcn.vgpr
    %ds6 = amdgcn.alloca : !amdgcn.vgpr
    %ds7 = amdgcn.alloca : !amdgcn.vgpr
    %ds8 = amdgcn.alloca : !amdgcn.vgpr
    %ds9 = amdgcn.alloca : !amdgcn.vgpr
    %ds10 = amdgcn.alloca : !amdgcn.vgpr
    %ds11 = amdgcn.alloca : !amdgcn.vgpr
    %ds_addr = amdgcn.alloca : !amdgcn.vgpr

    // Allocate registers for mfma (a: 2, b: 2, c: 4)
    %a0 = amdgcn.alloca : !amdgcn.vgpr
    %a1 = amdgcn.alloca : !amdgcn.vgpr
    %b0 = amdgcn.alloca : !amdgcn.vgpr
    %b1 = amdgcn.alloca : !amdgcn.vgpr
    %c0 = amdgcn.alloca : !amdgcn.vgpr
    %c1 = amdgcn.alloca : !amdgcn.vgpr
    %c2 = amdgcn.alloca : !amdgcn.vgpr
    %c3 = amdgcn.alloca : !amdgcn.vgpr

    // Allocate registers for stores and final loads
    %s0 = amdgcn.alloca : !amdgcn.vgpr
    %s1 = amdgcn.alloca : !amdgcn.vgpr
    %s2 = amdgcn.alloca : !amdgcn.vgpr
    %s3 = amdgcn.alloca : !amdgcn.vgpr
    %s4 = amdgcn.alloca : !amdgcn.vgpr
    %s5 = amdgcn.alloca : !amdgcn.vgpr

    %f0 = amdgcn.alloca : !amdgcn.vgpr
    %f1 = amdgcn.alloca : !amdgcn.vgpr
    %f2 = amdgcn.alloca : !amdgcn.vgpr
    %f3 = amdgcn.alloca : !amdgcn.vgpr
    %f4 = amdgcn.alloca : !amdgcn.vgpr
    %f5 = amdgcn.alloca : !amdgcn.vgpr

    %addr = amdgcn.alloca : !amdgcn.sgpr
    %addr2 = amdgcn.alloca : !amdgcn.sgpr
    %addr_range = amdgcn.make_register_range %addr, %addr2 : !amdgcn.sgpr, !amdgcn.sgpr


    // Here we have 0 vmcnt and 0 lgkmcnt in flight.

    // 1. Six global loads
    //     CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK-COUNT-6:   amdgcn.flat.global_load
    %g_range0 = amdgcn.make_register_range %g0 : !amdgcn.vgpr
    %g_range1 = amdgcn.make_register_range %g1 : !amdgcn.vgpr
    %g_range2 = amdgcn.make_register_range %g2 : !amdgcn.vgpr
    %g_range3 = amdgcn.make_register_range %g3 : !amdgcn.vgpr
    %g_range4 = amdgcn.make_register_range %g4 : !amdgcn.vgpr
    %g_range5 = amdgcn.make_register_range %g5 : !amdgcn.vgpr
    %gl0 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %g_range0, %addr_range, offset = 0
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>
    %gl1 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %g_range1, %addr_range, offset = 4
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>
    %gl2 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %g_range2, %addr_range, offset = 8
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>
    %gl3 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %g_range3, %addr_range, offset = 12
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>
    %gl4 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %g_range4, %addr_range, offset = 16
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>
    %gl5 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %g_range5, %addr_range, offset = 20
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>

    // Here we have 6 vmcnt and 0 lgkmcnt in flight.

    // 2. Six ds_write operations depending on loads 0, 2, 4 (non-trivial pattern)
    // Write 0 uses load 2, needs to wait for 0, 1, 2 -> vmcnt reaches 3 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 3 expcnt = 0 lgkmcnt = 0
    // CHECK:   amdgcn.ds.write
    %c0_ds = arith.constant 0 : i32
    %c4_ds = arith.constant 4 : i32
    %c8_ds = arith.constant 8 : i32
    %c12_ds = arith.constant 12 : i32
    %c16_ds = arith.constant 16 : i32
    %c20_ds = arith.constant 20 : i32
    %split_gl0 = amdgcn.split_register_range %gl2 : !amdgcn.vgpr_range<[? + 1]>
    %ds_wr_range0 = amdgcn.make_register_range %split_gl0 : !amdgcn.vgpr
    amdgcn.ds.write #amdgcn.inst<ds_write_b32> %ds_wr_range0, %ds_addr, offset = %c0_ds
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr, i32
    // Write 1 uses load 0 (already covered)
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.ds.write
    %split_gl0_2 = amdgcn.split_register_range %gl0 : !amdgcn.vgpr_range<[? + 1]>
    %ds_wr_range1 = amdgcn.make_register_range %split_gl0_2 : !amdgcn.vgpr
    amdgcn.ds.write #amdgcn.inst<ds_write_b32> %ds_wr_range1, %ds_addr, offset = %c4_ds
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr, i32
    // Write 2 uses load 2 (already covered)
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.ds.write
    %split_gl2 = amdgcn.split_register_range %gl2 : !amdgcn.vgpr_range<[? + 1]>
    %ds_wr_range2 = amdgcn.make_register_range %split_gl2 : !amdgcn.vgpr
    amdgcn.ds.write #amdgcn.inst<ds_write_b32> %ds_wr_range2, %ds_addr, offset = %c8_ds
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr, i32
    // Write 3 uses load 2 (already covered)
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.ds.write
    %split_gl2_2 = amdgcn.split_register_range %gl2 : !amdgcn.vgpr_range<[? + 1]>
    %ds_wr_range3 = amdgcn.make_register_range %split_gl2_2 : !amdgcn.vgpr
    amdgcn.ds.write #amdgcn.inst<ds_write_b32> %ds_wr_range3, %ds_addr, offset = %c12_ds
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr, i32
    // Write 4 uses load 4, need to wait for 3, 4 -> vmcnt reaches 1 in flight.
    //                                            -> lgkmcnt has 4 in flight (no wait needed)
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 4
    // CHECK:   amdgcn.ds.write
    %split_gl4 = amdgcn.split_register_range %gl4 : !amdgcn.vgpr_range<[? + 1]>
    %ds_wr_range4 = amdgcn.make_register_range %split_gl4 : !amdgcn.vgpr
    amdgcn.ds.write #amdgcn.inst<ds_write_b32> %ds_wr_range4, %ds_addr, offset = %c16_ds
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr, i32
    // Write 5 uses load 4 (already covered)
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.ds.write
    %split_gl4_2 = amdgcn.split_register_range %gl4 : !amdgcn.vgpr_range<[? + 1]>
    %ds_wr_range5 = amdgcn.make_register_range %split_gl4_2 : !amdgcn.vgpr
    amdgcn.ds.write #amdgcn.inst<ds_write_b32> %ds_wr_range5, %ds_addr, offset = %c20_ds
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr, i32

    // Here we have 1 vmcnt and 6 lgkmcnt in flight.

    // 3. Six ds_read operations depending on writes 1, 3 and 5 respectively.
    //    This time, we use b64 (2 VGPRs) due to MFMA, which will alias more.
    // ds_read_b64 [4 .. 12) aliases with ds_write 4 and 8
    // need to wait for {0, 1, 2} -> vmcnt remains 1 in flight (no wait needed)
    //                            -> lgkmcnt reaches 3 in flight
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 3
    // CHECK:   ds_read_b64
    %ds_rd_range0 = amdgcn.make_register_range %ds0, %ds1 : !amdgcn.vgpr, !amdgcn.vgpr
    %dr0 = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %ds_rd_range0, %ds_addr, offset = %c4_ds
      : !amdgcn.vgpr, i32 -> !amdgcn.vgpr_range<[? + 2]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   ds_read_b64
    %ds_rd_range1 = amdgcn.make_register_range %ds2, %ds3 : !amdgcn.vgpr, !amdgcn.vgpr
    %dr1 = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %ds_rd_range1, %ds_addr, offset = %c4_ds
      : !amdgcn.vgpr, i32 -> !amdgcn.vgpr_range<[? + 2]>
    // ds_read_b64 [12 .. 20) aliases with ds_write 12 and 16
    //   need to wait for {3, 4} with {5} still in flight.
    //   also need to account for the 2 ds_read in flight
    //   -> vmcnt remains 1, lgkmcnt reaches 3 in flight
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 3
    // CHECK:   ds_read_b64
    %ds_rd_range2 = amdgcn.make_register_range %ds4, %ds5 : !amdgcn.vgpr, !amdgcn.vgpr
    %dr2 = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %ds_rd_range2, %ds_addr, offset = %c12_ds
      : !amdgcn.vgpr, i32 -> !amdgcn.vgpr_range<[? + 2]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   ds_read_b64
    %ds_rd_range3 = amdgcn.make_register_range %ds6, %ds7 : !amdgcn.vgpr, !amdgcn.vgpr
    %dr3 = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %ds_rd_range3, %ds_addr, offset = %c12_ds
      : !amdgcn.vgpr, i32 -> !amdgcn.vgpr_range<[? + 2]>
    // now wait for {5} to complete, but we have 2 new ds_read in flight
    //   -> vmcnt remains 1, lgkmcnt reaches 4 in flight
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 4
    // CHECK:   ds_read_b64
    %ds_rd_range4 = amdgcn.make_register_range %ds8, %ds9 : !amdgcn.vgpr, !amdgcn.vgpr
    %dr4 = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %ds_rd_range4, %ds_addr, offset = %c20_ds
      : !amdgcn.vgpr, i32 -> !amdgcn.vgpr_range<[? + 2]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   ds_read_b64
    %ds_rd_range5 = amdgcn.make_register_range %ds10, %ds11 : !amdgcn.vgpr, !amdgcn.vgpr
    %dr5 = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %ds_rd_range5, %ds_addr, offset = %c20_ds
      : !amdgcn.vgpr, i32 -> !amdgcn.vgpr_range<[? + 2]>

    // Here we have 1 vmcnt and 6 lgkmcnt in flight (4 + 2 new ds_read).

    // 4. Six mfma operations
    %c_range0 = amdgcn.make_register_range %c0, %c1, %c2, %c3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    // mfma0 uses read 0, needs to wait for {0} -> lgkmcnt reaches 5 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 5
    // CHECK:   amdgcn.vop3p.vop3p_mai
    %mfma0 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_range0, %dr0, %dr0, %c_range0
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.vop3p.vop3p_mai
    %mfma1 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_range0, %dr0, %dr0, %c_range0
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>
    // mfma0 uses read 2, needs to wait for {1, 2} -> lgkmcnt reaches 3 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 3
    // CHECK:   amdgcn.vop3p.vop3p_mai
    %mfma2 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_range0, %dr2, %dr2, %c_range0
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.vop3p.vop3p_mai
    %mfma3 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_range0, %dr2, %dr2, %c_range0
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>
    // mfma0 uses read 4, needs to wait for {3, 4} -> lgkmcnt reaches 1 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 1
    // CHECK:   amdgcn.vop3p.vop3p_mai
    %mfma4 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_range0, %dr4, %dr4, %c_range0
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.vop3p.vop3p_mai
    %mfma5 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_range0, %dr4, %dr4, %c_range0
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>

    // Here we have 1 vmcnt and 1 lgkmcnt in flight.

    // 5. Six global_store operations depending on mfma 1, 3, 5
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.flat.global_store
    %mfma1_0, %mfma1_1, %mfma1_2, %mfma1_3 = amdgcn.split_register_range %mfma1 : !amdgcn.vgpr_range<[? + 4]>
    %s_range0 = amdgcn.make_register_range %mfma1_0 : !amdgcn.vgpr
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %s_range0, %addr_range, offset = 24
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.flat.global_store
    %mfma1_0_2, %mfma1_1_2, %mfma1_2_2, %mfma1_3_2 = amdgcn.split_register_range %mfma1 : !amdgcn.vgpr_range<[? + 4]>
    %s_range1 = amdgcn.make_register_range %mfma1_0_2 : !amdgcn.vgpr
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %s_range1, %addr_range, offset = 28
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.flat.global_store
    %mfma3_0, %mfma3_1, %mfma3_2, %mfma3_3 = amdgcn.split_register_range %mfma3 : !amdgcn.vgpr_range<[? + 4]>
    %s_range2 = amdgcn.make_register_range %mfma3_0 : !amdgcn.vgpr
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %s_range2, %addr_range, offset = 32
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.flat.global_store
    %mfma3_0_2, %mfma3_1_2, %mfma3_2_2, %mfma3_3_2 = amdgcn.split_register_range %mfma3 : !amdgcn.vgpr_range<[? + 4]>
    %s_range3 = amdgcn.make_register_range %mfma3_0_2 : !amdgcn.vgpr
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %s_range3, %addr_range, offset = 36
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.flat.global_store
    %mfma5_0, %mfma5_1, %mfma5_2, %mfma5_3 = amdgcn.split_register_range %mfma5 : !amdgcn.vgpr_range<[? + 4]>
    %s_range4 = amdgcn.make_register_range %mfma5_0 : !amdgcn.vgpr
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %s_range4, %addr_range, offset = 40
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.flat.global_store
    %mfma5_0_2, %mfma5_1_2, %mfma5_2_2, %mfma5_3_2 = amdgcn.split_register_range %mfma5 : !amdgcn.vgpr_range<[? + 4]>
    %s_range5 = amdgcn.make_register_range %mfma5_0_2 : !amdgcn.vgpr
    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %s_range5, %addr_range, offset = 44
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]>

    // Here we have 7 (1 + 6) vmcnt and 1 lgkmcnt in flight.

    // 6. Six global_load operations depending on stores
    //   offset 24 aliases with store {0}, need to wait for {0} and the old prior
    //   vcmnt global_load
    //     -> vmcnt reaches 5 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 5 expcnt = 0 lgkmcnt = 1
    // CHECK:   amdgcn.flat.global_load
    %f_range0 = amdgcn.make_register_range %f0 : !amdgcn.vgpr
    %fl0 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %f_range0, %addr_range, offset = 24
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.flat.global_load
    %f_range1 = amdgcn.make_register_range %f1 : !amdgcn.vgpr
    %fl1 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %f_range1, %addr_range, offset = 24
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>
    //   offset 24 aliases with store {2}, need to wait for {1, 2} and we have 2 new global_load in flight
    //     -> vmcnt stays at 5 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 5 expcnt = 0 lgkmcnt = 1
    // CHECK:   amdgcn.flat.global_load
    %f_range2 = amdgcn.make_register_range %f2 : !amdgcn.vgpr
    %fl2 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %f_range2, %addr_range, offset = 32
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.flat.global_load
    %f_range3 = amdgcn.make_register_range %f3 : !amdgcn.vgpr
    %fl3 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %f_range3, %addr_range, offset = 32
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>
    //   offset 36 aliases with store {3}, need to wait for {3} and we have 2 new global_load in flight
    //     -> vmcnt goes up to 6 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 6 expcnt = 0 lgkmcnt = 1
    // CHECK:   amdgcn.flat.global_load
    %f_range4 = amdgcn.make_register_range %f4 : !amdgcn.vgpr
    %fl4 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %f_range4, %addr_range, offset = 36
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>
    //   offset 44 aliases with store {5}, need to wait for {4, 5} and we have 1 new global_load in flight
    //     -> vmcnt reaches 5 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 5 expcnt = 0 lgkmcnt = 1
    // CHECK:   amdgcn.flat.global_load
    %f_range5 = amdgcn.make_register_range %f5 : !amdgcn.vgpr
    %fl5 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %f_range5, %addr_range, offset = 44
      : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.vgpr_range<[? + 1]>

    // Here we have 6 (5 + 1) vmcnt and 1 lgkmcnt in flight.

    //      CHECK:   amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
    // CHECK-NEXT:   end_kernel
    amdgcn.end_kernel
  }
}
