// Test proper counting of multiple global_load operations
// RUN: aster-opt %s --amdgcn-optimize-straight-line-waits --split-input-file | FileCheck %s

//   CHECK-LABEL: amdgcn.module @test_global_load_counting
//     CHECK-NOT:   amdgcn.sopp.s_waitcnt
// CHECK-COUNT-3:   load global_load_dword
// Wait for first 2 -> vmcnt reaches 1
//         CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1
//         CHECK:   amdgcn.vop1.vop1
//         CHECK:   store global_store_dword
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
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %c12 = arith.constant 12 : i32
    %addr_range = amdgcn.make_register_range %4, %5 : !amdgcn.sgpr, !amdgcn.sgpr

    // Three global loads
    %dst_range0 = amdgcn.make_register_range %0 : !amdgcn.vgpr
    %6, %tok6 = amdgcn.load global_load_dword dest %dst_range0 addr %addr_range : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<flat>
    %dst_range1 = amdgcn.make_register_range %1 : !amdgcn.vgpr
    %7, %tok7 = amdgcn.load global_load_dword dest %dst_range1 addr %addr_range offset c(%c4) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
    %dst_range2 = amdgcn.make_register_range %2 : !amdgcn.vgpr
    %8, %tok8 = amdgcn.load global_load_dword dest %dst_range2 addr %addr_range offset c(%c8) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>

    // Use second load (index 1) - should wait for 2 operations (indices 0 and 1)
    %split1 = amdgcn.split_register_range %7 : !amdgcn.vgpr
    %9 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %3, %split1
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %data_range = amdgcn.make_register_range %9 : !amdgcn.vgpr
    %tok_store = amdgcn.store global_store_dword data %data_range addr %addr_range offset c(%c12) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.write_token<flat>

    amdgcn.end_kernel
  }
}

// -----

// CHECK-LABEL: amdgcn.module @test_ds_read_counting
//     CHECK-NOT:   amdgcn.sopp.s_waitcnt
// CHECK-COUNT-3:   load ds_read_b32
// Wait for first 2 -> lgkmcnt reaches 1
//         CHECK:   amdgcn.sopp.s_waitcnt {{.*}} lgkmcnt = 1
//         CHECK:   amdgcn.vop1.vop1
//         CHECK:   store ds_write_b32
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
    %5, %tok5 = amdgcn.load ds_read_b32 dest %dst_range0 addr %4 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
    %dst_range1 = amdgcn.make_register_range %1 : !amdgcn.vgpr
    %6, %tok6 = amdgcn.load ds_read_b32 dest %dst_range1 addr %4 offset c(%c4) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
    %dst_range2 = amdgcn.make_register_range %2 : !amdgcn.vgpr
    %7, %tok7 = amdgcn.load ds_read_b32 dest %dst_range2 addr %4 offset c(%c8) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>

    // Use second read (index 1) - should wait for 2 operations (indices 0 and 1)
    %split1 = amdgcn.split_register_range %6 : !amdgcn.vgpr
    %8 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %3, %split1
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %data_range = amdgcn.make_register_range %8 : !amdgcn.vgpr
    %tok_write = amdgcn.store ds_write_b32 data %data_range addr %4 offset c(%c12) : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

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
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %c12 = arith.constant 12 : i32
    %addr_range = amdgcn.make_register_range %4, %5 : !amdgcn.sgpr, !amdgcn.sgpr

    // Two global loads
    %dst_range0 = amdgcn.make_register_range %0 : !amdgcn.vgpr
    %6, %tok6 = amdgcn.load global_load_dword dest %dst_range0 addr %addr_range offset c(%c0) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
    // Store at non aliasing location
    %data_range0 = amdgcn.make_register_range %2 : !amdgcn.vgpr
    %tok_s0 = amdgcn.store global_store_dword data %data_range0 addr %addr_range offset c(%c8) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.write_token<flat>
    %dst_range1 = amdgcn.make_register_range %1 : !amdgcn.vgpr
    %7, %tok7 = amdgcn.load global_load_dword dest %dst_range1 addr %addr_range offset c(%c4) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>

    //     CHECK-NOT:   amdgcn.sopp.s_waitcnt
    //         CHECK:   load global_load_dword
    //         CHECK:   store global_store_dword
    //         CHECK:   load global_load_dword
    // Use second load (index 1) - needs to  wait for all 3 memory operations
    //         CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 0
    //         CHECK:   amdgcn.vop1.vop1
    %split1 = amdgcn.split_register_range %7 : !amdgcn.vgpr
    %8 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %3, %split1
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    //         CHECK:   store global_store_dword
    //         CHECK:   amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
    //    CHECK-NEXT:   end_kernel
    %data_range1 = amdgcn.make_register_range %8 : !amdgcn.vgpr
    %tok_s1 = amdgcn.store global_store_dword data %data_range1 addr %addr_range offset c(%c12) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.write_token<flat>

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
    // CHECK-COUNT-6:   load global_load_dword
    %c0_g = arith.constant 0 : i32
    %c4_g = arith.constant 4 : i32
    %c8_g = arith.constant 8 : i32
    %c12_g = arith.constant 12 : i32
    %c16_g = arith.constant 16 : i32
    %c20_g = arith.constant 20 : i32
    %c24_g = arith.constant 24 : i32
    %c28_g = arith.constant 28 : i32
    %c32_g = arith.constant 32 : i32
    %c36_g = arith.constant 36 : i32
    %c40_g = arith.constant 40 : i32
    %c44_g = arith.constant 44 : i32
    %g_range0 = amdgcn.make_register_range %g0 : !amdgcn.vgpr
    %g_range1 = amdgcn.make_register_range %g1 : !amdgcn.vgpr
    %g_range2 = amdgcn.make_register_range %g2 : !amdgcn.vgpr
    %g_range3 = amdgcn.make_register_range %g3 : !amdgcn.vgpr
    %g_range4 = amdgcn.make_register_range %g4 : !amdgcn.vgpr
    %g_range5 = amdgcn.make_register_range %g5 : !amdgcn.vgpr
    %gl0, %tok_gl0 = amdgcn.load global_load_dword dest %g_range0 addr %addr_range offset c(%c0_g) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
    %gl1, %tok_gl1 = amdgcn.load global_load_dword dest %g_range1 addr %addr_range offset c(%c4_g) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
    %gl2, %tok_gl2 = amdgcn.load global_load_dword dest %g_range2 addr %addr_range offset c(%c8_g) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
    %gl3, %tok_gl3 = amdgcn.load global_load_dword dest %g_range3 addr %addr_range offset c(%c12_g) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
    %gl4, %tok_gl4 = amdgcn.load global_load_dword dest %g_range4 addr %addr_range offset c(%c16_g) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
    %gl5, %tok_gl5 = amdgcn.load global_load_dword dest %g_range5 addr %addr_range offset c(%c20_g) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>

    // Here we have 6 vmcnt and 0 lgkmcnt in flight.

    // 2. Six ds_write operations depending on loads 0, 2, 4 (non-trivial pattern)
    // Write 0 uses load 2, needs to wait for 0, 1, 2 -> vmcnt reaches 3 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 3 expcnt = 0 lgkmcnt = 0
    // CHECK:   store ds_write_b32
    %c0_ds = arith.constant 0 : i32
    %c4_ds = arith.constant 4 : i32
    %c8_ds = arith.constant 8 : i32
    %c12_ds = arith.constant 12 : i32
    %c16_ds = arith.constant 16 : i32
    %c20_ds = arith.constant 20 : i32
    %split_gl0 = amdgcn.split_register_range %gl2 : !amdgcn.vgpr
    %ds_wr_range0 = amdgcn.make_register_range %split_gl0 : !amdgcn.vgpr
    %tok_dsw0 = amdgcn.store ds_write_b32 data %ds_wr_range0 addr %ds_addr offset c(%c0_ds) : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
    // Write 1 uses load 0 (already covered)
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   store ds_write_b32
    %split_gl0_2 = amdgcn.split_register_range %gl0 : !amdgcn.vgpr
    %ds_wr_range1 = amdgcn.make_register_range %split_gl0_2 : !amdgcn.vgpr
    %tok_dsw1 = amdgcn.store ds_write_b32 data %ds_wr_range1 addr %ds_addr offset c(%c4_ds) : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
    // Write 2 uses load 2 (already covered)
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   store ds_write_b32
    %split_gl2 = amdgcn.split_register_range %gl2 : !amdgcn.vgpr
    %ds_wr_range2 = amdgcn.make_register_range %split_gl2 : !amdgcn.vgpr
    %tok_dsw2 = amdgcn.store ds_write_b32 data %ds_wr_range2 addr %ds_addr offset c(%c8_ds) : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
    // Write 3 uses load 2 (already covered)
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   store ds_write_b32
    %split_gl2_2 = amdgcn.split_register_range %gl2 : !amdgcn.vgpr
    %ds_wr_range3 = amdgcn.make_register_range %split_gl2_2 : !amdgcn.vgpr
    %tok_dsw3 = amdgcn.store ds_write_b32 data %ds_wr_range3 addr %ds_addr offset c(%c12_ds) : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
    // Write 4 uses load 4, need to wait for 3, 4 -> vmcnt reaches 1 in flight.
    //                                            -> lgkmcnt has 4 in flight (no wait needed)
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 4
    // CHECK:   store ds_write_b32
    %split_gl4 = amdgcn.split_register_range %gl4 : !amdgcn.vgpr
    %ds_wr_range4 = amdgcn.make_register_range %split_gl4 : !amdgcn.vgpr
    %tok_dsw4 = amdgcn.store ds_write_b32 data %ds_wr_range4 addr %ds_addr offset c(%c16_ds) : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
    // Write 5 uses load 4 (already covered)
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   store ds_write_b32
    %split_gl4_2 = amdgcn.split_register_range %gl4 : !amdgcn.vgpr
    %ds_wr_range5 = amdgcn.make_register_range %split_gl4_2 : !amdgcn.vgpr
    %tok_dsw5 = amdgcn.store ds_write_b32 data %ds_wr_range5 addr %ds_addr offset c(%c20_ds) : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // Here we have 1 vmcnt and 6 lgkmcnt in flight.

    // 3. Six ds_read operations depending on writes 1, 3 and 5 respectively.
    //    This time, we use b64 (2 VGPRs) due to MFMA, which will alias more.
    // ds_read_b64 [4 .. 12) aliases with ds_write 4 and 8
    // need to wait for {0, 1, 2} -> vmcnt remains 1 in flight (no wait needed)
    //                            -> lgkmcnt reaches 3 in flight
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 3
    // CHECK:   load ds_read_b64
    %ds_rd_range0 = amdgcn.make_register_range %ds0, %ds1 : !amdgcn.vgpr, !amdgcn.vgpr
    %dr0, %tok_dr0 = amdgcn.load ds_read_b64 dest %ds_rd_range0 addr %ds_addr offset c(%c4_ds) : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   load ds_read_b64
    %ds_rd_range1 = amdgcn.make_register_range %ds2, %ds3 : !amdgcn.vgpr, !amdgcn.vgpr
    %dr1, %tok_dr1 = amdgcn.load ds_read_b64 dest %ds_rd_range1 addr %ds_addr offset c(%c4_ds) : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
    // ds_read_b64 [12 .. 20) aliases with ds_write 12 and 16
    //   need to wait for {3, 4} with {5} still in flight.
    //   also need to account for the 2 ds_read in flight
    //   -> vmcnt remains 1, lgkmcnt reaches 3 in flight
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 3
    // CHECK:   load ds_read_b64
    %ds_rd_range2 = amdgcn.make_register_range %ds4, %ds5 : !amdgcn.vgpr, !amdgcn.vgpr
    %dr2, %tok_dr2 = amdgcn.load ds_read_b64 dest %ds_rd_range2 addr %ds_addr offset c(%c12_ds) : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   load ds_read_b64
    %ds_rd_range3 = amdgcn.make_register_range %ds6, %ds7 : !amdgcn.vgpr, !amdgcn.vgpr
    %dr3, %tok_dr3 = amdgcn.load ds_read_b64 dest %ds_rd_range3 addr %ds_addr offset c(%c12_ds) : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
    // now wait for {5} to complete, but we have 2 new ds_read in flight
    //   -> vmcnt remains 1, lgkmcnt reaches 4 in flight
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 4
    // CHECK:   load ds_read_b64
    %ds_rd_range4 = amdgcn.make_register_range %ds8, %ds9 : !amdgcn.vgpr, !amdgcn.vgpr
    %dr4, %tok_dr4 = amdgcn.load ds_read_b64 dest %ds_rd_range4 addr %ds_addr offset c(%c20_ds) : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   load ds_read_b64
    %ds_rd_range5 = amdgcn.make_register_range %ds10, %ds11 : !amdgcn.vgpr, !amdgcn.vgpr
    %dr5, %tok_dr5 = amdgcn.load ds_read_b64 dest %ds_rd_range5 addr %ds_addr offset c(%c20_ds) : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>

    // Here we have 1 vmcnt and 6 lgkmcnt in flight (4 + 2 new ds_read).

    // 4. Six mfma operations
    %c_range0 = amdgcn.make_register_range %c0, %c1, %c2, %c3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    // mfma0 uses read 0, needs to wait for {0} -> lgkmcnt reaches 5 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 5
    // CHECK:   amdgcn.vop3p.vop3p_mai
    %mfma0 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_range0, %dr0, %dr0, %c_range0
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.vop3p.vop3p_mai
    %mfma1 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_range0, %dr0, %dr0, %c_range0
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>
    // mfma0 uses read 2, needs to wait for {1, 2} -> lgkmcnt reaches 3 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 3
    // CHECK:   amdgcn.vop3p.vop3p_mai
    %mfma2 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_range0, %dr2, %dr2, %c_range0
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.vop3p.vop3p_mai
    %mfma3 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_range0, %dr2, %dr2, %c_range0
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>
    // mfma0 uses read 4, needs to wait for {3, 4} -> lgkmcnt reaches 1 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 1 expcnt = 0 lgkmcnt = 1
    // CHECK:   amdgcn.vop3p.vop3p_mai
    %mfma4 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_range0, %dr4, %dr4, %c_range0
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   amdgcn.vop3p.vop3p_mai
    %mfma5 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_range0, %dr4, %dr4, %c_range0
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>

    // Here we have 1 vmcnt and 1 lgkmcnt in flight.

    // 5. Six global_store operations depending on mfma 1, 3, 5
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   store global_store_dword
    %mfma1_0, %mfma1_1, %mfma1_2, %mfma1_3 = amdgcn.split_register_range %mfma1 : !amdgcn.vgpr<[? + 4]>
    %s_range0 = amdgcn.make_register_range %mfma1_0 : !amdgcn.vgpr
    %tok_gs0 = amdgcn.store global_store_dword data %s_range0 addr %addr_range offset c(%c24_g) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.write_token<flat>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   store global_store_dword
    %mfma1_0_2, %mfma1_1_2, %mfma1_2_2, %mfma1_3_2 = amdgcn.split_register_range %mfma1 : !amdgcn.vgpr<[? + 4]>
    %s_range1 = amdgcn.make_register_range %mfma1_0_2 : !amdgcn.vgpr
    %tok_gs1 = amdgcn.store global_store_dword data %s_range1 addr %addr_range offset c(%c28_g) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.write_token<flat>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   store global_store_dword
    %mfma3_0, %mfma3_1, %mfma3_2, %mfma3_3 = amdgcn.split_register_range %mfma3 : !amdgcn.vgpr<[? + 4]>
    %s_range2 = amdgcn.make_register_range %mfma3_0 : !amdgcn.vgpr
    %tok_gs2 = amdgcn.store global_store_dword data %s_range2 addr %addr_range offset c(%c32_g) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.write_token<flat>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   store global_store_dword
    %mfma3_0_2, %mfma3_1_2, %mfma3_2_2, %mfma3_3_2 = amdgcn.split_register_range %mfma3 : !amdgcn.vgpr<[? + 4]>
    %s_range3 = amdgcn.make_register_range %mfma3_0_2 : !amdgcn.vgpr
    %tok_gs3 = amdgcn.store global_store_dword data %s_range3 addr %addr_range offset c(%c36_g) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.write_token<flat>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   store global_store_dword
    %mfma5_0, %mfma5_1, %mfma5_2, %mfma5_3 = amdgcn.split_register_range %mfma5 : !amdgcn.vgpr<[? + 4]>
    %s_range4 = amdgcn.make_register_range %mfma5_0 : !amdgcn.vgpr
    %tok_gs4 = amdgcn.store global_store_dword data %s_range4 addr %addr_range offset c(%c40_g) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.write_token<flat>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   store global_store_dword
    %mfma5_0_2, %mfma5_1_2, %mfma5_2_2, %mfma5_3_2 = amdgcn.split_register_range %mfma5 : !amdgcn.vgpr<[? + 4]>
    %s_range5 = amdgcn.make_register_range %mfma5_0_2 : !amdgcn.vgpr
    %tok_gs5 = amdgcn.store global_store_dword data %s_range5 addr %addr_range offset c(%c44_g) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.write_token<flat>

    // Here we have 7 (1 + 6) vmcnt and 1 lgkmcnt in flight.

    // 6. Six global_load operations depending on stores
    //   offset 24 aliases with store {0}, need to wait for {0} and the old prior
    //   vcmnt global_load
    //     -> vmcnt reaches 5 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 5 expcnt = 0 lgkmcnt = 1
    // CHECK:   load global_load_dword
    %f_range0 = amdgcn.make_register_range %f0 : !amdgcn.vgpr
    %fl0, %tok_fl0 = amdgcn.load global_load_dword dest %f_range0 addr %addr_range offset c(%c24_g) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   load global_load_dword
    %f_range1 = amdgcn.make_register_range %f1 : !amdgcn.vgpr
    %fl1, %tok_fl1 = amdgcn.load global_load_dword dest %f_range1 addr %addr_range offset c(%c24_g) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
    //   offset 24 aliases with store {2}, need to wait for {1, 2} and we have 2 new global_load in flight
    //     -> vmcnt stays at 5 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 5 expcnt = 0 lgkmcnt = 1
    // CHECK:   load global_load_dword
    %f_range2 = amdgcn.make_register_range %f2 : !amdgcn.vgpr
    %fl2, %tok_fl2 = amdgcn.load global_load_dword dest %f_range2 addr %addr_range offset c(%c32_g) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
    // CHECK-NOT:   amdgcn.sopp.s_waitcnt
    // CHECK:   load global_load_dword
    %f_range3 = amdgcn.make_register_range %f3 : !amdgcn.vgpr
    %fl3, %tok_fl3 = amdgcn.load global_load_dword dest %f_range3 addr %addr_range offset c(%c32_g) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
    //   offset 36 aliases with store {3}, need to wait for {3} and we have 2 new global_load in flight
    //     -> vmcnt goes up to 6 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 6 expcnt = 0 lgkmcnt = 1
    // CHECK:   load global_load_dword
    %f_range4 = amdgcn.make_register_range %f4 : !amdgcn.vgpr
    %fl4, %tok_fl4 = amdgcn.load global_load_dword dest %f_range4 addr %addr_range offset c(%c36_g) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
    //   offset 44 aliases with store {5}, need to wait for {4, 5} and we have 1 new global_load in flight
    //     -> vmcnt reaches 5 in flight.
    // CHECK:   amdgcn.sopp.s_waitcnt {{.*}} vmcnt = 5 expcnt = 0 lgkmcnt = 1
    // CHECK:   load global_load_dword
    %f_range5 = amdgcn.make_register_range %f5 : !amdgcn.vgpr
    %fl5, %tok_fl5 = amdgcn.load global_load_dword dest %f_range5 addr %addr_range offset c(%c44_g) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>

    // Here we have 6 (5 + 1) vmcnt and 1 lgkmcnt in flight.

    //      CHECK:   amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
    // CHECK-NEXT:   end_kernel
    amdgcn.end_kernel
  }
}
