// RUN: aster-opt %s --aster-codegen | FileCheck %s
//
// Uniform get_lds_offset values rotated as cf loop block-args must materialize
// as SGPR carriers (not VGPR), so in-loop addressing stays
// v_add_u32(vgpr_swizzle, sgpr_offset) rather than carrying VGPR offsets.

// CHECK-LABEL: kernel @test_lds_offset_cf_carry
// CHECK: get_lds_offset
// CHECK: lsir.cond_br {{.*}} ^bb1({{.*}}, {{.*}}, {{.*}} : {{.*}}, !amdgcn.sgpr, !amdgcn.vgpr)
// CHECK: ^bb1(%{{.*}}: !amdgcn.sgpr, %[[OFF:.*]]: !amdgcn.sgpr, {{.*}}: !amdgcn.vgpr)
// CHECK-NOT: reg_cast {{.*}} !amdgcn.sgpr -> !amdgcn.vgpr
// CHECK: lsir.addi i32 {{.*}}, %[[OFF]], {{.*}} : !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr

amdgcn.module @test_lds_offset_cf_carry target = <gfx942> {
  kernel @test_lds_offset_cf_carry {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %buf = amdgcn.alloc_lds 4096 alignment 256
    %off = amdgcn.get_lds_offset %buf : i32
    %lane = aster_utils.thread_id  x
    %sw = arith.muli %lane, %c1024_i32 overflow<nsw, nuw> : i32
    %cmp = arith.cmpi sgt, %lane, %c0_i32 : i32
    cf.cond_br %cmp, ^bb1(%c0_i32, %off, %sw : i32, i32, i32), ^bb2
  ^bb1(%iv: i32, %lds_off: i32, %swizzle: i32):  // 2 preds: ^bb0, ^bb1
    %addr = arith.addi %lds_off, %swizzle overflow<nsw, nuw> : i32
    %addr_reg = lsir.to_reg %addr : i32 -> !amdgcn.vgpr
    amdgcn.test_inst ins %addr_reg : (!amdgcn.vgpr) -> ()
    %next = arith.addi %iv, %c1_i32 : i32
    %loop_cmp = arith.cmpi slt, %next, %c1024_i32 : i32
    cf.cond_br %loop_cmp, ^bb1(%next, %lds_off, %swizzle : i32, i32, i32), ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    end_kernel
  }
}
