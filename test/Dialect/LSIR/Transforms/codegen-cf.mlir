// RUN: aster-opt %s --aster-codegen | FileCheck %s

//===----------------------------------------------------------------------===//
// Test CF dialect patterns: arith.cmpi/cmpf conversion and block argument
// handling in control flow operations. Verifies that:
// 1. arith.cmpi is converted to lsir.cmpi (DPS, SCC/VCC dst)
// 2. Scalar block arguments are converted to register types
// 3. Branch operands are properly wrapped in alloca+mov when needed
// 4. cf.cond_br with register condition is lowered to lsir.cond_br
//===----------------------------------------------------------------------===//

// CHECK-LABEL: amdgcn.module @test_uniform_loop
// CHECK:         kernel @test_uniform_loop
// CHECK:           %[[C0:.*]] = arith.constant 0 : i32
// CHECK:           %[[C1:.*]] = arith.constant 1 : i32
// CHECK:           %[[LOAD:.*]] = load_arg 1
// CHECK:           s_waitcnt
// CHECK:           split_register_range
// CHECK:           %[[SCC_INIT:.*]] = lsir.alloca : !amdgcn.scc
// CHECK:           %[[CMP_INIT:.*]] = lsir.cmpi i32 sgt %[[SCC_INIT]], %{{.*}}, %[[C0]] : !amdgcn.scc, !amdgcn.sgpr, i32
// CHECK:           %[[ALLOCA_INIT:.*]] = lsir.alloca : !amdgcn.sgpr
// CHECK:           %[[MOV_INIT:.*]] = lsir.mov %[[ALLOCA_INIT]], %[[C0]]
// CHECK:           lsir.cond_br %[[CMP_INIT]] : !amdgcn.scc, ^bb1(%[[MOV_INIT]] : !amdgcn.sgpr), ^bb2
// CHECK:         ^bb1(%[[LOOP_ARG:.*]]: !amdgcn.sgpr):
// CHECK:           test_inst ins %[[LOOP_ARG]]
// CHECK:           %[[ALLOCA_LOOP:.*]] = lsir.alloca : !amdgcn.sgpr
// CHECK:           %[[LOOP_ADDI:.*]] = lsir.addi i32 %[[ALLOCA_LOOP]], %[[LOOP_ARG]], %[[C1]]
// CHECK:           %[[SCC_LOOP:.*]] = lsir.alloca : !amdgcn.scc
// CHECK:           %[[CMP_LOOP:.*]] = lsir.cmpi i32 slt %[[SCC_LOOP]], %[[LOOP_ADDI]], %{{.*}} : !amdgcn.scc, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           lsir.cond_br %[[CMP_LOOP]] : !amdgcn.scc, ^bb1(%[[LOOP_ADDI]] : !amdgcn.sgpr), ^bb2
// CHECK:         ^bb2:
// CHECK:           end_kernel

amdgcn.module @test_uniform_loop target = <gfx942> {
  kernel @test_uniform_loop arguments <[#amdgcn.buffer_arg<address_space = generic, access = read_only>, #amdgcn.buffer_arg<address_space = generic>]> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0
    %arg0, %arg1 = amdgcn.split_register_range %0 : !amdgcn.sgpr<[? + 2]>
    %1 = aster_utils.assume_uniform %arg0 : !amdgcn.sgpr
    %2 = lsir.from_reg %1 : !amdgcn.sgpr -> i32
    %3 = arith.cmpi sgt, %2, %c0_i32 : i32
    cf.cond_br %3, ^bb1(%c0_i32 : i32), ^bb2
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb1
    %5 = lsir.to_reg %4 : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %5 : (!amdgcn.sgpr) -> ()
    %6 = arith.addi %4, %c1_i32 : i32
    %7 = arith.cmpi slt, %6, %2 : i32
    cf.cond_br %7, ^bb1(%6 : i32), ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    end_kernel
  }
}

// -----

// CHECK-LABEL: amdgcn.module @test_uniform_loop_with_load
// CHECK:         kernel @test_uniform_loop_with_load
// CHECK:           %[[C2:.*]] = arith.constant 2 : i32
// CHECK:           %[[C1:.*]] = arith.constant 1 : i32
// CHECK:           %[[C0:.*]] = arith.constant 0 : i32
// CHECK:           load_arg 0
// CHECK:           load_arg 1
// CHECK:           s_waitcnt
// CHECK:           alloca
// CHECK:           %[[LOAD_RESULT:.*]], %{{.*}} = s_load_dword
// CHECK:           s_waitcnt
// CHECK:           %[[SCC_INIT2:.*]] = lsir.alloca : !amdgcn.scc
// CHECK:           %[[CMP_INIT2:.*]] = lsir.cmpi i32 sgt %[[SCC_INIT2]], %[[LOAD_RESULT]], %[[C0]] : !amdgcn.scc, !amdgcn.sgpr, i32
// CHECK:           %[[ALLOCA_INIT2:.*]] = lsir.alloca : !amdgcn.sgpr
// CHECK:           %[[MOV_INIT2:.*]] = lsir.mov %[[ALLOCA_INIT2]], %[[C0]]
// CHECK:           lsir.cond_br %[[CMP_INIT2]] : !amdgcn.scc, ^bb1(%[[MOV_INIT2]] : !amdgcn.sgpr), ^bb2
// CHECK:         ^bb1(%[[LOOP_ARG2:.*]]: !amdgcn.sgpr):
// CHECK:           %[[ALLOCA_SHLI:.*]] = lsir.alloca : !amdgcn.sgpr
// CHECK:           %[[LOOP_SHLI:.*]] = lsir.shli i32 %[[ALLOCA_SHLI]], %[[LOOP_ARG2]], %[[C2]]
// CHECK:           alloca
// CHECK:           v_mov_b32
// CHECK:           global_store_dword
// CHECK:           %[[ALLOCA_ADDI:.*]] = lsir.alloca : !amdgcn.sgpr
// CHECK:           %[[LOOP_ADDI2:.*]] = lsir.addi i32 %[[ALLOCA_ADDI]], %[[LOOP_ARG2]], %[[C1]]
// CHECK:           %[[SCC_LOOP2:.*]] = lsir.alloca : !amdgcn.scc
// CHECK:           %[[CMP_LOOP2:.*]] = lsir.cmpi i32 slt %[[SCC_LOOP2]], %[[LOOP_ADDI2]], %[[LOAD_RESULT]] : !amdgcn.scc, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           lsir.cond_br %[[CMP_LOOP2]] : !amdgcn.scc, ^bb1(%[[LOOP_ADDI2]] : !amdgcn.sgpr), ^bb2
// CHECK:         ^bb2:
// CHECK:           end_kernel

amdgcn.module @test_uniform_loop_with_load target = <gfx942> {
  kernel @test_uniform_loop_with_load arguments <[#amdgcn.buffer_arg<address_space = generic, access = read_only>, #amdgcn.buffer_arg<address_space = generic>]> {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %1 = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0
    %2 = amdgcn.alloca : !amdgcn.sgpr
    %result, %token = amdgcn.s_load_dword dest %2 addr %0 offset c(%c0_i32) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<constant>
    amdgcn.s_waitcnt lgkmcnt = 0
    %3 = lsir.from_reg %result : !amdgcn.sgpr -> i32
    %4 = arith.cmpi sgt, %3, %c0_i32 : i32
    cf.cond_br %4, ^bb1(%c0_i32 : i32), ^bb2
  ^bb1(%5: i32):  // 2 preds: ^bb0, ^bb1
    %6 = arith.shli %5, %c2_i32 : i32
    %7 = lsir.to_reg %6 : i32 -> !amdgcn.sgpr
    %8 = amdgcn.alloca : !amdgcn.vgpr
    %9 = amdgcn.v_mov_b32 outs(%8) ins(%7) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %10 = amdgcn.global_store_dword data %9 addr %1 offset d(%9) + c(%c0_i32) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %11 = arith.addi %5, %c1_i32 : i32
    %12 = arith.cmpi slt, %11, %3 : i32
    cf.cond_br %12, ^bb1(%11 : i32), ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    end_kernel
  }
}

// -----

//===----------------------------------------------------------------------===//
// Test arith.cmpi + arith.select -> lsir.cmpi (DPS) + lsir.select(SCC)
// Verifies that:
// 1. arith.cmpi is converted to lsir.cmpi (DPS) with SCC dst
// 2. arith.select with register condition is converted to lsir.select
// 3. No unrealized_conversion_cast is inserted
//===----------------------------------------------------------------------===//

// CHECK-LABEL: amdgcn.module @test_select_i1
// CHECK:         kernel @test_select_i1
// CHECK:           %[[SCC:.*]] = lsir.alloca : !amdgcn.scc
// CHECK:           %[[CMP:.*]] = lsir.cmpi i32 eq %[[SCC]], %{{.*}}, %{{.*}} : !amdgcn.scc, !amdgcn.sgpr, i32
// CHECK:           %[[ALLOCA:.*]] = lsir.alloca : !amdgcn.sgpr
// CHECK:           lsir.select %[[ALLOCA]], %[[CMP]], %{{.*}}, %{{.*}} : !amdgcn.sgpr, !amdgcn.scc, i32, i32
// CHECK-NOT:       unrealized_conversion_cast

amdgcn.module @test_select_i1 target = <gfx942> {
  kernel @test_select_i1 arguments <[#amdgcn.buffer_arg<address_space = generic, access = read_only>, #amdgcn.buffer_arg<address_space = generic>]> {
    %c0_i32 = arith.constant 0 : i32
    %c42_i32 = arith.constant 42 : i32
    %c99_i32 = arith.constant 99 : i32
    %0 = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0
    %arg0, %arg1 = amdgcn.split_register_range %0 : !amdgcn.sgpr<[? + 2]>
    %1 = aster_utils.assume_uniform %arg0 : !amdgcn.sgpr
    %2 = lsir.from_reg %1 : !amdgcn.sgpr -> i32
    %cmp = arith.cmpi eq, %2, %c0_i32 : i32
    %sel = arith.select %cmp, %c42_i32, %c99_i32 : i32
    %3 = lsir.to_reg %sel : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %3 : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}
