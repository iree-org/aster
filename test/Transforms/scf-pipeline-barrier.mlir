// RUN: aster-opt --aster-scf-pipeline -split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @barrier_at_stage1

// Prologue: stage 0 store only -- NO barrier here
// CHECK:       %[[P_WTOK:.*]] = amdgcn.ds_write_b32
// CHECK-NOT:   amdgcn.s_barrier

// Kernel: store, barrier, wait+read present in every iteration
// CHECK:       %[[KER:.*]]:5 = scf.for {{.*}} iter_args(%[[A_WTOK:.*]] = %[[P_WTOK]]
// CHECK:         amdgcn.ds_write_b32
// CHECK:         amdgcn.s_barrier
// CHECK:         amdgcn.wait deps %[[A_WTOK]]
// CHECK:         amdgcn.ds_read_b32
// CHECK:         scf.yield

// Epilogue: barrier present, before the wait+read drain
// CHECK:       amdgcn.s_barrier
// CHECK:       amdgcn.wait deps %[[KER]]#0
// CHECK:       amdgcn.ds_read_b32
// CHECK:       return

func.func @barrier_at_stage1(%data_in: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %s_out = amdgcn.alloca : !amdgcn.vgpr
  scf.for %i = %c0 to %c4 step %c1 {
    %lds = amdgcn.alloc_lds 256 {sched.stage = 0 : i32}
    %lds_off = amdgcn.get_lds_offset %lds {sched.stage = 0 : i32} : i32
    %lds_addr = lsir.to_reg %lds_off {sched.stage = 0 : i32} : i32 -> !amdgcn.vgpr
    %wtok = amdgcn.ds_write_b32 data %data_in addr %lds_addr offset c(%c0_i32) {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    amdgcn.s_barrier {sched.stage = 1 : i32}

    amdgcn.wait deps %wtok {sched.stage = 1 : i32} : !amdgcn.write_token<shared>
    %dest = amdgcn.alloca {sched.stage = 1 : i32} : !amdgcn.vgpr
    %c0_i32_mig1 = arith.constant 0 : i32
    %read_data, %rtok = amdgcn.ds_read_b32 dest %dest addr %lds_addr offset c(%c0_i32_mig1) {sched.stage = 1 : i32} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %rtok {sched.stage = 1 : i32} : !amdgcn.read_token<shared>
    %result = amdgcn.test_inst outs %s_out ins %read_data {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    amdgcn.dealloc_lds %lds {sched.stage = 1 : i32}
  }
  return
}

// -----

func.func @barrier_missing_stage(%data_in: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %s_out = amdgcn.alloca : !amdgcn.vgpr
  scf.for %i = %c0 to %c4 step %c1 {
    %lds = amdgcn.alloc_lds 256 {sched.stage = 0 : i32}
    %lds_off = amdgcn.get_lds_offset %lds {sched.stage = 0 : i32} : i32
    %lds_addr = lsir.to_reg %lds_off {sched.stage = 0 : i32} : i32 -> !amdgcn.vgpr
    %wtok = amdgcn.ds_write_b32 data %data_in addr %lds_addr offset c(%c0_i32) {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // expected-error @below {{amdgcn.s_barrier in a pipelined loop body requires an explicit sched.stage attribute}}
    amdgcn.s_barrier

    amdgcn.wait deps %wtok {sched.stage = 1 : i32} : !amdgcn.write_token<shared>
    %dest = amdgcn.alloca {sched.stage = 1 : i32} : !amdgcn.vgpr
    %c0_i32_mig2 = arith.constant 0 : i32
    %read_data, %rtok = amdgcn.ds_read_b32 dest %dest addr %lds_addr offset c(%c0_i32_mig2) {sched.stage = 1 : i32} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %rtok {sched.stage = 1 : i32} : !amdgcn.read_token<shared>
    %result = amdgcn.test_inst outs %s_out ins %read_data {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    amdgcn.dealloc_lds %lds {sched.stage = 1 : i32}
  }
  return
}
