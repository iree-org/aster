// Without bufferization, regalloc assigns the same VGPR to the loop block arg
// (carrying previous iteration's value) and the new stage-0 write.
// regalloc-only: shows the clobber (block arg and new write share same VGPR)
// RUN: aster-opt %s --cse --amdgcn-register-allocation --cse --aster-disable-verifiers --aster-suppress-disabled-verifier-warning --split-input-file | FileCheck %s --check-prefix=REGONLY

// With bufferization, explicit v_mov_b32 copies separate the lifetimes.
// bufferization + regalloc: uses different VGPRs
// RUN: aster-opt %s --amdgcn-bufferization --cse --amdgcn-register-allocation --cse --aster-disable-verifiers --aster-suppress-disabled-verifier-warning --split-input-file | FileCheck %s --check-prefix=BUFREGALLOC



// REGONLY-LABEL: kernel @iter_arg_clobber_2stage {
//       REGONLY:   cf.br ^bb1({{.*}} : !amdgcn.vgpr<0>)
//       REGONLY: ^bb1(%[[CARRIED:.*]]: !amdgcn.vgpr<0>):
// No copies inserted -- this is the bug
//  REGONLY-NOT:   v_mov_b32
//       REGONLY:   %[[NEW:.*]] = test_inst outs {{.*}} : (!amdgcn.vgpr<0>,
//       REGONLY:   test_inst ins %[[CARRIED]] : (!amdgcn.vgpr<0>)
//  REGONLY-NOT:   v_mov_b32
//       REGONLY:   cf.cond_br {{.*}}, ^bb1(%[[NEW]] : !amdgcn.vgpr<0>)

// BUFREGALLOC-LABEL: kernel @iter_arg_clobber_2stage {
//       BUFREGALLOC:   cf.br ^bb1({{.*}} : !amdgcn.vgpr<0>)
//       BUFREGALLOC: ^bb1(%[[CARRIED:.*]]: !amdgcn.vgpr<0>):
// New write goes to a different VGPR, carried value is safe
//       BUFREGALLOC:   %[[NEW:.*]] = test_inst outs {{.*}} : (!amdgcn.vgpr<1>,
//       BUFREGALLOC:   test_inst ins %[[CARRIED]] : (!amdgcn.vgpr<0>)
// Copy rotates new value into carried register for next iteration
//       BUFREGALLOC:   amdgcn.vop1.vop1 <v_mov_b32_e32> {{.*}}, %[[NEW]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<0>
//       BUFREGALLOC:   cf.cond_br {{.*}}, ^bb1({{.*}} : !amdgcn.vgpr<0>)
amdgcn.module @iter_arg_clobber_2stage_mod target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @iter_arg_clobber_2stage {
    %alloc_init = alloca : !amdgcn.vgpr
    %alloc_new = alloca : !amdgcn.vgpr
    %s0 = alloca : !amdgcn.sgpr
    %init = test_inst outs %alloc_init ins %s0 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %cond = func.call @rand() : () -> i1
    cf.br ^loop(%init : !amdgcn.vgpr)
  ^loop(%carried: !amdgcn.vgpr):
    // Stage 0: write new value. Without bufferization, clobbers %carried.
    %new_val = test_inst outs %alloc_new ins %s0 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    // Stage 1: read carried value from previous iteration.
    test_inst ins %carried : (!amdgcn.vgpr) -> ()
    %cond2 = func.call @rand() : () -> i1
    cf.cond_br %cond2, ^loop(%new_val : !amdgcn.vgpr), ^exit
  ^exit:
    end_kernel
  }
}

// -----

// REGONLY-LABEL: kernel @iter_arg_clobber_3stage {
//       REGONLY:   cf.br ^bb1({{.*}} !amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
//       REGONLY: ^bb1(%[[C0:.*]]: !amdgcn.vgpr<0>, %[[C1:.*]]: !amdgcn.vgpr<1>):
// No copies -- clobber occurs
//  REGONLY-NOT:   v_mov_b32
// Stage 0 writes to vgpr<0>, same register as carried0 -- clobber!
//       REGONLY:   test_inst outs {{.*}} : (!amdgcn.vgpr<0>,
// Stage 1 reads carried0 via %[[C0]], but it was already overwritten
//       REGONLY:   test_inst outs {{.*}} ins %[[C0]] : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>)
// Stage 2 reads carried1
//       REGONLY:   test_inst ins %[[C1]] : (!amdgcn.vgpr<1>)
//       REGONLY:   cf.cond_br {{.*}}, ^bb1({{.*}} !amdgcn.vgpr<0>, !amdgcn.vgpr<1>)

// BUFREGALLOC-LABEL: kernel @iter_arg_clobber_3stage {
//       BUFREGALLOC:   cf.br ^bb1({{.*}} !amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
//       BUFREGALLOC: ^bb1(%[[C0:.*]]: !amdgcn.vgpr<0>, %[[C1:.*]]: !amdgcn.vgpr<1>):
// Bufferization copies carried0 to vgpr<2> before stage 0 write
//       BUFREGALLOC:   amdgcn.vop1.vop1 <v_mov_b32_e32> {{.*}}, %[[C0]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<0>)
// Stage 0 writes to vgpr<0>, but carried0 is safe in vgpr<2>
//       BUFREGALLOC:   test_inst outs {{.*}} : (!amdgcn.vgpr<0>,
// Bufferization copies carried1 to vgpr<3>
//       BUFREGALLOC:   amdgcn.vop1.vop1 <v_mov_b32_e32> {{.*}}, %[[C1]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<1>)
// Stage 1 reads the safe copy of carried0 from vgpr<2>
//       BUFREGALLOC:   test_inst outs {{.*}} ins {{.*}} : (!amdgcn.vgpr<1>, !amdgcn.vgpr<2>)
// Stage 2 reads the safe copy of carried1 from vgpr<3>
//       BUFREGALLOC:   test_inst ins {{.*}} : (!amdgcn.vgpr<3>)
//       BUFREGALLOC:   cf.cond_br
amdgcn.module @iter_arg_clobber_3stage_mod target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @iter_arg_clobber_3stage {
    %a0 = alloca : !amdgcn.vgpr
    %a1 = alloca : !amdgcn.vgpr
    %s0 = alloca : !amdgcn.sgpr
    %init0 = test_inst outs %a0 ins %s0 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %init1 = test_inst outs %a1 ins %s0 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    cf.br ^loop(%init0, %init1 : !amdgcn.vgpr, !amdgcn.vgpr)
  ^loop(%carried0: !amdgcn.vgpr, %carried1: !amdgcn.vgpr):
    // Stage 0: produce new value. Without bufferization, clobbers carried0.
    %new0 = test_inst outs %a0 ins %s0 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    // Stage 1: consume carried0, produce new value. Reads clobbered data
    // without bufferization.
    %new1 = test_inst outs %a1 ins %carried0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    // Stage 2: consume carried1.
    test_inst ins %carried1 : (!amdgcn.vgpr) -> ()
    %cond = func.call @rand() : () -> i1
    cf.cond_br %cond, ^loop(%new0, %new1 : !amdgcn.vgpr, !amdgcn.vgpr), ^exit
  ^exit:
    end_kernel
  }
}

// -----

// No clobber when carried value is read BEFORE the new write.
// Both regalloc-only and bufferization+regalloc produce correct results here as
// no copies are inserted.

// REGONLY-LABEL: kernel @iter_arg_no_clobber_read_first {
//       REGONLY: ^bb1(%[[CARRIED:.*]]: !amdgcn.vgpr<0>):
// Read carried value first -- safe regardless of what happens next
//       REGONLY:   test_inst ins %[[CARRIED]] : (!amdgcn.vgpr<0>)
// Write new value after read -- no clobber
//       REGONLY:   test_inst outs {{.*}} : (!amdgcn.vgpr<0>,

// BUFREGALLOC-LABEL: kernel @iter_arg_no_clobber_read_first {
//       BUFREGALLOC: ^bb1({{.*}}: !amdgcn.vgpr<0>):
//   BUFREGALLOC-NOT:   v_mov_b32
//       BUFREGALLOC:   test_inst ins {{.*}} : (!amdgcn.vgpr<0>)
//       BUFREGALLOC:   test_inst outs {{.*}} : (!amdgcn.vgpr<
amdgcn.module @iter_arg_no_clobber_read_first_mod target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @iter_arg_no_clobber_read_first {
    %alloc = alloca : !amdgcn.vgpr
    %s0 = alloca : !amdgcn.sgpr
    %init = test_inst outs %alloc ins %s0 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    cf.br ^loop(%init : !amdgcn.vgpr)
  ^loop(%carried: !amdgcn.vgpr):
    // Read carried value FIRST -- safe regardless of register assignment
    test_inst ins %carried : (!amdgcn.vgpr) -> ()
    // Write new value AFTER read -- no clobber
    %new_val = test_inst outs %alloc ins %s0 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %cond = func.call @rand() : () -> i1
    cf.cond_br %cond, ^loop(%new_val : !amdgcn.vgpr), ^exit
  ^exit:
    end_kernel
  }
}
