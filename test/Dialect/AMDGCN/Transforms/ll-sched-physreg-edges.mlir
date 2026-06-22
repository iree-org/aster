// RUN: aster-opt %s --pass-pipeline="builtin.module(test-amdgcn-sched-graph)" 2>&1 | FileCheck %s

amdgcn.module @physreg_raw target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @physreg_raw
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "v_mov_b32/test.writer -> v_mov_b32/test.reader"
  // CHECK-NOT:     label = "v_mov_b32/test.reader -> v_mov_b32/test.writer"
  // CHECK:       }
  amdgcn.kernel @physreg_raw {
    %wsrc = amdgcn.alloca : !amdgcn.vgpr<35>
    %w33  = amdgcn.alloca : !amdgcn.vgpr<33>
    amdgcn.v_mov_b32 outs(%w33) ins(%wsrc) {test.writer} : outs(!amdgcn.vgpr<33>) ins(!amdgcn.vgpr<35>)
    %r33  = amdgcn.alloca : !amdgcn.vgpr<33>
    %rdst = amdgcn.alloca : !amdgcn.vgpr<34>
    amdgcn.v_mov_b32 outs(%rdst) ins(%r33) {test.reader} : outs(!amdgcn.vgpr<34>) ins(!amdgcn.vgpr<33>)
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @physreg_war target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @physreg_war
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "v_mov_b32/test.reader -> v_mov_b32/test.writer"
  // CHECK-NOT:     label = "v_mov_b32/test.writer -> v_mov_b32/test.reader"
  // CHECK:       }
  amdgcn.kernel @physreg_war {
    %r33  = amdgcn.alloca : !amdgcn.vgpr<33>
    %rdst = amdgcn.alloca : !amdgcn.vgpr<34>
    amdgcn.v_mov_b32 outs(%rdst) ins(%r33) {test.reader} : outs(!amdgcn.vgpr<34>) ins(!amdgcn.vgpr<33>)
    %w33  = amdgcn.alloca : !amdgcn.vgpr<33>
    %wsrc = amdgcn.alloca : !amdgcn.vgpr<35>
    amdgcn.v_mov_b32 outs(%w33) ins(%wsrc) {test.writer} : outs(!amdgcn.vgpr<33>) ins(!amdgcn.vgpr<35>)
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @physreg_waw target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @physreg_waw
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "v_mov_b32/test.first -> v_mov_b32/test.second"
  // CHECK-NOT:     label = "v_mov_b32/test.second -> v_mov_b32/test.first"
  // CHECK:       }
  amdgcn.kernel @physreg_waw {
    %src1 = amdgcn.alloca : !amdgcn.vgpr<34>
    %w1   = amdgcn.alloca : !amdgcn.vgpr<33>
    amdgcn.v_mov_b32 outs(%w1) ins(%src1) {test.first}  : outs(!amdgcn.vgpr<33>) ins(!amdgcn.vgpr<34>)
    %src2 = amdgcn.alloca : !amdgcn.vgpr<35>
    %w2   = amdgcn.alloca : !amdgcn.vgpr<33>
    amdgcn.v_mov_b32 outs(%w2) ins(%src2) {test.second} : outs(!amdgcn.vgpr<33>) ins(!amdgcn.vgpr<35>)
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @physreg_partial_overlap target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @physreg_partial_overlap
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "global_load_dwordx2 -> v_mov_b32"
  // CHECK:       }
  amdgcn.kernel @physreg_partial_overlap {
    %a0   = amdgcn.alloca : !amdgcn.vgpr<0>
    %a1   = amdgcn.alloca : !amdgcn.vgpr<1>
    %addr = amdgcn.make_register_range %a0, %a1 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
    %d32  = amdgcn.alloca : !amdgcn.vgpr<32>
    %d33  = amdgcn.alloca : !amdgcn.vgpr<33>
    %pair = amdgcn.make_register_range %d32, %d33 : !amdgcn.vgpr<32>, !amdgcn.vgpr<33>
    %c0   = arith.constant 0 : i32
    %tok = amdgcn.global_load_dwordx2 dest %pair addr %addr offset c(%c0) : outs(!amdgcn.vgpr<[32 : 34]>) ins(!amdgcn.vgpr<[0 : 2]>) mods(i32) -> !amdgcn.read_token<flat>
    %r33  = amdgcn.alloca : !amdgcn.vgpr<33>
    %rdst = amdgcn.alloca : !amdgcn.vgpr<40>
    amdgcn.v_mov_b32 outs(%rdst) ins(%r33) : outs(!amdgcn.vgpr<40>) ins(!amdgcn.vgpr<33>)
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @physreg_distinct_index target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @physreg_distinct_index
  // CHECK:       digraph SchedGraph
  // CHECK-NOT:     label = "v_mov_b32/test.a -> v_mov_b32/test.b"
  // CHECK-NOT:     label = "v_mov_b32/test.b -> v_mov_b32/test.a"
  // CHECK:       }
  amdgcn.kernel @physreg_distinct_index {
    %asrc = amdgcn.alloca : !amdgcn.vgpr<35>
    %a33  = amdgcn.alloca : !amdgcn.vgpr<33>
    amdgcn.v_mov_b32 outs(%a33) ins(%asrc) {test.a} : outs(!amdgcn.vgpr<33>) ins(!amdgcn.vgpr<35>)
    %bsrc = amdgcn.alloca : !amdgcn.vgpr<36>
    %b34  = amdgcn.alloca : !amdgcn.vgpr<34>
    amdgcn.v_mov_b32 outs(%b34) ins(%bsrc) {test.b} : outs(!amdgcn.vgpr<34>) ins(!amdgcn.vgpr<36>)
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @physreg_distinct_file target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @physreg_distinct_file
  // CHECK:       digraph SchedGraph
  // CHECK-NOT:     label = "v_mov_b32 -> s_mov_b32"
  // CHECK-NOT:     label = "s_mov_b32 -> v_mov_b32"
  // CHECK:       }
  amdgcn.kernel @physreg_distinct_file {
    %vsrc = amdgcn.alloca : !amdgcn.vgpr<35>
    %v33  = amdgcn.alloca : !amdgcn.vgpr<33>
    amdgcn.v_mov_b32 outs(%v33) ins(%vsrc) : outs(!amdgcn.vgpr<33>) ins(!amdgcn.vgpr<35>)
    %c0   = arith.constant 0 : i32
    %s33  = amdgcn.alloca : !amdgcn.sgpr<33>
    amdgcn.s_mov_b32 outs(%s33) ins(%c0) : outs(!amdgcn.sgpr<33>) ins(i32)
    amdgcn.end_kernel
  }
}
