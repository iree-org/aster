// RUN: aster-opt --amdgcn-hoist-iter-arg-waits %s | FileCheck %s --check-prefix=HOIST
// RUN: aster-opt --amdgcn-hoist-iter-arg-waits --canonicalize %s | FileCheck %s --check-prefix=MERGE

// HOIST-LABEL: func.func @hoist_pure_iter_arg_wait
// HOIST:       scf.for {{.*}} iter_args(%[[TOK:.*]] = %{{.*}}, %[[DATA:.*]] = %{{.*}})
// HOIST-NEXT:    amdgcn.wait deps %[[TOK]] : !amdgcn.read_token<flat>
// HOIST:         amdgcn.global_load_dword
// HOIST:         amdgcn.test_inst
// HOIST:         scf.yield

func.func @hoist_pure_iter_arg_wait(%addr: !amdgcn.vgpr<[? + 2]>) {
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %s_compute = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32_mig1 = arith.constant 0 : i32
  %pro_data, %pro_tok = amdgcn.global_load_dword dest %dest addr %addr offset c(%c0_i32_mig1) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %res:2 = scf.for %i = %c1 to %c4 step %c1
      iter_args(%iter_tok = %pro_tok, %iter_data = %pro_data)
      -> (!amdgcn.read_token<flat>, !amdgcn.vgpr) {
    %c0_i32_mig2 = arith.constant 0 : i32
    %new_data, %new_tok = amdgcn.global_load_dword dest %dest addr %addr offset c(%c0_i32_mig2) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    amdgcn.wait deps %iter_tok : !amdgcn.read_token<flat>
    %result = amdgcn.test_inst outs %s_compute ins %iter_data
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    scf.yield %new_tok, %new_data : !amdgcn.read_token<flat>, !amdgcn.vgpr
  }
  amdgcn.wait deps %res#0 : !amdgcn.read_token<flat>
  %final = amdgcn.test_inst outs %s_compute ins %res#1
    : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  return
}

// HOIST-LABEL: func.func @clone_mixed_dep_wait
// HOIST:       scf.for {{.*}} iter_args(%[[ITOK:.*]] = %{{.*}}, %[[IDATA:.*]] = %{{.*}})
// HOIST-NEXT:    amdgcn.wait deps %[[ITOK]] : !amdgcn.read_token<flat>
// HOIST:         %[[ND:.*]], %[[NT:.*]] = amdgcn.global_load_dword
// HOIST:         amdgcn.wait deps %[[NT]] : !amdgcn.read_token<flat>
// HOIST:         amdgcn.test_inst
// HOIST:         scf.yield

func.func @clone_mixed_dep_wait(%addr: !amdgcn.vgpr<[? + 2]>) {
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %dest2 = amdgcn.alloca : !amdgcn.vgpr
  %s_compute = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32_mig3 = arith.constant 0 : i32
  %pro_data, %pro_tok = amdgcn.global_load_dword dest %dest addr %addr offset c(%c0_i32_mig3) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %res:2 = scf.for %i = %c1 to %c4 step %c1
      iter_args(%iter_tok = %pro_tok, %iter_data = %pro_data)
      -> (!amdgcn.read_token<flat>, !amdgcn.vgpr) {
    %c0_i32_mig4 = arith.constant 0 : i32
    %new_data, %new_tok = amdgcn.global_load_dword dest %dest2 addr %addr offset c(%c0_i32_mig4) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    amdgcn.wait deps %iter_tok, %new_tok
      : !amdgcn.read_token<flat>, !amdgcn.read_token<flat>
    %result = amdgcn.test_inst outs %s_compute ins %iter_data
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    scf.yield %new_tok, %new_data : !amdgcn.read_token<flat>, !amdgcn.vgpr
  }
  amdgcn.wait deps %res#0 : !amdgcn.read_token<flat>
  %final = amdgcn.test_inst outs %s_compute ins %res#1
    : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  return
}

// HOIST-LABEL: func.func @hoist_multiple_waits
// HOIST:       scf.for {{.*}} iter_args(
// HOIST-NEXT:    amdgcn.wait deps %{{.*}} : !amdgcn.write_token<shared>
// HOIST-NEXT:    amdgcn.wait deps %{{.*}} : !amdgcn.write_token<shared>
// HOIST:         amdgcn.ds_write_b32
// HOIST:         amdgcn.ds_read_b32
// HOIST:         amdgcn.test_inst
// HOIST:         scf.yield

// MERGE-LABEL: func.func @hoist_multiple_waits
// MERGE:       scf.for {{.*}} iter_args(
// MERGE-NEXT:    amdgcn.wait deps %{{.*}}, %{{.*}} : !amdgcn.write_token<shared>, !amdgcn.write_token<shared>
// MERGE:         amdgcn.ds_write_b32
// MERGE:         amdgcn.ds_read_b32
// MERGE:         amdgcn.test_inst
// MERGE:         scf.yield

func.func @hoist_multiple_waits(%data_in: !amdgcn.vgpr, %lds_addr_a: !amdgcn.vgpr, %lds_addr_b: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c0_i32 = arith.constant 0 : i32
  %s_out = amdgcn.alloca : !amdgcn.vgpr
  %s_read = amdgcn.alloca : !amdgcn.vgpr
  %wtok_a = amdgcn.ds_write_b32 data %data_in addr %lds_addr_a offset c(%c0_i32) : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
  %wtok_b = amdgcn.ds_write_b32 data %data_in addr %lds_addr_b offset c(%c0_i32) : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
  %res:2 = scf.for %i = %c1 to %c6 step %c1
      iter_args(%iter_wtok_a = %wtok_a, %iter_wtok_b = %wtok_b)
      -> (!amdgcn.write_token<shared>, !amdgcn.write_token<shared>) {
    %new_wtok_a = amdgcn.ds_write_b32 data %data_in addr %lds_addr_a offset c(%c0_i32) : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    amdgcn.wait deps %iter_wtok_a : !amdgcn.write_token<shared>
    %c0_i32_mig1 = arith.constant 0 : i32
    %read_data, %rtok = amdgcn.ds_read_b32 dest %s_read addr %lds_addr_a offset c(%c0_i32_mig1) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    %new_wtok_b = amdgcn.ds_write_b32 data %data_in addr %lds_addr_b offset c(%c0_i32) : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    amdgcn.wait deps %iter_wtok_b : !amdgcn.write_token<shared>
    %result = amdgcn.test_inst outs %s_out ins %read_data
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    scf.yield %new_wtok_a, %new_wtok_b
      : !amdgcn.write_token<shared>, !amdgcn.write_token<shared>
  }
  amdgcn.wait deps %res#0, %res#1
    : !amdgcn.write_token<shared>, !amdgcn.write_token<shared>
  return
}

// Intra-iteration-only wait must NOT be hoisted.
// HOIST-LABEL: func.func @no_hoist_intra_only_wait
// HOIST:       scf.for {{.*}} iter_args(%[[TOK:.*]] = %{{.*}}, %[[DATA:.*]] = %{{.*}})
// HOIST:         %[[ND:.*]], %[[NT:.*]] = amdgcn.global_load_dword
// HOIST-NEXT:    amdgcn.wait deps %[[NT]] : !amdgcn.read_token<flat>
// HOIST:         scf.yield

func.func @no_hoist_intra_only_wait(%addr: !amdgcn.vgpr<[? + 2]>) {
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %s_compute = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32_mig5 = arith.constant 0 : i32
  %pro_data, %pro_tok = amdgcn.global_load_dword dest %dest addr %addr offset c(%c0_i32_mig5) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %res:2 = scf.for %i = %c1 to %c4 step %c1
      iter_args(%iter_tok = %pro_tok, %iter_data = %pro_data)
      -> (!amdgcn.read_token<flat>, !amdgcn.vgpr) {
    // Wait depends only on intra-iteration token -- must stay after load
    %c0_i32_mig6 = arith.constant 0 : i32
    %new_data, %new_tok = amdgcn.global_load_dword dest %dest addr %addr offset c(%c0_i32_mig6) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    amdgcn.wait deps %new_tok : !amdgcn.read_token<flat>
    %result = amdgcn.test_inst outs %s_compute ins %new_data
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    scf.yield %new_tok, %new_data : !amdgcn.read_token<flat>, !amdgcn.vgpr
  }
  amdgcn.wait deps %res#0 : !amdgcn.read_token<flat>
  return
}

// Nested loop: inner loop's iter_arg wait must not be hoisted to outer loop.
// HOIST-LABEL: func.func @no_hoist_across_nested_loops
// HOIST:       scf.for
// HOIST-NEXT:    amdgcn.wait deps %{{.*}} : !amdgcn.read_token<flat>
// HOIST:         scf.for
// HOIST-NEXT:      amdgcn.wait deps %{{.*}} : !amdgcn.write_token<shared>
// HOIST:           scf.yield
// HOIST:         scf.yield

func.func @no_hoist_across_nested_loops(%addr: !amdgcn.vgpr<[? + 2]>, %lds_addr: !amdgcn.vgpr) {
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %s_out = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %c0_i32_mig7 = arith.constant 0 : i32
  %pro_data, %pro_tok = amdgcn.global_load_dword dest %dest addr %addr offset c(%c0_i32_mig7) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  // Outer loop with iter_arg token
  %res:2 = scf.for %i = %c1 to %c4 step %c1
      iter_args(%outer_tok = %pro_tok, %outer_data = %pro_data)
      -> (!amdgcn.read_token<flat>, !amdgcn.vgpr) {
    %c0_i32_mig8 = arith.constant 0 : i32
    %new_data, %new_tok = amdgcn.global_load_dword dest %dest addr %addr offset c(%c0_i32_mig8) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    amdgcn.wait deps %outer_tok : !amdgcn.read_token<flat>
    // Inner loop with its own iter_arg token
    %init_wtok = amdgcn.ds_write_b32 data %outer_data addr %lds_addr offset c(%c0_i32) : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    %inner_res = scf.for %j = %c0 to %c4 step %c1
        iter_args(%inner_wtok = %init_wtok) -> (!amdgcn.write_token<shared>) {
      amdgcn.wait deps %inner_wtok : !amdgcn.write_token<shared>
      %new_inner_wtok = amdgcn.ds_write_b32 data %outer_data addr %lds_addr offset c(%c0_i32) : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
      scf.yield %new_inner_wtok : !amdgcn.write_token<shared>
    }
    scf.yield %new_tok, %new_data : !amdgcn.read_token<flat>, !amdgcn.vgpr
  }
  amdgcn.wait deps %res#0 : !amdgcn.read_token<flat>
  return
}

// Barrier with no waits after it is moved after the hoisted waits.
// HOIST-LABEL: func.func @hoist_barrier_after_waits
// HOIST:       scf.for {{.*}} iter_args(%[[TOK:.*]] = %{{.*}}, %[[DATA:.*]] = %{{.*}})
// HOIST-NEXT:    amdgcn.wait deps %[[TOK]] : !amdgcn.write_token<shared>
// HOIST-NEXT:    amdgcn.s_barrier
// HOIST:         %{{.*}}, %{{.*}} = amdgcn.ds_read_b32
// HOIST:         amdgcn.ds_write_b32
// HOIST:         scf.yield

func.func @hoist_barrier_after_waits(%data_in: !amdgcn.vgpr, %lds_addr: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %s_read = amdgcn.alloca : !amdgcn.vgpr
  %init_wtok = amdgcn.ds_write_b32 data %data_in addr %lds_addr offset c(%c0_i32) : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
  %c0_i32_mig2 = arith.constant 0 : i32
  %init_data, %init_rtok = amdgcn.ds_read_b32 dest %s_read addr %lds_addr offset c(%c0_i32_mig2) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  %res:2 = scf.for %i = %c1 to %c4 step %c1
      iter_args(%iter_wtok = %init_wtok, %iter_data = %init_data)
      -> (!amdgcn.write_token<shared>, !amdgcn.vgpr) {
    amdgcn.wait deps %iter_wtok : !amdgcn.write_token<shared>
    amdgcn.s_barrier
    %c0_i32_mig3 = arith.constant 0 : i32
    %read_data, %rtok = amdgcn.ds_read_b32 dest %s_read addr %lds_addr offset c(%c0_i32_mig3) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    %new_wtok = amdgcn.ds_write_b32 data %data_in addr %lds_addr offset c(%c0_i32) : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    scf.yield %new_wtok, %read_data
      : !amdgcn.write_token<shared>, !amdgcn.vgpr
  }
  amdgcn.wait deps %res#0 : !amdgcn.write_token<shared>
  return
}

// Barrier with a wait AFTER it must NOT be moved.
// HOIST-LABEL: func.func @no_move_barrier_with_wait_after
// HOIST:       scf.for {{.*}} iter_args(%[[TOK:.*]] = %{{.*}}, %[[DATA:.*]] = %{{.*}})
// HOIST-NEXT:    amdgcn.wait deps %[[TOK]] : !amdgcn.write_token<shared>
// HOIST:         amdgcn.ds_write_b32
// HOIST-NEXT:    amdgcn.s_barrier
// HOIST:         amdgcn.wait deps %{{.*}} : !amdgcn.read_token<shared>
// HOIST:         scf.yield

func.func @no_move_barrier_with_wait_after(%data_in: !amdgcn.vgpr, %lds_addr: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %s_read = amdgcn.alloca : !amdgcn.vgpr
  %init_wtok = amdgcn.ds_write_b32 data %data_in addr %lds_addr offset c(%c0_i32) : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
  %c0_i32_mig4 = arith.constant 0 : i32
  %init_data, %init_rtok = amdgcn.ds_read_b32 dest %s_read addr %lds_addr offset c(%c0_i32_mig4) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  %res:2 = scf.for %i = %c1 to %c4 step %c1
      iter_args(%iter_wtok = %init_wtok, %iter_data = %init_data)
      -> (!amdgcn.write_token<shared>, !amdgcn.vgpr) {
    amdgcn.wait deps %iter_wtok : !amdgcn.write_token<shared>
    %new_wtok = amdgcn.ds_write_b32 data %data_in addr %lds_addr offset c(%c0_i32) : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
    amdgcn.s_barrier
    %c0_i32_mig5 = arith.constant 0 : i32
    %read_data, %rtok = amdgcn.ds_read_b32 dest %s_read addr %lds_addr offset c(%c0_i32_mig5) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %rtok : !amdgcn.read_token<shared>
    scf.yield %new_wtok, %read_data
      : !amdgcn.write_token<shared>, !amdgcn.vgpr
  }
  amdgcn.wait deps %res#0 : !amdgcn.write_token<shared>
  return
}

// Wait inside scf.if (not directly under scf.for) must NOT be hoisted.
// HOIST-LABEL: func.func @no_hoist_wait_inside_scf_if
// HOIST:       scf.for {{.*}} iter_args(%[[TOK:.*]] = %{{.*}}, %[[DATA:.*]] = %{{.*}})
// HOIST:         amdgcn.global_load_dword
// HOIST:         scf.if
// HOIST:           amdgcn.wait deps %[[TOK]]
// HOIST:         scf.yield

func.func @no_hoist_wait_inside_scf_if(%addr: !amdgcn.vgpr<[? + 2]>, %cond: i1) {
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %s_compute = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32_mig9 = arith.constant 0 : i32
  %pro_data, %pro_tok = amdgcn.global_load_dword dest %dest addr %addr offset c(%c0_i32_mig9) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %res:2 = scf.for %i = %c1 to %c4 step %c1
      iter_args(%iter_tok = %pro_tok, %iter_data = %pro_data)
      -> (!amdgcn.read_token<flat>, !amdgcn.vgpr) {
    %c0_i32_mig10 = arith.constant 0 : i32
    %new_data, %new_tok = amdgcn.global_load_dword dest %dest addr %addr offset c(%c0_i32_mig10) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    // Wait is inside scf.if -- not a direct child of the for body
    scf.if %cond {
      amdgcn.wait deps %iter_tok : !amdgcn.read_token<flat>
      %result = amdgcn.test_inst outs %s_compute ins %iter_data
        : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    }
    scf.yield %new_tok, %new_data : !amdgcn.read_token<flat>, !amdgcn.vgpr
  }
  amdgcn.wait deps %res#0 : !amdgcn.read_token<flat>
  return
}

// Wait with only count attributes (no token deps) must NOT be hoisted.
// HOIST-LABEL: func.func @no_hoist_count_only_wait
// HOIST:       scf.for {{.*}} iter_args(%[[TOK:.*]] = %{{.*}}, %[[DATA:.*]] = %{{.*}})
// HOIST:         amdgcn.global_load_dword
// HOIST:         amdgcn.wait vm_cnt 0
// HOIST:         scf.yield

func.func @no_hoist_count_only_wait(%addr: !amdgcn.vgpr<[? + 2]>) {
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %s_compute = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32_mig11 = arith.constant 0 : i32
  %pro_data, %pro_tok = amdgcn.global_load_dword dest %dest addr %addr offset c(%c0_i32_mig11) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %res:2 = scf.for %i = %c1 to %c4 step %c1
      iter_args(%iter_tok = %pro_tok, %iter_data = %pro_data)
      -> (!amdgcn.read_token<flat>, !amdgcn.vgpr) {
    %c0_i32_mig12 = arith.constant 0 : i32
    %new_data, %new_tok = amdgcn.global_load_dword dest %dest addr %addr offset c(%c0_i32_mig12) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    // Count-only wait has no token deps -- must not be hoisted
    amdgcn.wait vm_cnt 0
    %result = amdgcn.test_inst outs %s_compute ins %iter_data
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    scf.yield %new_tok, %new_data : !amdgcn.read_token<flat>, !amdgcn.vgpr
  }
  amdgcn.wait deps %res#0 : !amdgcn.read_token<flat>
  return
}
