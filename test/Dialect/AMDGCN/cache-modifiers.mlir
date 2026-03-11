// RUN: aster-opt %s --verify-roundtrip

// Test cache modifier (glc, slc, scc) roundtrip on load and store ops.

// Global load with single modifier
func.func @test_global_load_glc(%dest: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %result, %token = amdgcn.load global_load_dword dest %dest addr %addr
    offset c(%c0) {glc = true}
    : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>, i32)
      -> !amdgcn.read_token<flat>
  return
}

// Global load with slc
func.func @test_global_load_slc(%dest: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %result, %token = amdgcn.load global_load_dword dest %dest addr %addr
    offset c(%c0) {slc = true}
    : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>, i32)
      -> !amdgcn.read_token<flat>
  return
}

// Global load with all three modifiers
func.func @test_global_load_all(%dest: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %result, %token = amdgcn.load global_load_dword dest %dest addr %addr
    offset c(%c0) {glc = true, scc = true, slc = true}
    : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>, i32)
      -> !amdgcn.read_token<flat>
  return
}

// Global store with scc
func.func @test_global_store_scc(%data: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %token = amdgcn.store global_store_dword data %data addr %addr
    offset c(%c0) {scc = true}
    : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32)
      -> !amdgcn.write_token<flat>
  return
}

// Buffer load with glc
func.func @test_buffer_load_glc(
    %dest: !amdgcn.vgpr,
    %rsrc: !amdgcn.sgpr<[? + 4]>,
    %soffset: !amdgcn.sgpr,
    %voffset: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %result, %token = amdgcn.load buffer_load_dword dest %dest addr %rsrc
    offset u(%soffset) + d(%voffset) + c(%c0) {glc = true}
    : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.read_token<flat>
  return
}

// Buffer store with glc and slc
func.func @test_buffer_store_glc_slc(
    %data: !amdgcn.vgpr,
    %rsrc: !amdgcn.sgpr<[? + 4]>,
    %soffset: !amdgcn.sgpr,
    %voffset: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %token = amdgcn.store buffer_store_dword data %data addr %rsrc
    offset u(%soffset) + d(%voffset) + c(%c0) {glc = true, slc = true}
    : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.write_token<flat>
  return
}

// No modifiers - backward compat
func.func @test_no_modifiers(%dest: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %result, %token = amdgcn.load global_load_dword dest %dest addr %addr
    offset c(%c0)
    : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>, i32)
      -> !amdgcn.read_token<flat>
  return
}
