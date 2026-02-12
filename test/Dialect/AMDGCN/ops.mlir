// RUN: aster-opt %s --verify-roundtrip

func.func @test_alloca_vgpr() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  return
}

func.func @test_make_register_range() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %3 = amdgcn.make_register_range %0, %1, %2 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return
}

func.func @test_make_register_range_single() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.make_register_range %0 : !amdgcn.vgpr
  return
}

amdgcn.module @test_module target = #amdgcn.target<gfx940> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    %0 = amdgcn.alloca : !amdgcn.vgpr
    amdgcn.end_kernel
  }

  amdgcn.kernel @empty_kernel {
    amdgcn.end_kernel
  }
}

amdgcn.module @named_module target = #amdgcn.target<gfx940> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @kernel_in_named_module {
    amdgcn.end_kernel
  }
}

// Test kernel with ptr argument
amdgcn.module @kernel_with_ptr target = #amdgcn.target<gfx940> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @kernel_ptr arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>,
    #amdgcn.buffer_arg<address_space = private, access = read_only, flags = const | volatile>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write, type = !ptr.ptr<#ptr.generic_space>>
  ]> {
    amdgcn.end_kernel
  }
}

// Test kernel with by value argument
amdgcn.module @kernel_with_int target = #amdgcn.target<gfx940> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @kernel_by_val arguments <[
    #amdgcn.by_val_arg<size = 4, name = "int_arg", type = i32>,
    #amdgcn.by_val_arg<size = 8, alignment = 8, name = "long_arg", type = i64>
  ]> {
    amdgcn.end_kernel
  }
}

// Test gfx950/cdna4 module
amdgcn.module @cdna4_module target = #amdgcn.target<gfx950> isa = #amdgcn.isa<cdna4> {
  amdgcn.kernel @cdna4_kernel {
    amdgcn.end_kernel
  }
}

// Test library with cdna4 isa
amdgcn.library @library_cdna4 isa = [#amdgcn.isa<cdna4>] {
  func.func @cdna4_func() {
    return
  }
}

// Test library with both cdna3 and cdna4
amdgcn.library @library_cdna3_cdna4 isa = [#amdgcn.isa<cdna3>, #amdgcn.isa<cdna4>] {
  func.func @cross_cdna_func() {
    return
  }
}

// Test library operations (target-agnostic, no isa attribute)
amdgcn.library @empty_library {
}

amdgcn.library @library_with_func {
  func.func @compute_offset(%bidx: index, %tidx: index) -> index {
    %c64 = arith.constant 64 : index
    %widx = arith.divui %tidx, %c64 : index
    %lidx = arith.remui %tidx, %c64 : index
    %result = arith.addi %widx, %lidx : index
    return %result : index
  }
}

amdgcn.library @library_with_multiple_funcs {
  func.func private @helper(%x: index) -> index {
    %c2 = arith.constant 2 : index
    %result = arith.muli %x, %c2 : index
    return %result : index
  }

  func.func @main_func(%a: index, %b: index) -> index {
    %sum = arith.addi %a, %b : index
    %result = func.call @helper(%sum) : (index) -> index
    return %result : index
  }
}

// Test library operations (target-specific, with isa attribute)
amdgcn.library @library_multi_isa isa = [#amdgcn.isa<cdna3>, #amdgcn.isa<rdna4>] {
  func.func @multi_target_func() {
    return
  }
}

// Target-agnostic library with arith operations (no isa attribute)
amdgcn.library @library_arith_no_isa {
  func.func @arith_ops(%a: i32, %b: i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    %prod = arith.muli %sum, %b : i32
    return %prod : i32
  }
}

// Target-specific library with arith operations (with isa attribute)
amdgcn.library @library_arith_with_isa isa = [#amdgcn.isa<cdna3>] {
  func.func @arith_ops_cdna3(%a: i32, %b: i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    %diff = arith.subi %a, %b : i32
    %result = arith.muli %sum, %diff : i32
    return %result : i32
  }
}

// Library with isa attribute can contain both arith and AMDGCN instructions
amdgcn.library @library_mixed_ops isa = [#amdgcn.isa<cdna3>] {
  func.func @mixed_arith_and_amdgcn(%x: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %doubled = arith.muli %x, %c2 : i32
    amdgcn.vop1.v_nop
    return %doubled : i32
  }
}

// Library with isa attribute can contain AMDGCN instructions
amdgcn.library @library_with_amdgcn_insts isa = [#amdgcn.isa<cdna3>] {
  func.func @func_with_nop() {
    amdgcn.vop1.v_nop
    return
  }
}

func.func @test_allocated_registers() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<5>
  %1 = amdgcn.alloca : !amdgcn.agpr<5>
  %2 = amdgcn.alloca : !amdgcn.sgpr<5>
  return
}

func.func @test_make_register_ranges() -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<2>
  %1 = amdgcn.alloca : !amdgcn.vgpr<3>
  %2 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
  %4, %5 = amdgcn.split_register_range %2 : !amdgcn.vgpr_range<[2 : 4]>
  return %4, %5: !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
}

func.func @test_make_register_ranges_relocatable() -> (!amdgcn.vgpr, !amdgcn.vgpr) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  %4, %5 = amdgcn.split_register_range %2 : !amdgcn.vgpr_range<[? + 2]>
  return %4, %5: !amdgcn.vgpr, !amdgcn.vgpr
}

//===----------------------------------------------------------------------===//
// VOP1 Operations
//===----------------------------------------------------------------------===//

func.func @test_vop1_nop() {
  amdgcn.vop1.v_nop
  return
}

//===----------------------------------------------------------------------===//
// SOPP Operations
//===----------------------------------------------------------------------===//

func.func @test_sopp_nop() {
  amdgcn.sopp.sopp #amdgcn.inst<s_nop>
  return
}

func.func @test_sopp_nop_with_imm() {
  amdgcn.sopp.sopp #amdgcn.inst<s_nop> , imm = 0
  return
}

func.func @test_sopp_nop_with_imm_max() {
  amdgcn.sopp.sopp #amdgcn.inst<s_nop> , imm = 15
  return
}

//===----------------------------------------------------------------------===//
// Wait Operation
//===----------------------------------------------------------------------===//

func.func @test_waits(
    %rt1: !amdgcn.read_token<flat>,
    %rt2: !amdgcn.read_token<shared>,
    %wt1: !amdgcn.write_token<flat>) {
  // Wait for tokens.
  amdgcn.wait deps %rt1 : !amdgcn.read_token<flat>
  amdgcn.wait deps %rt1, %rt2 : !amdgcn.read_token<flat>, !amdgcn.read_token<shared>
  amdgcn.wait deps %rt1, %wt1 : !amdgcn.read_token<flat>, !amdgcn.write_token<flat>
  // Mixed wait with counts and tokens.
  amdgcn.wait vm_cnt 0 lgkm_cnt 1 deps %rt1, %wt1 : !amdgcn.read_token<flat>, !amdgcn.write_token<flat>
  // Wait with only counts.
  amdgcn.wait vm_cnt 2
  return
}

//===----------------------------------------------------------------------===//
// LDS Buffer Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Buffer Resource Descriptor Operations
//===----------------------------------------------------------------------===//

func.func @test_make_buffer_rsrc(
    %base : !amdgcn.sgpr_range<[? + 2]>,
    %num_records : !amdgcn.sgpr) {
  // Raw buffer (stride=0, no swizzle)
  %stride = arith.constant 0 : i32
  %rsrc = amdgcn.make_buffer_rsrc %base, %num_records, %stride,
    cache_swizzle = false, swizzle_enable = false, flags = 0
    : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr_range<[? + 4]>
  return
}

func.func @test_make_buffer_rsrc_with_stride(
    %base : !amdgcn.sgpr_range<[? + 2]>,
    %num_records : !amdgcn.sgpr) {
  // Structured buffer with 16-byte stride (e.g., float4 elements)
  %stride = arith.constant 16 : i32
  %rsrc = amdgcn.make_buffer_rsrc %base, %num_records, %stride,
    cache_swizzle = false, swizzle_enable = false, flags = 131076
    : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr_range<[? + 4]>
  return
}

func.func @test_make_buffer_rsrc_swizzled(
    %base : !amdgcn.sgpr_range<[? + 2]>,
    %num_records : !amdgcn.sgpr) {
  // Swizzled buffer access with stride=4 (f32 elements)
  %stride = arith.constant 4 : i32
  %rsrc = amdgcn.make_buffer_rsrc %base, %num_records, %stride,
    cache_swizzle = true, swizzle_enable = true, flags = 0
    : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr_range<[? + 4]>
  return
}

func.func @test_make_buffer_rsrc_allocated(
    %base : !amdgcn.sgpr_range<[0 : 2]>,
    %num_records : !amdgcn.sgpr<2>) {
  %stride = arith.constant 0 : i32
  %rsrc = amdgcn.make_buffer_rsrc %base, %num_records, %stride,
    cache_swizzle = false, swizzle_enable = false, flags = 0
    : (!amdgcn.sgpr_range<[0 : 2]>, !amdgcn.sgpr<2>, i32) -> !amdgcn.sgpr_range<[? + 4]>
  return
}

//===----------------------------------------------------------------------===//
// Buffer Load/Store Operations (MUBUF)
//===----------------------------------------------------------------------===//

func.func @test_buffer_load_dword(
    %dest : !amdgcn.vgpr,
    %rsrc : !amdgcn.sgpr_range<[? + 4]>,
    %vaddr : !amdgcn.vgpr,
    %soffset : !amdgcn.sgpr,
    %off : i32) {
  %result, %tok = amdgcn.load buffer_load_dword dest %dest addr %rsrc
    offset u(%soffset) + d(%vaddr) + c(%off)
    : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.read_token<flat>
  return
}

func.func @test_buffer_load_dwordx2(
    %dest : !amdgcn.vgpr_range<[? + 2]>,
    %rsrc : !amdgcn.sgpr_range<[? + 4]>,
    %vaddr : !amdgcn.vgpr,
    %soffset : !amdgcn.sgpr,
    %off : i32) {
  %result, %tok = amdgcn.load buffer_load_dwordx2 dest %dest addr %rsrc
    offset u(%soffset) + d(%vaddr) + c(%off)
    : dps(!amdgcn.vgpr_range<[? + 2]>) ins(!amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.read_token<flat>
  return
}

func.func @test_buffer_load_dwordx3(
    %dest : !amdgcn.vgpr_range<[? + 3]>,
    %rsrc : !amdgcn.sgpr_range<[? + 4]>,
    %vaddr : !amdgcn.vgpr,
    %soffset : !amdgcn.sgpr,
    %off : i32) {
  %result, %tok = amdgcn.load buffer_load_dwordx3 dest %dest addr %rsrc
    offset u(%soffset) + d(%vaddr) + c(%off)
    : dps(!amdgcn.vgpr_range<[? + 3]>) ins(!amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.read_token<flat>
  return
}

func.func @test_buffer_load_dwordx4(
    %dest : !amdgcn.vgpr_range<[? + 4]>,
    %rsrc : !amdgcn.sgpr_range<[? + 4]>,
    %vaddr : !amdgcn.vgpr,
    %soffset : !amdgcn.sgpr,
    %off : i32) {
  %result, %tok = amdgcn.load buffer_load_dwordx4 dest %dest addr %rsrc
    offset u(%soffset) + d(%vaddr) + c(%off)
    : dps(!amdgcn.vgpr_range<[? + 4]>) ins(!amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.read_token<flat>
  return
}

func.func @test_buffer_store_dword(
    %data : !amdgcn.vgpr,
    %rsrc : !amdgcn.sgpr_range<[? + 4]>,
    %vaddr : !amdgcn.vgpr,
    %soffset : !amdgcn.sgpr,
    %off : i32) {
  %tok = amdgcn.store buffer_store_dword data %data addr %rsrc
    offset u(%soffset) + d(%vaddr) + c(%off)
    : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.write_token<flat>
  return
}

func.func @test_buffer_store_dwordx2(
    %data : !amdgcn.vgpr_range<[? + 2]>,
    %rsrc : !amdgcn.sgpr_range<[? + 4]>,
    %vaddr : !amdgcn.vgpr,
    %soffset : !amdgcn.sgpr,
    %off : i32) {
  %tok = amdgcn.store buffer_store_dwordx2 data %data addr %rsrc
    offset u(%soffset) + d(%vaddr) + c(%off)
    : ins(!amdgcn.vgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.write_token<flat>
  return
}

func.func @test_buffer_store_dwordx3(
    %data : !amdgcn.vgpr_range<[? + 3]>,
    %rsrc : !amdgcn.sgpr_range<[? + 4]>,
    %vaddr : !amdgcn.vgpr,
    %soffset : !amdgcn.sgpr,
    %off : i32) {
  %tok = amdgcn.store buffer_store_dwordx3 data %data addr %rsrc
    offset u(%soffset) + d(%vaddr) + c(%off)
    : ins(!amdgcn.vgpr_range<[? + 3]>, !amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.write_token<flat>
  return
}

func.func @test_buffer_store_dwordx4(
    %data : !amdgcn.vgpr_range<[? + 4]>,
    %rsrc : !amdgcn.sgpr_range<[? + 4]>,
    %vaddr : !amdgcn.vgpr,
    %soffset : !amdgcn.sgpr,
    %off : i32) {
  %tok = amdgcn.store buffer_store_dwordx4 data %data addr %rsrc
    offset u(%soffset) + d(%vaddr) + c(%off)
    : ins(!amdgcn.vgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.write_token<flat>
  return
}

func.func @lds_buffer_ops(%arg0: index, %arg1: index) {
  %0 = amdgcn.alloc_lds %arg0
  amdgcn.get_lds_offset %0 : i32
  amdgcn.dealloc_lds %0
  %1 = amdgcn.alloc_lds %arg1
  amdgcn.get_lds_offset %0 : index
  amdgcn.dealloc_lds %1
  %2 = amdgcn.alloc_lds 32 offset 64
  return
}
