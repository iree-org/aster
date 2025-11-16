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
