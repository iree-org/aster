// Unit tests for loop lowering

amdgcn.module @test_uniform_loop target = <gfx942> isa = <cdna3> {
  kernel @test_uniform_loop arguments <[#amdgcn.buffer_arg<address_space = generic, access = read_only>, #amdgcn.buffer_arg<address_space = generic>]> {
    %0 = load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %1 = load_arg 1 : !amdgcn.sgpr<[? + 2]>
    wait lgkm_cnt 0
    %2 = alloca : !amdgcn.sgpr
    %result, %token = load s_load_dword dest %2 addr %0 : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<constant>
    wait deps %token : !amdgcn.read_token<constant>
    %3 = lsir.from_reg %result : !amdgcn.sgpr -> i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    scf.for %arg0 = %c0_i32 to %3 step %c1_i32 : i32 {
      %4 = arith.muli %arg0, %c4_i32 : i32
      %5 = lsir.to_reg %4 : i32 -> !amdgcn.sgpr
      %6 = amdgcn.alloca : !amdgcn.vgpr
      %7 = amdgcn.vop1.vop1 <v_mov_b32_e32> %6, %5 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
      %8 = amdgcn.store global_store_dword data %7 addr %1 offset d(%7) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) -> !amdgcn.write_token<flat>
    }
    end_kernel
  }
}
