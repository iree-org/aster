// RUN: aster-opt %s --amdgcn-backend --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s --check-prefix=ASM

// ASM-LABEL: copy_kernel:
//   ASM:        s_load_dwordx2
//   ASM:        s_load_dwordx2
//   ASM:        s_waitcnt lgkmcnt(0)
//   ASM:        global_load_dword
//   ASM:        s_waitcnt vmcnt(0)
//   ASM:        global_store_dword
//   ASM:        s_endpgm

module attributes {dlti.dl_spec = #dlti.dl_spec<!ptr.ptr<#amdgcn.addr_space<local, read_write>> = #ptr.spec<size = 32, abi = 32, preferred = 32>, !ptr.ptr<#amdgcn.addr_space<global, read_write>> = #ptr.spec<size = 64, abi = 64, preferred = 64>>} {
  amdgcn.module @m target = <gfx942> attributes {normal_forms = [#amdgcn.no_reg_cast_ops]} {
    kernel @copy_kernel arguments <[#amdgcn.buffer_arg<type = !ptr.ptr<#ptr.generic_space>>, #amdgcn.buffer_arg<type = !ptr.ptr<#ptr.generic_space>>, #amdgcn.block_dim_arg<x>, #amdgcn.block_dim_arg<y>, #amdgcn.block_dim_arg<z>, #amdgcn.grid_dim_arg<x>, #amdgcn.grid_dim_arg<y>, #amdgcn.grid_dim_arg<z>]> {
      %0 = load_arg 1 : !amdgcn.sgpr<[? + 2]>
      %1 = load_arg 0 : !amdgcn.sgpr<[? + 2]>
      %2 = alloca : !amdgcn.vgpr
      %c0_i32_mig1 = arith.constant 0 : i32
      %voff = alloca : !amdgcn.vgpr
      %dest_res, %token = amdgcn.global_load_dword dest %2 addr %1 offset d(%voff) + c(%c0_i32_mig1) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
      %voff2 = alloca : !amdgcn.vgpr
      %3 = amdgcn.global_store_dword data %dest_res addr %0 offset d(%voff2) + c(%c0_i32_mig1) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
      end_kernel
    }
  }
}
