// Hello ASTER: compute 32 + 10 = 42, self-check, trap if wrong.
// Compare MLIR directly to assembly output -- this is WYSIWYG.
//
// Key concepts:
//   amdgcn.module  - target GPU module (gfx942 = MI300X)
//   kernel / end_kernel - kernel entry/exit
//   alloca : !amdgcn.vgpr<N> - allocate physical register vN
//   v_add_u32 outs(%dst) ins(%a, %b) - DPS: destination-passing style
//   cmpi v_cmp_ne_i32 - vector compare (sets VCC register)
//   cbranch s_cbranch_vccnz - conditional branch on VCC
//   s_trap 2 - hardware trap (GPU abort)

module {
  amdgcn.module @hello target = <gfx942> {
    kernel @kernel {
    ^entry:
      %v0 = alloca : !amdgcn.vgpr<0>
      %v1 = alloca : !amdgcn.vgpr<1>
      %v2 = alloca : !amdgcn.vgpr<2>

      %c32 = arith.constant 32 : i32
      amdgcn.v_mov_b32 outs(%v1) ins(%c32) : outs(!amdgcn.vgpr<1>) ins(i32)

      %c10 = arith.constant 10 : i32
      amdgcn.v_mov_b32 outs(%v2) ins(%c10) : outs(!amdgcn.vgpr<2>) ins(i32)

      // v0 = v1 + v2 = 32 + 10 = 42
      amdgcn.v_add_u32 outs(%v0) ins(%v1, %v2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>, !amdgcn.vgpr<2>)

      // Self-check: trap if result != 42
      %vcc = amdgcn.alloca : !amdgcn.vcc
      %c42 = arith.constant 42 : i32
      amdgcn.v_cmp_ne_i32 outs(%vcc) ins(%c42, %v0) : outs(!amdgcn.vcc) ins(i32, !amdgcn.vgpr<0>)
      amdgcn.cbranch s_cbranch_vccnz %vcc ^trap fallthrough(^ok)
        : !amdgcn.vcc

    ^ok:
      end_kernel

    ^trap:
      s_trap 2
      end_kernel
    }
  }
}
