// Common register allocation functions for AMDGCN kernels.

!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!sx4 = !amdgcn.sgpr_range<[? + 4]>

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

!a   = !amdgcn.agpr
!ax2 = !amdgcn.agpr_range<[? + 2]>
!ax4 = !amdgcn.agpr_range<[? + 4]>

amdgcn.library @common_register_init isa = [#amdgcn.isa<cdna3>] {

  //===--------------------------------------------------------------------===//
  // xGPR allocation functions
  //===--------------------------------------------------------------------===//
  // Allocate a VGPRx2 range
  func.func private @alloc_vgprx2() -> !vx2 {
    %r0 = amdgcn.alloca : !v
    %r1 = amdgcn.alloca : !v
    %range = amdgcn.make_register_range %r0, %r1 : !v, !v
    return %range : !vx2
  }

  // Allocate a VGPRx4 range
  func.func private @alloc_vgprx4() -> !vx4 {
    %r0 = amdgcn.alloca : !v
    %r1 = amdgcn.alloca : !v
    %r2 = amdgcn.alloca : !v
    %r3 = amdgcn.alloca : !v
    %range = amdgcn.make_register_range %r0, %r1, %r2, %r3 : !v, !v, !v, !v
    return %range : !vx4
  }

  // Allocate a SGPRx2 range
  func.func private @alloc_sgprx2() -> !sx2 {
    %r0 = amdgcn.alloca : !s
    %r1 = amdgcn.alloca : !s
    %range = amdgcn.make_register_range %r0, %r1 : !s, !s
    return %range : !sx2
  }

  // Allocate a SGPRx4 range
  func.func private @alloc_sgprx4() -> !sx4 {
    %r0 = amdgcn.alloca : !s
    %r1 = amdgcn.alloca : !s
    %r2 = amdgcn.alloca : !s
    %r3 = amdgcn.alloca : !s
    %range = amdgcn.make_register_range %r0, %r1, %r2, %r3 : !s, !s, !s, !s
    return %range : !sx4
  }

  // Allocate an AGPRx2 range
  func.func private @alloc_agprx2() -> !ax2 {
    %r0 = amdgcn.alloca : !a
    %r1 = amdgcn.alloca : !a
    %range = amdgcn.make_register_range %r0, %r1 : !a, !a
    return %range : !ax2
  }

  // Allocate an AGPRx4 range
  func.func private @alloc_agprx4() -> !ax4 {
    %r0 = amdgcn.alloca : !a
    %r1 = amdgcn.alloca : !a
    %r2 = amdgcn.alloca : !a
    %r3 = amdgcn.alloca : !a
    %range = amdgcn.make_register_range %r0, %r1, %r2, %r3 : !a, !a, !a, !a
    return %range : !ax4
  }


  //===--------------------------------------------------------------------===//
  // VGPR initialization functions
  //===--------------------------------------------------------------------===//
  // Initialize a VGPRx2 range to %cst
  func.func private @init_vgprx2(%cst: i32) -> !vx2 {
    %r0 = amdgcn.alloca : !v
    %r1 = amdgcn.alloca : !v
    %v0 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r0, %cst : (!v, i32) -> !v
    %v1 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r1, %cst : (!v, i32) -> !v
    %range = amdgcn.make_register_range %v0, %v1 : !v, !v
    return %range : !vx2
  }

  // Initialize a VGPRx4 range to %cst
  func.func private @init_vgprx4(%cst: i32) -> !vx4 {
    %r0 = amdgcn.alloca : !v
    %r1 = amdgcn.alloca : !v
    %r2 = amdgcn.alloca : !v
    %r3 = amdgcn.alloca : !v
    %v0 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r0, %cst : (!v, i32) -> !v
    %v1 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r1, %cst : (!v, i32) -> !v
    %v2 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r2, %cst : (!v, i32) -> !v
    %v3 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r3, %cst : (!v, i32) -> !v
    %range = amdgcn.make_register_range %v0, %v1, %v2, %v3 : !v, !v, !v, !v
    return %range : !vx4
  }

  // Initialize a VGPRx4 range to %reg
  func.func private @init_vgprx4_reg(%reg: !v) -> !vx4 {
    %r0 = amdgcn.alloca : !v
    %r1 = amdgcn.alloca : !v
    %r2 = amdgcn.alloca : !v
    %r3 = amdgcn.alloca : !v
    %v0 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r0, %reg : (!v, !v) -> !v
    %v1 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r1, %reg : (!v, !v) -> !v
    %v2 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r2, %reg : (!v, !v) -> !v
    %v3 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r3, %reg : (!v, !v) -> !v
    %range = amdgcn.make_register_range %v0, %v1, %v2, %v3 : !v, !v, !v, !v
    return %range : !vx4
  }

  // TODO: SGPR initialization requires s_mov_b32 (SOP1) which is not yet implemented
  // TODO: AGPR initialization requires v_accvgpr_write_b32 which is not yet implemented
}
