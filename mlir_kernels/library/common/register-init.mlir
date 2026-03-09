// Common register allocation functions for AMDGCN kernels.

!s   = !amdgcn.sgpr
!sx1 = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>
!sx3 = !amdgcn.sgpr<[? + 3]>
!sx4 = !amdgcn.sgpr<[? + 4]>

!v   = !amdgcn.vgpr
!vx1 = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx3 = !amdgcn.vgpr<[? + 3]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!vx8 = !amdgcn.vgpr<[? + 8]>
!vx16 = !amdgcn.vgpr<[? + 16]>

!a   = !amdgcn.agpr
!ax1 = !amdgcn.agpr
!ax2 = !amdgcn.agpr<[? + 2]>
!ax3 = !amdgcn.agpr<[? + 3]>
!ax4 = !amdgcn.agpr<[? + 4]>
!ax16 = !amdgcn.agpr<[? + 16]>

amdgcn.library @common_register_init isa = [#amdgcn.isa<cdna3>] {

  //===--------------------------------------------------------------------===//
  // xGPR allocation functions
  //===--------------------------------------------------------------------===//
  // Allocate a single VGPR
  func.func private @alloc_vgpr() -> !v {
    %r = amdgcn.alloca : !v
    return %r : !v
  }

  // Allocate a VGPRx1 range
  func.func private @alloc_vgprx1() -> !vx1 {
    %r0 = amdgcn.alloca : !v
    %range = amdgcn.make_register_range %r0 : !v
    return %range : !vx1
  }

  // Allocate a VGPRx2 range
  func.func private @alloc_vgprx2() -> !vx2 {
    %r0 = amdgcn.alloca : !v
    %r1 = amdgcn.alloca : !v
    %range = amdgcn.make_register_range %r0, %r1 : !v, !v
    return %range : !vx2
  }

  // Allocate a VGPRx3 range
  func.func private @alloc_vgprx3() -> !vx3 {
    %r0 = amdgcn.alloca : !v
    %r1 = amdgcn.alloca : !v
    %r2 = amdgcn.alloca : !v
    %range = amdgcn.make_register_range %r0, %r1, %r2 : !v, !v, !v
    return %range : !vx3
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

  // Allocate a VGPRx8 range
  func.func private @alloc_vgprx8() -> !vx8 {
    %r0 = amdgcn.alloca : !v
    %r1 = amdgcn.alloca : !v
    %r2 = amdgcn.alloca : !v
    %r3 = amdgcn.alloca : !v
    %r4 = amdgcn.alloca : !v
    %r5 = amdgcn.alloca : !v
    %r6 = amdgcn.alloca : !v
    %r7 = amdgcn.alloca : !v
    %range = amdgcn.make_register_range %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7
      : !v, !v, !v, !v, !v, !v, !v, !v
    return %range : !vx8
  }

  // Allocate a VGPRx16 range
  func.func private @alloc_vgprx16() -> !vx16 {
    %r0  = amdgcn.alloca : !v
    %r1  = amdgcn.alloca : !v
    %r2  = amdgcn.alloca : !v
    %r3  = amdgcn.alloca : !v
    %r4  = amdgcn.alloca : !v
    %r5  = amdgcn.alloca : !v
    %r6  = amdgcn.alloca : !v
    %r7  = amdgcn.alloca : !v
    %r8  = amdgcn.alloca : !v
    %r9  = amdgcn.alloca : !v
    %r10 = amdgcn.alloca : !v
    %r11 = amdgcn.alloca : !v
    %r12 = amdgcn.alloca : !v
    %r13 = amdgcn.alloca : !v
    %r14 = amdgcn.alloca : !v
    %r15 = amdgcn.alloca : !v
    %range = amdgcn.make_register_range
      %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7,
      %r8, %r9, %r10, %r11, %r12, %r13, %r14, %r15
      : !v, !v, !v, !v, !v, !v, !v, !v,
        !v, !v, !v, !v, !v, !v, !v, !v
    return %range : !vx16
  }

  // Allocate a SGPRx1 range
  func.func private @alloc_sgprx1() -> !sx1 {
    %r0 = amdgcn.alloca : !s
    %range = amdgcn.make_register_range %r0 : !s
    return %range : !sx1
  }

  // Allocate a SGPRx2 range
  func.func private @alloc_sgprx2() -> !sx2 {
    %r0 = amdgcn.alloca : !s
    %r1 = amdgcn.alloca : !s
    %range = amdgcn.make_register_range %r0, %r1 : !s, !s
    return %range : !sx2
  }

  // Allocate a SGPRx3 range
  func.func private @alloc_sgprx3() -> !sx3 {
    %r0 = amdgcn.alloca : !s
    %r1 = amdgcn.alloca : !s
    %r2 = amdgcn.alloca : !s
    %range = amdgcn.make_register_range %r0, %r1, %r2 : !s, !s, !s
    return %range : !sx3
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

  // Allocate an AGPRx1 range
  func.func private @alloc_agprx1() -> !ax1 {
    %r0 = amdgcn.alloca : !a
    %range = amdgcn.make_register_range %r0 : !a
    return %range : !ax1
  }

  // Allocate an AGPRx2 range
  func.func private @alloc_agprx2() -> !ax2 {
    %r0 = amdgcn.alloca : !a
    %r1 = amdgcn.alloca : !a
    %range = amdgcn.make_register_range %r0, %r1 : !a, !a
    return %range : !ax2
  }

  // Allocate an AGPRx3 range
  func.func private @alloc_agprx3() -> !ax3 {
    %r0 = amdgcn.alloca : !a
    %r1 = amdgcn.alloca : !a
    %r2 = amdgcn.alloca : !a
    %range = amdgcn.make_register_range %r0, %r1, %r2 : !a, !a, !a
    return %range : !ax3
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

  // Initialize a VGPRx3 range to %cst
  func.func private @init_vgprx3(%cst: i32) -> !vx3 {
    %r0 = amdgcn.alloca : !v
    %r1 = amdgcn.alloca : !v
    %r2 = amdgcn.alloca : !v
    %v0 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r0, %cst : (!v, i32) -> !v
    %v1 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r1, %cst : (!v, i32) -> !v
    %v2 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r2, %cst : (!v, i32) -> !v
    %range = amdgcn.make_register_range %v0, %v1, %v2 : !v, !v, !v
    return %range : !vx3
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

  // Initialize a VGPRx8 range to %cst
  func.func private @init_vgprx8(%cst: i32) -> !vx8 {
    %r0 = amdgcn.alloca : !v
    %r1 = amdgcn.alloca : !v
    %r2 = amdgcn.alloca : !v
    %r3 = amdgcn.alloca : !v
    %r4 = amdgcn.alloca : !v
    %r5 = amdgcn.alloca : !v
    %r6 = amdgcn.alloca : !v
    %r7 = amdgcn.alloca : !v
    %v0 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r0, %cst : (!v, i32) -> !v
    %v1 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r1, %cst : (!v, i32) -> !v
    %v2 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r2, %cst : (!v, i32) -> !v
    %v3 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r3, %cst : (!v, i32) -> !v
    %v4 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r4, %cst : (!v, i32) -> !v
    %v5 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r5, %cst : (!v, i32) -> !v
    %v6 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r6, %cst : (!v, i32) -> !v
    %v7 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r7, %cst : (!v, i32) -> !v
    %range = amdgcn.make_register_range %v0, %v1, %v2, %v3, %v4, %v5, %v6, %v7
      : !v, !v, !v, !v, !v, !v, !v, !v
    return %range : !vx8
  }

  // Initialize a VGPRx16 range to %cst
  func.func private @init_vgprx16(%cst: i32) -> !vx16 {
    %r0  = amdgcn.alloca : !v
    %r1  = amdgcn.alloca : !v
    %r2  = amdgcn.alloca : !v
    %r3  = amdgcn.alloca : !v
    %r4  = amdgcn.alloca : !v
    %r5  = amdgcn.alloca : !v
    %r6  = amdgcn.alloca : !v
    %r7  = amdgcn.alloca : !v
    %r8  = amdgcn.alloca : !v
    %r9  = amdgcn.alloca : !v
    %r10 = amdgcn.alloca : !v
    %r11 = amdgcn.alloca : !v
    %r12 = amdgcn.alloca : !v
    %r13 = amdgcn.alloca : !v
    %r14 = amdgcn.alloca : !v
    %r15 = amdgcn.alloca : !v
    %v0  = amdgcn.vop1.vop1 <v_mov_b32_e32> %r0,  %cst : (!v, i32) -> !v
    %v1  = amdgcn.vop1.vop1 <v_mov_b32_e32> %r1,  %cst : (!v, i32) -> !v
    %v2  = amdgcn.vop1.vop1 <v_mov_b32_e32> %r2,  %cst : (!v, i32) -> !v
    %v3  = amdgcn.vop1.vop1 <v_mov_b32_e32> %r3,  %cst : (!v, i32) -> !v
    %v4  = amdgcn.vop1.vop1 <v_mov_b32_e32> %r4,  %cst : (!v, i32) -> !v
    %v5  = amdgcn.vop1.vop1 <v_mov_b32_e32> %r5,  %cst : (!v, i32) -> !v
    %v6  = amdgcn.vop1.vop1 <v_mov_b32_e32> %r6,  %cst : (!v, i32) -> !v
    %v7  = amdgcn.vop1.vop1 <v_mov_b32_e32> %r7,  %cst : (!v, i32) -> !v
    %v8  = amdgcn.vop1.vop1 <v_mov_b32_e32> %r8,  %cst : (!v, i32) -> !v
    %v9  = amdgcn.vop1.vop1 <v_mov_b32_e32> %r9,  %cst : (!v, i32) -> !v
    %v10 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r10, %cst : (!v, i32) -> !v
    %v11 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r11, %cst : (!v, i32) -> !v
    %v12 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r12, %cst : (!v, i32) -> !v
    %v13 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r13, %cst : (!v, i32) -> !v
    %v14 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r14, %cst : (!v, i32) -> !v
    %v15 = amdgcn.vop1.vop1 <v_mov_b32_e32> %r15, %cst : (!v, i32) -> !v
    %range = amdgcn.make_register_range
      %v0, %v1, %v2, %v3, %v4, %v5, %v6, %v7,
      %v8, %v9, %v10, %v11, %v12, %v13, %v14, %v15
      : !v, !v, !v, !v, !v, !v, !v, !v,
        !v, !v, !v, !v, !v, !v, !v, !v
    return %range : !vx16
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

  //===--------------------------------------------------------------------===//
  // AGPR initialization functions
  //===--------------------------------------------------------------------===//
  // Initialize an AGPRx1 range to %cst
  func.func private @init_agprx1(%cst: i32) -> !ax1 {
    %r0 = amdgcn.alloca : !a
    %a0 = amdgcn.vop3p v_accvgpr_write_b32 outs %r0 ins %cst : !amdgcn.agpr, i32
    %range = amdgcn.make_register_range %a0 : !a
    return %range : !ax1
  }

  // Initialize an AGPRx2 range to %cst
  func.func private @init_agprx2(%cst: i32) -> !ax2 {
    %r0 = amdgcn.alloca : !a
    %r1 = amdgcn.alloca : !a
    %a0 = amdgcn.vop3p v_accvgpr_write_b32 outs %r0 ins %cst : !amdgcn.agpr, i32
    %a1 = amdgcn.vop3p v_accvgpr_write_b32 outs %r1 ins %cst : !amdgcn.agpr, i32
    %range = amdgcn.make_register_range %a0, %a1 : !a, !a
    return %range : !ax2
  }

  // Initialize an AGPRx3 range to %cst
  func.func private @init_agprx3(%cst: i32) -> !ax3 {
    %r0 = amdgcn.alloca : !a
    %r1 = amdgcn.alloca : !a
    %r2 = amdgcn.alloca : !a
    %a0 = amdgcn.vop3p v_accvgpr_write_b32 outs %r0 ins %cst : !amdgcn.agpr, i32
    %a1 = amdgcn.vop3p v_accvgpr_write_b32 outs %r1 ins %cst : !amdgcn.agpr, i32
    %a2 = amdgcn.vop3p v_accvgpr_write_b32 outs %r2 ins %cst : !amdgcn.agpr, i32
    %range = amdgcn.make_register_range %a0, %a1, %a2 : !a, !a, !a
    return %range : !ax3
  }

  // Initialize an AGPRx4 range to %cst
  func.func private @init_agprx4(%cst: i32) -> !ax4 {
    %r0 = amdgcn.alloca : !a
    %r1 = amdgcn.alloca : !a
    %r2 = amdgcn.alloca : !a
    %r3 = amdgcn.alloca : !a
    %a0 = amdgcn.vop3p v_accvgpr_write_b32 outs %r0 ins %cst : !amdgcn.agpr, i32
    %a1 = amdgcn.vop3p v_accvgpr_write_b32 outs %r1 ins %cst : !amdgcn.agpr, i32
    %a2 = amdgcn.vop3p v_accvgpr_write_b32 outs %r2 ins %cst : !amdgcn.agpr, i32
    %a3 = amdgcn.vop3p v_accvgpr_write_b32 outs %r3 ins %cst : !amdgcn.agpr, i32
    %range = amdgcn.make_register_range %a0, %a1, %a2, %a3 : !a, !a, !a, !a
    return %range : !ax4
  }

  // Initialize an AGPRx16 range to %cst (for 32x32 MFMA accumulators)
  func.func private @init_agprx16(%cst: i32) -> !ax16 {
    %r0  = amdgcn.alloca : !a
    %r1  = amdgcn.alloca : !a
    %r2  = amdgcn.alloca : !a
    %r3  = amdgcn.alloca : !a
    %r4  = amdgcn.alloca : !a
    %r5  = amdgcn.alloca : !a
    %r6  = amdgcn.alloca : !a
    %r7  = amdgcn.alloca : !a
    %r8  = amdgcn.alloca : !a
    %r9  = amdgcn.alloca : !a
    %r10 = amdgcn.alloca : !a
    %r11 = amdgcn.alloca : !a
    %r12 = amdgcn.alloca : !a
    %r13 = amdgcn.alloca : !a
    %r14 = amdgcn.alloca : !a
    %r15 = amdgcn.alloca : !a
    %a0  = amdgcn.vop3p v_accvgpr_write_b32 outs %r0  ins %cst : !amdgcn.agpr, i32
    %a1  = amdgcn.vop3p v_accvgpr_write_b32 outs %r1  ins %cst : !amdgcn.agpr, i32
    %a2  = amdgcn.vop3p v_accvgpr_write_b32 outs %r2  ins %cst : !amdgcn.agpr, i32
    %a3  = amdgcn.vop3p v_accvgpr_write_b32 outs %r3  ins %cst : !amdgcn.agpr, i32
    %a4  = amdgcn.vop3p v_accvgpr_write_b32 outs %r4  ins %cst : !amdgcn.agpr, i32
    %a5  = amdgcn.vop3p v_accvgpr_write_b32 outs %r5  ins %cst : !amdgcn.agpr, i32
    %a6  = amdgcn.vop3p v_accvgpr_write_b32 outs %r6  ins %cst : !amdgcn.agpr, i32
    %a7  = amdgcn.vop3p v_accvgpr_write_b32 outs %r7  ins %cst : !amdgcn.agpr, i32
    %a8  = amdgcn.vop3p v_accvgpr_write_b32 outs %r8  ins %cst : !amdgcn.agpr, i32
    %a9  = amdgcn.vop3p v_accvgpr_write_b32 outs %r9  ins %cst : !amdgcn.agpr, i32
    %a10 = amdgcn.vop3p v_accvgpr_write_b32 outs %r10 ins %cst : !amdgcn.agpr, i32
    %a11 = amdgcn.vop3p v_accvgpr_write_b32 outs %r11 ins %cst : !amdgcn.agpr, i32
    %a12 = amdgcn.vop3p v_accvgpr_write_b32 outs %r12 ins %cst : !amdgcn.agpr, i32
    %a13 = amdgcn.vop3p v_accvgpr_write_b32 outs %r13 ins %cst : !amdgcn.agpr, i32
    %a14 = amdgcn.vop3p v_accvgpr_write_b32 outs %r14 ins %cst : !amdgcn.agpr, i32
    %a15 = amdgcn.vop3p v_accvgpr_write_b32 outs %r15 ins %cst : !amdgcn.agpr, i32
    %range = amdgcn.make_register_range
      %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7,
      %a8, %a9, %a10, %a11, %a12, %a13, %a14, %a15
      : !a, !a, !a, !a, !a, !a, !a, !a,
        !a, !a, !a, !a, !a, !a, !a, !a
    return %range : !ax16
  }

  // TODO: SGPR initialization requires s_mov_b32 (SOP1) which is not yet implemented
}
