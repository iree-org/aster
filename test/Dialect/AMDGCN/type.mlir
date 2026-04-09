// RUN: aster-opt %s --verify-roundtrip

!sgpr1 = !amdgcn.sgpr<[0 : 4]>
!sgpr2 = !amdgcn.sgpr<[0 : 5]>
!sgpr3 = !amdgcn.sgpr<[0 : 3]>
!sgpr4 = !amdgcn.sgpr<[0 : 4 align 8]>

!vcc = !amdgcn.vcc<0>
!scc = !amdgcn.scc<0>
!exec = !amdgcn.exec<0>
!execz = !amdgcn.execz<0>

func.func private @test(
  !amdgcn.vgpr<*>, !amdgcn.vgpr<?>, !amdgcn.vgpr<5>,
  !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? : ? + 4]>, !amdgcn.vgpr<[0 : 4]>,
  !amdgcn.vgpr<[? + 4 align 8]>, !amdgcn.vgpr<[? : ? + 4 align 8]>, !amdgcn.vgpr<[0 : 4 align 8]>
)

// lds_buffer as memref element type (enabled by MemRefElementTypeInterface)
func.func private @test_lds_buffer_memref(memref<2 x !amdgcn.lds_buffer>)
