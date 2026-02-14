// RUN: aster-opt %s --verify-roundtrip

!sgpr1 = !amdgcn.sgpr<[0 : 4]>
!sgpr2 = !amdgcn.sgpr<[0 : 5]>
!sgpr3 = !amdgcn.sgpr<[0 : 3]>
!sgpr4 = !amdgcn.sgpr<[0 : 4 align 8]>

!vcc = !amdgcn.vcc
!scc = !amdgcn.scc
!exec = !amdgcn.exec
!execz = !amdgcn.execz

func.func private @test(
  !amdgcn.vgpr<*>, !amdgcn.vgpr<?>, !amdgcn.vgpr<5>,
  !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? : ? + 4]>, !amdgcn.vgpr<[0 : 4]>,
  !amdgcn.vgpr<[? + 4 align 8]>, !amdgcn.vgpr<[? : ? + 4 align 8]>, !amdgcn.vgpr<[0 : 4 align 8]>
)
