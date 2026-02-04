// RUN: aster-opt %s --verify-roundtrip

!sgpr1 = !amdgcn.sgpr_range<[0 : 4]>
!sgpr2 = !amdgcn.sgpr_range<[0 : 5]>
!sgpr3 = !amdgcn.sgpr_range<[0 : 3]>
!sgpr4 = !amdgcn.sgpr_range<[0 : 4 align 8]>

!vcc = !amdgcn.vcc
!scc = !amdgcn.scc
!exec = !amdgcn.exec
!execz = !amdgcn.execz

func.func private @test(
  !amdgcn.vgpr<*>, !amdgcn.vgpr<?>, !amdgcn.vgpr<5>,
  !amdgcn.vgpr_range<[? + 4]>, !amdgcn.vgpr_range<[? : ? + 4]>, !amdgcn.vgpr_range<[0 : 4]>,
  !amdgcn.vgpr_range<[? + 4 align 8]>, !amdgcn.vgpr_range<[? : ? + 4 align 8]>, !amdgcn.vgpr_range<[0 : 4 align 8]>
)
