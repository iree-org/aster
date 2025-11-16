// RUN: aster-opt %s --verify-roundtrip

!sgpr1 = !amdgcn.sgpr_range<[0 : 4]>
!sgpr2 = !amdgcn.sgpr_range<[0 : 5]>
!sgpr3 = !amdgcn.sgpr_range<[0 : 3]>
!sgpr4 = !amdgcn.sgpr_range<[0 : 4 align 8]>
