// RUN: aster-opt %s --verify-diagnostics --split-input-file

// expected-error@+1 {{VGPR range size must be positive}}
!invalid_2 = !amdgcn.vgpr<[1 : 1]>

// -----

// expected-error@+1 {{begin VGPR is invalid}}
!invalid_3 = !amdgcn.vgpr<[-1 : 1]>

// -----

// expected-error@+1 {{align must be positive, got 0}}
!invalid_4 = !amdgcn.vgpr<[1 : 2 align 0]>

// -----

// expected-error@+1 {{align must be a power of 2, got 3}}
!invalid_5 = !amdgcn.vgpr<[0 : 2 align 3]>

// -----

// size == 5 -> align to next power of 2 == 8 by default
// expected-error@+1 {{index begin (3) must be aligned to align (8)}}
!invalid_6 = !amdgcn.sgpr<[3 : 8]>
