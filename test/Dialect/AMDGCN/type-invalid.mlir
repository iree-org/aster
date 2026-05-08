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

// -----

// EXEC rejects value semantics.
// expected-error@+3 {{exec does not accept value semantics}}
func.func @exec_value_semantics(
  %arg0: !amdgcn.exec
) { return }

// -----

// EXEC rejects unallocated semantics.
func.func @exec_unalloc_semantics(
  // expected-error@+1 {{exec does not accept unallocated semantics}}
  %arg0: !amdgcn.exec<?>
) { return }

// -----

// EXEC_LO rejects value semantics.
// expected-error@+3 {{exec_lo does not accept value semantics}}
func.func @exec_lo_value_semantics(
  %arg0: !amdgcn.exec_lo
) { return }

// -----

// EXEC_LO rejects unallocated semantics.
func.func @exec_lo_unalloc_semantics(
  // expected-error@+1 {{exec_lo does not accept unallocated semantics}}
  %arg0: !amdgcn.exec_lo<?>
) { return }

// -----

// EXEC_HI rejects value semantics.
// expected-error@+3 {{exec_hi does not accept value semantics}}
func.func @exec_hi_value_semantics(
  %arg0: !amdgcn.exec_hi
) { return }

// -----

// EXEC_HI rejects unallocated semantics.
func.func @exec_hi_unalloc_semantics(
  // expected-error@+1 {{exec_hi does not accept unallocated semantics}}
  %arg0: !amdgcn.exec_hi<?>
) { return }

// -----

// VCCZ rejects unallocated semantics.
func.func @vccz_unalloc_semantics(
  // expected-error@+1 {{vccz does not accept unallocated semantics}}
  %arg0: !amdgcn.vccz<?>
) { return }

// -----

// EXECZ rejects unallocated semantics.
func.func @execz_unalloc_semantics(
  // expected-error@+1 {{execz does not accept unallocated semantics}}
  %arg0: !amdgcn.execz<?>
) { return }

// -----

// All SREGs reject unallocated semantics (VCC).
func.func @vcc_unalloc_semantics(
  // expected-error@+1 {{vcc does not accept unallocated semantics}}
  %arg0: !amdgcn.vcc<?>
) { return }

// -----

// All SREGs reject unallocated semantics (VCC_LO).
func.func @vcc_lo_unalloc_semantics(
  // expected-error@+1 {{vcc_lo does not accept unallocated semantics}}
  %arg0: !amdgcn.vcc_lo<?>
) { return }

// -----

// All SREGs reject unallocated semantics (VCC_HI).
func.func @vcc_hi_unalloc_semantics(
  // expected-error@+1 {{vcc_hi does not accept unallocated semantics}}
  %arg0: !amdgcn.vcc_hi<?>
) { return }

// -----

// All SREGs reject unallocated semantics (M0).
func.func @m0_unalloc_semantics(
  // expected-error@+1 {{m0 does not accept unallocated semantics}}
  %arg0: !amdgcn.m0<?>
) { return }

// -----

// All SREGs reject unallocated semantics (SCC).
func.func @scc_unalloc_semantics(
  // expected-error@+1 {{scc does not accept unallocated semantics}}
  %arg0: !amdgcn.scc<?>
) { return }

// -----

// IsReadOnly register cannot be allocated with value semantics.
func.func @alloca_readonly_vccz() {
  // expected-error@+1 {{'amdgcn.alloca' op cannot allocate read-only register with value semantics}}
  %0 = amdgcn.alloca : !amdgcn.vccz
  return
}

// -----

// IsReadOnly register cannot be allocated with value semantics.
func.func @alloca_readonly_execz() {
  // expected-error@+1 {{'amdgcn.alloca' op cannot allocate read-only register with value semantics}}
  %0 = amdgcn.alloca : !amdgcn.execz
  return
}

// -----

// IsReadOnly register with allocated semantics is fine.
func.func @alloca_readonly_allocated_ok() {
  %0 = amdgcn.alloca : !amdgcn.vccz<0>
  %1 = amdgcn.alloca : !amdgcn.execz<0>
  return
}
