// RUN: aster-opt %s --split-input-file --verify-diagnostics

// amdgcn.wait is incompatible with gfx1250.
amdgcn.module @gfx1250_old_wait target = #amdgcn.target<gfx1250> {
  func.func @f(%arg0: !amdgcn.vgpr<[? + 2]>) {
    %0 = amdgcn.alloca : !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %r, %tok = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    // expected-error @below {{'amdgcn.wait' op is not compatible with module target gfx1250}}
    amdgcn.wait deps %tok : !amdgcn.read_token<flat>
    return
  }
}

// -----

// amdgcn.wait_gfx1250 is incompatible with gfx942.
amdgcn.module @cdna3_new_wait target = #amdgcn.target<gfx942> {
  func.func @f(%arg0: !amdgcn.vgpr<[? + 2]>) {
    %0 = amdgcn.alloca : !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %r, %tok = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    // expected-error @below {{'amdgcn.wait_gfx1250' op is not compatible with module target gfx942}}
    amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<flat>
    return
  }
}

// -----

// amdgcn.wait_gfx1250 is incompatible with gfx950.
amdgcn.module @cdna4_new_wait target = #amdgcn.target<gfx950> {
  func.func @f(%arg0: !amdgcn.vgpr<[? + 2]>) {
    %0 = amdgcn.alloca : !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %r, %tok = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    // expected-error @below {{'amdgcn.wait_gfx1250' op is not compatible with module target gfx950}}
    amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<flat>
    return
  }
}

// -----

amdgcn.module @cdna3_cluster_id target = #amdgcn.target<gfx942> {
  func.func @f() {
    // expected-error @below {{'amdgcn.cluster_id' op is not compatible with module target gfx942}}
    %0 = amdgcn.cluster_id x : !amdgcn.sgpr
    return
  }
}

// -----

amdgcn.module @cdna4_cluster_id target = #amdgcn.target<gfx950> {
  func.func @f() {
    // expected-error @below {{'amdgcn.cluster_id' op is not compatible with module target gfx950}}
    %0 = amdgcn.cluster_id x : !amdgcn.sgpr
    return
  }
}
