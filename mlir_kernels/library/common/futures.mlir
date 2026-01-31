// Helper functions for working with futures (async read/write tokens).
// These functions wait on async operations and extract values.

//===----------------------------------------------------------------------===//
// Type aliases (required for futures)
//===----------------------------------------------------------------------===//

// Vector General Purpose Registers (VGPR)
!vx2 = !amdgcn.vgpr_range<[? + 2]>

// Future types
!future_global_read_any = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>
!future_lds_write = !amdgcn.write_token<shared>
!future_lds_read_any = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

amdgcn.library @common_futures isa = [#amdgcn.isa<cdna3>] {

  //===--------------------------------------------------------------------===//
  // Wait on futures and return values
  //===--------------------------------------------------------------------===//

  func.func private @get_global_load_value_vx2(%future: !future_global_read_any) -> !vx2 {
    %value_any, %token = aster_utils.struct_extract %future ["value", "token"] : !future_global_read_any -> !aster_utils.any, !amdgcn.read_token<flat>
    amdgcn.wait deps %token : !amdgcn.read_token<flat>
    %value = aster_utils.from_any %value_any : !vx2
    return %value : !vx2
  }

  func.func private @get_global_load_value_vx2_1d(
      %futures: memref<?x!future_global_read_any>, %idx: index) -> !vx2 {
    %future = memref.load %futures[%idx] : memref<?x!future_global_read_any>
    %value_any, %token = aster_utils.struct_extract %future ["value", "token"]
      : !future_global_read_any -> !aster_utils.any, !amdgcn.read_token<flat>
    amdgcn.wait deps %token : !amdgcn.read_token<flat>
    %value = aster_utils.from_any %value_any : !vx2
    return %value : !vx2
  }

  func.func private @get_lds_read_value_vx2(%future: !future_lds_read_any) -> !vx2 {
    %value_any, %token = aster_utils.struct_extract %future ["value", "token"] : !future_lds_read_any -> !aster_utils.any, !amdgcn.read_token<shared>
    amdgcn.wait deps %token : !amdgcn.read_token<shared>
    %value = aster_utils.from_any %value_any : !vx2
    return %value : !vx2
  }

  func.func private @get_lds_read_value_vx2_1d(
      %futures: memref<?x!future_lds_read_any>, %idx: index) -> !vx2 {
    %future = memref.load %futures[%idx] : memref<?x!future_lds_read_any>
    %value_any, %token = aster_utils.struct_extract %future ["value", "token"]
      : !future_lds_read_any -> !aster_utils.any, !amdgcn.read_token<shared>
    amdgcn.wait deps %token : !amdgcn.read_token<shared>
    %value = aster_utils.from_any %value_any : !vx2
    return %value : !vx2
  }

  func.func private @wait_lds_write(%token: !future_lds_write) {
    amdgcn.wait deps %token : !amdgcn.write_token<shared>
    return
  }

  //===--------------------------------------------------------------------===//
  // Conditional wait helpers for async operations
  //===--------------------------------------------------------------------===//
  // These wait on all outstanding operations of a given type if the valid flag is true.
  // Used with conditional future variants where operations may or may not have executed.

  // Wait for all outstanding LDS writes (lgkmcnt=0) if valid
  func.func private @maybe_wait_all_lds_writes(%valid: i1, %tokens: memref<?x!future_lds_write>) {
    scf.if %valid {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %n = memref.dim %tokens, %c0 : memref<?x!future_lds_write>
      scf.for %idx = %c0 to %n step %c1 {
        %token = memref.load %tokens[%idx] : memref<?x!future_lds_write>
        amdgcn.wait deps %token : !amdgcn.write_token<shared>
      } {aster.constexpr}
    }
    return
  }

  //===--------------------------------------------------------------------===//
  // Conditional wait and extract helpers
  //===--------------------------------------------------------------------===//
  // These wait on futures and extract values if the valid flag is true.

  // Wait for global load futures and extract values into a 2D memref
  // %k: outer index for output memref
  // %n: number of futures to process
  // %output: memref<?x?x!vx2> - output[k, idx] = extracted value
  // %futures: memref<?x!future_global_read_any> - futures[idx]
  // %valid: i1 - only process if true
  func.func private @maybe_get_global_load_values_vx2(
    %k: index, %n: index,
    %output: memref<?x?x!vx2>,
    %futures: memref<?x!future_global_read_any>,
    %valid: i1
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    scf.if %valid {
      scf.for %idx = %c0 to %n step %c1 {
        %future = memref.load %futures[%idx] : memref<?x!future_global_read_any>
        %value = func.call @get_global_load_value_vx2(%future) : (!future_global_read_any) -> !vx2
        memref.store %value, %output[%k, %idx] : memref<?x?x!vx2>
      } {aster.constexpr}
    }
    return
  }

  // Wait for LDS read future and extract value into a 3D memref
  // %k, %ii, %jj: indices for output memref
  // %output: memref<?x?x?x!vx2> - output[k, ii, jj] = extracted value
  // %futures: memref<?x?x?x!future_lds_read_any> - futures[k, ii, jj]
  // %valid: i1 - only process if true
  func.func private @maybe_get_lds_read_value_vx2(
    %k: index, %ii: index, %jj: index,
    %output: memref<?x?x?x!vx2>,
    %futures: memref<?x?x?x!future_lds_read_any>,
    %valid: i1
  ) {
    scf.if %valid {
      // Wait on LDS reads via s_waitcnt
      amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

      %future = memref.load %futures[%k, %ii, %jj] : memref<?x?x?x!future_lds_read_any>
      %value_any, %token = aster_utils.struct_extract %future ["value", "token"] : !future_lds_read_any -> !aster_utils.any, !amdgcn.read_token<shared>
      %value = aster_utils.from_any %value_any : !vx2
      memref.store %value, %output[%k, %ii, %jj] : memref<?x?x?x!vx2>
    }
    return
  }
}
