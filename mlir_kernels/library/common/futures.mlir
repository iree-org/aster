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
}
