// RUN: aster-opt %s

// Only used in preload-library.mlir test, nothing to see here beyond the IR is
// well-formed.
amdgcn.library @indexing {
  // Compute wave index from thread index (assuming 64 threads per wave).
  func.func @wave_index(%tidx: index) -> index {
    %c64 = arith.constant 64 : index
    %widx = arith.divui %tidx, %c64 : index
    return %widx : index
  }

  // Compute lane index within a wave (0-63).
  func.func @lane_index(%tidx: index) -> index {
    %c64 = arith.constant 64 : index
    %lidx = arith.remui %tidx, %c64 : index
    return %lidx : index
  }

  // Compute number of waves from block dimension.
  func.func @num_waves(%bdimx: index) -> index {
    %c64 = arith.constant 64 : index
    %num_waves = arith.divui %bdimx, %c64 : index
    return %num_waves : index
  }
}
