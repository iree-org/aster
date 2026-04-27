// RUN: aster-opt %s --amdgcn-preload-library --split-input-file | FileCheck %s

// CHECK-LABEL: amdgcn.module @mod
// CHECK:         func.func @compute(%arg0: index) -> index {
// CHECK:           arith.muli
// CHECK-NOT:       func.func private @compute
module {
  amdgcn.library @mylib {
    func.func @compute(%x: index) -> index {
      %c2 = arith.constant 2 : index
      %r = arith.muli %x, %c2 : index
      return %r : index
    }
  }

  amdgcn.module @mod target = #amdgcn.target<gfx942> {
    func.func private @compute(index) -> index

    func.func @use_compute(%x: index) -> index {
      %r = func.call @compute(%x) : (index) -> index
      return %r : index
    }
  }
}

// -----

// CHECK-LABEL: amdgcn.module @mod_with_own_def_takes_priority
// CHECK:         func.func @helper(%arg0: index) -> index {
// CHECK:           arith.constant 99
// CHECK-NOT:       arith.constant 10
module {
  amdgcn.library @lib2 {
    func.func @helper(%x: index) -> index {
      %c10 = arith.constant 10 : index
      %r = arith.addi %x, %c10 : index
      return %r : index
    }
  }

  amdgcn.module @mod_with_own_def_takes_priority target = #amdgcn.target<gfx942> {
    func.func @helper(%x: index) -> index {
      %c99 = arith.constant 99 : index
      return %c99 : index
    }
  }
}
