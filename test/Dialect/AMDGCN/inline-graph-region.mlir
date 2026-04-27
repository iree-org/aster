// RUN: aster-opt %s --inline --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @caller
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         arith.addi %{{.*}}, %[[C1]] : index
// CHECK-NOT:     call @helper
module {
  amdgcn.module @mod target = #amdgcn.target<gfx942> {
    func.func @helper(%x: index) -> index {
      %c1 = arith.constant 1 : index
      %r = arith.addi %x, %c1 : index
      return %r : index
    }

    func.func @caller(%x: index) -> index {
      %r = func.call @helper(%x) : (index) -> index
      return %r : index
    }
  }
}

// -----


// CHECK-LABEL: kernel @kernel_main
// CHECK-NOT:     call @kernel_helper
module {
  amdgcn.module @mod2 target = #amdgcn.target<gfx942> {
    func.func @kernel_helper(%x: index) -> index {
      %c2 = arith.constant 2 : index
      %r = arith.addi %x, %c2 : index
      return %r : index
    }

    amdgcn.kernel @kernel_main {
      %c0 = arith.constant 0 : index
      %result = func.call @kernel_helper(%c0) : (index) -> index
      amdgcn.end_kernel
    }
  }
}
