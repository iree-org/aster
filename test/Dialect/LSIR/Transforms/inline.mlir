// RUN: aster-opt %s --inline | FileCheck %s

//===----------------------------------------------------------------------===//
// Test inline pass
//===----------------------------------------------------------------------===//

func.func @helper(%x: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.alloca : !amdgcn.vgpr
  %1 = lsir.addi i32 %0, %x, %x : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %1 : !amdgcn.vgpr
}

// CHECK-LABEL: func.func @main
//       CHECK:   lsir.alloca
//       CHECK:   lsir.addi
//   CHECK-NOT:   call
func.func @main(%arg: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = call @helper(%arg) : (!amdgcn.vgpr) -> !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}
