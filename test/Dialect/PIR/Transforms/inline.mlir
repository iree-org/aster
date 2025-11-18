// RUN: aster-opt %s --inline | FileCheck %s

//===----------------------------------------------------------------------===//
// Test inline pass
//===----------------------------------------------------------------------===//

func.func @helper(%x: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.alloca : !amdgcn.vgpr
  %1 = pir.addi i32 %0, %x, %x : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %1 : !amdgcn.vgpr
}

// CHECK-LABEL: func.func @main
//       CHECK:   pir.alloca
//       CHECK:   pir.addi
//   CHECK-NOT:   call
func.func @main(%arg: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = call @helper(%arg) : (!amdgcn.vgpr) -> !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}
