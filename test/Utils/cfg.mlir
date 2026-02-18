// RUN: aster-opt %s --test-cfg-walker | FileCheck %s

// CHECK-LABEL:  function: "cfg"
// CHECK:  cfg: entry -> Block<op = func.func @cfg() {...}, region = 0, bb = ^bb0, args = []>
// CHECK:      %{{.*}} = arith.constant 0 : i32
// CHECK:      %{{.*}} = arith.constant 1 : i32
// CHECK:      %{{.*}} = arith.constant 10 : i32
// CHECK:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}  : i32 {...}
// CHECK:  cfg: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}  : i32 {...} -> Block<op = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}  : i32 {...}, region = 0, bb = ^bb0, args = [%{{.*}}]>
// CHECK:      %{{.*}} = func.call @rand() : () -> i1
// CHECK:      scf.if %{{.*}} {...} else {...}
// CHECK:  cfg: scf.if %{{.*}} {...} else {...} -> Block<op = scf.if %{{.*}} {...} else {...}, region = 0, bb = ^bb0, args = []>
// CHECK:      %{{.*}} = func.call @foo(%{{.*}}) : (i32) -> i32
// CHECK:      scf.yield
// CHECK:  cfg: scf.yield -> scf.if %{{.*}} {...} else {...}
// CHECK:  cfg: scf.if %{{.*}} {...} else {...} -> Block<op = scf.if %{{.*}} {...} else {...}, region = 1, bb = ^bb0, args = []>
// CHECK:      %{{.*}} = func.call @foo(%{{.*}}) : (i32) -> i32
// CHECK:      scf.yield
// CHECK:  cfg: scf.yield -> scf.if %{{.*}} {...} else {...}
// CHECK:      scf.yield
// CHECK:  cfg: scf.yield -> Block<op = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}  : i32 {...}, region = 0, bb = ^bb0, args = [%{{.*}}]>
// CHECK:  cfg: scf.yield -> scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}  : i32 {...}
// CHECK:  cfg: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}  : i32 {...} -> scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}  : i32 {...}
// CHECK:      cf.br ^bb1
// CHECK:  cfg: cf.br ^bb1 -> Block<op = func.func @cfg() {...}, region = 0, bb = ^bb1, args = []>
// CHECK:      %{{.*}} = func.call @rand() : () -> i1
// CHECK:      %{{.*}} = func.call @foo(%{{.*}}) : (i32) -> i32
// CHECK:      cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:  cfg: cf.cond_br %{{.*}}, ^bb1, ^bb2 -> Block<op = func.func @cfg() {...}, region = 0, bb = ^bb1, args = []>
// CHECK:  cfg: cf.cond_br %{{.*}}, ^bb1, ^bb2 -> Block<op = func.func @cfg() {...}, region = 0, bb = ^bb2, args = []>
// CHECK:      %{{.*}} = func.call @rand() : () -> i1
// CHECK:      %{{.*}} = func.call @foo(%{{.*}}) : (i32) -> i32
// CHECK:      cf.cond_br %{{.*}}, ^bb3, ^bb1
// CHECK:  cfg: cf.cond_br %{{.*}}, ^bb3, ^bb1 -> Block<op = func.func @cfg() {...}, region = 0, bb = ^bb3, args = []>
// CHECK:      %{{.*}} = func.call @foo(%{{.*}}) : (i32) -> i32
// CHECK:      func.return
// CHECK:  cfg: cf.cond_br %{{.*}}, ^bb3, ^bb1 -> Block<op = func.func @cfg() {...}, region = 0, bb = ^bb1, args = []>
func.func private @rand() -> i1
func.func private @foo(i32) -> i32
func.func @cfg() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  scf.for %arg0 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
    %5 = func.call @rand() : () -> i1
    scf.if %5 {
      %6 = func.call @foo(%c0_i32) : (i32) -> i32
    } else {
      %6 = func.call @foo(%c1_i32) : (i32) -> i32
    }
  }
  cf.br ^bb1
^bb1:  // 3 preds: ^bb0, ^bb1, ^bb2
  %0 = call @rand() : () -> i1
  %1 = call @foo(%c0_i32) : (i32) -> i32
  cf.cond_br %0, ^bb1, ^bb2
^bb2:  // pred: ^bb1
  %2 = call @rand() : () -> i1
  %3 = call @foo(%c1_i32) : (i32) -> i32
  cf.cond_br %2, ^bb3, ^bb1
^bb3:  // pred: ^bb2
  %4 = call @foo(%c10_i32) : (i32) -> i32
  return
}
