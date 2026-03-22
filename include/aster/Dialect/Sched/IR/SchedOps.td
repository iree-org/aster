//===- SchedOps.td - Sched operations ----------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sched dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_SCHED_IR_SCHEDOPS_TD
#define ASTER_DIALECT_SCHED_IR_SCHEDOPS_TD

include "aster/Dialect/Sched/IR/SchedDialect.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def Sched_YieldOp : Sched_Op<"yield", [Terminator, ReturnLike]> {
  let summary = "Yields values from a sched region";
  let description = [{
    The `sched.yield` operation yields values from a `sched` region back to
    the parent operation. It is the terminator for all `sched` regions.

    Example:
    ```mlir
    sched.yield %0, %1 : i32, f32
    sched.yield
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}

//===----------------------------------------------------------------------===//
// UnitOp
//===----------------------------------------------------------------------===//

def Sched_UnitOp : Sched_Op<"unit", [
    IsolatedFromAbove,
    RecursiveMemoryEffects,
    AffineScope,
    SingleBlock,
    DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorRegions", "getSuccessorInputs"]>
  ]> {
  let summary = "A schedulable unit of work";
  let description = [{
    The `sched.unit` operation encapsulates a single block of computation as a
    schedulable unit. It is isolated from above, has recursive memory effects,
    and forms an affine scope.

    The `stage` attribute identifies which pipeline stage this unit belongs to.
    The optional `barrier` attribute indicates that this unit places a
    scheduling barrier between all preceding units and all subsequent units.

    The unit receives a variadic number of `inputs` (its operands) and produces
    a variadic number of results. Values yielded by the terminating `sched.yield`
    become the op's results.

    Example:
    ```mlir
    %0, %1 = sched.unit(%a, %b : i32, f32) -> (i32, f32) stage 0 {
      sched.yield %a, %b : i32, f32
    }
    %2 = sched.unit(%c : i32) -> i32 stage 1 barrier {
      sched.yield %c : i32
    }
    ```
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    IndexAttr:$stage,
    UnitAttr:$barrier
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    (`(` $inputs^ `:` type($inputs) `)`)?
    (`->` `(` type($results)^ `)`)?
    `stage` $stage (`barrier` $barrier^)?
    attr-dict-with-keyword $body
  }];
}

//===----------------------------------------------------------------------===//
// LoopResourceOp
//===----------------------------------------------------------------------===//

def Sched_LoopResourceOp : Sched_Op<"loop_resource", [IsolatedFromAbove]> {
  let summary = "A resource with an explicit loop lifecycle";
  let description = [{
    The `sched.loop_resource` operation models a resource with an explicit
    lifecycle tied to an enclosing loop. It must appear directly inside a
    loop-like operation.

    The op has four regions, all with a single block terminated by `sched.yield`:

    - `allocate`: Always present. Executed once before the loop. Must yield at
      least one value (the "resource handles"). No block arguments.

    - `deallocate`: Optional. Executed once after the loop has finished. Receives
      the values yielded by `allocate` as block arguments.

    - `forward`: Always present. Determines which values to forward as loop
      `iter_args` (i.e., the next iteration's view of the resource). Receives
      the values yielded by `allocate` as block arguments. The op's results are
      the values yielded by this region.

    - `fence`: Optional. Invoked at the end of each loop iteration to wait for
      the resource to become available before the next iteration (relevant when
      there is only one resource instance). Receives the values yielded by
      `allocate` as block arguments.

    Example:
    ```mlir
    %token = sched.loop_resource -> i32 {
    ^allocate:
      %buf = gpu.alloc() : i32
      sched.yield %buf : i32
    } deallocate {
    ^bb0(%buf : i32):
      gpu.dealloc %buf : i32
      sched.yield
    } forward {
    ^bb0(%buf : i32):
      sched.yield %buf : i32
    } fence {
    ^bb0(%buf : i32):
      gpu.wait %buf : i32
      sched.yield
    }
    ```
  }];
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region
    SizedRegion<1>:$allocate,
    MaxSizedRegion<1>:$deallocate,
    SizedRegion<1>:$forward,
    MaxSizedRegion<1>:$fence
  );
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

#endif // ASTER_DIALECT_SCHED_IR_SCHEDOPS_TD
