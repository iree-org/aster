# AMDGCN Register Allocation

This document describes the AMDGCN register allocation pipeline and its constituent passes: bufferization, to-register semantics, and register coloring. It lists pre-conditions and results for each step and includes sample IR from the tests.

## Pipeline Overview

The **`amdgcn-reg-alloc`** pipeline performs register allocation for AMDGCN kernels by running the following passes in sequence:

1. **Bufferization** (`aster-amdgcn-bufferization`) – inserts copies to remove potentially clobbered values and eliminates phi-nodes with register value semantics.
2. **ToRegisterSemantics** (`amdgcn-to-register-semantics`) – converts value allocas to unallocated register semantics and propagates them through the IR.
3. **RegisterDCE** (`amdgcn-register-dce`) – dead code elimination for register copy/mov operations.
4. **RegisterColoring** (`amdgcn-register-coloring`) – assigns physical registers using liveness and interference.

**Important:** Both bufferization and to-register semantics expect IR in **value semantics**. Register coloring expects IR in **register semantics** (after to-register semantics and register DCE).

---

## 1. Bufferization (`aster-amdgcn-bufferization`)

### Summary

Inserts phi-breaking copies to prepare for register allocation. It can be viewed as a “bufferization” pass that moves the IR away from DPS/value semantics toward side-effects before register allocation.

### Pre-conditions

- **IR in value semantics**: Kernels use `alloca` with register types (`!amdgcn.vgpr`, `!amdgcn.sgpr`) and SSA value semantics.
- Block arguments may carry register-typed values (phi-nodes).
- Operations may produce and consume values that flow through phis or may clobber allocas.

### Results

- **Phi-nodes removed**: Block arguments with register value semantics are eliminated by inserting allocations at dominator points and copying values into them; branch operands for those phis are removed.
- **Clobber copies inserted**: When a value may be clobbered (e.g., same alloca written twice, or value used across blocks after another write), the pass inserts `lsir.copy` and fresh allocas so each “version” has its own storage.
- **Still value semantics**: Allocas and copies remain in value-semantic form; no physical register numbers yet.

### Sample: Diamond CFG (phi-breaking)

**Before** (value semantics, phi at merge):

```mlir
amdgcn.module @bufferization_phi_copies_1 target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @bufferization_phi_copies_1 {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb3(%1 : !amdgcn.vgpr)
  ^bb2:
    cf.br ^bb3(%2 : !amdgcn.vgpr)
  ^bb3(%3: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
    %4 = test_inst outs %3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    end_kernel
  }
}
```

**After** (phi removed, copies and merge alloca inserted):

```mlir
// CHECK:         %[[VAL_0:.*]] = alloca : !amdgcn.vgpr
// CHECK:         %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK:         %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:         cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:       ^bb1:
// CHECK:         %[[VAL_3:.*]] = alloca : !amdgcn.vgpr
// CHECK:         %[[COPY_0:.*]] = lsir.copy %[[VAL_3]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:         lsir.copy %[[VAL_2]], %[[COPY_0]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:         cf.br ^bb3
// CHECK:       ^bb2:
// CHECK:         %[[VAL_4:.*]] = alloca : !amdgcn.vgpr
// CHECK:         %[[COPY_2:.*]] = lsir.copy %[[VAL_4]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:         lsir.copy %[[VAL_2]], %[[COPY_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:         cf.br ^bb3
// CHECK:       ^bb3:
// CHECK:         %[[VAL_5:.*]] = dealloc_cast %[[VAL_2]] : !amdgcn.vgpr<?>
// CHECK:         %[[VAL_6:.*]] = test_inst outs %[[VAL_5]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
```

### Sample: Cross-block clobber

When the same alloca is written twice and the first result is used in a successor block, bufferization inserts a copy so that use sees the correct value:

**Before:**

```mlir
%v1 = test_inst outs %0 ins %1 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
%v2 = test_inst outs %0 ins %1 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  test_inst ins %v1 : (!amdgcn.vgpr) -> ()
^bb2:
  test_inst ins %v2 : (!amdgcn.vgpr) -> ()
```

**After:** A copy is inserted so the first write’s value is preserved for `%v1`:

```mlir
// CHECK:         %[[COPY_A:.*]] = alloca : !amdgcn.vgpr
// CHECK:         %[[COPY:.*]] = lsir.copy %[[COPY_A]], %[[A]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:         %[[V1:.*]] = test_inst outs %[[A]] ins %[[S]]
// CHECK:         %[[V2:.*]] = test_inst outs %[[COPY]] ins %[[S]]
```

---

## 2. To-Register Semantics (`amdgcn-to-register-semantics`)

### Summary

Propagates **non-value** (unallocated) register semantics through the IR. Converts value allocas to unallocated allocas and ensures instruction results are not used as SSA values by other operations (values flow through side-effects on allocas).

### Pre-conditions

- **IR in value semantics**: Same as bufferization output — value allocas, `lsir.copy`, possibly `dealloc_cast`, block arguments already eliminated for register types by bufferization.
- Runs **after** bufferization, **before** register coloring.

### Results

- **Value allocas → unallocated allocas**: Types change from e.g. `!amdgcn.vgpr` to `!amdgcn.vgpr<?>` (unallocated).
- **No SSA use of instruction results**: After the pass, instruction results are not used as operands of other operations; all uses go through allocas (register semantics).
- **`amdgcn.alloca`**: Value `alloca` may be represented as `amdgcn.alloca` with `!amdgcn.vgpr<?>` (or similar) in the tests; the key is unallocated semantics and use only through side-effects.

### Sample: Value allocas to unallocated

**Before** (value semantics, results used as SSA values):

```mlir
func.func @range_amdgcn.allocations() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  // ...
  %7 = lsir.copy %6, %0 : !amdgcn.vgpr, !amdgcn.vgpr
  %8 = amdgcn.test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  // ...
  amdgcn.test_inst ins %8, %11 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  func.return
}
```

**After** (unallocated semantics, `?` in type):

```mlir
// CHECK-DAG:     %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:     %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// ...
// CHECK:         lsir.copy %[[ALLOCA_6]], %[[ALLOCA_0]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:         amdgcn.test_inst outs %[[ALLOCA_0]] : (!amdgcn.vgpr<?>) -> ()
// ...
// CHECK:         amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
```

### Sample: Existing physical registers preserved

When the input already has fixed register indices (e.g. `!amdgcn.vgpr<0>`, `!amdgcn.sgpr<2>`), to-register semantics preserves them and only converts unallocated slots to `?`:

```mlir
// Input
%0 = amdgcn.alloca : !amdgcn.vgpr<0>
%2 = amdgcn.alloca : !amdgcn.sgpr<0>
// ...
// Output: same fixed indices, unallocated ones become <?>
// CHECK-DAG: %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK-DAG: %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<0>
```

---

## 3. Register DCE (`amdgcn-register-dce`)

### Summary

Dead code elimination for register copy/mov operations: removes `lsir.copy` and AMDGCN mov instructions whose destination register is not live after the operation.

### Pre-conditions

- IR in **register semantics** (after to-register semantics): allocas with unallocated semantics, no SSA use of instruction results for register values.

### Results

- Redundant or dead copies and movs are removed, reducing register pressure and code size before coloring.

---

## 4. Register Coloring (`amdgcn-register-coloring`)

### Summary

Performs register allocation by building liveness, building an interference graph, and assigning physical register indices to each alloca. Supports optional “full” graph build mode via pipeline option `--amdgcn-reg-alloc=mode=full`.

### Pre-conditions

- **IR in register semantics**: All register storage is in the form of allocas (or equivalent) with **unallocated** semantics (e.g. `!amdgcn.vgpr<?>`, `!amdgcn.sgpr<?>`).
- No block arguments carrying register types (already removed by bufferization).
- Runs **after** bufferization, to-register semantics, and register DCE.

### Results

- **Allocas get physical registers**: Types change from `!amdgcn.vgpr<?>` to `!amdgcn.vgpr<N>` (and similarly for SGPR/AGPR).
- **`lsir.copy` lowered**: Copies are lowered to machine movs (e.g. `amdgcn.vop1.vop1 <v_mov_b32_e32>`, `amdgcn.sop1 s_mov_b32`).
- **Register ranges**: `make_register_range` operands get concrete indices; the IR is in allocated form suitable for later codegen/legalization.

### Sample: Allocas colored to fixed indices

**Before** (register semantics, unallocated):

```mlir
amdgcn.kernel @range_allocations {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  // ...
  test_inst outs %0 : (!amdgcn.vgpr<?>) -> ()
  test_inst ins %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}
```

**After** (physical registers):

```mlir
// CHECK-DAG:     %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// ...
// CHECK:         test_inst outs %[[VAL_0]] : (!amdgcn.vgpr<0>) -> ()
// CHECK:         test_inst ins %[[VAL_0]], %[[VAL_1]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
```

### Sample: Copies lowered to movs

**Before:**

```mlir
lsir.copy %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
lsir.copy %2, %3 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
```

**After:**

```mlir
// CHECK:         amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:         amdgcn.sop1 s_mov_b32 outs %[[ALLOCA_2]] ins %[[ALLOCA_3]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
```

### Sample: Full pipeline (reg-alloc)

**Input** (value semantics, phi):

```mlir
amdgcn.module @reg_alloc target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @reg_alloc {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb3(%1 : !amdgcn.vgpr)
  ^bb2:
    cf.br ^bb3(%2 : !amdgcn.vgpr)
  ^bb3(%3: !amdgcn.vgpr):
    %4 = test_inst outs %3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    end_kernel
  }
}
```

**After `--amdgcn-reg-alloc`**: Phi and block args gone; allocas colored (e.g. `!amdgcn.vgpr<0>`); no unallocated `!amdgcn.vgpr<1>` in minimal mode:

```mlir
// CHECK-NOT: alloca : !amdgcn.vgpr<1>
// CHECK:     alloca : !amdgcn.vgpr<0>
```

---

## Summary Table

| Step                 | Pre-condition        | Result |
|----------------------|----------------------|--------|
| **Bufferization**    | Value semantics IR   | No phis for registers; clobber copies inserted; still value semantics |
| **ToRegisterSemantics** | Value semantics IR (after bufferization) | Unallocated allocas (`?`); register semantics; no SSA use of inst results |
| **RegisterDCE**      | Register semantics   | Dead copies/movs removed |
| **RegisterColoring** | Register semantics (unallocated) | Physical registers assigned; copies lowered to movs |

---

## Running the pipeline

```bash
aster-opt input.mlir --amdgcn-reg-alloc -o output.mlir
```

With full interference graph build mode:

```bash
aster-opt input.mlir --amdgcn-reg-alloc=mode=full -o output.mlir
```

Individual passes (for debugging or custom pipelines):

```bash
aster-opt input.mlir --aster-amdgcn-bufferization --split-input-file | FileCheck %s
aster-opt input.mlir --amdgcn-to-register-semantics --split-input-file | FileCheck %s
aster-opt input.mlir --amdgcn-register-coloring --cse --split-input-file | FileCheck %s
```

Test locations:

- Bufferization: `aster/test/Dialect/AMDGCN/Transforms/bufferization.mlir`
- To-register semantics: `aster/test/Dialect/AMDGCN/Transforms/to-register-semantics.mlir`
- Register coloring: `aster/test/Dialect/AMDGCN/Transforms/register-coloring.mlir`
- Full pipeline: `aster/test/Dialect/AMDGCN/Transforms/reg-alloc.mlir`, `chained-select-dps-violation.mlir`
