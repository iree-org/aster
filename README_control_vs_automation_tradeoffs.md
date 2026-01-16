# Control / Automation Tradeoffs in ASTER

ASTER embraces "Control First, Automatic Optimizations Second" - WYSIWYG assembly
production where what you write is what you get. However, mixing manual control
with compiler automation introduces subtle interactions that power users should
understand.

## Core Philosophy

ASTER provides three main paths for IR generation:
1. **Direct MLIR authoring** - Type-safe, WYSIWYG for small kernels
2. **Python metaprogramming** - Variable declarations, loops, comprehensions for code generation
3. **MLIR passes and lowerings** - Higher-level abstractions with automatic transformations

Paths 2 and 3 can easily break WYSIWYG properties: dead-code elimination (DCE)
kicks in by default when performing MLIR dataflow analyses or using the MLIR
greedy pattern rewriter.

### WYSIWYG vs Automatic Transformations

Higher-level abstractions and MLIR passes provide productivity through automation
while maintaining various degrees of control but one needs to be aware of the
following compiler:

- **Greedy pattern rewriter** may trigger DCE as a side effect
- **Canonicalization and CSE passes** may eliminate "useless" code you wanted to
benchmark or reuse values leading to different register allocation behaviors
- **Dataflow analyses** may mark unreachable code as dead

**Recommendation**: For nanobenchmarking or hardware exploration where you need
perfect control over emitted instructions, prefer low-level MLIR authoring with
explicitly allocated registers or Python metaprogramming (see 
[demo/README.md](demo/README.md)) over MLIR passes.

## Automation Caveats: DCE and `test_inst` Interactions with Register Allocation

To avoid undesirable DCE on certain parts of the program, a common strategy is to
use transient ops. ASTER introduces transient ops such as `amdgcn.test_inst` to
allow expressing def-use chains and prevent DCE (e.g. to produce asm that only
loads values and stress test the load properties of the HW).
Such instructions are removed by a late pass like `RemoveTestInst`.

When using `amdgcn.test_inst` to prevent DCE of otherwise dead code, the placement
of `test_inst` relative to operations that can complete out-of-order---such as
memory operations---affects register allocation, even though `test_inst` is
erased before assembly translation.

**The Issue**: `test_inst` is a transient op that establishes SSA def-use chains
to inform liveness analysis and register allocation; it does not produce any 
assembly instruction (not even `nop`).
In the case of memory operations such as `global_load`, special care needs to be
taken to ensure the semantics implied by the existence of transient ops, is
materialized during execution.

Concretely, issues may arise when these conditions hold:
1. some SSA def-use chains are disjoint 
2. memory operations whose effect is observed later in the program depend on 
values computed by 1. 
3. register allocation aliases storage for values with disjoint lifetimes such as 1.
4. the transient ops are removed: aliasing remains without corresponding register-level
WAR, RAW or WAW dependencies that the needs to resolve
dependence chains
5. faster operations issued after finish executing earlier

In the case of `global_load`, these can cause the address of multiple loads in
flight to be allocated the same destination register, leading to write-after-write
(WAW) hazards and segfaults.

### Incorrect Pattern - a.k.a. Careful What Automation + Control You Wish For

Consider the following pseudo-IR:
```mlir
%loaded_0 = global_load %a_0 ...
test_inst %loaded_0              // <- %loaded_0 appears consumed here
%loaded_1 = global_load %a_1 ... // <- regalloc may reuse ***%a_0***'s register for %a_1. 
                                 // In the absence of a s_waitcnt, this may lead to a race.
test_inst %loaded_1      
%loaded_2 = global_load %a_2 ...
test_inst %loaded_2
%loaded_3 = global_load %a_3 ...
test_inst %loaded_3
amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 // wait on the loads to actually finish.

// repeat the above
%loaded_0 = global_load %a_0 ...
test_inst %loaded_0             
%loaded_1 = global_load %a_1 ...
test_inst %loaded_1      
%loaded_2 = global_load %a_2 ...
test_inst %loaded_2
%loaded_3 = global_load %a_3 ...
test_inst %loaded_3
amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 // wait on the loads to actually finish.
```

which produces assembly resembling:
```
  v_add_u32 v2, v1, v0
  global_load_dword v0, v2, s[2:3] // <- v0 used as return value for global_load
  v_add_u32 v0, 2, v2              // <- WAW race on v0
  global_load_dword v1, v0, s[2:3] // <- Technically v0 could contain `@v2[s[2:3]]` (instead of the intended `v2 + 2`).
                                         The probability this occurs is very low but this is still a bug.
  v_add_u32 v1, 4, v2
  global_load_dword v3, v1, s[2:3]
  v_add_u32 v3, 6, v2
  global_load_dword v4, v3, s[2:3]
  s_waitcnt vmcnt(0)               // <- After this instruction, the previous global_load are guaranteed to have computed.
                                   // <- If v0 previously held value `v2 + 2`, now it it sure to hold `@v2[s[2:3]]` and the program would crash.
                                   //    otherwise v0 would have held `@v2[s[2:3]]` and would already have crashed.
  global_load_dword v4, v2, s[2:3] // <- This almost surely crashes.
  global_load_dword v2, v0, s[2:3]
  global_load_dword v0, v1, s[2:3]
  global_load_dword v0, v3, s[2:3]
  s_waitcnt vmcnt(0)
```

In other words, the program is **guaranteed to crash** !

**Use case** - Such a pattern may appear when trying to create multiple async
loads that are concurrently in-flight (e.g. to nanobenchmark the limits of the HW
load units) while ignoring their return value and preventing DCE  and still
benefiting from automatic register allocation.

### Correct Patterns In The Presence Of `amdgcn.test_inst

**Note** - It is sufficient (but not necessary), to synchronizing the `global_load`.
Synchronizing the `global_load` guarantees the **address register** `v0` has been
written by `global_load` first and then has been overwritten by `v_add_u32`.
However this also over-synchronizes and defeats the purpose as shown by the
following pseudo-IR:
```mlir
%loaded_0 = global_load %a_0 ...
test_inst %loaded_0
amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
%loaded_1 = global_load %a_1 ...                                
test_inst %loaded_1      
amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
%loaded_2 = global_load %a_2 ...
test_inst %loaded_2
amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
%loaded_3 = global_load %a_3 ...
test_inst %loaded_3
amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
```

which results in the following asm:
```
  v_add_u32 v2, v1, v0
  global_load_dword v0, v2, s[2:3]
  s_waitcnt vmcnt(0)
  v_add_u32 v0, 2, v2 
  global_load_dword v1, v0, s[2:3]
  s_waitcnt vmcnt(0)
  v_add_u32 v1, 4, v2
  global_load_dword v3, v1, s[2:3]
  s_waitcnt vmcnt(0)
  v_add_u32 v3, 6, v2
  global_load_dword v4, v3, s[2:3]
  s_waitcnt vmcnt(0)
  global_load_dword v4, v2, s[2:3]
  s_waitcnt vmcnt(0)
  global_load_dword v2, v0, s[2:3]
  s_waitcnt vmcnt(0)
  global_load_dword v0, v1, s[2:3]
  s_waitcnt vmcnt(0)
  global_load_dword v0, v3, s[2:3]
  s_waitcnt vmcnt(0)
```

**Correct Pattern** - Performing all loads, then all uses ensure the liveness of `%a_0`
overlaps with that of `%a_1` etc :
```mlir
%loaded_0 = global_load %a_0 ...
%loaded_1 = global_load %a_1 ...
%loaded_2 = global_load %a_2 ...
%loaded_3 = global_load %a_3 ...

test_inst %loaded_0
test_inst %loaded_1
test_inst %loaded_2
test_inst %loaded_3
```

Each load gets a unique destination register because all values are live
simultaneously at the `test_inst` uses.
