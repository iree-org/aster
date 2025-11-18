// RUN: aster-opt %s --split-input-file --pass-pipeline="builtin.module(amdgcn.module(amdgcn-register-allocation))" --aster-disable-verifiers --aster-suppress-disabled-verifier-warning --verify-diagnostics

amdgcn.module @invalid target = <gfx942> isa = <cdna3> {
// expected-error@above {{Range constraints are not satisfiable}}
  kernel @invalid_alignment {
    // expected-error@+1 {{Unsatisfiable alignment constraint}}
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.vgpr
    %4 = alloca : !amdgcn.vgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
    // It's impossible to satisfy an alignment of 4 for %0 and %2
    %7 = make_register_range %0, %1, %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %8 = make_register_range %2, %3, %4, %5 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    end_kernel
  }
}

// -----

amdgcn.module @invalid target = <gfx942> isa = <cdna3> {
// expected-error@above {{There are un-allocatable ranges}}
// expected-error@above {{Range constraints are not satisfiable}}
  kernel @unallocatable_range {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.vgpr
    %4 = alloca : !amdgcn.vgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
    // It's impossible to allocate these variable because %6 and %7 overlap
    %7 = make_register_range %1, %0, %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    end_kernel
  }
}

// -----

amdgcn.module @invalid target = <gfx942> isa = <cdna3> {
// expected-error@above {{ill-formed IR detected}}
// expected-error@above {{Failed to create interference graph}}
  func.func private @rand() -> i1
  kernel @unallocatable_range {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    cf.br ^bb3(%1 : !amdgcn.vgpr)
  ^bb2:  // pred: ^bb0
    cf.br ^bb3(%2 : !amdgcn.vgpr)
  ^bb3(%3: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
    %4 = test_inst outs %3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    end_kernel
  }
}

// -----

amdgcn.module @invalid target = <gfx942> isa = <cdna3> {
  kernel @too_many_regs {
    // It maxes out available registers.
    // expected-error@below {{failed to allocate register range}}
    %0 = alloca : !amdgcn.sgpr
    %1 = alloca : !amdgcn.sgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.sgpr
    %5 = alloca : !amdgcn.sgpr
    %6 = alloca : !amdgcn.sgpr
    %7 = alloca : !amdgcn.sgpr
    %8 = alloca : !amdgcn.sgpr
    %9 = alloca : !amdgcn.sgpr
    %10 = alloca : !amdgcn.sgpr
    %11 = alloca : !amdgcn.sgpr
    %12 = alloca : !amdgcn.sgpr
    %13 = alloca : !amdgcn.sgpr
    %14 = alloca : !amdgcn.sgpr
    %15 = alloca : !amdgcn.sgpr
    %16 = alloca : !amdgcn.sgpr
    %17 = alloca : !amdgcn.sgpr
    %18 = alloca : !amdgcn.sgpr
    %19 = alloca : !amdgcn.sgpr
    %20 = alloca : !amdgcn.sgpr
    %21 = alloca : !amdgcn.sgpr
    %22 = alloca : !amdgcn.sgpr
    %23 = alloca : !amdgcn.sgpr
    %24 = alloca : !amdgcn.sgpr
    %25 = alloca : !amdgcn.sgpr
    %26 = alloca : !amdgcn.sgpr
    %27 = alloca : !amdgcn.sgpr
    %28 = alloca : !amdgcn.sgpr
    %29 = alloca : !amdgcn.sgpr
    %30 = alloca : !amdgcn.sgpr
    %31 = alloca : !amdgcn.sgpr
    %32 = alloca : !amdgcn.sgpr
    %33 = alloca : !amdgcn.sgpr
    %34 = alloca : !amdgcn.sgpr
    %35 = alloca : !amdgcn.sgpr
    %36 = alloca : !amdgcn.sgpr
    %37 = alloca : !amdgcn.sgpr
    %38 = alloca : !amdgcn.sgpr
    %39 = alloca : !amdgcn.sgpr
    %40 = alloca : !amdgcn.sgpr
    %41 = alloca : !amdgcn.sgpr
    %42 = alloca : !amdgcn.sgpr
    %43 = alloca : !amdgcn.sgpr
    %44 = alloca : !amdgcn.sgpr
    %45 = alloca : !amdgcn.sgpr
    %46 = alloca : !amdgcn.sgpr
    %47 = alloca : !amdgcn.sgpr
    %48 = alloca : !amdgcn.sgpr
    %49 = alloca : !amdgcn.sgpr
    %50 = alloca : !amdgcn.sgpr
    %51 = alloca : !amdgcn.sgpr
    %52 = alloca : !amdgcn.sgpr
    %53 = alloca : !amdgcn.sgpr
    %54 = alloca : !amdgcn.sgpr
    %55 = alloca : !amdgcn.sgpr
    %56 = alloca : !amdgcn.sgpr
    %57 = alloca : !amdgcn.sgpr
    %58 = alloca : !amdgcn.sgpr
    %59 = alloca : !amdgcn.sgpr
    %60 = alloca : !amdgcn.sgpr
    %61 = alloca : !amdgcn.sgpr
    %62 = alloca : !amdgcn.sgpr
    %63 = alloca : !amdgcn.sgpr
    %64 = alloca : !amdgcn.sgpr
    %65 = alloca : !amdgcn.sgpr
    %66 = alloca : !amdgcn.sgpr
    %67 = alloca : !amdgcn.sgpr
    %68 = alloca : !amdgcn.sgpr
    %69 = alloca : !amdgcn.sgpr
    %70 = alloca : !amdgcn.sgpr
    %71 = alloca : !amdgcn.sgpr
    %72 = alloca : !amdgcn.sgpr
    %73 = alloca : !amdgcn.sgpr
    %74 = alloca : !amdgcn.sgpr
    %75 = alloca : !amdgcn.sgpr
    %76 = alloca : !amdgcn.sgpr
    %77 = alloca : !amdgcn.sgpr
    %78 = alloca : !amdgcn.sgpr
    %79 = alloca : !amdgcn.sgpr
    %80 = alloca : !amdgcn.sgpr
    %81 = alloca : !amdgcn.sgpr
    %82 = alloca : !amdgcn.sgpr
    %83 = alloca : !amdgcn.sgpr
    %84 = alloca : !amdgcn.sgpr
    %85 = alloca : !amdgcn.sgpr
    %86 = alloca : !amdgcn.sgpr
    %87 = alloca : !amdgcn.sgpr
    %88 = alloca : !amdgcn.sgpr
    %89 = alloca : !amdgcn.sgpr
    %90 = alloca : !amdgcn.sgpr
    %91 = alloca : !amdgcn.sgpr
    %92 = alloca : !amdgcn.sgpr
    %93 = alloca : !amdgcn.sgpr
    %94 = alloca : !amdgcn.sgpr
    %95 = alloca : !amdgcn.sgpr
    %96 = alloca : !amdgcn.sgpr
    %97 = alloca : !amdgcn.sgpr
    %98 = alloca : !amdgcn.sgpr
    %99 = alloca : !amdgcn.sgpr
    %100 = alloca : !amdgcn.sgpr
    %range = make_register_range %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    test_inst ins %range : (!amdgcn.sgpr_range<[? + 101]>) -> ()
    end_kernel
  }
}
