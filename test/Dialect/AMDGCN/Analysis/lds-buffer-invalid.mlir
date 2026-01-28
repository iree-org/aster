// RUN: aster-opt --pass-pipeline="builtin.module(func.func(test-lds-interference-graph))" --split-input-file --verify-diagnostics %s

func.func @use_after_dealloc() {
  %0 = amdgcn.alloc_lds 128
  amdgcn.dealloc_lds %0
  //  expected-error@+1 {{get_lds_offset operates on a dead buffer}}
  %2 = amdgcn.get_lds_offset %0 : i32
  return
}

// -----

func.func @offset_use_after_dealloc() -> i32 {
  %0 = amdgcn.alloc_lds 128
  //  expected-error@+1 {{get_lds_offset use operates on a non-live buffer}}
  %2 = amdgcn.get_lds_offset %0 : i32
  amdgcn.dealloc_lds %0
  return %2 : i32
}
