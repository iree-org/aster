// RUN: aster-opt %s --verify-roundtrip

func.func @cmpi(%scc: !amdgcn.scc, %vcc: !amdgcn.vcc, %a: i32, %b: i32,
    %v1: !amdgcn.vgpr, %dst: !amdgcn.sgpr, %dstAlloc: !amdgcn.sgpr<0>) {
  amdgcn.cmpi s_cmp_eq_i32 outs %scc ins %a, %b : outs(!amdgcn.scc) ins(i32, i32)
  amdgcn.cmpi v_cmp_eq_i32 outs %vcc ins %a, %v1 : outs(!amdgcn.vcc) ins(i32, !amdgcn.vgpr)
  %0 = amdgcn.cmpi v_cmp_eq_i32_e64 outs %dst ins %a, %v1 : dps(!amdgcn.sgpr) ins(i32, !amdgcn.vgpr)
  amdgcn.cmpi v_cmp_eq_i32_e64 outs %dstAlloc ins %a, %v1 : outs(!amdgcn.sgpr<0>) ins(i32, !amdgcn.vgpr)
  return
}
