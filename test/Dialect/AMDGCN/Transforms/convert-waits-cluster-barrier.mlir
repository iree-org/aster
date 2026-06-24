// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn-convert-waits))" | FileCheck %s

// CHECK-LABEL: kernel @cbar
//       CHECK:   %[[SCC:.*]] = alloca : !amdgcn.scc<0>
//       CHECK:   s_barrier_signal_isfirst outs(%[[SCC]]) id(-1 : i16)
//       CHECK:   s_barrier_wait -1
//       CHECK:   lsir.cond_br %[[SCC]] : !amdgcn.scc<0>, ^[[SIG:.*]], ^[[JOIN:.*]]
//       CHECK: ^[[SIG]]:
//       CHECK:   s_barrier_signal -3
//       CHECK:   lsir.br ^[[JOIN]]
//       CHECK: ^[[JOIN]]:
//       CHECK:   s_barrier_wait -3
amdgcn.module @m_cbar target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @cbar {
    amdgcn.barrier scope(#amdgcn.barrier_scope<cluster>)
    amdgcn.end_kernel
  }
}

// CHECK-LABEL: kernel @ctbar
//   CHECK-NOT:   token_barrier
//       CHECK:   s_barrier_signal_isfirst outs(%{{.*}}) id(-1 : i16)
//       CHECK:   s_barrier_wait -1
//       CHECK:   lsir.cond_br %{{.*}} : !amdgcn.scc<0>, ^[[SIG2:.*]], ^[[JOIN2:.*]]
//       CHECK: ^[[SIG2]]:
//       CHECK:   s_barrier_signal -3
//       CHECK:   lsir.br ^[[JOIN2]]
//       CHECK: ^[[JOIN2]]:
//       CHECK:   s_barrier_wait -3
amdgcn.module @m_ctbar target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @ctbar {
    %t = amdgcn.token_barrier scope(#amdgcn.barrier_scope<cluster>)
    amdgcn.end_kernel
  }
}
