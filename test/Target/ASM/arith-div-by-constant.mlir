// RUN: aster-opt %s \
// RUN:     --aster-optimize-arith --aster-to-int-arith \
// RUN:     --aster-codegen --canonicalize \
// RUN:     --aster-to-amdgcn --amdgcn-backend --amdgcn-remove-test-inst \
// RUN:   2>/dev/null \
// RUN:   | aster-translate --mlir-to-asm 2>/dev/null \
// RUN:   | python3 -c "import sys; from aster.compiler.core import compile_to_hsaco; \
// RUN:     asm=sys.stdin.read(); r=compile_to_hsaco(asm); \
// RUN:     assert r is not None, 'HSACO assembly failed'"

amdgcn.module @arith_div_by_const target = #amdgcn.target<gfx942> {

  amdgcn.kernel @sdiv_by_3 {
    %n = amdgcn.alloca : !amdgcn.vgpr
    %n_i32 = lsir.from_reg %n : !amdgcn.vgpr -> i32
    %c3 = arith.constant 3 : i32
    %q = arith.divsi %n_i32, %c3 : i32
    %r = lsir.to_reg %q : i32 -> !amdgcn.vgpr
    amdgcn.test_inst ins %r : (!amdgcn.vgpr) -> ()
    end_kernel
  }

  amdgcn.kernel @sdiv_by_5 {
    %n = amdgcn.alloca : !amdgcn.vgpr
    %n_i32 = lsir.from_reg %n : !amdgcn.vgpr -> i32
    %c5 = arith.constant 5 : i32
    %q = arith.divsi %n_i32, %c5 : i32
    %r = lsir.to_reg %q : i32 -> !amdgcn.vgpr
    amdgcn.test_inst ins %r : (!amdgcn.vgpr) -> ()
    end_kernel
  }

  amdgcn.kernel @sdiv_by_7 {
    %n = amdgcn.alloca : !amdgcn.vgpr
    %n_i32 = lsir.from_reg %n : !amdgcn.vgpr -> i32
    %c7 = arith.constant 7 : i32
    %q = arith.divsi %n_i32, %c7 : i32
    %r = lsir.to_reg %q : i32 -> !amdgcn.vgpr
    amdgcn.test_inst ins %r : (!amdgcn.vgpr) -> ()
    end_kernel
  }

  amdgcn.kernel @sdiv_by_6 {
    %n = amdgcn.alloca : !amdgcn.vgpr
    %n_i32 = lsir.from_reg %n : !amdgcn.vgpr -> i32
    %c6 = arith.constant 6 : i32
    %q = arith.divsi %n_i32, %c6 : i32
    %r = lsir.to_reg %q : i32 -> !amdgcn.vgpr
    amdgcn.test_inst ins %r : (!amdgcn.vgpr) -> ()
    end_kernel
  }

  amdgcn.kernel @sdiv_by_12 {
    %n = amdgcn.alloca : !amdgcn.vgpr
    %n_i32 = lsir.from_reg %n : !amdgcn.vgpr -> i32
    %c12 = arith.constant 12 : i32
    %q = arith.divsi %n_i32, %c12 : i32
    %r = lsir.to_reg %q : i32 -> !amdgcn.vgpr
    amdgcn.test_inst ins %r : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}
