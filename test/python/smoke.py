# RUN: %PYTHON %s | FileCheck %s

from aster import ir


with ir.Context() as ctx, ir.Location.unknown() as loc:
    mod = ir.Module.parse(
        """
    func.func @smoke(%A: memref<2xf32>) -> f32 {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %f0 = arith.constant 0.0 : f32
        %res = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %f0) -> (f32) {
            %a = memref.load %A[%i]: memref<2xf32>
            scf.yield %a: f32
        }
        return %res: f32
    }
    """
    )
    # CHECK-LABEL: func.func @smoke
    #       CHECK:   scf.for
    print(mod)
