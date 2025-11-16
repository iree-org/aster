# RUN: %PYTHON %s | FileCheck %s

from aster import ir


with ir.Context() as ctx, ir.Location.unknown() as loc:
    mod = ir.Module.parse(
        """
    func.func @contract(
        %A: memref<2x3x5xf32>, %B: memref<2x5x7xf32>, %C: memref<2x3x7xf32>) {
    linalg.contract
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
        ins(%A, %B : memref<2x3x5xf32>, memref<2x5x7xf32>)
        outs(%C: memref<2x3x7xf32>)
    return
    }
    """
    )
    # CHECK: linalg.contract
    print(mod)
