# RUN: %PYTHON %s | FileCheck %s

from aster import ir
from aster.dialects import builtin, amdgcn as a_d

with ir.Context() as ctx, ir.Location.unknown() as loc:
    m = builtin.ModuleOp()
    with ir.InsertionPoint(m.body):
        modOp = a_d.ModuleOp(a_d.Target.GFX942, a_d.ISAVersion.CDNA3, "module")
        modOp.body_region.blocks.append()
    print(str(m))
    # CHECK-LABEL: amdgcn.module @module target = <gfx942> isa = <cdna3>

    # Test register types: VGPR, SGPR, AGPR
    # CHECK: vgpr<0>
    # CHECK-NEXT: sgpr<1>
    # CHECK-NEXT: agpr<2>
    print(str(a_d.VGPRType.get(ctx, 0)))
    print(str(a_d.SGPRType.get(ctx, 1)))
    print(str(a_d.AGPRType.get(ctx, 2)))

    # Test relocatable register types
    # CHECK-NEXT: vgpr
    # CHECK-NEXT: sgpr
    print(str(a_d.VGPRType.get(ctx, None)))
    print(str(a_d.SGPRType.get(ctx, None)))

    # Test register range types with default alignment
    # CHECK-NEXT: vgpr_range<[0 : 4]>
    # CHECK-NEXT: sgpr_range<[0 : 3]>
    # CHECK-NEXT: agpr_range<[0 : 2]>
    print(str(a_d.VGPRRangeType.get(ctx, 4, 0)))
    print(str(a_d.SGPRRangeType.get(ctx, 3, 0)))
    print(str(a_d.AGPRRangeType.get(ctx, 2, 0)))

    # Test register range types with explicit alignment (different from default)
    # CHECK-NEXT: vgpr_range<[0 : 4 align 8]>
    # CHECK-NEXT: sgpr_range<[0 : 3 align 8]>
    print(str(a_d.VGPRRangeType.get(ctx, 4, 0, 8)))
    print(str(a_d.SGPRRangeType.get(ctx, 3, 0, 8)))

    # Test relocatable register range types
    # CHECK-NEXT: vgpr_range<[? + 4]>
    # CHECK-NEXT: agpr_range<[? + 2]>
    print(str(a_d.VGPRRangeType.get(ctx, 4, None)))
    print(str(a_d.AGPRRangeType.get(ctx, 2, None)))
