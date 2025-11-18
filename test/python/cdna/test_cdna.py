# RUN: %PYTHON %s | FileCheck %s

from aster import ir
from aster.dialects import amdgcn, arith, builtin


def build_test_module(ctx: ir.Context) -> builtin.ModuleOp:
    module = builtin.ModuleOp()

    with ir.InsertionPoint(module.body):
        amdgcn_mod = amdgcn.ModuleOp(
            amdgcn.Target.GFX942, amdgcn.ISAVersion.CDNA3, "ds_kernels"
        )
        amdgcn_mod.body_region.blocks.append()
        # CHECK-LABEL:   amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
        # CHECK:           kernel @ds_all_kernel {
        # CHECK:             %[[VAL_0:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_2:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_3:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_4:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_5:.*]] = make_register_range %[[VAL_1]] : !amdgcn.vgpr
        # CHECK:             %[[VAL_6:.*]] = amdgcn.ds.read <ds_read_b32> %[[VAL_5]], %[[VAL_0]], offset = %{{.*}} : !amdgcn.vgpr, i32 -> <[? + 1]>
        # CHECK:             amdgcn.ds.write <ds_write_b32> %[[VAL_6]], %[[VAL_0]], offset = %{{.*}} : <[? + 1]>, !amdgcn.vgpr, i32
        # CHECK:             %[[VAL_7:.*]] = make_register_range %[[VAL_1]], %[[VAL_2]] : !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_8:.*]] = amdgcn.ds.read <ds_read_b64> %[[VAL_7]], %[[VAL_0]], offset = %{{.*}} : !amdgcn.vgpr, i32 -> <[? + 2]>
        # CHECK:             amdgcn.ds.write <ds_write_b64> %[[VAL_8]], %[[VAL_0]], offset = %{{.*}} : <[? + 2]>, !amdgcn.vgpr, i32
        # CHECK:             %[[VAL_9:.*]] = make_register_range %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_10:.*]] = amdgcn.ds.read <ds_read_b96> %[[VAL_9]], %[[VAL_0]], offset = %{{.*}} : !amdgcn.vgpr, i32 -> <[? + 3]>
        # CHECK:             amdgcn.ds.write <ds_write_b96> %[[VAL_10]], %[[VAL_0]], offset = %{{.*}} : <[? + 3]>, !amdgcn.vgpr, i32
        # CHECK:             %[[VAL_11:.*]] = make_register_range %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_12:.*]] = amdgcn.ds.read <ds_read_b128> %[[VAL_11]], %[[VAL_0]], offset = %{{.*}} : !amdgcn.vgpr, i32 -> <[? + 4]>
        # CHECK:             amdgcn.ds.write <ds_write_b128> %[[VAL_12]], %[[VAL_0]], offset = %{{.*}} : <[? + 4]>, !amdgcn.vgpr, i32
        with ir.InsertionPoint(amdgcn_mod.body_region.blocks[0]):
            kernel = amdgcn.KernelOp("ds_all_kernel")
            kernel.body_region.blocks.append()

            with ir.InsertionPoint(kernel.body_region.blocks[0]):
                # Allocate registers once
                addr = amdgcn.api.alloca_vgpr()
                data1 = amdgcn.api.alloca_vgpr()
                data2 = amdgcn.api.alloca_vgpr()
                data3 = amdgcn.api.alloca_vgpr()
                data4 = amdgcn.api.alloca_vgpr()

                # B32: data needs 1 register
                data_range_b32 = amdgcn.api.make_register_range([data1])
                result_b32 = amdgcn.api.ds_read_b32(data_range_b32, addr, offset=0)
                amdgcn.api.ds_write_b32(result_b32, addr, offset=0)

                # B64: data needs 2 registers
                data_range_b64 = amdgcn.api.make_register_range([data1, data2])
                result_b64 = amdgcn.api.ds_read_b64(data_range_b64, addr, offset=0)
                amdgcn.api.ds_write_b64(result_b64, addr, offset=0)

                # B96: data needs 3 registers
                data_range_b96 = amdgcn.api.make_register_range([data1, data2, data3])
                result_b96 = amdgcn.api.ds_read_b96(data_range_b96, addr, offset=4)
                amdgcn.api.ds_write_b96(result_b96, addr, offset=4)

                # B128: data needs 4 registers
                data_range_b128 = amdgcn.api.make_register_range(
                    [data1, data2, data3, data4]
                )
                result_b128 = amdgcn.api.ds_read_b128(data_range_b128, addr, offset=8)
                amdgcn.api.ds_write_b128(result_b128, addr, offset=8)

                amdgcn.EndKernelOp()

        # CHECK:           kernel @vop3p_mai_kernel {
        # CHECK:             %[[VAL_13:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_14:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_15:.*]] = make_register_range %[[VAL_13]], %[[VAL_14]] : !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_16:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_17:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_18:.*]] = make_register_range %[[VAL_16]], %[[VAL_17]] : !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_19:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_20:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_21:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_22:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_23:.*]] = make_register_range %[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_24:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_25:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_26:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_27:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_28:.*]] = make_register_range %[[VAL_24]], %[[VAL_25]], %[[VAL_26]], %[[VAL_27]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_29:.*]] = amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[VAL_28]], %[[VAL_15]], %[[VAL_18]], %[[VAL_23]] : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>
        # CHECK:             %[[VAL_30:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_31:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_32:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_33:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_34:.*]] = make_register_range %[[VAL_30]], %[[VAL_31]], %[[VAL_32]], %[[VAL_33]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_35:.*]] = amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_bf16> %[[VAL_34]], %[[VAL_15]], %[[VAL_18]], %[[VAL_23]] : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>
        with ir.InsertionPoint(amdgcn_mod.body_region.blocks[0]):
            kernel_mai = amdgcn.KernelOp("vop3p_mai_kernel")
            kernel_mai.body_region.blocks.append()

            with ir.InsertionPoint(kernel_mai.body_region.blocks[0]):
                # Allocate registers: A (2), B (2), C (4)
                a1 = amdgcn.api.alloca_vgpr()
                a2 = amdgcn.api.alloca_vgpr()
                a_range = amdgcn.api.make_register_range([a1, a2])

                b1 = amdgcn.api.alloca_vgpr()
                b2 = amdgcn.api.alloca_vgpr()
                b_range = amdgcn.api.make_register_range([b1, b2])

                c1 = amdgcn.api.alloca_vgpr()
                c2 = amdgcn.api.alloca_vgpr()
                c3 = amdgcn.api.alloca_vgpr()
                c4 = amdgcn.api.alloca_vgpr()
                c_range = amdgcn.api.make_register_range([c1, c2, c3, c4])

                # F16 MFMA - allocate destination range
                d1_f16 = amdgcn.api.alloca_vgpr()
                d2_f16 = amdgcn.api.alloca_vgpr()
                d3_f16 = amdgcn.api.alloca_vgpr()
                d4_f16 = amdgcn.api.alloca_vgpr()
                d_range_f16 = amdgcn.api.make_register_range(
                    [d1_f16, d2_f16, d3_f16, d4_f16]
                )

                result_f16 = amdgcn.api.v_mfma_f32_16x16x16_f16(
                    d_range_f16, a_range, b_range, c_range
                )

                # BF16 MFMA - allocate destination range
                d1_bf16 = amdgcn.api.alloca_vgpr()
                d2_bf16 = amdgcn.api.alloca_vgpr()
                d3_bf16 = amdgcn.api.alloca_vgpr()
                d4_bf16 = amdgcn.api.alloca_vgpr()
                d_range_bf16 = amdgcn.api.make_register_range(
                    [d1_bf16, d2_bf16, d3_bf16, d4_bf16]
                )

                result_bf16 = amdgcn.api.v_mfma_f32_16x16x16_bf16(
                    d_range_bf16, a_range, b_range, c_range
                )

                # Store all results to prevent dead-code elimination
                amdgcn.api.global_store_dwordx4(result_f16, a_range, offset=0)
                amdgcn.api.global_store_dwordx4(result_bf16, a_range, offset=0)

                amdgcn.EndKernelOp()

        # CHECK:           kernel @global_all_kernel {
        # CHECK:             %[[VAL_36:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_37:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_38:.*]] = make_register_range %[[VAL_36]], %[[VAL_37]] : !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_39:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_40:.*]] = make_register_range %[[VAL_39]] : !amdgcn.vgpr
        # CHECK:             %[[VAL_41:.*]] = amdgcn.flat.global_load <global_load_dword> %[[VAL_40]], %[[VAL_38]] : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]> -> <[? + 1]>
        # CHECK:             amdgcn.flat.global_store <global_store_dword> %[[VAL_41]], %[[VAL_38]] : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>
        # CHECK:             %[[VAL_42:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_43:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_44:.*]] = make_register_range %[[VAL_42]], %[[VAL_43]] : !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_45:.*]] = amdgcn.flat.global_load <global_load_dwordx2> %[[VAL_44]], %[[VAL_38]] : !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 2]> -> <[? + 2]>
        # CHECK:             amdgcn.flat.global_store <global_store_dwordx2> %[[VAL_45]], %[[VAL_38]] : !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 2]>
        # CHECK:             %[[VAL_46:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_47:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_48:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_49:.*]] = make_register_range %[[VAL_46]], %[[VAL_47]], %[[VAL_48]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_50:.*]] = amdgcn.flat.global_load <global_load_dwordx3> %[[VAL_49]], %[[VAL_38]], offset = 4 : !amdgcn.vgpr_range<[? + 3]>, !amdgcn.vgpr_range<[? + 2]> -> <[? + 3]>
        # CHECK:             amdgcn.flat.global_store <global_store_dwordx3> %[[VAL_50]], %[[VAL_38]], offset = 4 : !amdgcn.vgpr_range<[? + 3]>, !amdgcn.vgpr_range<[? + 2]>
        # CHECK:             %[[VAL_51:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_52:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_53:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_54:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_55:.*]] = make_register_range %[[VAL_51]], %[[VAL_52]], %[[VAL_53]], %[[VAL_54]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_56:.*]] = amdgcn.flat.global_load <global_load_dwordx4> %[[VAL_55]], %[[VAL_38]], offset = 8 : !amdgcn.vgpr_range<[? + 4]>, !amdgcn.vgpr_range<[? + 2]> -> <[? + 4]>
        # CHECK:             amdgcn.flat.global_store <global_store_dwordx4> %[[VAL_56]], %[[VAL_38]], offset = 8 : !amdgcn.vgpr_range<[? + 4]>, !amdgcn.vgpr_range<[? + 2]>
        with ir.InsertionPoint(amdgcn_mod.body_region.blocks[0]):
            kernel_global = amdgcn.KernelOp("global_all_kernel")
            kernel_global.body_region.blocks.append()

            with ir.InsertionPoint(kernel_global.body_region.blocks[0]):
                # Allocate registers
                addr = amdgcn.api.alloca_vgpr()
                addr_hi = amdgcn.api.alloca_vgpr()
                addr_range = amdgcn.api.make_register_range([addr, addr_hi])

                # DWORD: data needs 1 register, addr needs 2 registers
                data1 = amdgcn.api.alloca_vgpr()
                data_range_dword = amdgcn.api.make_register_range([data1])
                result_dword = result_dword = amdgcn.api.global_load_dword(
                    data_range_dword, addr_range, offset=0
                )
                amdgcn.api.global_store_dword(result_dword, addr_range, offset=0)

                # DWORDX2: data needs 2 registers, addr needs 2 registers
                data2 = amdgcn.api.alloca_vgpr()
                data3 = amdgcn.api.alloca_vgpr()
                data_range_dwordx2 = amdgcn.api.make_register_range([data2, data3])
                result_dwordx2 = amdgcn.api.global_load_dwordx2(
                    data_range_dwordx2, addr_range, offset=0
                )
                amdgcn.api.global_store_dwordx2(result_dwordx2, addr_range, offset=0)

                # DWORDX3: data needs 3 registers, addr needs 2 registers
                data4 = amdgcn.api.alloca_vgpr()
                data5 = amdgcn.api.alloca_vgpr()
                data6 = amdgcn.api.alloca_vgpr()
                data_range_dwordx3 = amdgcn.api.make_register_range(
                    [data4, data5, data6]
                )
                result_dwordx3 = amdgcn.api.global_load_dwordx3(
                    data_range_dwordx3, addr_range, offset=4
                )
                amdgcn.api.global_store_dwordx3(result_dwordx3, addr_range, offset=4)

                # DWORDX4: data needs 4 registers, addr needs 2 registers
                data7 = amdgcn.api.alloca_vgpr()
                data8 = amdgcn.api.alloca_vgpr()
                data9 = amdgcn.api.alloca_vgpr()
                data10 = amdgcn.api.alloca_vgpr()
                data_range_dwordx4 = amdgcn.api.make_register_range(
                    [data7, data8, data9, data10]
                )
                result_dwordx4 = amdgcn.api.global_load_dwordx4(
                    data_range_dwordx4, addr_range, offset=8
                )
                amdgcn.api.global_store_dwordx4(result_dwordx4, addr_range, offset=8)

                amdgcn.EndKernelOp()

        # CHECK:           kernel @smem_all_kernel {
        # CHECK:             %[[VAL_57:.*]] = alloca : !amdgcn.sgpr
        # CHECK:             %[[VAL_58:.*]] = alloca : !amdgcn.sgpr
        # CHECK:             %[[VAL_59:.*]] = make_register_range %[[VAL_57]], %[[VAL_58]] : !amdgcn.sgpr, !amdgcn.sgpr
        # CHECK:             %[[VAL_60:.*]] = alloca : !amdgcn.sgpr
        # CHECK:             %[[VAL_63:.*]] = alloca : !amdgcn.sgpr
        # CHECK:             %[[VAL_64:.*]] = alloca : !amdgcn.sgpr
        # CHECK:             %[[VAL_72:.*]] = alloca : !amdgcn.sgpr
        # CHECK:             %[[VAL_73:.*]] = alloca : !amdgcn.sgpr
        # CHECK:             %[[VAL_74:.*]] = alloca : !amdgcn.sgpr
        # CHECK:             %[[VAL_75:.*]] = alloca : !amdgcn.sgpr
        # CHECK:             %[[VAL_61:.*]] = make_register_range %[[VAL_60]] : !amdgcn.sgpr
        # CHECK:             %[[VAL_62:.*]] = amdgcn.smem.load <s_load_dword> %[[VAL_61]], %[[VAL_59]] : !amdgcn.sgpr_range<[? + 1]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.sgpr_range<[? + 1]>
        # CHECK:             amdgcn.smem.store <s_store_dword> %[[VAL_62]], %[[VAL_59]] : <[? + 1]>, <[? + 2]>
        # CHECK:             %[[VAL_65:.*]] = make_register_range %[[VAL_63]], %[[VAL_64]] : !amdgcn.sgpr, !amdgcn.sgpr
        # CHECK:             %[[VAL_66:.*]] = amdgcn.smem.load <s_load_dwordx2> %[[VAL_65]], %[[VAL_59]] : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.sgpr_range<[? + 2]>
        # CHECK:             amdgcn.smem.store <s_store_dwordx2> %[[VAL_66]], %[[VAL_59]] : <[? + 2]>, <[? + 2]>
        # CHECK:             %[[VAL_76:.*]] = make_register_range %[[VAL_72]], %[[VAL_73]], %[[VAL_74]], %[[VAL_75]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
        # CHECK:             %[[VAL_77:.*]] = amdgcn.smem.load <s_load_dwordx4> %[[VAL_76]], %[[VAL_59]] offset = 8 : !amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.sgpr_range<[? + 4]>
        # CHECK:             amdgcn.smem.store <s_store_dwordx4> %[[VAL_77]], %[[VAL_59]], offset = 8 : <[? + 4]>, <[? + 2]>
        # CHECK:             %[[VAL_78:.*]] = alloca : !amdgcn.sgpr
        # CHECK:             %[[VAL_79:.*]] = alloca : !amdgcn.sgpr
        # CHECK:             %[[VAL_80:.*]] = make_register_range %[[VAL_78]], %[[VAL_79]] : !amdgcn.sgpr, !amdgcn.sgpr
        # CHECK:             %[[VAL_81:.*]] = amdgcn.smem.load <s_memtime> %[[VAL_80]] : !amdgcn.sgpr_range<[? + 2]> -> !amdgcn.sgpr_range<[? + 2]>
        with ir.InsertionPoint(amdgcn_mod.body_region.blocks[0]):
            kernel_smem = amdgcn.KernelOp("smem_all_kernel")
            kernel_smem.body_region.blocks.append()

            with ir.InsertionPoint(kernel_smem.body_region.blocks[0]):
                # Allocate registers
                addr = amdgcn.api.alloca_sgpr()
                addr_hi = amdgcn.api.alloca_sgpr()
                addr_range = amdgcn.api.make_register_range([addr, addr_hi])
                data1 = amdgcn.api.alloca_sgpr()
                data2 = amdgcn.api.alloca_sgpr()
                data3 = amdgcn.api.alloca_sgpr()
                data4 = amdgcn.api.alloca_sgpr()
                data5 = amdgcn.api.alloca_sgpr()
                data6 = amdgcn.api.alloca_sgpr()
                data7 = amdgcn.api.alloca_sgpr()

                # DWORD: single register
                data_range_dword = amdgcn.api.make_register_range([data1])
                result_dword = amdgcn.api.s_load_dword(
                    data_range_dword, addr_range, offset=0
                )
                amdgcn.api.s_store_dword(result_dword, addr_range, offset=0)

                # DWORDX2: two registers
                data_range_dwordx2 = amdgcn.api.make_register_range([data2, data3])
                result_dwordx2 = amdgcn.api.s_load_dwordx2(
                    data_range_dwordx2, addr_range, offset=0
                )
                amdgcn.api.s_store_dwordx2(result_dwordx2, addr_range, offset=0)

                # DWORDX4: four registers
                data_range_dwordx4 = amdgcn.api.make_register_range(
                    [data4, data5, data6, data7]
                )
                result_dwordx4 = amdgcn.api.s_load_dwordx4(
                    data_range_dwordx4, addr_range, offset=8
                )
                amdgcn.api.s_store_dwordx4(result_dwordx4, addr_range, offset=8)

                # MEMTIME: read memory clock into 2 SGPRs (no address needed)
                memtime1 = amdgcn.api.alloca_sgpr()
                memtime2 = amdgcn.api.alloca_sgpr()
                memtime_range = amdgcn.api.make_register_range([memtime1, memtime2])
                result_memtime = amdgcn.api.s_memtime(memtime_range, addr=None)
                # store result_memtime to prevent dead-code elimination
                amdgcn.api.s_store_dwordx2(result_memtime, addr_range, offset=0)

                # SOPP operations
                amdgcn.api.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
                amdgcn.api.s_waitcnt(vmcnt=5, expcnt=2, lgkmcnt=1)
                amdgcn.api.s_trap(imm=2)
                amdgcn.api.s_barrier()

                amdgcn.EndKernelOp()

        # CHECK:           kernel @vop2_kernel {
        # CHECK:             %[[VAL_82:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_83:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_84:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_85:.*]] = vop2 v_lshrrev_b32_e32 outs %[[VAL_84]] ins %[[VAL_82]], %[[VAL_83]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_86:.*]] = alloca : !amdgcn.sgpr
        # CHECK:             %[[VAL_87:.*]] = vop2 v_lshrrev_b32_e32 outs %[[VAL_84]] ins %[[VAL_86]], %[[VAL_83]] : !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_88:.*]] = arith.constant 8 : i32
        # CHECK:             %[[VAL_89:.*]] = vop2 v_lshrrev_b32_e32 outs %[[VAL_84]] ins %[[VAL_88]], %[[VAL_83]] : !amdgcn.vgpr, i32, !amdgcn.vgpr
        # CHECK:             %[[VAL_90:.*]] = vop2 v_lshlrev_b32_e32 outs %[[VAL_84]] ins %[[VAL_82]], %[[VAL_83]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_91:.*]] = vop2 v_lshlrev_b32_e32 outs %[[VAL_84]] ins %[[VAL_86]], %[[VAL_83]] : !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr
        # CHECK:             %[[VAL_92:.*]] = vop2 v_lshlrev_b32_e32 outs %[[VAL_84]] ins %[[VAL_88]], %[[VAL_83]] : !amdgcn.vgpr, i32, !amdgcn.vgpr
        with ir.InsertionPoint(amdgcn_mod.body_region.blocks[0]):
            kernel_vop2 = amdgcn.KernelOp("vop2_kernel")
            kernel_vop2.body_region.blocks.append()

            with ir.InsertionPoint(kernel_vop2.body_region.blocks[0]):
                # Allocate registers for VOP2
                src0_vgpr = amdgcn.api.alloca_vgpr()
                vsrc1 = amdgcn.api.alloca_vgpr()
                dst = amdgcn.api.alloca_vgpr()

                # VOP2 lshrrev_b32_e32 with VGPR src0: dst = vsrc1 >> src0
                result1 = amdgcn.api.v_lshrrev_b32_e32(dst, src0_vgpr, vsrc1)

                # VOP2 lshrrev_b32_e32 with SGPR src0
                src0_sgpr = amdgcn.api.alloca_sgpr()
                result2 = amdgcn.api.v_lshrrev_b32_e32(dst, src0_sgpr, vsrc1)

                # VOP2 lshrrev_b32_e32 with immediate src0
                int_type = ir.IntegerType.get_signless(32, ctx)
                imm8 = arith.constant(int_type, 8)
                result3 = amdgcn.api.v_lshrrev_b32_e32(dst, imm8, vsrc1)

                # VOP2 lshlrev_b32_e32 with VGPR src0
                result4 = amdgcn.api.v_lshlrev_b32_e32(dst, src0_vgpr, vsrc1)

                # VOP2 lshlrev_b32_e32 with SGPR src0
                result5 = amdgcn.api.v_lshlrev_b32_e32(dst, src0_sgpr, vsrc1)

                # VOP2 lshlrev_b32_e32 with immediate src0
                result6 = amdgcn.api.v_lshlrev_b32_e32(dst, imm8, vsrc1)

                # Store all results to prevent dead-code elimination
                addr = amdgcn.api.alloca_vgpr()
                result1_range = amdgcn.api.make_register_range([result1])
                result2_range = amdgcn.api.make_register_range([result2])
                result3_range = amdgcn.api.make_register_range([result3])
                result4_range = amdgcn.api.make_register_range([result4])
                result5_range = amdgcn.api.make_register_range([result5])
                result6_range = amdgcn.api.make_register_range([result6])
                amdgcn.api.ds_write_b32(result1_range, addr, offset=0)
                amdgcn.api.ds_write_b32(result2_range, addr, offset=4)
                amdgcn.api.ds_write_b32(result3_range, addr, offset=8)
                amdgcn.api.ds_write_b32(result4_range, addr, offset=12)
                amdgcn.api.ds_write_b32(result5_range, addr, offset=16)
                amdgcn.api.ds_write_b32(result6_range, addr, offset=20)

                amdgcn.EndKernelOp()

        # CHECK:           kernel @vop1_kernel {
        # CHECK:             %[[VAL_93:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_94:.*]] = alloca : !amdgcn.vgpr
        # CHECK:             %[[VAL_95:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[VAL_94]], %[[VAL_93]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
        # CHECK:             %[[VAL_96:.*]] = alloca : !amdgcn.sgpr
        # CHECK:             %[[VAL_97:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[VAL_94]], %[[VAL_96]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
        # CHECK:             %[[VAL_98:.*]] = arith.constant 42 : i32
        # CHECK:             %[[VAL_99:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[VAL_94]], %[[VAL_98]] : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
        with ir.InsertionPoint(amdgcn_mod.body_region.blocks[0]):
            kernel_vop1 = amdgcn.KernelOp("vop1_kernel")
            kernel_vop1.body_region.blocks.append()

            with ir.InsertionPoint(kernel_vop1.body_region.blocks[0]):
                # Allocate registers for VOP1
                src0_vgpr = amdgcn.api.alloca_vgpr()
                dst = amdgcn.api.alloca_vgpr()

                # VOP1 mov_b32_e32 with VGPR src0: dst = src0
                result1 = amdgcn.api.v_mov_b32_e32(dst, src0_vgpr)

                # VOP1 mov_b32_e32 with SGPR src0
                src0_sgpr = amdgcn.api.alloca_sgpr()
                result2 = amdgcn.api.v_mov_b32_e32(dst, src0_sgpr)

                # VOP1 mov_b32_e32 with immediate src0
                int_type = ir.IntegerType.get_signless(32, ctx)
                imm42 = arith.constant(int_type, 42)
                result3 = amdgcn.api.v_mov_b32_e32(dst, imm42)

                # Store all results to prevent dead-code elimination
                addr = amdgcn.api.alloca_vgpr()
                result1_range = amdgcn.api.make_register_range([result1])
                result2_range = amdgcn.api.make_register_range([result2])
                result3_range = amdgcn.api.make_register_range([result3])
                amdgcn.api.ds_write_b32(result1_range, addr, offset=0)
                amdgcn.api.ds_write_b32(result2_range, addr, offset=4)
                amdgcn.api.ds_write_b32(result3_range, addr, offset=8)

                amdgcn.EndKernelOp()

    module.verify()
    return module


if __name__ == "__main__":
    with ir.Context() as ctx, ir.Location.unknown():
        module = build_test_module(ctx)
        print(str(module))
