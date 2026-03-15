#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._lsir_ops_gen import *
from ._ods_common import _cext as _ods_cext

_ods_ir = _ods_cext.ir


def to_reg(result_type, operand, *, loc=None, ip=None):
    """lsir.to_reg: bridge from standard MLIR types into ASTER register types."""
    return ToRegOp(result_type, operand, loc=loc, ip=ip).result


def from_reg(result_type, operand, *, loc=None, ip=None):
    """lsir.from_reg: bridge from ASTER register types to standard MLIR types."""
    return FromRegOp(result_type, operand, loc=loc, ip=ip).result
