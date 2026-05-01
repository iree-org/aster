#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from aster.dialects._pattern_ops_gen import *  # noqa: F403
from aster.dialects._ods_common import _cext as _ods_cext

_ods_ir = _ods_cext.ir
