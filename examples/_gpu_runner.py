#!/usr/bin/env python3
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""GPU execution subprocess -- avoids LLVM symbol conflicts with ROCm.

Called by common.execute_or_skip with LD_LIBRARY_PATH set so ROCm's HIP runtime is
findable. This process does NOT import ASTER's compiler (which bundles its own LLVM),
only the execution module.
"""

import json
import sys

import numpy as np

from aster.execution.core import InputArray, OutputArray, execute_hsaco
from aster.execution.helpers import hsaco_file

config = json.loads(sys.argv[1])

args = []
for f in config.get("input_files", []):
    args.append(InputArray(np.load(f)))

output_arrays = []
for f in config.get("output_files", []):
    arr = np.load(f, allow_pickle=False)
    output_arrays.append(arr)
    args.append(OutputArray(arr))

with hsaco_file(config["hsaco_path"]):
    times = execute_hsaco(
        hsaco_path=config["hsaco_path"],
        kernel_name=config["kernel_name"],
        arguments=args,
        grid_dim=tuple(config["grid"]),
        block_dim=tuple(config["block"]),
    )

for f, arr in zip(config.get("output_files", []), output_arrays):
    np.save(f, arr)

print(json.dumps({"times": times}))
