# ruff: noqa: F821
#
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil

import lit.formats
from lit.llvm import llvm_config

config.name = "ASTER-STINKYTOFU"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# .stir is the StinkyTofu textual IR suffix.
config.suffixes = [".stir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(
    config.aster_obj_root, "contrib", "stinkytofu", "test"
)

config.excludes = ["lit.cfg.py", "lit.site.cfg.py"]

llvm_config.use_default_substitutions()
llvm_config.with_system_environment(["HOME", "TMP", "TEMP"])
config.environment["FILECHECK_OPTS"] = "--dump-input=fail"

# stinkytofu-opt may live in the install bin or the contrib build tree.
candidate_dirs = [
    os.path.join(config.aster_obj_root, "bin"),
    os.path.join(
        config.aster_obj_root,
        "contrib",
        "stinkytofu",
        "stinkytofu",
        "tools",
        "stinkytofu-opt",
    ),
]
tool_dirs = [d for d in candidate_dirs if os.path.isdir(d)] + [config.llvm_tools_dir]

# Guard: if the tool was not built (option OFF or build skipped), mark the
# feature absent so REQUIRES-gated tests are UNSUPPORTED, not ERROR. Only the
# build/install dirs are authoritative -- do not fall back to a PATH lookup.
if any(shutil.which("stinkytofu-opt", path=d) for d in candidate_dirs):
    config.available_features.add("stinkytofu-opt")

llvm_config.add_tool_substitutions(["stinkytofu-opt"], tool_dirs)
