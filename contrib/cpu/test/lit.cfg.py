# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

import lit.formats
from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "ASTER-X86"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]
if config.aster_python_enabled.lower() == "on":
    config.suffixes += [".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.aster_obj_root, "contrib", "cpu", "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = [
    "lit.cfg.py",
    "lit.site.cfg.py",
    "__init__.py",
    "integration",
]

config.aster_tools_dir = os.path.join(config.aster_obj_root, "bin")

# Allow LLVM_TOOLS_DIR to be overridden from environment for sandboxed runs.
if os.environ.get("LLVM_TOOLS_DIR"):
    config.llvm_tools_dir = os.environ.get("LLVM_TOOLS_DIR")

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PYTHONPATH", config.aster_python_root, append_path=True)
config.environment["LLVM_TOOLS_DIR"] = config.llvm_tools_dir
config.environment["FILECHECK_OPTS"] = "--dump-input=fail"

tool_dirs = [config.aster_tools_dir, config.llvm_tools_dir]
tools = ["aster-cpu-opt", "aster-cpu-translate"]
llvm_config.add_tool_substitutions(tools, tool_dirs)

config.substitutions.append(("%llvm_mc", config.aster_llvm_mc))
config.substitutions.append(("%llvm_objdump", config.aster_llvm_objdump))

sys.path.append(config.aster_python_root)
