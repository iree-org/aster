# -*- Python -*-

import os
import sys

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "ASTER"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)


# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]
if config.aster_python_enabled.lower() == "on":
    config.suffixes += [".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.aster_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.extend(
    [
        ("%PYTHON", sys.executable),
        ("%llvm_shlibdir", config.llvm_lib_dir),
        ("%llvm_obj_root", config.llvm_obj_root),
    ]
)
llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite.
config.excludes = [
    "lit.cfg.py",
    "lit.site.cfg.py",
    "__init__.py",
    "integration",
]

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.aster_obj_root, "test")
config.aster_tools_dir = os.path.join(config.aster_obj_root, "bin")
config.aster_libs_dir = os.path.join(config.aster_obj_root, "lib")

config.substitutions.append(("%aster_libs", config.aster_libs_dir))

# Allow LLVM_TOOLS_DIR to be overridden from environment
if os.environ.get("LLVM_TOOLS_DIR"):
    config.llvm_tools_dir = os.environ.get("LLVM_TOOLS_DIR")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PYTHONPATH", config.aster_python_root, append_path=True)
config.environment["LLVM_TOOLS_DIR"] = config.llvm_tools_dir
config.environment["FILECHECK_OPTS"] = "--dump-input=fail"

tool_dirs = [config.aster_tools_dir, config.llvm_tools_dir]
tools = ["aster-opt", "aster-translate"]

llvm_config.add_tool_substitutions(tools, tool_dirs)

sys.path.append(config.aster_python_root)
