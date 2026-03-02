# -*- Python -*-
#
# Configuration for ASTER C++ unit tests.
# Adapted from LLVM/MLIR upstream (mlir/test/Unit/lit.cfg.py).
# Lit discovers and runs Google Test executables in the Unit output directory.

import os

import lit.formats

# name: The name of this test suite.
config.name = "ASTER-Unit"

# suffixes: A list of file extensions to treat as test files.
# Empty for GoogleTest format - lit discovers executables, not source files.
config.suffixes = []

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests (executables) should be run.
# Matches MLIR layout: unittests/ at top level, executables in build/unittests/
config.test_exec_root = os.path.join(config.aster_obj_root, "unittests")
config.test_source_root = config.test_exec_root

# testFormat: The test format to use - GoogleTest discovers and runs gtest binaries.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Tests")

# Propagate environment for tests.
if "TMP" in os.environ:
    config.environment["TMP"] = os.environ["TMP"]
if "TEMP" in os.environ:
    config.environment["TEMP"] = os.environ["TEMP"]
if "HOME" in os.environ:
    config.environment["HOME"] = os.environ["HOME"]

# Propagate sanitizer options.
for var in [
    "ASAN_SYMBOLIZER_PATH",
    "HWASAN_SYMBOLIZER_PATH",
    "MSAN_SYMBOLIZER_PATH",
    "TSAN_SYMBOLIZER_PATH",
    "UBSAN_SYMBOLIZER_PATH",
    "ASAN_OPTIONS",
    "HWASAN_OPTIONS",
    "MSAN_OPTIONS",
    "TSAN_OPTIONS",
    "UBSAN_OPTIONS",
]:
    if var in os.environ:
        config.environment[var] = os.environ[var]
