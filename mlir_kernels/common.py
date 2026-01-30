"""Common utilities for mlir_kernels tests, benchmarks, and nanobenchmarks."""

import os

_MLIR_KERNELS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_library_paths():
    """Get paths to all required library files."""
    library_dir = os.path.join(_MLIR_KERNELS_DIR, "library", "common")
    return [
        os.path.join(library_dir, "register-init.mlir"),
        os.path.join(library_dir, "indexing.mlir"),
        os.path.join(library_dir, "simple-copies.mlir"),
        os.path.join(library_dir, "copies.mlir"),
        os.path.join(library_dir, "multi-tile-copies.mlir"),
        os.path.join(library_dir, "conditional-multi-tile-copies.mlir"),
        os.path.join(library_dir, "simple-multi-tile-copies.mlir"),
        os.path.join(library_dir, "futures.mlir"),
    ]
