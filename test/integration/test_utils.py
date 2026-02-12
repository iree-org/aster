"""Backward-compatibility shim for integration test imports.

All testing utilities have moved to aster.testing. This file only re-exports FlushLLC
(which lives in integration.flush_llc and depends on HIP runtime bindings that may not
always be available).
"""

from aster.testing.flush_llc import FlushLLC
