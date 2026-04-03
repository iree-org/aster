"""GPU availability utilities."""

import re
import subprocess


def system_has_gpu(mcpu: str) -> bool:
    """Check if a GPU matching mcpu is available via rocminfo.

    Does NOT import aster/MLIR/LLVM. This is the single canonical
    implementation — system_has_mcpu is an alias for this function.
    """
    import shutil

    base_mcpu = mcpu.split(":")[0]
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        print(
            "WARNING: rocminfo not found on PATH. "
            "Install ROCm or add its bin/ directory to PATH."
        )
        return False
    except subprocess.TimeoutExpired:
        rocminfo_path = shutil.which("rocminfo")
        print(f"WARNING: rocminfo timed out after 30s (path: {rocminfo_path}).")
        return False

    if result.returncode != 0:
        print(f"WARNING: rocminfo exited with code {result.returncode}.")
        return False

    raw_matches = re.findall(r"gfx[0-9]{3,4}[a-z0-9]*", result.stdout)
    archs = set(a.split(":")[0] for a in raw_matches)
    return base_mcpu in archs


def system_has_mcpu(mcpu: str) -> bool:
    """Alias for system_has_gpu; checks GPU availability via rocminfo."""
    return system_has_gpu(mcpu)
