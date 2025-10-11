"""Compatibility helpers for package initialization."""
from __future__ import annotations

import os


def ensure_openmp_compat() -> None:
    """Allow environments with conflicting OpenMP runtimes to run.

    Some optional dependencies (notably PyTorch) ship their own OpenMP
    implementation. When those wheels are combined with the system runtime the
    interpreter aborts during import with ``OMP: Error #15``. The upstream
    recommendation is to make sure only a single runtime is loaded, but that is
    not always feasible for end users executing the CLI. The most reliable
    mitigation is to set ``KMP_DUPLICATE_LIB_OK`` before any of the affected
    libraries are imported.
    """

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
