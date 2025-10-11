"""Test helper to ensure local package import works.

This project keeps its source code inside the ``src`` directory, which means
Python needs that directory on ``sys.path`` to import the ``hugging_llama``
package without an editable install.  The test environment imports the module
directly, so we insert the directory at interpreter start-up by providing a
``sitecustomize`` module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_openmp_compat() -> None:
    """Allow environments with conflicting OpenMP runtimes to run.

    Some optional dependencies (notably PyTorch) ship their own OpenMP
    implementation.  When those wheels are combined with the system runtime the
    interpreter aborts during import with ``OMP: Error #15``.  The upstream
    recommendation is to make sure only a single runtime is loaded, but that is
    not always feasible for end users executing the CLI.  The most reliable
    mitigation is to set ``KMP_DUPLICATE_LIB_OK`` before any of the affected
    libraries are imported.  ``sitecustomize`` runs early in interpreter start-up,
    giving us a central place to apply the workaround so commands like
    ``hugging-llama ps`` continue working out of the box.
    """

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _ensure_src_on_path() -> None:
    """Insert the repository ``src`` directory at the front of ``sys.path``.

    ``sitecustomize`` is imported automatically after ``site`` during
    interpreter start-up.  By adding this file we make sure ``src`` is
    available for imports even when the project is not installed as a package.
    ``sys.path`` may already contain the directory, but ``insert`` will move it
    to the front, preserving import semantics while avoiding duplicates.
    """

    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"

    if src_dir.is_dir():
        src_path = os.fspath(src_dir)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        else:
            # Move the existing entry to the front for consistency.
            sys.path.remove(src_path)
            sys.path.insert(0, src_path)


_ensure_openmp_compat()
_ensure_src_on_path()

