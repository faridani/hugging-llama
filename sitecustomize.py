"""Test helper to ensure local package import works.

This project keeps its source code inside the ``src`` directory, which means
Python needs that directory on ``sys.path`` to import the ``ollama_local``
package without an editable install.  The test environment imports the module
directly, so we insert the directory at interpreter start-up by providing a
``sitecustomize`` module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


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


_ensure_src_on_path()

