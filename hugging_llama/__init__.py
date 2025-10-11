"""Compatibility wrapper to allow importing without installing the package."""

from __future__ import annotations

from pathlib import Path


_CURRENT_DIR = Path(__file__).resolve().parent
_SRC_PACKAGE = _CURRENT_DIR.parent / "src" / "hugging_llama"

if not _SRC_PACKAGE.is_dir():  # pragma: no cover - defensive guard
    raise ModuleNotFoundError(
        "Expected to find the hugging_llama sources in the src directory."
    )

# Allow importing submodules such as ``hugging_llama.server``.
__path__ = [str(_SRC_PACKAGE)]

# Execute the real package ``__init__`` in this module's namespace so that all
# exports are preserved.  This mirrors the behaviour of installing the package
# in editable mode while keeping the repository layout unchanged.
with (_SRC_PACKAGE / "__init__.py").open("rb") as _init_file:
    exec(compile(_init_file.read(), str(_SRC_PACKAGE / "__init__.py"), "exec"))

