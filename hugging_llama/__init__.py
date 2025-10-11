"""Compatibility wrapper to allow importing without installing the package."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_CURRENT_DIR = Path(__file__).resolve().parent
_SRC_PACKAGE = _CURRENT_DIR.parent / "src" / "hugging_llama"

if not _SRC_PACKAGE.is_dir():  # pragma: no cover - defensive guard
    raise ModuleNotFoundError(
        "Expected to find the hugging_llama sources in the src directory."
    )


def _load_src_package() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        __name__,
        _SRC_PACKAGE / "__init__.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError("Unable to locate hugging_llama source package")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_src_package()

# Allow importing submodules such as ``hugging_llama.server``.
__path__ = [str(_SRC_PACKAGE)]

_EXCLUDED_EXPORTS = {
    "__name__",
    "__loader__",
    "__package__",
    "__spec__",
    "__file__",
    "__cached__",
}

for _key, _value in vars(_MODULE).items():
    if _key in _EXCLUDED_EXPORTS:
        continue
    globals()[_key] = _value

__doc__ = _MODULE.__doc__
__all__ = getattr(_MODULE, "__all__", None)  # type: ignore[assignment]

# Avoid leaking temporary globals.
del _key, _value, _MODULE, _EXCLUDED_EXPORTS

