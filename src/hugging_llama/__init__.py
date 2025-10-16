"""hugging-llama package."""
from __future__ import annotations

from ._compat import ensure_asyncio_compat, ensure_openmp_compat

ensure_openmp_compat()
ensure_asyncio_compat()

__all__ = ["ensure_openmp_compat", "ensure_asyncio_compat"]
