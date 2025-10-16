"""Compatibility helpers for package initialization."""
from __future__ import annotations

import asyncio
import os
import sys


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


def ensure_asyncio_compat() -> None:
    """Ensure asyncio works consistently across platforms.

    Windows defaults to the ``ProactorEventLoop`` policy which has known
    compatibility issues with libraries that rely on selector-based features.
    The HTTP streaming used by the CLI hangs under that policy. Switching to
    ``WindowsSelectorEventLoopPolicy`` mirrors the behaviour on Unix platforms
    and restores streaming support.
    """

    if not sys.platform.startswith("win"):
        return

    policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_cls is None:
        return

    current_policy = asyncio.get_event_loop_policy()
    if isinstance(current_policy, policy_cls):
        return

    asyncio.set_event_loop_policy(policy_cls())
