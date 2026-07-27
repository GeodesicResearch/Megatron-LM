# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers for interacting with the experimental nvidia-resiliency-ext API."""

from importlib import import_module
from typing import Any, Callable, Dict

try:
    from packaging.version import Version as PkgVersion

    HAVE_PACKAGING = True
except ImportError:
    HAVE_PACKAGING = False

NVRX_MIN_VERSION = "0.6.0"


def has_nvrx_async_support() -> bool:
    """Checks whether the NVRx async checkpointing symbols Megatron uses are importable."""
    try:
        core = import_module("nvidia_resiliency_ext.checkpointing.async_ckpt.core")
        cached_metadata_reader = import_module(
            "nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader"
        )
        filesystem_async = import_module(
            "nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async"
        )
        state_dict_saver = import_module(
            "nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver"
        )
    except (ImportError, ModuleNotFoundError):
        return False

    required_symbols = (
        getattr(core, "AsyncCallsQueue", None),
        getattr(core, "AsyncRequest", None),
        getattr(cached_metadata_reader, "CachedMetadataFileSystemReader", None),
        getattr(filesystem_async, "FileSystemWriterAsync", None),
        getattr(filesystem_async, "get_write_results_queue", None),
        getattr(state_dict_saver, "CheckpointMetadataCache", None),
        getattr(state_dict_saver, "save_state_dict_async_finalize", None),
        getattr(state_dict_saver, "save_state_dict_async_plan", None),
    )
    # Geodesic/Isambard: a version below the async-checkpointing minimum must mean
    # "async support unavailable", not a crash at import time. This helper runs at
    # MODULE SCOPE of strategies/torch.py (HAVE_NVRX = has_nvrx_async_support()), which
    # sits on the import path of megatron.core.dist_checkpointing and therefore of
    # essentially everything. With nvidia-resiliency-ext 0.4.1 in the frozen NGC image
    # BOTH failure modes are real: the original bare `assert is_nvrx_min_version()`
    # raised AssertionError, and is_nvrx_min_version() itself raises AttributeError
    # first — 0.4.1 is a namespace package with no __version__ attribute. We never
    # enable async_save, so False is the correct capability answer either way,
    # matching how the import failures just above are handled.
    try:
        if not is_nvrx_min_version():
            return False
    except Exception:
        return False

    return all(symbol is not None for symbol in required_symbols) and hasattr(
        filesystem_async, "_results_queue"
    )


def make_nvrx_async_request(
    async_request_cls: type,
    async_fn: Callable[..., Any],
    async_fn_args: Any,
    finalize_fns: list[Callable[..., Any]],
    async_fn_kwargs: Dict[str, Any] | None = None,
    preload_fn: Callable[..., Any] | None = None,
):
    """Builds an AsyncRequest using the expected NVRx API."""
    return async_request_cls(
        async_fn,
        async_fn_args,
        finalize_fns,
        async_fn_kwargs=async_fn_kwargs or {},
        preload_fn=preload_fn,
    )


def is_nvrx_min_version(version: str = NVRX_MIN_VERSION) -> bool:
    """Check if minimum version of `NVRx` is installed."""
    if not HAVE_PACKAGING:
        raise ImportError(
            "packaging is not installed. Please install it with `pip install packaging`."
        )

    try:
        import nvidia_resiliency_ext as nvrx

        HAVE_NVRX = True
    except (ImportError, ModuleNotFoundError):
        HAVE_NVRX = False

    nvrx_version = str(nvrx.__version__) if HAVE_NVRX else "0.0.0"

    return PkgVersion(nvrx_version) >= PkgVersion(version)
