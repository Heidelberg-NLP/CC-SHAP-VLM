"""Vision-language model adapters and the factory that loads one by name."""

from __future__ import annotations

from .base import Prompt, SamplePredictions, VLMAdapter

_LLAVA_FAMILY = ("bakllava", "llava_mistral", "llava_vicuna")


def load_adapter(name: str) -> VLMAdapter:
    if "mplug" in name:
        from .mplug_owl3 import MplugOwl3Adapter
        return MplugOwl3Adapter.load(name)
    if name in _LLAVA_FAMILY:
        from .llava import LlavaFamilyAdapter
        return LlavaFamilyAdapter.load(name)
    raise NotImplementedError(f"Model {name!r} not implemented yet.")


__all__ = ["Prompt", "SamplePredictions", "VLMAdapter", "load_adapter"]
