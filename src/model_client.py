"""Thin cached wrapper around the SDK's ``get_logits_from_input_ids``."""

from collections import OrderedDict
from typing import Any, Protocol, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from .errors import ModelError

FloatArray = npt.NDArray[np.float32]


class LogitsModel(Protocol):
    """The only SDK method the decoding loop needs."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return the next-token logits for *input_ids*."""


class CachedModel(BaseModel):
    """LRU cache keyed by the full token sequence.

    The SDK has no KV-cache, so identical contexts (retries, repeated
    prompts, the "decide" step after a number) are served from memory.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    model: Any
    max_entries: int = 64

    def __init__(self, model: LogitsModel, max_entries: int = 64) -> None:
        super().__init__(model=model, max_entries=max_entries)
        self._model = model
        self._max = max(1, max_entries)
        self._cache: OrderedDict[tuple[int, ...], FloatArray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def logits(self, ids: Sequence[int]) -> FloatArray:
        """Next-token logits for *ids* as a float32 numpy array."""
        key = tuple(ids)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return cached
        self.misses += 1
        try:
            raw = self._model.get_logits_from_input_ids(list(key))
        except Exception as exc:
            raise ModelError(f"the model failed: {exc}") from exc
        array: FloatArray = np.asarray(raw, dtype=np.float32)
        self._cache[key] = array
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return array
