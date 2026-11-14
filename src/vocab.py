"""Pre-computed views of the vocabulary used by constrained decoding.

Everything here is computed once per run, so that each generation step
only needs numpy masks and a few short lists.
"""

from collections import defaultdict

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from .automata import STR_NORMAL, scan_string
from .tokenizer import BPETokenizer

BoolArray = npt.NDArray[np.bool_]


class VocabIndex(BaseModel):
    """Token texts plus the indexes the decoder relies on."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    tokenizer: BPETokenizer

    def __init__(self, tokenizer: BPETokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        size = tokenizer.vocab_size
        self._size = size
        self._texts: list[str | None] = [
            tokenizer.token_text(i) for i in range(size)
        ]
        self._text_to_id: dict[str, int] = {}
        self._first: dict[str, list[int]] = defaultdict(list)
        plain = np.zeros(size, dtype=bool)
        special: list[int] = []
        for token_id, text in enumerate(self._texts):
            if not text:
                continue
            self._text_to_id.setdefault(text, token_id)
            self._first[text[0]].append(token_id)
            if any(ord(c) < 0x20 for c in text):
                continue  # raw control characters are illegal in strings
            if '"' in text or "\\" in text:
                special.append(token_id)
            else:
                plain[token_id] = True
        #: tokens that are pure string content (no quote, no backslash)
        self._plain_string: BoolArray = plain
        #: tokens containing a quote or a backslash (need a real scan)
        self._string_special: list[int] = special
        self._normal_scan: list[tuple[int, int, int | None, str]] | None
        self._normal_scan = None
        self._starting_cache: dict[str, list[int]] = {}

    @property
    def size(self) -> int:
        """Public read access to the precomputed ``size`` table."""
        return self._size

    @property
    def texts(self) -> list[str | None]:
        """Public read access to the precomputed ``texts`` table."""
        return self._texts

    @property
    def text_to_id(self) -> dict[str, int]:
        """Public read access to the precomputed ``text_to_id`` table."""
        return self._text_to_id

    @property
    def plain_string(self) -> BoolArray:
        """Public read access to the precomputed ``plain_string`` table."""
        return self._plain_string

    @property
    def string_special(self) -> list[int]:
        """Public read access to the precomputed ``string_special`` table."""
        return self._string_special

    def text(self, token_id: int) -> str:
        """Text of *token_id* ('' for unusable ids)."""
        if 0 <= token_id < self._size:
            return self._texts[token_id] or ""
        return ""

    def tokens_starting(self, chars: str) -> list[int]:
        """Ids of the tokens whose first character is in *chars*."""
        cached = self._starting_cache.get(chars)
        if cached is None:
            found: list[int] = []
            for char in dict.fromkeys(chars):
                found.extend(self._first.get(char, ()))
            cached = sorted(set(found))
            self._starting_cache[chars] = cached
        return cached

    def normal_string_scan(self) -> list[tuple[int, int, int | None, str]]:
        """Scan of ``string_special`` tokens from the normal state.

        Items are ``(token_id, state_after, close_index, text)``.  The
        result never changes, so it is computed once and cached.
        """
        if self._normal_scan is None:
            hits: list[tuple[int, int, int | None, str]] = []
            for token_id in self._string_special:
                text = self.text(token_id)
                res = scan_string(STR_NORMAL, text)
                if res is not None:
                    hits.append((token_id, res[0], res[1], text))
            self._normal_scan = hits
        return self._normal_scan
