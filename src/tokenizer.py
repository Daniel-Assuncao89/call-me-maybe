"""Byte-level BPE tokenizer rebuilt from ``vocab.json`` + ``merges.txt``.

Bonus part: this module re-implements ``encode`` and ``decode`` so the
main code does not depend on the SDK tokenizer.  The algorithm is the
GPT-2 / Qwen2 one:

1. normalise to NFC;
2. split the text into "pre-tokens" with the Qwen2 regular expression;
3. map every byte of a pre-token to a printable unicode character;
4. apply the BPE merges by increasing rank;
5. look every resulting piece up in the vocabulary.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import ModelError, TokenizerError

# Python's ``re`` has no \p{L} / \p{N}; the classes below are equivalent
# for letters ([^\W\d_]) and decimal digits (\d).  An optional leading
# character is any char that is not CR/LF, a letter or a digit.
_PRETOKEN = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|(?:(?![\r\n])[\W_])?[^\W\d_]+"
    r"|\d"
    r"| ?(?:(?!\s)[\W_])+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 table mapping every byte to a printable unicode character."""
    kept = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table: dict[int, str] = {}
    extra = 0
    for byte in range(256):
        if byte in kept:
            table[byte] = chr(byte)
        else:
            table[byte] = chr(256 + extra)
            extra += 1
    return table


def pretokenize(text: str) -> list[str]:
    """Split *text* into pre-tokens (never merged across boundaries)."""
    return _PRETOKEN.findall(text)


class VocabData(BaseModel):
    """Validated content of ``vocab.json`` and ``merges.txt``."""

    model_config = ConfigDict(frozen=True)

    vocab: dict[str, int]
    merges: list[tuple[str, str]] = Field(default_factory=list)

    @field_validator("vocab")
    @classmethod
    def _check_vocab(cls, value: dict[str, int]) -> dict[str, int]:
        if not value:
            raise ValueError("the vocabulary is empty")
        if min(value.values()) < 0:
            raise ValueError("negative token id in the vocabulary")
        return value


def load_vocab_data(
    vocab_path: str | Path, merges_path: str | Path | None
) -> VocabData:
    """Read the tokenizer files from disk."""
    try:
        with open(vocab_path, "r", encoding="utf-8") as handle:
            vocab = json.load(handle)
        merges: list[tuple[str, str]] = []
        if merges_path is not None:
            with open(merges_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.rstrip("\n")
                    if not line or line.startswith("#version"):
                        continue
                    parts = line.split(" ")
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
        return VocabData(vocab=vocab, merges=merges)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise TokenizerError(f"cannot load tokenizer files: {exc}") from exc


class BPETokenizer(BaseModel):
    """Public ``encode`` / ``decode`` plus token-level helpers."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    data: VocabData

    def __init__(self, data: VocabData, **fields: Any) -> None:
        super().__init__(data=data, **fields)
        self._token_to_id: dict[str, int] = dict(data.vocab)
        self._id_to_token: dict[int, str] = {
            i: t for t, i in data.vocab.items()
        }
        self._ranks: dict[tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(data.merges)
        }
        self._byte_to_char = bytes_to_unicode()
        self._char_to_byte = {c: b for b, c in self._byte_to_char.items()}
        self._bpe_cache: dict[str, tuple[str, ...]] = {}
        self._size = max(self._id_to_token) + 1

    @property
    def vocab_size(self) -> int:
        """Number of ids (highest id + 1)."""
        return self._size

    # -- token level -----------------------------------------------------
    def token_bytes(self, token_id: int) -> bytes | None:
        """Raw bytes of a token, or None for unknown ids."""
        token = self._id_to_token.get(token_id)
        if token is None:
            return None
        try:
            return bytes(self._char_to_byte[c] for c in token)
        except KeyError:
            return None

    def token_text(self, token_id: int) -> str | None:
        """Text of a token, or None when it is not valid UTF-8 on its own.

        Byte-level BPE can split a multi-byte character in several
        tokens; such fragments cannot be used for text constraints.
        """
        raw = self.token_bytes(token_id)
        if raw is None:
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return None

    # -- text level --------------------------------------------------------
    def _bpe(self, word: str) -> tuple[str, ...]:
        cached = self._bpe_cache.get(word)
        if cached is not None:
            return cached
        parts = list(word)
        while len(parts) > 1:
            best_rank = -1
            best_pair: tuple[str, str] | None = None
            for pair in zip(parts, parts[1:]):
                rank = self._ranks.get(pair)
                if rank is not None and (
                    best_pair is None or rank < best_rank
                ):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            merged: list[str] = []
            i = 0
            while i < len(parts):
                if (
                    i < len(parts) - 1
                    and (parts[i], parts[i + 1]) == best_pair
                ):
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        result = tuple(parts)
        self._bpe_cache[word] = result
        return result

    def encode(self, text: str) -> list[int]:
        """Convert *text* to token ids (no special tokens are added)."""
        ids: list[int] = []
        for piece in pretokenize(unicodedata.normalize("NFC", text)):
            mapped = "".join(
                self._byte_to_char[b] for b in piece.encode("utf-8")
            )
            for token in self._bpe(mapped):
                token_id = self._token_to_id.get(token)
                if token_id is not None:
                    ids.append(token_id)
                    continue
                for char in token:  # defensive: fall back to single bytes
                    single = self._token_to_id.get(char)
                    if single is None:
                        raise TokenizerError(
                            f"character {char!r} is not in the vocabulary"
                        )
                    ids.append(single)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Convert token ids back to text (unknown ids are skipped)."""
        raw = bytearray()
        for token_id in ids:
            chunk = self.token_bytes(int(token_id))
            if chunk is not None:
                raw.extend(chunk)
        return raw.decode("utf-8", errors="replace")


class SdkTokenizer(BPETokenizer):
    """Same token table, but ``encode``/``decode`` delegate to the SDK.

    Used with ``--use-sdk-tokenizer`` to compare both implementations.
    """

    sdk_model: Any = None

    def __init__(self, data: VocabData, model: Any) -> None:
        super().__init__(data, sdk_model=model)

    def encode(self, text: str) -> list[int]:
        """Encode through the SDK (public method of ``llm_sdk``)."""
        tensor = self.sdk_model.encode(text)
        return [int(i) for i in tensor[0].tolist()]

    def decode(self, ids: Sequence[int]) -> str:
        """Decode through the SDK (public method of ``llm_sdk``)."""
        return str(self.sdk_model.decode([int(i) for i in ids]))


def build_tokenizer(model: Any, use_sdk: bool = False) -> BPETokenizer:
    """Build a tokenizer from the files the SDK points to."""
    try:
        vocab_path = model.get_path_to_vocab_file()
        merges_path = model.get_path_to_merges_file()
    except Exception as exc:  # network, missing files, ...
        raise ModelError(f"cannot locate tokenizer files: {exc}") from exc
    data = load_vocab_data(vocab_path, merges_path)
    if use_sdk:
        return SdkTokenizer(data, model)
    return BPETokenizer(data)
