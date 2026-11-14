"""Test doubles: a toy byte-level BPE and a scripted "oracle" LLM."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tokenizer import BPETokenizer, VocabData, bytes_to_unicode

WORDS = [
    "fn_", "fn_add_numbers", "fn_greet", "fn_reverse_string",
    "fn_get_square_root", "fn_flag", "fn_list", "fn_nested", "fn_pick",
    "name", "parameters", '", "', '": ', '{"', '"}', "}}", '",', ' "',
    ", ", "true", "false", "null", " hello", "hello", " world", "world",
    "Answer", "Request", ": ", "\n", "  ", "é", "🙂", "h",
]
MARKER = 'Answer: {"name": "'


def build_toy_vocab(words: list[str]) -> VocabData:
    """Byte vocabulary + one merge chain per word."""
    table = bytes_to_unicode()
    vocab: dict[str, int] = {table[b]: b for b in range(256)}
    merges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for word in words:
        mapped = "".join(table[b] for b in word.encode("utf-8"))
        for k in range(1, len(mapped)):
            pair = (mapped[:k], mapped[k])
            if pair in seen:
                continue
            seen.add(pair)
            merges.append(pair)
            vocab.setdefault(mapped[: k + 1], len(vocab))
    return VocabData(vocab=vocab, merges=merges)


class _Row:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def tolist(self) -> list[int]:
        return list(self._ids)


class _Tensor:
    def __init__(self, ids: list[int]) -> None:
        self._row = _Row(ids)

    def __getitem__(self, index: int) -> _Row:
        return self._row


class FakeLLM:
    """Mimics the public API of ``Small_LLM_Model``.

    ``scripts`` maps a request text to the continuation the "model"
    wants to write after ``Answer: {"name": "``.  The model is greedy and
    pushes towards that text with a high logit; everything else gets
    small deterministic noise, so when the constraint blocks the wanted
    token the choice among the remaining ones is arbitrary but stable.
    """

    def __init__(self, tmp: Path, scripts: dict[str, str]) -> None:
        data = build_toy_vocab(WORDS)
        self.tokenizer = BPETokenizer(data)
        tmp.mkdir(parents=True, exist_ok=True)
        self._vocab_path = tmp / "vocab.json"
        self._merges_path = tmp / "merges.txt"
        self._vocab_path.write_text(json.dumps(dict(data.vocab)))
        self._merges_path.write_text(
            "#version: 0.2\n"
            + "\n".join(f"{a} {b}" for a, b in data.merges)
            + "\n"
        )
        self.scripts = scripts
        self.calls = 0
        self.size = self.tokenizer.vocab_size + 5  # padded like Qwen
        self._by_text: dict[str, int] = {}
        for i in range(self.tokenizer.vocab_size):
            text = self.tokenizer.token_text(i)
            if text:
                self._by_text.setdefault(text, i)

    # -- SDK-like API ----------------------------------------------------
    def get_path_to_vocab_file(self) -> str:
        return str(self._vocab_path)

    def get_path_to_merges_file(self) -> str:
        return str(self._merges_path)

    def encode(self, text: str) -> Any:
        return _Tensor(self.tokenizer.encode(text))

    def decode(self, ids: Any) -> str:
        return self.tokenizer.decode(list(ids))

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        self.calls += 1
        text = self.tokenizer.decode(input_ids)
        head, _, written = text.rpartition(MARKER)
        request = head.rpartition("Request: ")[2].rstrip("\n")
        target = self.scripts.get(request, "")
        rng = np.random.default_rng(len(input_ids))
        logits = rng.normal(0.0, 0.5, self.size)
        if target.startswith(written) and len(written) < len(target):
            wanted = target[len(written):]
            for k in range(min(len(wanted), 24), 0, -1):
                token_id = self._by_text.get(wanted[:k])
                if token_id is not None:
                    logits[token_id] = 20.0
                    break
        return [float(x) for x in logits]
