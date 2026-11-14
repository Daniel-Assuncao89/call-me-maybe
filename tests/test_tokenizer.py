from pathlib import Path

import pytest

from src.tokenizer import (
    BPETokenizer,
    SdkTokenizer,
    build_tokenizer,
    bytes_to_unicode,
    pretokenize,
)

from .fakes import FakeLLM, WORDS, build_toy_vocab


@pytest.fixture
def tok() -> BPETokenizer:
    return BPETokenizer(build_toy_vocab(WORDS))


def test_byte_table_is_a_bijection() -> None:
    table = bytes_to_unicode()
    assert len(table) == 256 and len(set(table.values())) == 256
    assert table[ord("a")] == "a" and table[ord(" ")] == "Ġ"


def test_pretokenize_matches_qwen_rules() -> None:
    assert pretokenize("Hello, world 123") == [
        "Hello", ",", " world", " ", "1", "2", "3"]
    assert pretokenize("don't") == ["don", "'t"]
    assert pretokenize("a\n\nb") == ["a", "\n\n", "b"]


@pytest.mark.parametrize("text", [
    "", "a", "Greet shrek", "What is the sum of 265 and 345?",
    "fn_add_numbers hello world", 'x = {"a": 1}\n', "héllo 日本語 🙂",
    "  spaces   and\ttabs  ",
])
def test_roundtrip(tok: BPETokenizer, text: str) -> None:
    assert tok.decode(tok.encode(text)) == text


def test_merges_are_applied(tok: BPETokenizer) -> None:
    assert len(tok.encode("hello")) == 1
    assert len(tok.encode(" world")) == 1
    assert len(tok.encode("fn_greet")) < len("fn_greet")


def test_token_text_hides_partial_utf8(tok: BPETokenizer) -> None:
    assert tok.token_text(ord("a")) == "a"
    assert tok.token_text(0xC3) is None  # first byte of a 2-byte char


def test_sdk_tokenizer_delegates(tmp_path: Path) -> None:
    llm = FakeLLM(tmp_path, {})
    own = build_tokenizer(llm)
    sdk = build_tokenizer(llm, use_sdk=True)
    assert isinstance(sdk, SdkTokenizer)
    for text in ["Greet john", "hello world", "é"]:
        assert sdk.encode(text) == own.encode(text)
        assert sdk.decode(own.encode(text)) == text
