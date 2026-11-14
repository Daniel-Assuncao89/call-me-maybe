"""Shared fixtures."""

from pathlib import Path
from typing import Callable

import pytest

from src.decoder import ConstrainedDecoder, DecoderConfig
from src.engine import FunctionCaller
from src.model_client import CachedModel
from src.models import FunctionDef
from src.trace import Trace
from src.vocab import VocabIndex

from .fakes import FakeLLM

FUNCTIONS = [
    {"name": "fn_add_numbers", "description": "Add two numbers.",
     "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
     "returns": {"type": "number"}},
    {"name": "fn_greet", "description": "Greet someone.",
     "parameters": {"name": {"type": "string"}},
     "returns": {"type": "string"}},
    {"name": "fn_reverse_string", "description": "Reverse a string.",
     "parameters": {"s": {"type": "string"}},
     "returns": {"type": "string"}},
    {"name": "fn_flag", "description": "Set a flag.",
     "parameters": {"on": {"type": "boolean"},
                    "n": {"type": "integer"}},
     "returns": {"type": "string"}},
    {"name": "fn_list", "description": "Sum a list.",
     "parameters": {"xs": {"type": "array", "items": {"type": "number"}}},
     "returns": {"type": "number"}},
    {"name": "fn_nested", "description": "Nested arguments.",
     "parameters": {"point": {"type": "object", "properties": {
         "x": {"type": "number"}, "tag": {"type": "string"}}},
         "after": {"type": "integer"}},
     "returns": {"type": "string"}},
    {"name": "fn_pick", "description": "Pick a colour.",
     "parameters": {"color": {"type": "string",
                              "enum": ["red", "green", "redish"]}},
     "returns": {"type": "string"}},
]


@pytest.fixture
def functions() -> list[FunctionDef]:
    return [FunctionDef.model_validate(f) for f in FUNCTIONS]


@pytest.fixture
def make_caller(
    tmp_path: Path, functions: list[FunctionDef]
) -> Callable[..., tuple[FunctionCaller, FakeLLM, Trace]]:
    def build(
        scripts: dict[str, str], **config: int
    ) -> tuple[FunctionCaller, FakeLLM, Trace]:
        llm = FakeLLM(tmp_path / "tok", scripts)
        vocab = VocabIndex(llm.tokenizer)
        trace = Trace()
        decoder = ConstrainedDecoder(
            CachedModel(llm), llm.tokenizer, vocab,
            DecoderConfig(**config), trace)
        return FunctionCaller(decoder, llm.tokenizer, functions, trace), \
            llm, trace
    return build
