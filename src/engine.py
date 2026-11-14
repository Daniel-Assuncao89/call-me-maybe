"""Turns one natural-language prompt into one validated function call."""

import json
from typing import Sequence

from pydantic import BaseModel, ConfigDict

from .decoder import ConstrainedDecoder
from .errors import CallError, DecodeError
from .models import FunctionCall, FunctionDef, conforms_call
from .prompt import build_prompt
from .tokenizer import BPETokenizer
from .trace import Trace

STYLES = ("full", "compact")


class FunctionCaller(BaseModel):
    """High level API: ``call(prompt) -> FunctionCall``."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    decoder: ConstrainedDecoder
    tokenizer: BPETokenizer
    functions: list[FunctionDef]
    trace: Trace | None = None

    def __init__(
        self,
        decoder: ConstrainedDecoder,
        tokenizer: BPETokenizer,
        functions: Sequence[FunctionDef],
        trace: Trace | None = None,
    ) -> None:
        super().__init__(
            decoder=decoder, tokenizer=tokenizer,
            functions=list(functions), trace=trace)
        self._decoder = decoder
        self._tok = tokenizer
        self._functions = list(functions)
        self._by_name = {f.name: f for f in self._functions}
        self._trace = trace

    def call(self, prompt: str) -> FunctionCall:
        """Return the function call for *prompt*.

        Error recovery: if constrained decoding dead-ends with the full
        prompt, the same request is retried with the compact prompt
        before giving up with a ``CallError``.
        """
        if self._trace is not None:
            self._trace.begin(prompt)
        failures: list[str] = []
        for style in STYLES:
            try:
                result = self._attempt(prompt, style)
            except DecodeError as exc:
                failures.append(f"{style}: {exc}")
                continue
            if self._trace is not None:
                self._trace.finish(json.dumps(result.model_dump()))
            return result
        raise CallError("; ".join(failures))

    def _attempt(self, prompt: str, style: str) -> FunctionCall:
        text = build_prompt(self._functions, prompt, style)
        self._decoder.reset(self._tok.encode(text))
        name = self._decoder.choose_literal(
            [f.name for f in self._functions], '"'
        )
        function = self._by_name[name]
        parameters = self._decoder.gen_object(
            function.parameters, '", "parameters": '
        )
        if not conforms_call(parameters, function):
            raise DecodeError(f"{name}: generated arguments do not conform")
        return FunctionCall(prompt=prompt, name=name, parameters=parameters)
