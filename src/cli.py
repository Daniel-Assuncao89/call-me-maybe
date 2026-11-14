"""Command line interface: ``python -m src [options]``."""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field, ValidationError

from .decoder import ConstrainedDecoder, DecoderConfig
from .engine import FunctionCaller
from .errors import CallError, CallMeMaybeError, ModelError
from .io_utils import load_functions, load_prompts, write_output
from .model_client import CachedModel
from .models import FunctionCall
from .tokenizer import build_tokenizer
from .trace import Trace
from .vocab import VocabIndex

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


class CliConfig(BaseModel):
    """Validated command line options."""

    functions_definition: Path = Path("data/input/functions_definition.json")
    input: Path = Path("data/input/function_calling_tests.json")
    output: Path = Path("data/output/function_calls.json")
    model: str = Field(default=DEFAULT_MODEL, min_length=1)
    trace: bool = False
    trace_html: Path | None = None
    use_sdk_tokenizer: bool = False
    quiet: bool = False
    max_string_tokens: int = Field(default=64, ge=1)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into function "
        "calls using constrained decoding.",
    )
    parser.add_argument(
        "--functions_definition", "--functions-definition",
        dest="functions_definition", metavar="FILE",
        help="functions definition JSON "
        "(default: data/input/functions_definition.json)")
    parser.add_argument(
        "--input", metavar="FILE",
        help="prompts JSON (default: data/input/function_calling_tests"
        ".json)")
    parser.add_argument(
        "--output", metavar="FILE",
        help="result JSON (default: data/output/function_calls.json)")
    parser.add_argument(
        "--model", metavar="NAME",
        help=f"Hugging Face model id (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--trace", action="store_true",
        help="print every constrained decoding step to stderr")
    parser.add_argument(
        "--trace-html", metavar="FILE", dest="trace_html",
        help="write the full decoding trace as an HTML page")
    parser.add_argument(
        "--use-sdk-tokenizer", action="store_true", dest="use_sdk_tokenizer",
        help="encode/decode with the SDK instead of the built-in BPE")
    parser.add_argument(
        "--max-string-tokens", type=int, dest="max_string_tokens",
        metavar="N", help="token budget of one string argument")
    parser.add_argument(
        "--quiet", action="store_true", help="do not print progress")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> CliConfig:
    """Parse and validate the command line."""
    namespace = build_parser().parse_args(argv)
    values = {k: v for k, v in vars(namespace).items() if v is not None}
    try:
        return CliConfig(**values)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
            for e in exc.errors()
        )
        raise SystemExit(f"error: invalid options ({details})") from None


def load_model(name: str) -> Any:
    """Instantiate the SDK model (imported lazily: it is heavy)."""
    try:
        from llm_sdk import Small_LLM_Model
    except ImportError as exc:
        raise ModelError(
            f"cannot import llm_sdk ({exc}); did you run 'uv sync'?"
        ) from exc
    try:
        return Small_LLM_Model(model_name=name)
    except Exception as exc:
        raise ModelError(f"cannot load model {name!r}: {exc}") from exc


def run(config: CliConfig, model: Any | None = None) -> int:
    """Execute the whole pipeline.  Returns the process exit code."""
    functions = load_functions(config.functions_definition)
    prompts = load_prompts(config.input)
    if model is None:
        model = load_model(config.model)
    tokenizer = build_tokenizer(model, config.use_sdk_tokenizer)
    vocab = VocabIndex(tokenizer)
    cached = CachedModel(model)
    trace = Trace() if (config.trace or config.trace_html) else None
    decoder = ConstrainedDecoder(
        cached,
        tokenizer,
        vocab,
        DecoderConfig(max_string_tokens=config.max_string_tokens),
        trace,
    )
    caller = FunctionCaller(decoder, tokenizer, functions, trace)

    results: list[FunctionCall] = []
    failures = 0
    started = time.perf_counter()
    try:
        for index, prompt in enumerate(prompts, 1):
            begin = time.perf_counter()
            try:
                call = caller.call(prompt)
            except CallError as exc:
                failures += 1
                print(f"warning: prompt #{index} skipped: {exc}",
                      file=sys.stderr)
                continue
            results.append(call)
            if trace is not None and config.trace:
                print(trace.render_text(), file=sys.stderr)
            if not config.quiet:
                print(
                    f"[{index}/{len(prompts)}] {call.name} "
                    f"({time.perf_counter() - begin:.1f}s)",
                    file=sys.stderr,
                )
    except KeyboardInterrupt:
        print("interrupted: writing the results obtained so far",
              file=sys.stderr)
    write_output(config.output, results)
    if trace is not None and config.trace_html is not None:
        config.trace_html.parent.mkdir(parents=True, exist_ok=True)
        config.trace_html.write_text(trace.render_html(), encoding="utf-8")
    if not config.quiet:
        print(
            f"done: {len(results)} call(s) written to {config.output}, "
            f"{failures} skipped, {time.perf_counter() - started:.1f}s, "
            f"cache {cached.hits} hit / {cached.misses} miss",
            file=sys.stderr,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; never lets an expected error escape as a traceback."""
    config = parse_args(argv)
    try:
        return run(config)
    except CallMeMaybeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
