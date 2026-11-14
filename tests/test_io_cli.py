"""Input/output handling and the command line."""

import json
from pathlib import Path

import pytest

from src.cli import parse_args, run
from src.errors import InputError
from src.io_utils import load_functions, load_prompts, write_output
from src.models import FunctionCall, conforms

from .conftest import FUNCTIONS
from .fakes import FakeLLM


def test_load_functions_ok(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps(FUNCTIONS))
    assert len(load_functions(path)) == len(FUNCTIONS)


@pytest.mark.parametrize("content", [
    "not json", "{}", "[]", '[{"description": "no name"}]',
    json.dumps([FUNCTIONS[0], FUNCTIONS[0]]),
])
def test_load_functions_errors(tmp_path: Path, content: str) -> None:
    path = tmp_path / "f.json"
    path.write_text(content)
    with pytest.raises(InputError):
        load_functions(path)


def test_missing_and_directory(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        load_functions(tmp_path / "nope.json")
    with pytest.raises(InputError):
        load_prompts(tmp_path)


def test_load_prompts_skips_invalid(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps([{"prompt": "ok"}, {"x": 1}, 3]))
    assert load_prompts(path) == ["ok"]


def test_write_output_creates_dirs(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "o.json"
    write_output(out, [FunctionCall(prompt="p", name="n", parameters={})])
    assert json.loads(out.read_text()) == [
        {"prompt": "p", "name": "n", "parameters": {}}]


def test_conforms() -> None:
    from src.models import TypeSpec
    assert conforms(1, TypeSpec(type="number"))
    assert not conforms(True, TypeSpec(type="number"))
    assert not conforms(1.5, TypeSpec(type="integer"))
    assert conforms("a", TypeSpec(type="string", enum=["a"]))


def test_parse_args_defaults_and_alias() -> None:
    cfg = parse_args(["--functions-definition", "f.json", "--quiet"])
    assert cfg.functions_definition == Path("f.json") and cfg.quiet
    assert cfg.output == Path("data/output/function_calls.json")
    with pytest.raises(SystemExit):
        parse_args(["--max-string-tokens", "0"])


def test_cli_end_to_end(tmp_path: Path) -> None:
    funcs = tmp_path / "f.json"
    funcs.write_text(json.dumps(FUNCTIONS))
    prompts = tmp_path / "p.json"
    prompts.write_text(json.dumps(
        [{"prompt": "add"}, {"prompt": "hi"}]))
    out = tmp_path / "out" / "r.json"
    page = tmp_path / "trace.html"
    llm = FakeLLM(tmp_path / "tok", {
        "add": 'fn_add_numbers", "parameters": {"a": 40, "b": 2}}',
        "hi": 'fn_greet", "parameters": {"name": "hello"}}',
    })
    cfg = parse_args(["--functions_definition", str(funcs), "--input",
                      str(prompts), "--output", str(out), "--quiet",
                      "--trace-html", str(page)])
    assert run(cfg, model=llm) == 0
    data = json.loads(out.read_text())
    assert data == [
        {"prompt": "add", "name": "fn_add_numbers",
         "parameters": {"a": 40, "b": 2}},
        {"prompt": "hi", "name": "fn_greet",
         "parameters": {"name": "hello"}},
    ]
    assert page.exists()
