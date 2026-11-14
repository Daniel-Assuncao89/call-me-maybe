"""End-to-end behaviour of the constrained decoder with a scripted LLM."""

import json
from typing import Any, Callable

import pytest

from src.errors import CallError

Maker = Callable[..., Any]

CASES = {
    "add 2 and 3": ('fn_add_numbers", "parameters": {"a": 2, "b": 3}}',
                    "fn_add_numbers", {"a": 2, "b": 3}),
    "negative": ('fn_add_numbers", "parameters": {"a": -1.5, "b": 1e2}}',
                 "fn_add_numbers", {"a": -1.5, "b": 100.0}),
    "greet": ('fn_greet", "parameters": {"name": "hello world"}}',
              "fn_greet", {"name": "hello world"}),
    "escape": ('fn_greet", "parameters": {"name": "a\\"b\\\\c\\n"}}',
               "fn_greet", {"name": 'a"b\\c\n'}),
    "unicode": ('fn_reverse_string", "parameters": {"s": "héllo 🙂"}}',
                "fn_reverse_string", {"s": "héllo 🙂"}),
    "bool": ('fn_flag", "parameters": {"on": true, "n": 7}}',
             "fn_flag", {"on": True, "n": 7}),
    "bool2": ('fn_flag", "parameters": {"on": false, "n": 0}}',
              "fn_flag", {"on": False, "n": 0}),
    "list": ('fn_list", "parameters": {"xs": [1, 2.5, 3]}}',
             "fn_list", {"xs": [1, 2.5, 3]}),
    "empty list": ('fn_list", "parameters": {"xs": []}}',
                   "fn_list", {"xs": []}),
    "nested": ('fn_nested", "parameters": {"point": {"x": 1, "tag": "t"}, '
               '"after": 4}}', "fn_nested",
               {"point": {"x": 1, "tag": "t"}, "after": 4}),
    "enum": ('fn_pick", "parameters": {"color": "redish"}}',
             "fn_pick", {"color": "redish"}),
}


@pytest.mark.parametrize("request_text", list(CASES))
def test_wanted_call_is_produced(
    make_caller: Maker, request_text: str
) -> None:
    script, name, params = CASES[request_text]
    caller, _, _ = make_caller({request_text: script})
    call = caller.call(request_text)
    assert call.name == name
    assert call.parameters == params
    assert call.prompt == request_text
    json.dumps(call.model_dump())


def test_model_wanting_garbage_still_yields_valid_json(
    make_caller: Maker, functions: Any
) -> None:
    """Whatever the model prefers, the output stays schema-valid."""
    from src.models import conforms_call

    scripts = {"x": 'not_a_function", "parameters": {"zzz": [[[}'}
    caller, _, _ = make_caller(scripts)
    call = caller.call("x")
    fn = next(f for f in functions if f.name == call.name)
    assert conforms_call(call.parameters, fn)


def test_wrong_type_wanted_is_blocked(make_caller: Maker) -> None:
    scripts = {"r": 'fn_add_numbers", "parameters": {"a": "oops", "b": 1}}'}
    caller, _, _ = make_caller(scripts)
    call = caller.call("r")
    assert all(isinstance(v, (int, float))
               for v in call.parameters.values())


def test_enum_cannot_escape(make_caller: Maker) -> None:
    scripts = {"e": 'fn_pick", "parameters": {"color": "purple"}}'}
    caller, _, _ = make_caller(scripts)
    assert caller.call("e").parameters["color"] in {"red", "green", "redish"}


def test_string_budget_is_respected(make_caller: Maker) -> None:
    long = "hello " * 200
    scripts = {"l": f'fn_greet", "parameters": {{"name": "{long}"}}}}'}
    caller, _, _ = make_caller(scripts, max_string_tokens=8)
    call = caller.call("l")
    assert isinstance(call.parameters["name"], str)


def test_llm_call_budget_raises_call_error(make_caller: Maker) -> None:
    caller, _, _ = make_caller({}, max_llm_calls=1)
    with pytest.raises(CallError):
        caller.call("anything")


def test_trace_records_steps(make_caller: Maker) -> None:
    script, _, _ = CASES["add 2 and 3"]
    caller, _, trace = make_caller({"add 2 and 3": script})
    caller.call("add 2 and 3")
    text = trace.render_text()
    assert "fn_add_numbers" in text and "result" in text
    page = trace.render_html()
    assert page.startswith("<!doctype html>") and "<table>" in page


def test_cache_avoids_recomputation(make_caller: Maker) -> None:
    script, _, _ = CASES["add 2 and 3"]
    caller, llm, _ = make_caller({"add 2 and 3": script})
    caller.call("add 2 and 3")
    first = llm.calls
    caller.call("add 2 and 3")
    assert llm.calls == first  # second run fully served by the cache
