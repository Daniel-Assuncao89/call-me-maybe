import pytest

from src.automata import (
    STR_ESCAPE,
    STR_NORMAL,
    number_step,
    scan_number,
    scan_string,
)


def feed(text: str, integer: bool = False) -> str | None:
    state: str | None = "start"
    for ch in text:
        assert state is not None
        state = number_step(state, ch, integer)
        if state is None:
            return None
    return state


@pytest.mark.parametrize("text", [
    "0", "7", "-7", "10", "0.5", "-0.25", "3.14", "1e5", "1E-5",
    "2.5e+10", "-0e0",
])
def test_valid_numbers(text: str) -> None:
    assert feed(text) in {"zero", "int", "frac", "exp", "exp2"}


@pytest.mark.parametrize("text", [
    "", "+1", "01", "-", "1.", ".5", "1e", "1e+", "--1", "1.2.3", "abc",
])
def test_invalid_or_incomplete(text: str) -> None:
    assert feed(text) not in {"zero", "int", "frac", "exp", "exp2"}


def test_integer_mode_rejects_fraction_and_exponent() -> None:
    assert feed("12", integer=True) == "int"
    assert feed("1.5", integer=True) is None
    assert feed("1e5", integer=True) is None


def test_scan_number_stops_on_follow_char() -> None:
    assert scan_number("int", ",", False, ",}") == ("int", 0, True)
    assert scan_number("start", "5,", False, ",}") == ("int", 1, True)
    assert scan_number("start", "-,", False, ",}") is None
    assert scan_number("int", "7", False, ",") == ("int", 1, False)
    assert scan_number("int", "x", False, ",") is None


def test_scan_string() -> None:
    assert scan_string(STR_NORMAL, "abc") == (STR_NORMAL, None)
    assert scan_string(STR_NORMAL, 'ab"cd') == (STR_NORMAL, 2)
    assert scan_string(STR_NORMAL, "a\\") == (STR_ESCAPE, None)
    assert scan_string(STR_NORMAL, '\\"x') == (STR_NORMAL, None)
    assert scan_string(STR_NORMAL, "\\u00e9") == (STR_NORMAL, None)
    assert scan_string(STR_NORMAL, "\\u00") is not None
    assert scan_string(STR_NORMAL, "\\x") is None
    assert scan_string(STR_NORMAL, "\\u00zz") is None
    assert scan_string(STR_NORMAL, "a\nb") is None
    assert scan_string(STR_ESCAPE, '"') == (STR_NORMAL, None)
