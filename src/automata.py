"""Character-level automata that say which text is a valid JSON prefix.

Constrained decoding needs to answer one question for every candidate
token: "if I append this text, can the result still become a valid JSON
value?"  Both automata below answer it one character at a time, so a
token (which is a string of several characters) is simply fed through
them.
"""

_DIGITS = "0123456789"
_HEX = "0123456789abcdefABCDEF"

# ---------------------------------------------------------------- numbers
NUMBER_ACCEPTING = frozenset({"zero", "int", "frac", "exp", "exp2"})


def _first_digit(char: str) -> str | None:
    if char == "0":
        return "zero"
    if char in _DIGITS:
        return "int"
    return None


def _after_integer_part(char: str, integer: bool) -> str | None:
    if integer:
        return None
    if char == ".":
        return "dot"
    if char in "eE":
        return "e"
    return None


def number_step(state: str, char: str, integer: bool) -> str | None:
    """Advance the JSON-number automaton by one character.

    ``-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?``.  States:
    start, minus, zero, int, dot, frac, e, esign, exp, exp2 (the exponent
    is capped at two digits so a float can never overflow).  Returns None when
    *char* is not allowed.
    """
    digit = char in _DIGITS
    if state == "start":
        return "minus" if char == "-" else _first_digit(char)
    if state == "minus":
        return _first_digit(char)
    if state == "zero":
        return _after_integer_part(char, integer)
    if state == "int":
        return "int" if digit else _after_integer_part(char, integer)
    if state == "dot":
        return "frac" if digit else None
    if state == "frac":
        if digit:
            return "frac"
        return "e" if char in "eE" else None
    if state == "e":
        if char in "+-":
            return "esign"
        return "exp" if digit else None
    if state == "esign":
        return "exp" if digit else None
    if state == "exp":  # at most two exponent digits: no float overflow
        return "exp2" if digit else None
    return None


def scan_number(
    state: str, text: str, integer: bool, follow: str
) -> tuple[str, int, bool] | None:
    """Feed *text* to the number automaton.

    Returns ``(state, consumed, stopped)``.  ``stopped`` is True when the
    text reached a character of *follow* while the number was already
    complete: the number ends there and ``consumed`` chars belong to it.
    Returns None when the text can never be part of a valid number.
    """
    for index, char in enumerate(text):
        nxt = number_step(state, char, integer)
        if nxt is None:
            if char in follow and state in NUMBER_ACCEPTING:
                return state, index, True
            return None
        state = nxt
    return state, len(text), False


# ---------------------------------------------------------------- strings
STR_NORMAL = 0
STR_ESCAPE = 1
STR_HEX = 10  # STR_HEX + n  == "n hex digits still expected"

ESCAPE_CHARS = '"\\/bfnrtu'


def scan_string(state: int, text: str) -> tuple[int, int | None] | None:
    """Feed *text* to the JSON-string-content automaton.

    Returns ``(state, close_index)`` where ``close_index`` is the index of
    the closing quote inside *text* (None when the string stays open), or
    None when the text is not valid string content.
    """
    for index, char in enumerate(text):
        if state == STR_NORMAL:
            if char == '"':
                return STR_NORMAL, index
            if char == "\\":
                state = STR_ESCAPE
            elif ord(char) < 0x20:
                return None
        elif state == STR_ESCAPE:
            if char == "u":
                state = STR_HEX + 4
            elif char in ESCAPE_CHARS:
                state = STR_NORMAL
            else:
                return None
        else:
            if char not in _HEX:
                return None
            state -= 1
            if state == STR_HEX:
                state = STR_NORMAL
    return state, None


def string_first_chars(state: int) -> str:
    """Characters that may legally come next in a non-normal state."""
    if state == STR_ESCAPE:
        return ESCAPE_CHARS
    return _HEX
