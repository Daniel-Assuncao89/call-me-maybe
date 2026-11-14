"""Prompt construction.

The prompt only *informs* the model (which functions exist and what the
answer looks like).  Reliability never depends on it: the structure of
the answer is enforced by constrained decoding.
"""

from typing import Sequence

from .models import FunctionDef, TypeSpec

ANSWER_PREFIX = 'Answer: {"name": "'

_INSTRUCTIONS = (
    "You translate a user request into ONE function call.\n"
    "Choose the most suitable function from the list and copy the "
    "argument values from the request, exactly as written.\n"
    "Never answer the request itself.\n"
)

_FORMAT_EXAMPLES = (
    "Format examples (these functions are NOT in the list above):\n"
    "Request: Multiply 6 by 7\n"
    'Answer: {"name": "fn_example_multiply", "parameters": '
    '{"x": 6, "y": 7}}\n'
    "Request: Say hello to Maria\n"
    'Answer: {"name": "fn_example_hello", "parameters": '
    '{"person": "Maria"}}\n'
)


def type_label(spec: TypeSpec) -> str:
    """Short human label of a type, e.g. ``array[number]``."""
    if spec.type == "array" and spec.items is not None:
        return f"array[{type_label(spec.items)}]"
    if spec.type == "object" and spec.properties:
        inner = ", ".join(
            f"{k}: {type_label(v)}" for k, v in spec.properties.items()
        )
        return "{" + inner + "}"
    return spec.type


def describe_function(fn: FunctionDef) -> str:
    """One line describing a function and its parameters."""
    params = ", ".join(
        f"{name}: {type_label(spec)}" for name, spec in fn.parameters.items()
    )
    text = f"- {fn.name}({params})"
    if fn.description:
        text += f": {fn.description}"
    return text


def build_prompt(
    functions: Sequence[FunctionDef], request: str, style: str = "full"
) -> str:
    """Return the text fed to the model, ending right after ``"name": "``.

    ``style="full"`` has instructions and format examples; ``"compact"``
    is the shorter fallback used when the first attempt fails.
    """
    listing = "\n".join(describe_function(f) for f in functions)
    if style == "compact":
        return (
            f"Functions:\n{listing}\n\n"
            f"Request: {request}\n{ANSWER_PREFIX}"
        )
    return (
        f"{_INSTRUCTIONS}\nFunctions:\n{listing}\n\n{_FORMAT_EXAMPLES}\n"
        f"Request: {request}\n{ANSWER_PREFIX}"
    )
