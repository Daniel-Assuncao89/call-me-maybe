"""Pydantic models for the input and output files.

The schema vocabulary is the one used by ``functions_definition.json``:
``number``, ``integer``, ``string``, ``boolean`` and ``null`` plus the
nested ``array`` (with ``items``) and ``object`` (with ``properties``)
types used by the bonus part.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

TYPE_ALIASES: dict[str, str] = {
    "int": "integer",
    "float": "number",
    "double": "number",
    "str": "string",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}
SUPPORTED_TYPES = frozenset(
    {"number", "integer", "string", "boolean", "null", "array", "object"}
)


class TypeSpec(BaseModel):
    """Schema of one value (a parameter, an array item, a property)."""

    model_config = ConfigDict(extra="ignore")

    type: str
    description: str | None = None
    items: TypeSpec | None = None
    properties: dict[str, TypeSpec] | None = None
    enum: list[Any] | None = None

    @field_validator("type")
    @classmethod
    def _normalise_type(cls, value: str) -> str:
        key = value.strip().lower()
        key = TYPE_ALIASES.get(key, key)
        if key not in SUPPORTED_TYPES:
            raise ValueError(f"unsupported type {value!r}")
        return key

    @model_validator(mode="after")
    def _check_shape(self) -> TypeSpec:
        if self.type == "object" and self.properties is None:
            raise ValueError("an 'object' type needs 'properties'")
        if self.type == "array" and self.items is None:
            self.items = TypeSpec(type="string")
        return self


class FunctionDef(BaseModel):
    """One entry of ``functions_definition.json``."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, TypeSpec] = Field(default_factory=dict)
    returns: TypeSpec | None = None


class PromptItem(BaseModel):
    """One entry of ``function_calling_tests.json``."""

    model_config = ConfigDict(extra="ignore")

    prompt: str


class FunctionCall(BaseModel):
    """One entry of the output file (exactly three keys)."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]


def conforms(value: Any, spec: TypeSpec) -> bool:
    """Return True when *value* satisfies *spec* (strict, recursive)."""
    if spec.enum is not None and value not in spec.enum:
        return False
    kind = spec.type
    if kind == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    if kind == "array":
        items = spec.items
        return (
            isinstance(value, list)
            and items is not None
            and all(conforms(v, items) for v in value)
        )
    props = spec.properties or {}
    return (
        isinstance(value, dict)
        and set(value) == set(props)
        and all(conforms(value[k], props[k]) for k in props)
    )


def conforms_call(parameters: dict[str, Any], fn: FunctionDef) -> bool:
    """Return True when *parameters* match the signature of *fn*."""
    if set(parameters) != set(fn.parameters):
        return False
    return all(conforms(parameters[k], s) for k, s in fn.parameters.items())


TypeSpec.model_rebuild()
