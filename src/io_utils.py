"""Reading the input files and writing the output file safely."""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .errors import InputError
from .models import FunctionCall, FunctionDef, PromptItem


def _read_json(path: Path, label: str) -> Any:
    """Load a JSON file, turning every failure into an InputError."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise InputError(f"{label}: file not found: {path}") from None
    except IsADirectoryError:
        raise InputError(f"{label}: {path} is a directory") from None
    except PermissionError:
        raise InputError(f"{label}: permission denied: {path}") from None
    except UnicodeDecodeError as exc:
        raise InputError(f"{label}: {path} is not valid UTF-8 ({exc})") \
            from None
    except json.JSONDecodeError as exc:
        raise InputError(
            f"{label}: invalid JSON in {path} "
            f"(line {exc.lineno}, column {exc.colno}: {exc.msg})"
        ) from None
    except OSError as exc:
        raise InputError(f"{label}: cannot read {path} ({exc})") from None


def load_functions(path: Path) -> list[FunctionDef]:
    """Load and validate ``functions_definition.json``."""
    data = _read_json(path, "functions definition")
    if not isinstance(data, list):
        raise InputError(
            f"functions definition: {path} must contain a JSON array"
        )
    functions: list[FunctionDef] = []
    for index, entry in enumerate(data):
        try:
            functions.append(FunctionDef.model_validate(entry))
        except ValidationError as exc:
            raise InputError(
                f"functions definition: entry #{index} is invalid:\n"
                f"{_short(exc)}"
            ) from None
    if not functions:
        raise InputError("functions definition: the list is empty")
    names = [f.name for f in functions]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise InputError(
            "functions definition: duplicated names: "
            + ", ".join(duplicated)
        )
    return functions


def load_prompts(path: Path) -> list[str]:
    """Load ``function_calling_tests.json``.

    Invalid entries are skipped with a warning so that one bad item does
    not prevent the others from being processed.
    """
    data = _read_json(path, "prompts")
    if not isinstance(data, list):
        raise InputError(f"prompts: {path} must contain a JSON array")
    prompts: list[str] = []
    for index, entry in enumerate(data):
        try:
            prompts.append(PromptItem.model_validate(entry).prompt)
        except ValidationError:
            print(
                f"warning: prompts: entry #{index} ignored "
                "(expected an object with a string 'prompt')",
                file=sys.stderr,
            )
    return prompts


def write_output(path: Path, calls: list[FunctionCall]) -> None:
    """Write the result atomically (temporary file, then rename)."""
    payload = [call.model_dump() for call in calls]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=".function_calls.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise InputError(f"cannot write output {path}: {exc}") from None


def _short(exc: ValidationError) -> str:
    """Compact, readable rendering of a pydantic error."""
    lines = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {where}: {err['msg']}")
    return "\n".join(lines)
