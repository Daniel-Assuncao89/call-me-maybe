"""Exception hierarchy used across the project."""


class CallMeMaybeError(Exception):
    """Base class for every error raised on purpose by this project."""


class InputError(CallMeMaybeError):
    """An input file is missing, unreadable or does not match its schema."""


class ModelError(CallMeMaybeError):
    """The language model (or its tokenizer files) could not be loaded."""


class TokenizerError(CallMeMaybeError):
    """The tokenizer files are malformed or unusable."""


class DecodeError(CallMeMaybeError):
    """Constrained decoding reached a dead end (no valid token left)."""


class CallError(CallMeMaybeError):
    """No valid function call could be produced for a prompt."""
