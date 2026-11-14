"""Schema-driven constrained decoding.

The decoder owns the growing list of token ids.  It never lets the model
write JSON freely: the *structure* (braces, keys, commas) is appended by
the program, and the model is only consulted for the decisions that
carry information (which function, which number, which characters of a
string, how many array items).  For each of those decisions the logits
of every token that would break the grammar are discarded (equivalent to
setting them to ``-inf``) before the choice is made.
"""

import json
import math
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from .automata import (
    NUMBER_ACCEPTING,
    STR_NORMAL,
    scan_number,
    scan_string,
    string_first_chars,
)
from .errors import DecodeError
from .model_client import CachedModel
from .models import TypeSpec
from .tokenizer import BPETokenizer
from .trace import Trace, TraceStep
from .vocab import VocabIndex

FloatArray = npt.NDArray[np.float32]
NEG_INF = float("-inf")
NUMBER_CHARS = "-+.eE0123456789"


class DecoderConfig(BaseModel):
    """Safety limits of the decoding loop."""

    max_string_tokens: int = Field(default=64, ge=1)
    max_number_tokens: int = Field(default=24, ge=1)
    max_array_items: int = Field(default=16, ge=1)
    max_llm_calls: int = Field(default=600, ge=1)
    top_k: int = Field(default=5, ge=1)


def _lse(values: Sequence[float] | FloatArray) -> float:
    """Numerically stable log-sum-exp (``-inf`` for an empty input)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return NEG_INF
    top = float(arr.max())
    if not math.isfinite(top):
        return NEG_INF
    return top + math.log(float(np.exp(arr - top).sum()))


def _start_chars(spec: TypeSpec) -> str:
    """Characters a value of type *spec* may start with."""
    if spec.enum is not None:
        return "".join(
            sorted({json.dumps(v, ensure_ascii=False)[0] for v in spec.enum})
        )
    return {
        "number": "-0123456789", "integer": "-0123456789", "string": '"',
        "boolean": "tf", "null": "n", "array": "[", "object": "{",
    }.get(spec.type, '"')


class ConstrainedDecoder(BaseModel):
    """Generates schema-valid values, one constrained token at a time."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    model: CachedModel
    tokenizer: BPETokenizer
    vocab: VocabIndex
    config: DecoderConfig
    trace: Trace | None = None

    def __init__(
        self,
        model: CachedModel,
        tokenizer: BPETokenizer,
        vocab: VocabIndex,
        config: DecoderConfig | None = None,
        trace: Trace | None = None,
    ) -> None:
        super().__init__(
            model=model, tokenizer=tokenizer, vocab=vocab,
            config=config or DecoderConfig(), trace=trace)
        self._model = model
        self._tok = tokenizer
        self._vocab = vocab
        self._cfg = config or DecoderConfig()
        self._trace = trace
        self._ids: list[int] = []
        self._pending = ""
        self._calls = 0
        self._number_cache: dict[str, list[int]] = {}

    # ------------------------------------------------------- bookkeeping
    @property
    def ids(self) -> list[int]:
        """Copy of the current context (prompt + generated tokens)."""
        return list(self._ids)

    @property
    def llm_calls(self) -> int:
        """Number of model calls made since the last ``reset``."""
        return self._calls

    def reset(self, prompt_ids: Sequence[int]) -> None:
        """Start a new generation from *prompt_ids*."""
        self._ids = list(prompt_ids)
        self._pending = ""
        self._calls = 0

    def _flush(self) -> None:
        if self._pending:
            self._ids.extend(self._tok.encode(self._pending))
            self._pending = ""

    def _push(self, token_id: int) -> None:
        self._flush()
        self._ids.append(token_id)

    def _append_text(self, text: str) -> None:
        if text:
            self._flush()
            self._ids.extend(self._tok.encode(text))

    def force(self, text: str) -> None:
        """Append *text* to the context without consulting the model."""
        full = self._pending + text
        self._pending = ""
        if full:
            self._ids.extend(self._tok.encode(full))
            self._record(TraceStep(kind="forced", text=full))

    def _logits(self) -> FloatArray:
        self._flush()
        if self._calls >= self._cfg.max_llm_calls:
            raise DecodeError("too many model calls for one prompt")
        self._calls += 1
        raw = self._model.logits(self._ids)
        size = self._vocab.size
        if raw.shape[0] >= size:
            return raw[:size]
        padded: FloatArray = np.full(size, NEG_INF, dtype=np.float32)
        padded[: raw.shape[0]] = raw
        return padded

    def _record(self, step: TraceStep) -> None:
        if self._trace is not None:
            self._trace.add(step)

    def _trace_choice(
        self,
        kind: str,
        text: str,
        ids: Sequence[int],
        logits: FloatArray,
        note: str = "",
    ) -> None:
        """Record a model-driven step with its best candidates."""
        if self._trace is None or not ids:
            return
        arr = np.asarray(ids, dtype=np.int64)
        vals = logits[arr].astype(np.float64)
        order = np.argsort(-vals)[: self._cfg.top_k]
        top = float(vals[order[0]])
        expo = np.exp(vals - top) if math.isfinite(top) else vals * 0
        total = float(expo.sum()) or 1.0
        cands = [
            (self._vocab.text(int(arr[i])), float(expo[i]) / total)
            for i in order
        ]
        self._record(
            TraceStep(
                kind=kind,
                text=text,
                candidates=cands,
                allowed=int(arr.size),
                vocab=self._vocab.size,
                note=note,
            )
        )

    @staticmethod
    def _best(ids: Sequence[int], logits: FloatArray) -> int:
        """Id with the highest logit among *ids* (first one on ties)."""
        arr = np.asarray(ids, dtype=np.int64)
        return int(arr[int(np.argmax(logits[arr]))])

    # ----------------------------------------------------------- literals
    def choose_literal(self, options: Sequence[str], stop_chars: str) -> str:
        """Let the model pick one of *options*, token by token.

        Only tokens whose text is a prefix of what is still missing from
        some option are allowed.  When the text typed so far is already
        a complete option *and* longer options exist, the model decides
        between stopping (a token starting with a char of *stop_chars*)
        and continuing.  The stop token itself is not consumed.
        """
        unique = list(dict.fromkeys(options))
        if not unique:
            raise DecodeError("no option to choose from")
        text = ""
        while True:
            longer = [
                o for o in unique if len(o) > len(text) and o.startswith(text)
            ]
            exact = text in unique
            if not longer:
                if exact:
                    return text
                raise DecodeError(f"dead end while writing {text!r}")
            if len(longer) == 1 and not exact:
                rest = longer[0][len(text):]
                self._append_text(rest)
                self._record(TraceStep(kind="forced", text=rest,
                                       note="only one option left"))
                return longer[0]
            logits = self._logits()
            cont = self._literal_continuations(text, longer)
            stop = self._vocab.tokens_starting(stop_chars) if exact else []
            if stop and (
                not cont
                or _lse(logits[np.asarray(stop)])
                >= _lse(logits[np.asarray(cont)])
            ):
                self._trace_choice("literal", "<stop>", stop, logits,
                                   f"{text!r} is complete")
                return text
            if not cont:
                char = longer[0][len(text)]
                self._append_text(char)
                text += char
                continue
            best = self._best(cont, logits)
            self._trace_choice("literal", self._vocab.text(best), cont,
                               logits)
            self._push(best)
            text += self._vocab.text(best)

    def _literal_continuations(
        self, text: str, longer: Sequence[str]
    ) -> list[int]:
        found: set[int] = set()
        for option in longer:
            rest = option[len(text):]
            for k in range(1, len(rest) + 1):
                token_id = self._vocab.text_to_id.get(rest[:k])
                if token_id is not None:
                    found.add(token_id)
        return sorted(found)

    def decide(self, first: str, second: str) -> str:
        """Ask the model which of two structural characters comes next."""
        logits = self._logits()
        ids_a = self._vocab.tokens_starting(first)
        ids_b = self._vocab.tokens_starting(second)
        mass_a = _lse(logits[np.asarray(ids_a, dtype=np.int64)])
        mass_b = _lse(logits[np.asarray(ids_b, dtype=np.int64)])
        if not (math.isfinite(mass_a) or math.isfinite(mass_b)):
            raise DecodeError("no token can continue the structure")
        chosen = first if mass_a >= mass_b else second
        ids = ids_a if chosen == first else ids_b
        self._trace_choice("decide", chosen, ids, logits,
                           f"{first!r} vs {second!r}")
        return chosen

    # ------------------------------------------------------------ numbers
    def gen_number(self, integer: bool, follow: str) -> float | int:
        """Generate a JSON number; *follow* are the chars that may end it."""
        candidates = self._number_cache.get(follow)
        if candidates is None:
            candidates = self._vocab.tokens_starting(NUMBER_CHARS + follow)
            self._number_cache[follow] = candidates
        state = "start"
        text = ""
        for _ in range(self._cfg.max_number_tokens):
            logits = self._logits()
            cont: list[tuple[int, str, int]] = []
            stop: list[tuple[int, int]] = []
            for token_id in candidates:
                tok_text = self._vocab.text(token_id)
                res = scan_number(state, tok_text, integer, follow)
                if res is None:
                    continue
                new_state, consumed, stopped = res
                if stopped:
                    stop.append((token_id, consumed))
                else:
                    cont.append((token_id, new_state, consumed))
            cont_ids = [c[0] for c in cont]
            stop_ids = [s[0] for s in stop]
            mass_cont = _lse(logits[np.asarray(cont_ids, dtype=np.int64)])
            mass_stop = _lse(logits[np.asarray(stop_ids, dtype=np.int64)])
            if not (math.isfinite(mass_cont) or math.isfinite(mass_stop)):
                raise DecodeError(f"dead end while writing number {text!r}")
            if stop_ids and mass_stop >= mass_cont:
                best = self._best(stop_ids, logits)
                consumed = dict(stop)[best]
                self._trace_choice("number", "<end>", stop_ids, logits,
                                   f"value so far {text!r}")
                part = self._vocab.text(best)[:consumed]
                text += part
                if consumed == len(self._vocab.text(best)):
                    self._push(best)
                else:
                    self._append_text(part)
                break
            best = self._best(cont_ids, logits)
            self._trace_choice("number", self._vocab.text(best), cont_ids,
                               logits)
            state = next(s for t, s, _ in cont if t == best)
            text += self._vocab.text(best)
            self._push(best)
        else:
            if state not in NUMBER_ACCEPTING:
                raise DecodeError("number is not finished")
        return self._to_number(text, integer)

    @staticmethod
    def _to_number(text: str, integer: bool) -> float | int:
        try:
            value: float | int = int(text) if integer else float(text)
        except ValueError as exc:
            raise DecodeError(f"invalid number {text!r}") from exc
        if isinstance(value, float) and not math.isfinite(value):
            raise DecodeError(f"number {text!r} is out of range")
        return value

    # ------------------------------------------------------------ strings
    def gen_string(self) -> str:
        """Generate string content; the opening quote is already forced.

        The closing quote is *not* consumed here: it is queued and written
        together with the next structural text, which gives the model the
        natural tokenisation (``",`` is a single token in Qwen).
        """
        raw = ""
        safe = ""
        state = STR_NORMAL
        vocab = self._vocab
        for _ in range(self._cfg.max_string_tokens):
            logits = self._logits()
            cont: list[tuple[int, int, str]] = []
            stop: list[tuple[int, str]] = []
            plain_best = -1
            plain_mass = NEG_INF
            if state == STR_NORMAL:
                masked = np.where(vocab.plain_string, logits, NEG_INF)
                top = int(np.argmax(masked))
                if math.isfinite(float(masked[top])):
                    plain_best = top
                    plain_mass = _lse(masked)
                for token_id, new_state, close, tok_text in (
                    vocab.normal_string_scan()
                ):
                    if close is None:
                        cont.append((token_id, new_state, tok_text))
                    else:
                        stop.append((token_id, tok_text[:close]))
            else:
                for token_id in vocab.tokens_starting(
                    string_first_chars(state)
                ):
                    tok_text = vocab.text(token_id)
                    res = scan_string(state, tok_text)
                    if res is None:
                        continue
                    if res[1] is None:
                        cont.append((token_id, res[0], tok_text))
                    else:
                        stop.append((token_id, tok_text[: res[1]]))
            cont_ids = [c[0] for c in cont]
            stop_ids = [s[0] for s in stop]
            mass_cont = float(
                np.logaddexp(
                    plain_mass,
                    _lse(logits[np.asarray(cont_ids, dtype=np.int64)]),
                )
            )
            mass_stop = _lse(logits[np.asarray(stop_ids, dtype=np.int64)])
            if not (math.isfinite(mass_cont) or math.isfinite(mass_stop)):
                raise DecodeError("dead end inside a string")
            if stop_ids and mass_stop >= mass_cont:
                best = self._best(stop_ids, logits)
                content = dict(stop)[best]
                self._trace_choice("string", "<end>", stop_ids, logits,
                                   f"value so far {raw!r}")
                raw += content
                if content == vocab.text(best):
                    self._push(best)
                else:
                    self._append_text(content)
                safe = raw
                break
            best_cont = self._best(cont_ids, logits) if cont_ids else -1
            if plain_best >= 0 and (
                best_cont < 0 or logits[plain_best] >= logits[best_cont]
            ):
                chosen = plain_best
                self._trace_choice(
                    "string", vocab.text(chosen),
                    [plain_best] + cont_ids, logits)
            else:
                chosen = best_cont
                self._trace_choice("string", vocab.text(chosen),
                                   cont_ids, logits)
                state = next(s for t, s, _ in cont if t == chosen)
            raw += vocab.text(chosen)
            self._push(chosen)
            if state == STR_NORMAL:
                safe = raw
        else:
            raw = safe  # token budget exhausted: drop a dangling escape
        self._pending += '"'
        return self._decode_string(raw)

    @staticmethod
    def _decode_string(raw: str) -> str:
        try:
            value = str(json.loads('"' + raw + '"'))
        except ValueError as exc:
            raise DecodeError(f"invalid string content {raw!r}") from exc
        if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            value = value.encode("utf-16", "surrogatepass").decode(
                "utf-16", "replace"
            )
        return value

    # ------------------------------------------------------ whole values
    def gen_value(self, spec: TypeSpec, prefix: str, follow: str) -> Any:
        """Generate one value of type *spec*.

        *prefix* is the structural text that must precede the value
        (``{"a": ``, ``, ``, ``[`` ...); *follow* are the characters that
        may legally come right after it.
        """
        if spec.enum is not None:
            self.force(prefix)
            options = [json.dumps(v, ensure_ascii=False) for v in spec.enum]
            return json.loads(self.choose_literal(options, follow))
        kind = spec.type
        if kind in ("number", "integer"):
            self.force(prefix)
            return self.gen_number(kind == "integer", follow)
        if kind == "string":
            self.force(prefix + '"')
            return self.gen_string()
        if kind == "boolean":
            self.force(prefix)
            return self.choose_literal(["true", "false"], follow) == "true"
        if kind == "null":
            self.force(prefix + "null")
            return None
        if kind == "array":
            return self._gen_array(spec, prefix)
        return self.gen_object(spec.properties or {}, prefix)

    def _gen_array(self, spec: TypeSpec, prefix: str) -> list[Any]:
        item_spec = spec.items or TypeSpec(type="string")
        out: list[Any] = []
        self.force(prefix + "[")
        if self.decide(_start_chars(item_spec), "]") == "]":
            self._pending += "]"
            return out
        lead = ""
        for index in range(self._cfg.max_array_items):
            out.append(self.gen_value(item_spec, lead, ",]"))
            if index == self._cfg.max_array_items - 1:
                break
            if self.decide(",", "]") == "]":
                break
            lead = ", "
        self._pending += "]"
        return out

    def gen_object(
        self, properties: dict[str, TypeSpec], prefix: str
    ) -> dict[str, Any]:
        """Generate an object with exactly the given properties."""
        out: dict[str, Any] = {}
        keys = list(properties)
        if not keys:
            self.force(prefix + "{")
            self._pending += "}"
            return out
        for index, key in enumerate(keys):
            last = index == len(keys) - 1
            lead = prefix + "{" if index == 0 else ", "
            key_text = json.dumps(key, ensure_ascii=False) + ": "
            out[key] = self.gen_value(
                properties[key], lead + key_text, "}" if last else ","
            )
        self._pending += "}"
        return out
