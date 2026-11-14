"""Visualisation of the generation process (bonus).

Every decoding step is recorded: which text was forced without calling
the model, and, for each real step, how many tokens survived the mask
and what the best candidates were.
"""

import html
from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    """One step of the generation."""

    kind: str
    text: str
    candidates: list[tuple[str, float]] = Field(default_factory=list)
    allowed: int = 0
    vocab: int = 0
    note: str = ""


class PromptTrace(BaseModel):
    """All steps recorded for one prompt."""

    prompt: str
    steps: list[TraceStep] = Field(default_factory=list)
    result: str = ""


class Trace(BaseModel):
    """Collects steps and renders them as text or HTML."""

    prompts: list[PromptTrace] = Field(default_factory=list)

    def begin(self, prompt: str) -> None:
        """Start recording a new prompt."""
        self.prompts.append(PromptTrace(prompt=prompt))

    def add(self, step: TraceStep) -> None:
        """Append a step to the current prompt."""
        if self.prompts:
            self.prompts[-1].steps.append(step)

    def finish(self, result: str) -> None:
        """Store the final JSON of the current prompt."""
        if self.prompts:
            self.prompts[-1].result = result

    # ---------------------------------------------------------------- text
    def render_text(self, last_only: bool = True) -> str:
        """Human readable trace (last prompt only by default)."""
        items = self.prompts[-1:] if last_only else self.prompts
        out: list[str] = []
        for item in items:
            out.append(f"=== {item.prompt!r}")
            for step in item.steps:
                if step.kind == "forced":
                    out.append(f"  [forced ] {step.text!r}")
                    continue
                cands = "  ".join(
                    f"{t!r}:{p:.0%}" for t, p in step.candidates
                )
                out.append(
                    f"  [{step.kind:<7}] -> {step.text!r}"
                    f"  (allowed {step.allowed}/{step.vocab})"
                    f"  {cands}  {step.note}".rstrip()
                )
            out.append(f"  result: {item.result}")
        return "\n".join(out)

    # ---------------------------------------------------------------- html
    def render_html(self) -> str:
        """Self-contained HTML page with every recorded prompt."""
        blocks: list[str] = []
        for item in self.prompts:
            rows: list[str] = []
            for step in item.steps:
                cands = " ".join(
                    f"<span class=c>{html.escape(t)} "
                    f"<b>{p:.0%}</b></span>"
                    for t, p in step.candidates
                )
                share = (
                    f"{step.allowed}/{step.vocab}" if step.vocab else ""
                )
                rows.append(
                    f"<tr class={step.kind}><td>{step.kind}</td>"
                    f"<td><code>{html.escape(step.text)}</code></td>"
                    f"<td>{share}</td><td>{cands}</td>"
                    f"<td>{html.escape(step.note)}</td></tr>"
                )
            blocks.append(
                f"<h2>{html.escape(item.prompt)}</h2>"
                "<table><tr><th>step</th><th>text</th><th>allowed</th>"
                "<th>top candidates (renormalised)</th><th>note</th></tr>"
                + "".join(rows)
                + f"</table><pre>{html.escape(item.result)}</pre>"
            )
        style = (
            "body{font:14px system-ui;background:#0f1117;color:#e6e6e6;"
            "margin:2rem}table{border-collapse:collapse;width:100%}"
            "td,th{border:1px solid #2b2f3a;padding:4px 8px;"
            "text-align:left}tr.forced{color:#8b93a7}"
            ".c{background:#1c2130;border-radius:4px;padding:1px 6px;"
            "margin-right:4px}pre{background:#1c2130;padding:8px}"
        )
        return (
            "<!doctype html><meta charset=utf-8><title>trace</title>"
            f"<style>{style}</style><h1>Constrained decoding trace</h1>"
            + "".join(blocks)
        )
