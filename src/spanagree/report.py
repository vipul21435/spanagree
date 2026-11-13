"""Markdown rendering helpers."""

from __future__ import annotations

from collections.abc import Sequence


def fmt(x: object, digits: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def table(header: Sequence[str], rows: Sequence[Sequence[object]], digits: int = 4) -> str:
    """A GitHub-style Markdown table."""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(c, digits) for c in row) + " |")
    return "\n".join(lines)
