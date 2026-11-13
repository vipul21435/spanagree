"""Span records and the JSON lines reader."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, order=True)
class Span:
    """A labelled half-open interval ``[start, end)`` in one document.

    Offsets may be characters or tokens; every function treats them as integer
    positions and reports boundary shifts in the same unit.
    """

    doc: str
    start: int
    end: int
    label: str

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"empty or reversed span {self.start}-{self.end} in {self.doc!r}")
        if self.start < 0:
            raise ValueError(f"negative offset in {self.doc!r}")

    @property
    def key(self) -> str:
        """A stable example id, ``doc:start-end``."""
        return f"{self.doc}:{self.start}-{self.end}"

    def overlap(self, other: Span) -> int:
        """Number of positions shared with ``other`` (0 when disjoint or in another doc)."""
        if self.doc != other.doc:
            return 0
        return max(0, min(self.end, other.end) - max(self.start, other.start))


@dataclass
class Corpus:
    """Span sets keyed by annotator, plus what is known about each document.

    ``lengths`` holds the number of offset positions per document (needed for
    token- or character-level agreement); ``tokens`` holds the token strings when
    the input was a column file, and ``breaks`` the sentence start offsets.
    """

    sets: dict[str, list[Span]] = field(default_factory=dict)
    lengths: dict[str, int] = field(default_factory=dict)
    tokens: dict[str, list[str]] = field(default_factory=dict)
    breaks: dict[str, list[int]] = field(default_factory=dict)
    unit: str = "token"

    @property
    def annotators(self) -> list[str]:
        return list(self.sets)

    def documents(self) -> list[str]:
        docs = set(self.lengths)
        for spans in self.sets.values():
            docs.update(s.doc for s in spans)
        return sorted(docs, key=_natural_key)

    def length(self, doc: str) -> int:
        if doc in self.lengths:
            return self.lengths[doc]
        ends = [s.end for spans in self.sets.values() for s in spans if s.doc == doc]
        return max(ends, default=0)

    def merge(self, other: Corpus) -> Corpus:
        sets = dict(self.sets)
        for name, spans in other.sets.items():
            if name in sets:
                raise ValueError(f"annotator {name!r} appears in more than one input")
            sets[name] = spans
        lengths = dict(self.lengths)
        for doc, n in other.lengths.items():
            lengths[doc] = max(n, lengths.get(doc, 0))
        tokens = dict(self.tokens)
        tokens.update({d: t for d, t in other.tokens.items() if d not in tokens})
        breaks = dict(self.breaks)
        breaks.update({d: b for d, b in other.breaks.items() if d not in breaks})
        unit = self.unit if self.sets else other.unit
        return Corpus(sets=sets, lengths=lengths, tokens=tokens, breaks=breaks, unit=unit)


def _natural_key(text: str) -> list[object]:
    import re

    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text)]


def spans_by_document(spans: Iterable[Span]) -> dict[str, list[Span]]:
    """Group spans by document, each list sorted by position."""
    out: dict[str, list[Span]] = {}
    for s in spans:
        out.setdefault(s.doc, []).append(s)
    for doc in out:
        out[doc].sort()
    return out


def load_jsonl(path: str | Path, annotator: str | None = None, unit: str = "char") -> Corpus:
    """Read spans from a JSON lines file.

    Each line is an object with ``document`` (or ``doc``), ``start``, ``end`` and
    ``label``, and optionally ``annotator``. A line with ``document`` and
    ``length`` but no ``start`` records a document's length, so that documents
    without spans still count, and a line with only ``annotator`` declares an
    annotator who may have marked nothing. Lines without ``annotator`` take
    ``annotator`` (the file stem when not given).
    """
    path = Path(path)
    default = annotator or path.stem
    corpus = Corpus(unit=unit)
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({exc.msg})") from None
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            doc = rec.get("document", rec.get("doc"))
            if doc is None and "annotator" in rec and "start" not in rec:
                # Declares an annotator, so that one who marked nothing still counts.
                corpus.sets.setdefault(str(rec["annotator"]), [])
                continue
            if doc is None:
                raise ValueError(f"{path}:{lineno}: missing 'document'")
            doc = str(doc)
            if "start" not in rec and "length" in rec:
                length = rec["length"]
                if not _is_int(length) or length < 0:
                    raise ValueError(f"{path}:{lineno}: length must be a non-negative integer")
                corpus.lengths[doc] = max(length, corpus.lengths.get(doc, 0))
                continue
            for key in ("start", "end", "label"):
                if key not in rec:
                    raise ValueError(f"{path}:{lineno}: missing {key!r}")
            start, end, label = rec["start"], rec["end"], rec["label"]
            if not (_is_int(start) and _is_int(end)):
                raise ValueError(f"{path}:{lineno}: start and end must be integers")
            if not isinstance(label, str) or not label:
                raise ValueError(f"{path}:{lineno}: label must be a non-empty string")
            try:
                span = Span(doc, start, end, label)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from None
            name = str(rec.get("annotator", default))
            corpus.sets.setdefault(name, []).append(span)
    for name in corpus.sets:
        corpus.sets[name].sort()
    return corpus


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def write_jsonl(corpus: Corpus, path: str | Path) -> None:
    """Write the annotators and document lengths, then every span of every annotator."""
    with Path(path).open("w", encoding="utf-8") as fh:
        for name in corpus.sets:
            fh.write(json.dumps({"annotator": name}) + "\n")
        for doc in corpus.documents():
            fh.write(json.dumps({"document": doc, "length": corpus.length(doc)}) + "\n")
        for name, spans in corpus.sets.items():
            for s in spans:
                rec = {"document": s.doc, "annotator": name, "start": s.start, "end": s.end,
                       "label": s.label}
                fh.write(json.dumps(rec) + "\n")
