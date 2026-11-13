"""BIO-style tag schemes: decoding, encoding, checking and the column-file reader.

Schemes:

``iob1``
    ``I-X`` inside every chunk; ``B-X`` only on the first token of a chunk that
    directly follows another chunk of the same type (the original CoNLL-2003
    files use this).
``iob2`` (alias ``bio``)
    ``B-X`` on the first token of every chunk, ``I-X`` on the rest.
``bilou`` (aliases ``bioes``, ``iobes``)
    ``B-X`` first, ``I-X`` inside, ``L-X`` (or ``E-X``) last, ``U-X`` (or ``S-X``)
    for one-token chunks.

Illegal transitions are counted and reported. They are repaired only when the
caller asks (``repair=True``): an ``I-X`` or ``L-X`` that continues nothing
starts a new chunk, and a chunk left open is closed where the next chunk or
``O`` begins.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from spanagree.spans import Corpus, Span

SCHEMES: tuple[str, ...] = ("iob1", "iob2", "bilou")
_ALIASES = {"bio": "iob2", "iob": "iob1", "bioes": "bilou", "iobes": "bilou"}
_PREFIXES = {"iob1": {"B", "I"}, "iob2": {"B", "I"}, "bilou": {"B", "I", "L", "U"}}


def scheme_name(name: str) -> str:
    """Canonical scheme name for ``name`` or an alias."""
    key = name.lower()
    key = _ALIASES.get(key, key)
    if key not in SCHEMES:
        raise ValueError(f"unknown scheme {name!r}; choose from iob1, iob2 (bio), bilou (bioes)")
    return key


class SchemeError(ValueError):
    """Raised when tags break the scheme and repair was not requested."""

    def __init__(self, message: str, violations: list[Violation]):
        super().__init__(message)
        self.violations = violations


@dataclass(frozen=True)
class Violation:
    """One illegal transition: ``prev`` (or ``None`` at a sentence start) to ``tag``."""

    doc: str
    position: int
    prev: str | None
    tag: str

    @property
    def kind(self) -> str:
        return f"{self.prev or '<start>'} -> {self.tag}"


@dataclass
class CheckReport:
    """Tag counts and illegal transitions found in one or more tag sequences."""

    scheme: str
    n_tags: int = 0
    n_chunks: int = 0
    violations: list[Violation] = field(default_factory=list)

    def counts(self) -> Counter[str]:
        return Counter(v.kind for v in self.violations)

    def as_dict(self, examples: int = 5) -> dict[str, object]:
        by_kind: dict[str, list[str]] = {}
        for v in self.violations:
            by_kind.setdefault(v.kind, [])
            if len(by_kind[v.kind]) < examples:
                by_kind[v.kind].append(f"{v.doc}:{v.position}")
        return {
            "scheme": self.scheme,
            "tags": self.n_tags,
            "chunks": self.n_chunks,
            "illegal_transitions": len(self.violations),
            "by_kind": {k: {"count": n, "examples": by_kind[k]}
                        for k, n in self.counts().most_common()},
        }


def parse_tag(tag: str, scheme: str) -> tuple[str, str | None]:
    """Split ``tag`` into prefix and type; ``("O", None)`` for outside."""
    if tag == "O":
        return "O", None
    if len(tag) < 3 or tag[1] not in "-_":
        raise ValueError(f"malformed tag {tag!r}")
    prefix, label = tag[0].upper(), tag[2:]
    if scheme == "bilou":
        prefix = {"E": "L", "S": "U"}.get(prefix, prefix)
    if prefix not in _PREFIXES[scheme]:
        raise ValueError(f"tag {tag!r} is not part of the {scheme} scheme")
    return prefix, label


def decode(tags: Sequence[str], scheme: str = "iob2", doc: str = "",
           offset: int = 0) -> tuple[list[Span], list[Violation]]:
    """Spans and illegal transitions of one sentence of tags.

    Spans are always produced with the repair policy described in the module
    docstring; the caller decides whether violations are acceptable.
    """
    scheme = scheme_name(scheme)
    spans: list[Span] = []
    viol: list[Violation] = []
    cur: tuple[int, str] | None = None
    prev: str | None = None

    def close(end: int) -> None:
        nonlocal cur
        if cur is not None:
            spans.append(Span(doc, offset + cur[0], offset + end, cur[1]))
            cur = None

    def bad(i: int, tag: str) -> None:
        viol.append(Violation(doc, offset + i, prev, tag))

    for i, tag in enumerate(tags):
        p, lab = parse_tag(tag, scheme)
        if scheme == "iob2":
            if p == "O":
                close(i)
            elif p == "B":
                close(i)
                cur = (i, lab)
            elif cur is None or cur[1] != lab:
                bad(i, tag)
                close(i)
                cur = (i, lab)
        elif scheme == "iob1":
            if p == "O":
                close(i)
            elif p == "I":
                if cur is None or cur[1] != lab:
                    close(i)
                    cur = (i, lab)
            else:
                if cur is None or cur[1] != lab:
                    bad(i, tag)
                close(i)
                cur = (i, lab)
        else:
            if p == "O":
                if cur is not None:
                    bad(i, tag)
                close(i)
            elif p == "U":
                if cur is not None:
                    bad(i, tag)
                close(i)
                spans.append(Span(doc, offset + i, offset + i + 1, lab))
            elif p == "B":
                if cur is not None:
                    bad(i, tag)
                close(i)
                cur = (i, lab)
            elif p == "I":
                if cur is None or cur[1] != lab:
                    bad(i, tag)
                    close(i)
                    cur = (i, lab)
            else:  # L
                if cur is None or cur[1] != lab:
                    bad(i, tag)
                    close(i)
                    cur = (i, lab)
                close(i + 1)
        prev = tag
    if scheme == "bilou" and cur is not None:
        viol.append(Violation(doc, offset + len(tags), prev, "<end>"))
    close(len(tags))
    return spans, viol


def encode(spans: Sequence[Span], n: int, scheme: str = "iob2", offset: int = 0) -> list[str]:
    """Tags for ``n`` positions starting at ``offset``; spans must not overlap."""
    scheme = scheme_name(scheme)
    tags = ["O"] * n
    last_end, last_label = None, None
    for s in sorted(spans):
        a, b = s.start - offset, s.end - offset
        if a < 0 or b > n:
            raise ValueError(f"span {s.key} lies outside the sentence")
        if any(t != "O" for t in tags[a:b]):
            raise ValueError(f"span {s.key} overlaps another span; tag schemes need flat spans")
        if scheme == "iob2":
            tags[a] = f"B-{s.label}"
            for k in range(a + 1, b):
                tags[k] = f"I-{s.label}"
        elif scheme == "iob1":
            adjacent = last_end == a and last_label == s.label
            tags[a] = f"B-{s.label}" if adjacent else f"I-{s.label}"
            for k in range(a + 1, b):
                tags[k] = f"I-{s.label}"
        else:
            if b - a == 1:
                tags[a] = f"U-{s.label}"
            else:
                tags[a] = f"B-{s.label}"
                for k in range(a + 1, b - 1):
                    tags[k] = f"I-{s.label}"
                tags[b - 1] = f"L-{s.label}"
        last_end, last_label = b, s.label
    return tags


@dataclass
class ColumnFile:
    """Parsed column file: documents of sentences of (token, tag)."""

    docs: list[tuple[str, list[list[tuple[str, str]]]]]


def read_columns(path: str | Path, tag_col: int = -1, token_col: int = 0,
                 doc_mode: str = "auto") -> ColumnFile:
    """Read a CoNLL-style column file.

    Sentences are separated by blank lines. ``-DOCSTART-`` lines start a new
    document; with ``doc_mode="sentence"`` (or ``"auto"`` when the file has no
    ``-DOCSTART-`` line) every sentence is its own document. Document ids are
    sequential: ``doc-1``, ``doc-2``, ...
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    has_docstart = any(ln.startswith("-DOCSTART-") for ln in lines)
    if doc_mode not in ("auto", "docstart", "sentence"):
        raise ValueError("doc_mode must be auto, docstart or sentence")
    per_sentence = doc_mode == "sentence" or (doc_mode == "auto" and not has_docstart)
    docs: list[list[list[tuple[str, str]]]] = []
    current: list[list[tuple[str, str]]] = []
    sent: list[tuple[str, str]] = []

    def end_sentence() -> None:
        nonlocal sent
        if sent:
            current.append(sent)
            sent = []
            if per_sentence:
                end_doc()

    def end_doc() -> None:
        nonlocal current
        if current:
            docs.append(current)
            current = []

    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            end_sentence()
            continue
        if line.startswith("-DOCSTART-"):
            end_sentence()
            if not per_sentence:
                end_doc()
            continue
        cols = line.split()
        if len(cols) < 2:
            raise ValueError(f"{path}:{lineno}: expected at least two columns")
        try:
            sent.append((cols[token_col], cols[tag_col]))
        except IndexError:
            raise ValueError(f"{path}:{lineno}: expected at least two columns") from None
    end_sentence()
    end_doc()
    return ColumnFile([(f"doc-{i}", d) for i, d in enumerate(docs, 1)])


def load_columns(path: str | Path, scheme: str = "iob2", annotator: str | None = None,
                 tag_col: int = -1, token_col: int = 0, doc_mode: str = "auto",
                 repair: bool = False) -> tuple[Corpus, CheckReport]:
    """Spans from a column file, with its scheme check.

    Raises :class:`SchemeError` when the tags break the scheme, unless ``repair``
    is true; the report always lists every violation.
    """
    scheme = scheme_name(scheme)
    cf = read_columns(path, tag_col=tag_col, token_col=token_col, doc_mode=doc_mode)
    name = annotator or Path(path).stem
    corpus = Corpus(unit="token")
    report = CheckReport(scheme)
    spans: list[Span] = []
    for doc, sentences in cf.docs:
        offset = 0
        toks: list[str] = []
        breaks: list[int] = []
        for sent in sentences:
            try:
                got, viol = decode([t for _, t in sent], scheme, doc=doc, offset=offset)
            except ValueError as exc:
                raise ValueError(f"{path}: {doc}: {exc}") from None
            spans.extend(got)
            report.violations.extend(viol)
            report.n_tags += len(sent)
            report.n_chunks += len(got)
            breaks.append(offset)
            toks.extend(tok for tok, _ in sent)
            offset += len(sent)
        corpus.lengths[doc] = offset
        corpus.tokens[doc] = toks
        corpus.breaks[doc] = breaks
    corpus.sets[name] = sorted(spans)
    if report.violations and not repair:
        counts = ", ".join(f"{k} x{n}" for k, n in report.counts().most_common(5))
        raise SchemeError(
            f"{path}: {len(report.violations)} illegal {scheme} transitions ({counts}); "
            "run 'spanagree check' to list them, or pass --repair to accept the repaired spans",
            report.violations,
        )
    return corpus, report


def write_columns(corpus: Corpus, annotator: str, path: str | Path, scheme: str = "iob2") -> None:
    """Write one annotator's spans as a two-column file (token, tag)."""
    scheme = scheme_name(scheme)
    by_doc: dict[str, list[Span]] = {}
    for s in corpus.sets[annotator]:
        by_doc.setdefault(s.doc, []).append(s)
    out: list[str] = []
    for doc in corpus.documents():
        n = corpus.length(doc)
        toks = corpus.tokens.get(doc) or [f"t{i}" for i in range(n)]
        breaks = corpus.breaks.get(doc) or [0]
        bounds = list(zip(breaks, breaks[1:] + [n], strict=True))
        out.append("-DOCSTART- -X- -X- O")
        out.append("")
        spans = by_doc.get(doc, [])
        for a, b in bounds:
            inside = [s for s in spans if s.start < b and s.end > a]
            if any(s.start < a or s.end > b for s in inside):
                raise ValueError(f"a span in {doc} crosses a sentence boundary")
            for tok, tag in zip(toks[a:b], encode(inside, b - a, scheme, offset=a), strict=True):
                out.append(f"{tok} {tag}")
            out.append("")
    Path(path).write_text("\n".join(out), encoding="utf-8")
