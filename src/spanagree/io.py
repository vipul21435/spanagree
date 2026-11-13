"""Load any mix of JSON lines and column files into one corpus."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from spanagree.bio import CheckReport, load_columns
from spanagree.spans import Corpus, load_jsonl

JSONL_SUFFIXES = (".jsonl", ".json", ".ndjson")


def is_jsonl(path: str | Path) -> bool:
    return Path(path).suffix.lower() in JSONL_SUFFIXES


def load_inputs(paths: Sequence[str | Path], scheme: str = "iob2", repair: bool = False,
                tag_col: int = -1, doc_mode: str = "auto", unit: str = "char",
                names: Sequence[str] | None = None) -> tuple[Corpus, list[CheckReport]]:
    """Merge every input into one corpus.

    A JSON lines file may hold several annotators; a column file holds one, named
    after the file stem (or the matching entry of ``names``). Column files must
    agree on documents and token counts.
    """
    corpus = Corpus()
    reports: list[CheckReport] = []
    seen_stems: dict[str, int] = {}
    for k, path in enumerate(paths):
        stem = Path(path).stem
        if names and k < len(names):
            name = names[k]
        else:
            seen_stems[stem] = seen_stems.get(stem, 0) + 1
            name = stem if seen_stems[stem] == 1 else f"{stem}-{seen_stems[stem]}"
        if is_jsonl(path):
            part = load_jsonl(path, annotator=name, unit=unit)
        else:
            part, report = load_columns(path, scheme=scheme, annotator=name, tag_col=tag_col,
                                        doc_mode=doc_mode, repair=repair)
            reports.append(report)
            if corpus.tokens:
                _check_same_tokens(corpus, part, path)
        corpus = corpus.merge(part)
    return corpus, reports


def _check_same_tokens(corpus: Corpus, part: Corpus, path: str | Path) -> None:
    if set(corpus.tokens) != set(part.tokens):
        raise ValueError(f"{path}: the documents differ from the first column file "
                         f"({len(part.tokens)} against {len(corpus.tokens)})")
    for doc, toks in part.tokens.items():
        if len(toks) != len(corpus.tokens[doc]):
            raise ValueError(f"{path}: {doc} has {len(toks)} tokens, the first column file "
                             f"has {len(corpus.tokens[doc])}")


def pick_pair(corpus: Corpus, ref: str | None, other: str | None) -> tuple[str, str]:
    """Resolve the reference and other annotator names (defaults: the first two)."""
    names = corpus.annotators
    if ref is None or other is None:
        rest = [n for n in names if n not in (ref, other)]
        if ref is None:
            ref = rest.pop(0) if rest else None
        if other is None:
            other = rest.pop(0) if rest else None
    if ref is None or other is None:
        raise ValueError(f"need two annotators, found {len(names)}: {', '.join(names) or 'none'}")
    for n in (ref, other):
        if n not in corpus.sets:
            raise ValueError(f"no annotator {n!r}; found {', '.join(names)}")
    return ref, other
