"""Agreement and typed error counts for labelled text spans."""

from spanagree.agreement import group_agreement, pair_agreement
from spanagree.bio import decode, encode, load_columns
from spanagree.confusion import confusion
from spanagree.diff import diff
from spanagree.match import REGIMES, align, score
from spanagree.spans import Corpus, Span, load_jsonl, spans_by_document

__all__ = [
    "REGIMES",
    "Corpus",
    "Span",
    "align",
    "confusion",
    "decode",
    "diff",
    "encode",
    "group_agreement",
    "load_columns",
    "load_jsonl",
    "pair_agreement",
    "score",
    "spans_by_document",
]
__version__ = "0.1.0"
