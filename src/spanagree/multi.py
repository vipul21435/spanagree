"""More than two annotators: pairwise matrix, each against the others' majority,
and the documents with the least agreement."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from spanagree.match import PRF, score
from spanagree.spans import Corpus, Span


def pairwise_matrix(corpus: Corpus, names: Sequence[str] | None = None,
                    regime: str = "strict") -> dict[str, dict[str, float | None]]:
    """Symmetric matrix of pairwise F; the diagonal is 1 for non-empty sets."""
    names = list(names or corpus.annotators)
    out: dict[str, dict[str, float | None]] = {a: {} for a in names}
    for a in names:
        out[a][a] = 1.0 if corpus.sets[a] else None
    for a, b in combinations(names, 2):
        f = score(corpus.sets[a], corpus.sets[b], regime).micro.f
        out[a][b] = out[b][a] = f
    return out


def majority(corpus: Corpus, names: Sequence[str]) -> list[Span]:
    """Spans (same document, boundaries and label) chosen by more than half of ``names``."""
    votes: Counter[Span] = Counter()
    for n in names:
        votes.update(set(corpus.sets[n]))
    need = len(names) / 2.0
    return sorted(s for s, v in votes.items() if v > need)


@dataclass
class AgainstMajority:
    annotator: str
    majority_spans: int
    result: PRF

    def as_dict(self) -> dict[str, object]:
        return {"annotator": self.annotator, "majority_spans": self.majority_spans,
                **self.result.as_dict()}


def against_majority(corpus: Corpus, names: Sequence[str] | None = None,
                     regime: str = "strict") -> list[AgainstMajority]:
    """Each annotator scored against the strict majority of the others."""
    names = list(names or corpus.annotators)
    if len(names) < 3:
        raise ValueError("a majority of the others needs at least three annotators")
    out = []
    for a in names:
        maj = majority(corpus, [n for n in names if n != a])
        out.append(AgainstMajority(a, len(maj), score(maj, corpus.sets[a], regime).micro))
    return out


def least_agreement(corpus: Corpus, names: Sequence[str] | None = None, regime: str = "strict",
                    top: int = 10) -> list[tuple[str, float, int]]:
    """Documents ranked by mean pairwise F, lowest first: (document, mean F, spans).

    Pairs where neither annotator marked anything in the document are left out of
    the mean; documents with no spans at all are skipped.
    """
    names = list(names or corpus.annotators)
    by_doc: dict[str, dict[str, list[Span]]] = {}
    for n in names:
        for s in corpus.sets[n]:
            by_doc.setdefault(s.doc, {}).setdefault(n, []).append(s)
    rows = []
    for doc, sets in by_doc.items():
        fs = []
        for a, b in combinations(names, 2):
            f = score(sets.get(a, []), sets.get(b, []), regime).micro.f
            if f is not None:
                fs.append(f)
        if fs:
            rows.append((doc, sum(fs) / len(fs), sum(len(v) for v in sets.values())))
    rows.sort(key=lambda r: (r[1], -r[2], r[0]))
    return rows[:top]
