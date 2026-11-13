"""Label confusion restricted to spans whose boundaries match exactly."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from spanagree.match import align
from spanagree.spans import Span


@dataclass
class Confusion:
    """Counts of (reference label, other label) over boundary-matched pairs."""

    counts: Counter[tuple[str, str]]
    unmatched_ref: int
    unmatched_other: int

    @property
    def labels(self) -> list[str]:
        return sorted({a for a, _ in self.counts} | {b for _, b in self.counts})

    @property
    def matched(self) -> int:
        return sum(self.counts.values())

    @property
    def agreeing(self) -> int:
        return sum(n for (a, b), n in self.counts.items() if a == b)

    def matrix(self) -> list[list[int]]:
        labs = self.labels
        return [[self.counts.get((a, b), 0) for b in labs] for a in labs]

    def as_dict(self) -> dict[str, object]:
        return {"labels": self.labels, "matrix": self.matrix(), "matched": self.matched,
                "same_label": self.agreeing, "ref_without_boundary_match": self.unmatched_ref,
                "other_without_boundary_match": self.unmatched_other}


def confusion(ref: Sequence[Span], other: Sequence[Span]) -> Confusion:
    """Rows are reference labels, columns other labels; only identical boundaries count."""
    al = align(ref, other, "exact")
    counts: Counter[tuple[str, str]] = Counter((a.label, b.label) for a, b in al.pairs)
    return Confusion(counts, len(al.unmatched_ref), len(al.unmatched_other))
