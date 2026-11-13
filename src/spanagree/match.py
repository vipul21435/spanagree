"""One-to-one alignment of two span sets and scoring under four matching regimes.

Regimes (the names follow the SemEval-2013 DDIExtraction evaluation):

``strict``
    same start, same end and same label.
``exact``
    same start and same end; the label is ignored.
``partial``
    the spans overlap; the label is ignored. A pair with identical boundaries earns
    full credit and any other overlapping pair half credit, as in the MUC scoring
    of partial matches.
``type``
    the spans overlap and carry the same label.

Within each regime the alignment is one-to-one and optimal: among all one-to-one
pairings of compatible spans it maximises, in this order, the number of pairs,
the number of pairs with identical boundaries, and the total overlap. It is
solved as an assignment problem (``scipy.optimize.linear_sum_assignment``) on
each connected group of overlapping spans, so it never depends on the order in
which spans were listed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from spanagree.spans import Span, spans_by_document

REGIMES: tuple[str, ...] = ("strict", "exact", "partial", "type")


def compatible(a: Span, b: Span, regime: str) -> bool:
    """Whether ``a`` and ``b`` may be paired under ``regime``."""
    if a.doc != b.doc:
        return False
    if regime == "strict":
        return a.start == b.start and a.end == b.end and a.label == b.label
    if regime == "exact":
        return a.start == b.start and a.end == b.end
    if regime == "partial":
        return a.overlap(b) > 0
    if regime == "type":
        return a.overlap(b) > 0 and a.label == b.label
    raise ValueError(f"unknown regime {regime!r}; choose from {', '.join(REGIMES)}")


def credit(a: Span, b: Span, regime: str) -> float:
    """Credit for an aligned pair: 0.5 for a non-exact partial match, else 1."""
    if regime == "partial" and (a.start != b.start or a.end != b.end):
        return 0.5
    return 1.0


@dataclass
class Alignment:
    """Result of aligning a reference set with another set."""

    pairs: list[tuple[Span, Span]] = field(default_factory=list)
    unmatched_ref: list[Span] = field(default_factory=list)
    unmatched_other: list[Span] = field(default_factory=list)


def overlap_groups(ref: Sequence[Span], other: Sequence[Span]) -> list[tuple[list[int], list[int]]]:
    """Connected groups of the overlap graph between two span lists of one document.

    Returns ``(ref_indices, other_indices)`` per group, including singletons, in
    position order.
    """
    items = [(s.start, s.end, 0, i) for i, s in enumerate(ref)]
    items += [(s.start, s.end, 1, i) for i, s in enumerate(other)]
    items.sort()
    groups: list[tuple[list[int], list[int]]] = []
    # Sweep: overlap between the two sides only; spans on the same side join a
    # group only through a span of the other side.
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    active: list[int] = []
    for k, (start, _end, _side, _i) in enumerate(items):
        active = [j for j in active if items[j][1] > start]
        for j in active:
            if items[j][2] != items[k][2]:
                parent[find(j)] = find(k)
        active.append(k)
    by_root: dict[int, tuple[list[int], list[int]]] = {}
    order: list[int] = []
    for k, (_s, _e, side, i) in enumerate(items):
        r = find(k)
        if r not in by_root:
            by_root[r] = ([], [])
            order.append(r)
        by_root[r][side].append(i)
    for r in order:
        groups.append(by_root[r])
    return groups


def _assign(ref: Sequence[Span], other: Sequence[Span], regime: str) -> list[tuple[int, int]]:
    """Optimal one-to-one pairing within one overlap group."""
    if not ref or not other:
        return []
    if len(ref) == 1 and len(other) == 1:
        return [(0, 0)] if compatible(ref[0], other[0], regime) else []
    ov = np.zeros((len(ref), len(other)))
    exact = np.zeros_like(ov)
    ok = np.zeros(ov.shape, dtype=bool)
    for i, a in enumerate(ref):
        for j, b in enumerate(other):
            if compatible(a, b, regime):
                ok[i, j] = True
                ov[i, j] = a.overlap(b)
                exact[i, j] = float(a.start == b.start and a.end == b.end)
    if not ok.any():
        return []
    big_b = ov[ok].sum() + 1.0
    big_a = (min(ov.shape) + 1) * big_b
    weight = np.where(ok, big_a + big_b * exact + ov, 0.0)
    rows, cols = linear_sum_assignment(weight, maximize=True)
    return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if ok[i, j]]


def align(ref: Sequence[Span], other: Sequence[Span], regime: str) -> Alignment:
    """Align two span sets (any number of documents) under ``regime``."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; choose from {', '.join(REGIMES)}")
    by_ref = spans_by_document(ref)
    by_other = spans_by_document(other)
    out = Alignment()
    for doc in sorted(set(by_ref) | set(by_other)):
        rs = by_ref.get(doc, [])
        os_ = by_other.get(doc, [])
        for gi, gj in overlap_groups(rs, os_):
            gr = [rs[i] for i in gi]
            go = [os_[j] for j in gj]
            pairs = _assign(gr, go, regime)
            used_r = {i for i, _ in pairs}
            used_o = {j for _, j in pairs}
            out.pairs.extend((gr[i], go[j]) for i, j in pairs)
            out.unmatched_ref.extend(s for i, s in enumerate(gr) if i not in used_r)
            out.unmatched_other.extend(s for j, s in enumerate(go) if j not in used_o)
    return out


@dataclass
class PRF:
    """Precision, recall and F from matched credit and set sizes.

    ``precision`` is ``None`` when the other set is empty, ``recall`` when the
    reference set is empty, and ``f`` when both are.
    """

    credit: float
    n_ref: int
    n_other: int

    @property
    def precision(self) -> float | None:
        return self.credit / self.n_other if self.n_other else None

    @property
    def recall(self) -> float | None:
        return self.credit / self.n_ref if self.n_ref else None

    @property
    def f(self) -> float | None:
        total = self.n_ref + self.n_other
        return 2.0 * self.credit / total if total else None

    def as_dict(self) -> dict[str, float | int | None]:
        return {"precision": self.precision, "recall": self.recall, "f": self.f,
                "credit": self.credit, "n_ref": self.n_ref, "n_other": self.n_other}


@dataclass
class Score:
    """Scores for one regime: micro, macro and per label."""

    regime: str
    micro: PRF
    per_label: dict[str, PRF]

    @property
    def macro_f(self) -> float | None:
        fs = [p.f for p in self.per_label.values() if p.f is not None]
        return sum(fs) / len(fs) if fs else None

    @property
    def macro_precision(self) -> float | None:
        ps = [p.precision for p in self.per_label.values() if p.precision is not None]
        return sum(ps) / len(ps) if ps else None

    @property
    def macro_recall(self) -> float | None:
        rs = [p.recall for p in self.per_label.values() if p.recall is not None]
        return sum(rs) / len(rs) if rs else None

    def as_dict(self) -> dict[str, object]:
        return {
            "regime": self.regime,
            "micro": self.micro.as_dict(),
            "macro": {"precision": self.macro_precision, "recall": self.macro_recall,
                      "f": self.macro_f},
            "per_label": {k: v.as_dict() for k, v in sorted(self.per_label.items())},
        }


def score(ref: Sequence[Span], other: Sequence[Span], regime: str = "strict") -> Score:
    """Precision, recall and F of ``other`` against ``ref`` under one regime.

    Per label, recall uses the reference span's label and precision the other
    span's label (they coincide for ``strict`` and ``type``). Macro values are
    unweighted means over the labels seen in either set.
    """
    al = align(ref, other, regime)
    ref_credit: dict[str, float] = defaultdict(float)
    oth_credit: dict[str, float] = defaultdict(float)
    total = 0.0
    for a, b in al.pairs:
        c = credit(a, b, regime)
        total += c
        ref_credit[a.label] += c
        oth_credit[b.label] += c
    n_ref: dict[str, int] = defaultdict(int)
    n_oth: dict[str, int] = defaultdict(int)
    for s in ref:
        n_ref[s.label] += 1
    for s in other:
        n_oth[s.label] += 1
    labels = sorted(set(n_ref) | set(n_oth))
    per_label = {}
    for lab in labels:
        # Precision and recall can carry different credit when labels differ
        # inside a pair, so keep both and derive F from them.
        per_label[lab] = _LabelPRF(ref_credit[lab], oth_credit[lab], n_ref[lab], n_oth[lab])
    return Score(regime, PRF(total, len(ref), len(other)), per_label)


class _LabelPRF(PRF):
    """PRF where recall and precision credit may differ (label-agnostic regimes)."""

    def __init__(self, ref_credit: float, other_credit: float, n_ref: int, n_other: int):
        super().__init__(credit=ref_credit, n_ref=n_ref, n_other=n_other)
        self.other_credit = other_credit

    @property
    def precision(self) -> float | None:
        return self.other_credit / self.n_other if self.n_other else None

    @property
    def f(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return 0.0 if (self.n_ref + self.n_other) else None
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    def as_dict(self) -> dict[str, float | int | None]:
        d = super().as_dict()
        d["credit"] = self.credit
        d["credit_other"] = self.other_credit
        return d


def score_all(ref: Sequence[Span], other: Sequence[Span]) -> dict[str, Score]:
    """Scores under every regime."""
    return {r: score(ref, other, r) for r in REGIMES}
