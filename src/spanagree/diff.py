"""Typed difference between a reference span set and another span set.

Spans of the two sets are grouped by overlap (connected components of the
bipartite overlap graph, per document). Each group is then typed:

==================  =====================================================
one ref, no other   ``missed``
no ref, one other   ``extra``
one ref, one other  ``identical``, ``label`` (same boundaries, other label),
                    ``boundary`` (same label, other boundaries) or ``both``
one ref, k others   ``split`` when the k others are disjoint and none has
                    the reference boundaries
k refs, one other   ``merged``, the mirror case
anything else       paired by the optimal one-to-one alignment of the
                    ``partial`` regime; each pair is typed as above and the
                    leftovers are ``missed`` or ``extra`` (the group is also
                    counted under ``tangled``)
==================  =====================================================

Boundary shifts are ``other.start - ref.start`` (left side) and
``other.end - ref.end`` (right side), in offset units: tokens for column files.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from spanagree.match import _assign, overlap_groups
from spanagree.spans import Span, spans_by_document

CATEGORIES: tuple[str, ...] = (
    "identical", "label", "boundary", "both", "missed", "extra", "split", "merged")


@dataclass(frozen=True)
class DiffItem:
    """One typed difference (or an identical pair)."""

    kind: str
    ref: tuple[Span, ...]
    other: tuple[Span, ...]

    @property
    def doc(self) -> str:
        return (self.ref or self.other)[0].doc

    @property
    def example_id(self) -> str:
        left = "+".join(f"{s.start}-{s.end} {s.label}" for s in self.ref) or "none"
        right = "+".join(f"{s.start}-{s.end} {s.label}" for s in self.other) or "none"
        return f"{self.doc}: {left} -> {right}"

    @property
    def shift(self) -> tuple[int, int] | None:
        """(left, right) boundary shift of a one-to-one pair, else ``None``."""
        if len(self.ref) == 1 and len(self.other) == 1:
            a, b = self.ref[0], self.other[0]
            return b.start - a.start, b.end - a.end
        return None

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "kind": self.kind, "document": self.doc,
            "ref": [[s.start, s.end, s.label] for s in self.ref],
            "other": [[s.start, s.end, s.label] for s in self.other],
        }
        if self.shift is not None and self.kind in ("boundary", "both"):
            d["shift"] = list(self.shift)
        return d


@dataclass
class DiffResult:
    """All typed items plus summary counts."""

    items: list[DiffItem] = field(default_factory=list)
    tangled_groups: int = 0

    def counts(self) -> dict[str, int]:
        c = Counter(it.kind for it in self.items)
        return {k: c.get(k, 0) for k in CATEGORIES}

    @property
    def n_differences(self) -> int:
        """Items that are not ``identical`` (a split or a merge counts once)."""
        return sum(1 for it in self.items if it.kind != "identical")

    def boundary_sides(self) -> dict[str, int]:
        """Boundary-changed pairs (``boundary`` and ``both``) by the side that moved."""
        out = {"left": 0, "right": 0, "both sides": 0}
        for it in self.items:
            if it.kind in ("boundary", "both") and it.shift is not None:
                left, right = it.shift
                out["both sides" if left and right else "left" if left else "right"] += 1
        return out

    def shifts(self) -> dict[str, dict[int, int]]:
        """Histogram of non-zero shifts per side."""
        left: Counter[int] = Counter()
        right: Counter[int] = Counter()
        for it in self.items:
            if it.kind in ("boundary", "both") and it.shift is not None:
                if it.shift[0]:
                    left[it.shift[0]] += 1
                if it.shift[1]:
                    right[it.shift[1]] += 1
        return {"left": dict(sorted(left.items())), "right": dict(sorted(right.items()))}

    def label_changes(self) -> Counter[tuple[str, str]]:
        """(ref label, other label) for ``label`` and ``both`` items."""
        return Counter((it.ref[0].label, it.other[0].label) for it in self.items
                       if it.kind in ("label", "both"))

    def examples(self, kind: str, limit: int = 5) -> list[str]:
        return [it.example_id for it in self.items if it.kind == kind][:limit]

    def as_dict(self, examples: int = 5, items: bool = False) -> dict[str, object]:
        d: dict[str, object] = {
            "counts": self.counts(),
            "differences": self.n_differences,
            "tangled_groups": self.tangled_groups,
            "boundary_sides": self.boundary_sides(),
            "shifts": {side: {str(k): v for k, v in h.items()}
                       for side, h in self.shifts().items()},
            "label_changes": [{"ref": a, "other": b, "count": n}
                              for (a, b), n in self.label_changes().most_common()],
            "examples": {k: self.examples(k, examples) for k in CATEGORIES if k != "identical"},
        }
        if items:
            d["items"] = [it.as_dict() for it in self.items]
        return d


def _pair_kind(a: Span, b: Span) -> str:
    same_bounds = a.start == b.start and a.end == b.end
    same_label = a.label == b.label
    if same_bounds:
        return "identical" if same_label else "label"
    return "boundary" if same_label else "both"


def _disjoint(spans: Sequence[Span]) -> bool:
    ordered = sorted(spans)
    return all(x.end <= y.start for x, y in zip(ordered, ordered[1:], strict=False))


def _same_bounds(a: Span, b: Span) -> bool:
    return a.start == b.start and a.end == b.end


def diff(ref: Sequence[Span], other: Sequence[Span]) -> DiffResult:
    """Typed difference of ``other`` against ``ref``."""
    out = DiffResult()
    by_ref = spans_by_document(ref)
    by_other = spans_by_document(other)
    for doc in sorted(set(by_ref) | set(by_other)):
        rs = by_ref.get(doc, [])
        os_ = by_other.get(doc, [])
        for gi, gj in overlap_groups(rs, os_):
            gr = [rs[i] for i in gi]
            go = [os_[j] for j in gj]
            out.items.extend(_type_group(gr, go, out))
    return out


def _type_group(gr: list[Span], go: list[Span], out: DiffResult) -> list[DiffItem]:
    if len(gr) == 1 and not go:
        return [DiffItem("missed", (gr[0],), ())]
    if len(go) == 1 and not gr:
        return [DiffItem("extra", (), (go[0],))]
    if len(gr) == 1 and len(go) == 1:
        return [DiffItem(_pair_kind(gr[0], go[0]), (gr[0],), (go[0],))]
    if len(gr) == 1 and _disjoint(go) and not any(_same_bounds(gr[0], b) for b in go):
        return [DiffItem("split", (gr[0],), tuple(sorted(go)))]
    if len(go) == 1 and _disjoint(gr) and not any(_same_bounds(a, go[0]) for a in gr):
        return [DiffItem("merged", tuple(sorted(gr)), (go[0],))]
    out.tangled_groups += 1
    pairs = _assign(gr, go, "partial")
    items = [DiffItem(_pair_kind(gr[i], go[j]), (gr[i],), (go[j],)) for i, j in pairs]
    used_r = {i for i, _ in pairs}
    used_o = {j for _, j in pairs}
    items += [DiffItem("missed", (s,), ()) for i, s in enumerate(gr) if i not in used_r]
    items += [DiffItem("extra", (), (s,)) for j, s in enumerate(go) if j not in used_o]
    return items
