import random

from spanagree.diff import CATEGORIES, diff
from spanagree.spans import Span


def S(start, end, label="PER", doc="d"):
    return Span(doc, start, end, label)


def test_every_category_on_a_hand_built_case():
    ref = [S(0, 2), S(3, 4, "LOC"), S(5, 7, "ORG"), S(8, 9, "MISC"), S(10, 14, "ORG"),
           S(15, 16), S(17, 18), S(20, 22, "LOC")]
    oth = [S(0, 2), S(3, 4, "ORG"), S(5, 8, "ORG"), S(8, 10, "LOC"), S(10, 12, "ORG"),
           S(12, 14, "ORG"), S(15, 18), S(24, 25)]
    res = diff(ref, oth)
    assert res.counts() == {"identical": 1, "label": 1, "boundary": 1, "both": 1, "missed": 1,
                            "extra": 1, "split": 1, "merged": 1}
    assert res.n_differences == 7
    assert res.boundary_sides() == {"left": 0, "right": 2, "both sides": 0}
    assert res.shifts() == {"left": {}, "right": {1: 2}}
    assert res.label_changes() == {("LOC", "ORG"): 1, ("MISC", "LOC"): 1}
    assert res.examples("missed") == ["d: 20-22 LOC -> none"]
    assert res.tangled_groups == 0


def test_left_and_both_side_shifts():
    res = diff([S(2, 5), S(10, 12)], [S(1, 5), S(11, 14)])
    assert res.boundary_sides() == {"left": 1, "right": 0, "both sides": 1}
    assert res.shifts() == {"left": {-1: 1, 1: 1}, "right": {2: 1}}


def test_one_ref_with_identical_and_extra_overlap_is_not_split():
    # The other set has the same span plus a nested one: a tangled group.
    res = diff([S(0, 4)], [S(0, 4), S(1, 2, "LOC")])
    assert res.counts()["identical"] == 1
    assert res.counts()["extra"] == 1
    assert res.counts()["split"] == 0
    assert res.tangled_groups == 1


def test_documents_are_kept_apart():
    res = diff([S(0, 2, doc="a")], [S(0, 2, doc="b")])
    assert res.counts()["missed"] == 1 and res.counts()["extra"] == 1


def test_json_shape():
    res = diff([S(0, 2)], [S(0, 3)])
    d = res.as_dict(items=True)
    assert d["counts"]["boundary"] == 1
    assert d["items"][0]["shift"] == [0, 1]
    assert set(d["examples"]) == set(CATEGORIES) - {"identical"}


def test_every_span_is_accounted_for_once():
    rng = random.Random(11)
    for _ in range(300):
        def rand(n):
            out = []
            for _ in range(n):
                s = rng.randrange(20)
                out.append(S(s, s + rng.randrange(1, 4), rng.choice("AB")))
            return out
        ref, oth = rand(rng.randrange(6)), rand(rng.randrange(6))
        res = diff(ref, oth)
        assert sorted(s for it in res.items for s in it.ref) == sorted(ref)
        assert sorted(s for it in res.items for s in it.other) == sorted(oth)


def test_identical_sets():
    spans = [S(0, 1), S(3, 5, "LOC")]
    res = diff(spans, list(spans))
    assert res.n_differences == 0
    assert res.counts()["identical"] == 2
