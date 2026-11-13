import pytest

from spanagree.confusion import confusion
from spanagree.multi import against_majority, least_agreement, majority, pairwise_matrix
from spanagree.spans import Corpus, Span


def S(start, end, label="PER", doc="d"):
    return Span(doc, start, end, label)


def corpus():
    return Corpus(sets={
        "a": [S(0, 2), S(4, 5, "LOC"), S(0, 1, doc="e")],
        "b": [S(0, 2), S(4, 5, "ORG"), S(0, 1, doc="e")],
        "c": [S(0, 2), S(4, 5, "LOC"), S(3, 4, doc="e")],
    })


def test_pairwise_matrix_symmetric():
    m = pairwise_matrix(corpus())
    assert m["a"]["b"] == pytest.approx(2 / 3)
    assert m["b"]["a"] == m["a"]["b"]
    assert m["a"]["a"] == 1.0
    assert m["a"]["c"] == pytest.approx(2 / 3)


def test_majority_and_against_majority():
    c = corpus()
    assert majority(c, ["a", "b", "c"]) == sorted([S(0, 2), S(4, 5, "LOC"), S(0, 1, doc="e")])
    # Two others: a span needs both of them.
    assert majority(c, ["b", "c"]) == [S(0, 2)]
    res = {r.annotator: r for r in against_majority(c)}
    assert res["a"].majority_spans == 1
    assert res["a"].result.recall == pytest.approx(1.0)
    assert res["a"].result.precision == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        against_majority(Corpus(sets={"a": [], "b": []}))


def test_least_agreement_order():
    rows = least_agreement(corpus())
    assert [r[0] for r in rows] == ["e", "d"]
    assert rows[0][1] == pytest.approx(1 / 3)


def test_confusion_only_counts_matching_boundaries():
    ref = [S(0, 2), S(4, 5, "LOC"), S(7, 9, "ORG")]
    oth = [S(0, 2), S(4, 5, "ORG"), S(7, 8, "ORG")]
    c = confusion(ref, oth)
    assert c.labels == ["LOC", "ORG", "PER"]
    assert c.matrix() == [[0, 1, 0], [0, 0, 0], [0, 0, 1]]
    assert c.matched == 2 and c.agreeing == 1
    assert c.unmatched_ref == 1 and c.unmatched_other == 1
    assert c.as_dict()["same_label"] == 1
