import itertools
import random

import pytest

from spanagree.match import REGIMES, align, compatible, overlap_groups, score
from spanagree.spans import Span


def S(start, end, label="PER", doc="d1"):
    return Span(doc, start, end, label)


def test_identical_sets_score_one_everywhere():
    ref = [S(0, 2), S(5, 6, "LOC"), S(3, 4, "ORG", doc="d2")]
    for regime in REGIMES:
        sc = score(ref, list(ref), regime)
        assert sc.micro.f == pytest.approx(1.0)
        assert sc.macro_f == pytest.approx(1.0)


def test_label_error_counts_only_in_label_aware_regimes():
    ref = [S(0, 2, "PER")]
    oth = [S(0, 2, "ORG")]
    assert score(ref, oth, "strict").micro.f == pytest.approx(0.0)
    assert score(ref, oth, "exact").micro.f == pytest.approx(1.0)
    assert score(ref, oth, "partial").micro.f == pytest.approx(1.0)
    assert score(ref, oth, "type").micro.f == pytest.approx(0.0)


def test_boundary_slip_gets_half_credit_in_partial():
    ref = [S(0, 3)]
    oth = [S(1, 3)]
    assert score(ref, oth, "strict").micro.f == pytest.approx(0.0)
    assert score(ref, oth, "exact").micro.f == pytest.approx(0.0)
    assert score(ref, oth, "partial").micro.f == pytest.approx(0.5)
    assert score(ref, oth, "type").micro.f == pytest.approx(1.0)


def test_precision_recall_hand_case():
    ref = [S(0, 1), S(2, 3), S(4, 5, "LOC"), S(6, 7, "LOC")]
    oth = [S(0, 1), S(4, 5, "LOC")]
    sc = score(ref, oth, "strict")
    assert sc.micro.precision == pytest.approx(1.0)
    assert sc.micro.recall == pytest.approx(0.5)
    assert sc.micro.f == pytest.approx(2 / 3)
    assert sc.per_label["PER"].recall == pytest.approx(0.5)
    assert sc.macro_f == pytest.approx(2 / 3)


def test_empty_sets_give_none():
    sc = score([], [], "strict")
    assert sc.micro.f is None and sc.micro.precision is None and sc.micro.recall is None
    sc = score([S(0, 1)], [], "strict")
    assert sc.micro.precision is None
    assert sc.micro.recall == pytest.approx(0.0)
    assert sc.micro.f == pytest.approx(0.0)


def test_alignment_is_one_to_one_and_prefers_more_pairs():
    # Largest-overlap-first would pair 0-10 with 5-10 (overlap 5) and leave
    # 8-10 and 0-4 unpaired; the optimal pairing keeps both pairs.
    ref = [S(0, 10), S(8, 10)]
    oth = [S(5, 10), S(0, 4)]
    al = align(ref, oth, "partial")
    assert len(al.pairs) == 2
    assert not al.unmatched_ref and not al.unmatched_other
    assert (S(0, 10), S(0, 4)) in al.pairs


def test_unknown_regime_rejected():
    with pytest.raises(ValueError):
        align([], [], "fuzzy")
    with pytest.raises(ValueError):
        compatible(S(0, 1), S(0, 1), "fuzzy")


def test_spans_in_different_documents_never_match():
    sc = score([S(0, 2, doc="a")], [S(0, 2, doc="b")], "partial")
    assert sc.micro.credit == pytest.approx(0.0)


def test_overlap_groups_cover_everything_once():
    ref = [S(0, 3), S(5, 8), S(20, 21)]
    oth = [S(2, 6), S(10, 11)]
    groups = overlap_groups(ref, oth)
    seen_r = sorted(i for g in groups for i in g[0])
    seen_o = sorted(j for g in groups for j in g[1])
    assert seen_r == [0, 1, 2] and seen_o == [0, 1]
    assert ([0, 1], [0]) in groups


def _objective(pairs):
    return (len(pairs), sum(a.start == b.start and a.end == b.end for a, b in pairs),
            sum(a.overlap(b) for a, b in pairs))


def _brute(ref, oth, regime):
    best = (0, 0, 0)
    cand = [(i, j) for i in range(len(ref)) for j in range(len(oth))
            if compatible(ref[i], oth[j], regime)]
    for k in range(len(cand) + 1):
        for combo in itertools.combinations(cand, k):
            if len({i for i, _ in combo}) < k or len({j for _, j in combo}) < k:
                continue
            best = max(best, _objective([(ref[i], oth[j]) for i, j in combo]))
    return best


def _random_spans(rng, n, length=12, labels=("A", "B")):
    out = []
    for _ in range(n):
        s = rng.randrange(length - 1)
        e = rng.randrange(s + 1, min(length, s + 5) + 1)
        out.append(S(s, e, rng.choice(labels)))
    return out


@pytest.mark.parametrize("regime", REGIMES)
def test_alignment_matches_brute_force(regime):
    rng = random.Random(7)
    for _ in range(150):
        ref = _random_spans(rng, rng.randrange(0, 5))
        oth = _random_spans(rng, rng.randrange(0, 5))
        al = align(ref, oth, regime)
        assert _objective(al.pairs) == _brute(ref, oth, regime)
        assert len(al.pairs) + len(al.unmatched_ref) == len(ref)
        assert len(al.pairs) + len(al.unmatched_other) == len(oth)


def test_alignment_ignores_input_order():
    rng = random.Random(3)
    for _ in range(50):
        ref = _random_spans(rng, 4)
        oth = _random_spans(rng, 4)
        a1 = align(ref, oth, "partial")
        a2 = align(list(reversed(ref)), list(reversed(oth)), "partial")
        assert _objective(a1.pairs) == _objective(a2.pairs)
