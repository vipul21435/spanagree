import krippendorff
import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters
from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss

from spanagree.agreement import (
    bootstrap,
    cohen_kappa,
    group_agreement,
    krippendorff_alpha_nominal,
    label_index,
    pair_agreement,
    position_labels,
)
from spanagree.match import score
from spanagree.spans import Corpus, Span


def random_corpus(rng, annotators=("a", "b"), docs=6, length=30, tokens=False):
    sets = {}
    for name in annotators:
        spans = []
        for d in range(docs):
            pos = 0
            while pos < length - 3:
                pos += int(rng.integers(0, 5))
                n = int(rng.integers(1, 4))
                if pos + n > length:
                    break
                spans.append(Span(f"doc-{d}", pos, pos + n, str(rng.choice(["PER", "LOC", "ORG"]))))
                pos += n
        sets[name] = spans
    lengths = {f"doc-{d}": length for d in range(docs)}
    toks = {}
    if tokens:
        toks = {f"doc-{d}": ["x" * int(rng.integers(1, 6)) for _ in range(length)]
                for d in range(docs)}
    return Corpus(sets=sets, lengths=lengths, tokens=toks)


def flat_labels(corpus, names):
    labels = label_index(corpus, names)
    arrays = position_labels(corpus, names, labels)
    return np.concatenate([arrays[d] for d in corpus.documents()], axis=1), labels


def test_cohen_kappa_matches_sklearn():
    rng = np.random.default_rng(1)
    for _ in range(5):
        c = random_corpus(rng)
        flat, labels = flat_labels(c, ["a", "b"])
        res = pair_agreement(c, "a", "b", n_boot=0)
        assert res.kappa_all.value == pytest.approx(cohen_kappa_score(flat[0], flat[1]), abs=1e-12)
        keep = (flat[0] != 0) | (flat[1] != 0)
        ref = cohen_kappa_score(flat[0][keep], flat[1][keep])
        assert res.kappa_marked.value == pytest.approx(ref, abs=1e-12)
        assert res.marked_positions == int(keep.sum())


def test_fleiss_matches_statsmodels():
    rng = np.random.default_rng(2)
    for _ in range(5):
        c = random_corpus(rng, annotators=("a", "b", "c", "d"))
        flat, labels = flat_labels(c, c.annotators)
        table, _ = aggregate_raters(flat.T, n_cat=len(labels))
        res = group_agreement(c, n_boot=0)
        assert res.fleiss_all.value == pytest.approx(sm_fleiss(table), abs=1e-12)
        keep = (flat != 0).any(axis=0)
        table, _ = aggregate_raters(flat[:, keep].T, n_cat=len(labels))
        assert res.fleiss_marked.value == pytest.approx(sm_fleiss(table), abs=1e-12)


def test_alpha_matches_krippendorff_package_on_characters():
    rng = np.random.default_rng(3)
    for names in (("a", "b"), ("a", "b", "c")):
        c = random_corpus(rng, annotators=names, tokens=True)
        flat, _ = flat_labels(c, list(names))
        widths = np.concatenate([[len(t) for t in c.tokens[d]] for d in c.documents()])
        chars = np.repeat(flat, widths, axis=1).astype(float)
        ref = krippendorff.alpha(reliability_data=chars, level_of_measurement="nominal")
        if len(names) == 2:
            got = pair_agreement(c, "a", "b", n_boot=0).alpha.value
        else:
            got = group_agreement(c, n_boot=0).alpha.value
        assert got == pytest.approx(ref, abs=1e-9)


def test_pair_f_equals_strict_score_and_is_symmetric():
    rng = np.random.default_rng(4)
    c = random_corpus(rng)
    ab = pair_agreement(c, "a", "b", n_boot=0)
    ba = pair_agreement(c, "b", "a", n_boot=0)
    for r in ("strict", "exact", "partial", "type"):
        assert ab.f[r].value == pytest.approx(score(c.sets["a"], c.sets["b"], r).micro.f)
        assert ab.f[r].value == pytest.approx(ba.f[r].value)


def test_identical_annotations_agree_perfectly():
    rng = np.random.default_rng(5)
    c = random_corpus(rng, annotators=("a",))
    c.sets["b"] = list(c.sets["a"])
    res = pair_agreement(c, "a", "b", n_boot=200)
    assert res.f["strict"].value == pytest.approx(1.0)
    assert res.kappa_all.value == pytest.approx(1.0)
    assert res.alpha.value == pytest.approx(1.0)
    assert res.f["strict"].low == pytest.approx(1.0)


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(6)
    c = random_corpus(rng, docs=20)
    r1 = pair_agreement(c, "a", "b", n_boot=300, seed=9)
    r2 = pair_agreement(c, "a", "b", n_boot=300, seed=9)
    r3 = pair_agreement(c, "a", "b", n_boot=300, seed=10)
    assert r1.as_dict() == r2.as_dict()
    assert r1.f["strict"].low != r3.f["strict"].low or r1.f["strict"].high != r3.f["strict"].high
    for est in (r1.f["strict"], r1.kappa_all, r1.alpha):
        assert est.low <= est.value <= est.high


def test_bootstrap_edge_cases():
    est = bootstrap(np.zeros((0, 3)), lambda s: s[:, 0], n_boot=10)
    assert est.low is None
    est = bootstrap(np.array([[1.0, 0.0]]), lambda s: np.full(s.shape[0], np.nan), n_boot=10)
    assert est.value is None


def test_degenerate_coefficients():
    assert cohen_kappa(np.zeros((2, 2))) is None
    assert cohen_kappa(np.array([[5.0, 0.0], [0.0, 0.0]])) is None
    assert krippendorff_alpha_nominal(np.zeros((2, 2))) is None


def test_span_past_document_length_is_an_error():
    c = Corpus(sets={"a": [Span("d", 0, 9, "X")], "b": []}, lengths={"d": 5})
    with pytest.raises(ValueError, match="past the document length"):
        pair_agreement(c, "a", "b", n_boot=0)
