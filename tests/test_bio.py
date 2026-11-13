import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from spanagree.bio import (
    SchemeError,
    decode,
    encode,
    load_columns,
    parse_tag,
    read_columns,
    scheme_name,
    write_columns,
)
from spanagree.spans import Span


def spans_of(tags, scheme):
    got, viol = decode(tags, scheme, doc="d")
    return [(s.start, s.end, s.label) for s in got], [v.kind for v in viol]


def test_iob2_decode_and_violation():
    got = spans_of(["B-PER", "I-PER", "O", "B-LOC"], "iob2")
    assert got == ([(0, 2, "PER"), (3, 4, "LOC")], [])
    got, viol = spans_of(["O", "I-PER", "I-PER"], "iob2")
    assert got == [(1, 3, "PER")]
    assert viol == ["O -> I-PER"]
    got, viol = spans_of(["B-PER", "I-LOC"], "iob2")
    assert got == [(0, 1, "PER"), (1, 2, "LOC")]
    assert viol == ["B-PER -> I-LOC"]


def test_iob1_decode():
    tags = ["I-PER", "I-PER", "B-PER", "O", "I-LOC", "I-ORG"]
    got, viol = spans_of(tags, "iob1")
    assert got == [(0, 2, "PER"), (2, 3, "PER"), (4, 5, "LOC"), (5, 6, "ORG")]
    assert viol == []
    got, viol = spans_of(["O", "B-PER"], "iob1")
    assert got == [(1, 2, "PER")]
    assert viol == ["O -> B-PER"]


def test_bilou_decode_and_violations():
    tags = ["U-PER", "B-LOC", "I-LOC", "L-LOC", "O", "S-ORG", "B-MISC", "E-MISC"]
    got, viol = spans_of(tags, "bilou")
    assert got == [(0, 1, "PER"), (1, 4, "LOC"), (5, 6, "ORG"), (6, 8, "MISC")]
    assert viol == []
    got, viol = spans_of(["B-PER", "O", "L-LOC", "B-ORG"], "bilou")
    assert got == [(0, 1, "PER"), (2, 3, "LOC"), (3, 4, "ORG")]
    assert viol == ["B-PER -> O", "O -> L-LOC", "B-ORG -> <end>"]
    got, viol = spans_of(["B-PER", "U-LOC", "I-ORG"], "bilou")
    assert viol == ["B-PER -> U-LOC", "U-LOC -> I-ORG", "I-ORG -> <end>"]


@pytest.mark.parametrize("tag", ["X", "B", "B-", "Q-PER", "BPER"])
def test_malformed_tags_raise(tag):
    with pytest.raises(ValueError):
        decode(["O", tag], "iob2")


def test_bilou_prefix_rejected_in_iob2():
    with pytest.raises(ValueError):
        parse_tag("U-PER", "iob2")


def test_scheme_aliases():
    assert scheme_name("BIO") == "iob2"
    assert scheme_name("bioes") == "bilou"
    with pytest.raises(ValueError):
        scheme_name("xyz")


def test_encode_rejects_overlap_and_outside():
    with pytest.raises(ValueError):
        encode([Span("d", 0, 3, "A"), Span("d", 2, 4, "B")], 5)
    with pytest.raises(ValueError):
        encode([Span("d", 3, 7, "A")], 5)


@st.composite
def flat_spans(draw):
    n = draw(st.integers(1, 14))
    cuts = sorted(draw(st.sets(st.integers(0, n), max_size=8)))
    spans = []
    for a, b in zip(cuts, cuts[1:], strict=False):
        if draw(st.booleans()):
            spans.append(Span("d", a, b, draw(st.sampled_from(["A", "B"]))))
    return n, spans


@settings(max_examples=150, deadline=None)
@given(flat_spans(), st.sampled_from(["iob1", "iob2", "bilou"]))
def test_round_trip_spans_tags_spans(data, scheme):
    n, spans = data
    tags = encode(spans, n, scheme)
    got, viol = decode(tags, scheme, doc="d")
    assert viol == []
    assert got == sorted(spans)


@settings(max_examples=150, deadline=None)
@given(flat_spans(), st.sampled_from(["iob1", "iob2", "bilou"]))
def test_round_trip_tags_spans_tags(data, scheme):
    n, spans = data
    tags = encode(spans, n, scheme)
    got, _ = decode(tags, scheme, doc="d")
    assert encode(got, n, scheme) == tags


@settings(max_examples=100, deadline=None)
@given(st.lists(st.sampled_from(["O", "B-A", "I-A", "B-B", "I-B"]), max_size=12))
def test_repair_output_is_legal_iob2(tags):
    got, _ = decode(tags, "iob2", doc="d")
    fixed = encode(got, len(tags), "iob2")
    _, viol = decode(fixed, "iob2", doc="d")
    assert viol == []


CONLL = """-DOCSTART- -X- -X- O

EU NNP B-NP I-ORG
rejects VBZ B-VP O
German JJ B-NP I-MISC

Peter NNP B-NP I-PER
Blackburn NNP I-NP I-PER

-DOCSTART- -X- -X- O

BRUSSELS NNP B-NP I-LOC
"""


def test_read_columns_documents_and_sentences(tmp_path):
    p = tmp_path / "a.conll"
    p.write_text(CONLL)
    cf = read_columns(p)
    assert [d for d, _ in cf.docs] == ["doc-1", "doc-2"]
    assert [len(s) for s in cf.docs[0][1]] == [3, 2]
    cf = read_columns(p, doc_mode="sentence")
    assert len(cf.docs) == 3


def test_load_columns_iob1(tmp_path):
    p = tmp_path / "a.conll"
    p.write_text(CONLL)
    corpus, report = load_columns(p, scheme="iob1", annotator="x")
    assert [(s.doc, s.start, s.end, s.label) for s in corpus.sets["x"]] == [
        ("doc-1", 0, 1, "ORG"), ("doc-1", 2, 3, "MISC"), ("doc-1", 3, 5, "PER"),
        ("doc-2", 0, 1, "LOC")]
    assert corpus.length("doc-1") == 5
    assert corpus.breaks["doc-1"] == [0, 3]
    assert report.n_tags == 6 and not report.violations


def test_load_columns_refuses_illegal_unless_repair(tmp_path):
    p = tmp_path / "a.conll"
    p.write_text(CONLL)
    with pytest.raises(SchemeError) as exc:
        load_columns(p, scheme="iob2")
    assert len(exc.value.violations) == 4
    corpus, report = load_columns(p, scheme="iob2", repair=True)
    assert len(corpus.sets["a"]) == 4
    assert report.as_dict()["illegal_transitions"] == 4


def test_load_columns_bad_line(tmp_path):
    p = tmp_path / "a.conll"
    p.write_text("onlyone\n")
    with pytest.raises(ValueError, match="two columns"):
        load_columns(p)


def test_write_columns_round_trip(tmp_path):
    p = tmp_path / "a.conll"
    p.write_text(CONLL)
    corpus, _ = load_columns(p, scheme="iob1", annotator="x")
    out = tmp_path / "b.conll"
    write_columns(corpus, "x", out, scheme="bilou")
    back, report = load_columns(out, scheme="bilou", annotator="x")
    assert back.sets["x"] == corpus.sets["x"]
    assert back.tokens == corpus.tokens


TAGS = {
    "iob1": ["O", "B-A", "I-A", "B-B", "I-B"],
    "iob2": ["O", "B-A", "I-A", "B-B", "I-B"],
    "bilou": ["O", "B-A", "I-A", "L-A", "U-A", "B-B", "I-B", "L-B", "U-B", "E-A", "S-B"],
}


@settings(max_examples=200, deadline=None)
@given(st.sampled_from(sorted(TAGS)).flatmap(
    lambda s: st.tuples(st.just(s), st.lists(st.sampled_from(TAGS[s]), max_size=15))))
def test_any_tag_sequence_decodes_to_flat_spans_that_re_encode_legally(case):
    scheme, tags = case
    got, viol = decode(tags, scheme, doc="d")
    for x, y in zip(got, got[1:], strict=False):
        assert x.end <= y.start
    assert all(0 <= s.start < s.end <= len(tags) for s in got)
    fixed = encode(got, len(tags), scheme)
    again, viol2 = decode(fixed, scheme, doc="d")
    assert viol2 == []
    assert again == got
    if not viol:
        assert fixed == [t.replace("E-", "L-").replace("S-", "U-") for t in tags]
