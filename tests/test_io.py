import pytest

from spanagree.io import load_inputs, pick_pair

A = "-DOCSTART- O\n\nJohn B-PER\nleft O\n\n-DOCSTART- O\n\nParis B-LOC\n"
FEWER_DOCS = "-DOCSTART- O\n\nJohn B-PER\nleft O\n"
FEWER_TOKENS = "-DOCSTART- O\n\nJohn B-PER\n\n-DOCSTART- O\n\nParis B-LOC\n"


def test_column_files_must_share_documents_and_tokens(tmp_path):
    a = tmp_path / "a.conll"
    a.write_text(A)
    b = tmp_path / "b.conll"
    b.write_text(FEWER_DOCS)
    with pytest.raises(ValueError, match="documents differ"):
        load_inputs([a, b])
    b.write_text(FEWER_TOKENS)
    with pytest.raises(ValueError, match="tokens"):
        load_inputs([a, b])


def test_same_stem_gets_a_suffix(tmp_path):
    (tmp_path / "x").mkdir()
    a = tmp_path / "a.conll"
    a.write_text(A)
    b = tmp_path / "x" / "a.conll"
    b.write_text(A)
    corpus, reports = load_inputs([a, b])
    assert corpus.annotators == ["a", "a-2"]
    assert len(reports) == 2


def test_pick_pair_errors(tmp_path):
    a = tmp_path / "a.conll"
    a.write_text(A)
    b = tmp_path / "b.conll"
    b.write_text(A)
    corpus, _ = load_inputs([a, b])
    assert pick_pair(corpus, None, None) == ("a", "b")
    assert pick_pair(corpus, "b", None) == ("b", "a")
    with pytest.raises(ValueError, match="no annotator"):
        pick_pair(corpus, "a", "zz")
