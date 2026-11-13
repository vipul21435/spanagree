import json

import pytest

from spanagree.spans import Corpus, Span, load_jsonl, write_jsonl


def test_span_validation():
    with pytest.raises(ValueError):
        Span("d", 3, 3, "X")
    with pytest.raises(ValueError):
        Span("d", -1, 2, "X")
    assert Span("d", 1, 4, "X").overlap(Span("d", 3, 9, "Y")) == 1
    assert Span("d", 1, 4, "X").key == "d:1-4"


def test_load_jsonl_groups_by_annotator(tmp_path):
    p = tmp_path / "spans.jsonl"
    rows = [
        {"document": "a", "length": 30},
        {"document": "a", "annotator": "r1", "start": 0, "end": 4, "label": "PER"},
        {"doc": "a", "annotator": "r2", "start": 0, "end": 5, "label": "PER"},
        {"document": "b", "start": 2, "end": 3, "label": "LOC"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    c = load_jsonl(p)
    assert set(c.sets) == {"r1", "r2", "spans"}
    assert c.length("a") == 30
    assert c.length("b") == 3
    assert c.documents() == ["a", "b"]


@pytest.mark.parametrize("line,msg", [
    ("{not json", "not valid JSON"),
    ("[1, 2]", "JSON object"),
    ('{"start": 1, "end": 2, "label": "X"}', "document"),
    ('{"document": "a", "start": 1, "label": "X"}', "end"),
    ('{"document": "a", "start": "x", "end": 2, "label": "X"}', "integers"),
    ('{"document": "a", "start": 4, "end": 2, "label": "X"}', "reversed"),
    ('{"document": "a", "start": 1.5, "end": 2, "label": "X"}', "integers"),
    ('{"document": "a", "start": true, "end": 2, "label": "X"}', "integers"),
    ('{"document": "a", "start": 1, "end": 2, "label": null}', "label"),
    ('{"document": "a", "start": 1, "end": 2, "label": ""}', "label"),
    ('{"document": "a", "length": -3}', "length"),
    ('{"document": "a", "length": "9"}', "length"),
])
def test_load_jsonl_rejects_malformed(tmp_path, line, msg):
    p = tmp_path / "bad.jsonl"
    p.write_text(line + "\n")
    with pytest.raises(ValueError, match=msg):
        load_jsonl(p)


def test_write_then_read_round_trip(tmp_path):
    c = Corpus(sets={"x": [Span("d", 0, 2, "A")], "y": [Span("d", 1, 2, "B")]},
               lengths={"d": 9})
    p = tmp_path / "out.jsonl"
    write_jsonl(c, p)
    back = load_jsonl(p)
    assert back.sets == c.sets
    assert back.length("d") == 9


def test_merge_rejects_duplicate_annotator():
    c = Corpus(sets={"x": []})
    with pytest.raises(ValueError):
        c.merge(Corpus(sets={"x": []}))
