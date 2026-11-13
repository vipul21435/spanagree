import json
from pathlib import Path

import pytest

from spanagree.cli import main

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "data" / "two_versions.jsonl"

CONLL_A = "-DOCSTART- O\n\nJohn B-PER\nSmith I-PER\nleft O\nParis B-LOC\n"
CONLL_B = "-DOCSTART- O\n\nJohn B-PER\nSmith O\nleft O\nParis B-ORG\n"
CONLL_BAD = "-DOCSTART- O\n\nJohn O\nSmith I-PER\nleft O\nParis B-LOC\n"


def test_score_markdown(capsys):
    assert main(["score", str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "| strict | 0.1667 | 0.2000 | 0.1818 |" in out
    assert "## Per label, partial" in out


def test_score_json_single_regime(capsys):
    assert main(["score", str(EXAMPLE), "--regime", "partial", "--format", "json",
                 "--ref", "second", "--other", "first"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ref"] == "second"
    assert data["regimes"]["partial"]["micro"]["credit"] == pytest.approx(3.0)


def test_score_column_files(tmp_path, capsys):
    a, b = tmp_path / "a.conll", tmp_path / "b.conll"
    a.write_text(CONLL_A)
    b.write_text(CONLL_B)
    assert main(["score", str(a), str(b), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["regimes"]["strict"]["micro"]["f"] == pytest.approx(0.0)
    assert data["regimes"]["partial"]["micro"]["f"] == pytest.approx(0.75)


def test_score_refuses_illegal_tags_without_repair(tmp_path, capsys):
    a, b = tmp_path / "a.conll", tmp_path / "bad.conll"
    a.write_text(CONLL_A)
    b.write_text(CONLL_BAD)
    assert main(["score", str(a), str(b)]) == 2
    assert "illegal iob2 transitions" in capsys.readouterr().err
    assert main(["score", str(a), str(b), "--repair"]) == 0


def test_check_exit_codes(tmp_path, capsys):
    good, bad = tmp_path / "a.conll", tmp_path / "bad.conll"
    good.write_text(CONLL_A)
    bad.write_text(CONLL_BAD)
    assert main(["check", str(good)]) == 0
    assert main(["check", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "O -> I-PER" in out
    assert main(["check", str(bad), "--format", "json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data[str(bad)]["illegal_transitions"] == 1


def test_missing_file_and_single_annotator(tmp_path, capsys):
    assert main(["score", str(tmp_path / "nope.jsonl")]) == 2
    one = tmp_path / "one.jsonl"
    one.write_text('{"document": "d", "start": 0, "end": 1, "label": "X"}\n')
    assert main(["score", str(one)]) == 2
    assert "need two annotators" in capsys.readouterr().err


def test_diff_markdown_and_json(capsys):
    assert main(["diff", str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "| split | 1 |" in out
    assert "- missed: n2: 6-7 MISC -> none" in out
    assert main(["diff", str(EXAMPLE), "--format", "json", "--items"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["differences"] == 5
    assert len(data["items"]) == 6


def test_agreement_two_annotators(capsys):
    assert main(["agreement", str(EXAMPLE), "--boot", "50", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["pair"]["f"]["strict"]["value"] == pytest.approx(2 / 11)
    assert main(["agreement", str(EXAMPLE), "--boot", "0"]) == 0
    assert "Cohen's kappa" in capsys.readouterr().out


def test_agreement_three_annotators(tmp_path, capsys):
    p = tmp_path / "three.jsonl"
    rows = []
    for name, label in (("a", "PER"), ("b", "PER"), ("c", "LOC")):
        rows.append({"document": "d1", "annotator": name, "start": 0, "end": 2, "label": label})
        rows.append({"document": "d2", "annotator": name, "start": 1, "end": 3, "label": "ORG"})
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert main(["agreement", str(p), "--boot", "20"]) == 0
    out = capsys.readouterr().out
    assert "Fleiss' kappa" in out and "majority of the others" in out
    assert main(["agreement", str(p), "--boot", "0", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["least_agreement"][0]["document"] == "d1"
    assert main(["agreement", str(p), "--ref", "a", "--other", "c", "--boot", "0"]) == 0


def test_agreement_needs_two(tmp_path, capsys):
    one = tmp_path / "one.jsonl"
    one.write_text('{"document": "d", "start": 0, "end": 1, "label": "X"}\n')
    assert main(["agreement", str(one)]) == 2


def test_confusion(capsys):
    assert main(["confusion", str(EXAMPLE)]) == 0
    assert "| LOC | 0 | 1 | 0 |" in capsys.readouterr().out
    assert main(["confusion", str(EXAMPLE), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["matched"] == 2


def test_convert_round_trip(tmp_path, capsys):
    a = tmp_path / "a.conll"
    a.write_text(CONLL_A)
    js = tmp_path / "a.jsonl"
    assert main(["convert", str(a), "-o", str(js)]) == 0
    back = tmp_path / "b.conll"
    assert main(["convert", str(js), "-o", str(back), "--tokens-from", str(a),
                 "--to-scheme", "bilou"]) == 0
    text = back.read_text()
    assert "John B-PER\nSmith L-PER" in text and "Paris U-LOC" in text
    bad = tmp_path / "bad.conll"
    bad.write_text(CONLL_BAD)
    assert main(["convert", str(bad), "-o", str(js)]) == 2
    assert main(["convert", str(bad), "-o", str(js), "--repair"]) == 0
    assert "repaired on request" in capsys.readouterr().out
    assert main(["convert", str(js), "-o", str(back), "--annotator", "zz"]) == 2
