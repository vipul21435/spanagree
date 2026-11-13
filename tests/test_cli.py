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
