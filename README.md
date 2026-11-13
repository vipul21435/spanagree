# spanagree

Compare two sets of labelled text spans over the same documents and report agreement under several
matching regimes.

Status: early. What works so far:

- reading spans from JSON lines (document, annotator, start, end, label) and from CoNLL-style column
  files in IOB1, IOB2 (BIO) or BILOU (BIOES);
- counting illegal tag transitions (`spanagree check`); a file with illegal transitions is refused
  unless `--repair` is passed;
- precision, recall and F under four matching regimes (strict, exact boundary, partial overlap,
  type), micro and macro, per label, on an optimal one-to-one alignment of overlapping spans
  (`spanagree score`).

The typed difference, agreement coefficients, bootstrap intervals and the other commands are not
written yet.

## Install

```
pip install git+https://github.com/vipul21435/spanagree
```

## Quickstart

```
$ spanagree score examples/data/two_versions.jsonl --regime all
# second against first

5 reference spans, 6 other spans, 2 documents.

| regime | precision | recall | F (micro) | F (macro) |
|---|---|---|---|---|
| strict | 0.1667 | 0.2000 | 0.1818 | 0.2500 |
| exact | 0.3333 | 0.4000 | 0.3636 | 0.2500 |
| partial | 0.5000 | 0.6000 | 0.5455 | 0.3750 |
| type | 0.5000 | 0.6000 | 0.5455 | 0.4167 |
...
```

## Licence

MIT. Copyright (c) 2026 Vipul Raj Jha.

Built by Vipul Raj Jha · https://vipul21435.github.io
