# spanagree

Compare two or more sets of labelled text spans over the same documents (two annotators, an
annotator and a gold set, an old and a corrected version of a dataset) and report agreement with
each kind of disagreement counted: boundary, label, missed and extra.

## Why this is hard

Spans have no fixed set of items. Chance-corrected coefficients such as Cohen's kappa need a list
of units that every annotator labelled, and with spans the units are whatever each annotator chose
to mark, so those coefficients do not apply directly. Exact-match scoring calls a one-token boundary
slip a double error (one missed span and one extra span). A single F-score hides whether two
annotators disagree about where an entity ends or about what it is. spanagree aligns the spans,
scores them under four matching regimes and types every difference, so each of these questions
gets its own count.

## Install

```
pip install git+https://github.com/vipul21435/spanagree
```

Python 3.10 or newer; depends on numpy and scipy.

## Quickstart

`examples/data/two_versions.jsonl` holds two small span sets, `first` and `second`, over two
documents. Each line is `{"document", "annotator", "start", "end", "label"}`; a line with
`document` and `length` records how many positions a document has, and a line with only
`annotator` declares an annotator who may have marked nothing.

```
$ spanagree diff examples/data/two_versions.jsonl
# Differences of second against first

5 reference spans, 6 other spans; 5 differences.

| kind | count |
|---|---|
| identical | 1 |
| label | 1 |
| boundary | 1 |
| both | 0 |
| missed | 1 |
| extra | 1 |
| split | 1 |
| merged | 0 |
...
## Examples

- label: n1: 4-5 LOC -> 4-5 ORG
- boundary: n1: 7-9 ORG -> 7-10 ORG
- missed: n2: 6-7 MISC -> none
- extra: n2: none -> 8-9 LOC
- split: n2: 1-4 ORG -> 1-2 ORG+3-4 ORG
```

```
$ spanagree score examples/data/two_versions.jsonl --regime all
...
| regime | precision | recall | F (micro) | F (macro) |
|---|---|---|---|---|
| strict | 0.1667 | 0.2000 | 0.1818 | 0.2500 |
| exact | 0.3333 | 0.4000 | 0.3636 | 0.2500 |
| partial | 0.5000 | 0.6000 | 0.5455 | 0.3750 |
| type | 0.5000 | 0.6000 | 0.5455 | 0.4167 |
...
```

With three or more annotators, `agreement` adds the pairwise matrix, each annotator against the
majority of the others and the documents with the least agreement:

```
$ spanagree agreement examples/data/three_annotators.jsonl --boot 0
# Agreement among 3 annotators

No bootstrap intervals (--boot 0).

| statistic | value [interval] |
|---|---|
| Fleiss' kappa, all positions | 0.6383 |
| Fleiss' kappa, positions marked by anyone | 0.4055 |
| Krippendorff's alpha, character labels | 0.6434 |

## Pairwise F (strict)

|  | r1 | r2 | r3 |
|---|---|---|---|
| r1 | 1.0000 | 0.5000 | 0.5000 |
| r2 | 0.5000 | 1.0000 | 0.2500 |
| r3 | 0.5000 | 0.2500 | 1.0000 |
...
## Documents with the least agreement (strict)

| document | mean pairwise F | spans |
|---|---|---|
| a3 | 0.0000 | 3 |
| a2 | 0.3333 | 3 |
| a1 | 0.6667 | 6 |
```

The commands:

| command | what it reports |
|---|---|
| `spanagree score` | precision, recall and F per regime, micro and macro, per label |
| `spanagree diff` | the typed difference: counts, boundary shifts, label changes, example ids |
| `spanagree agreement` | pairwise F, token-level Cohen's or Fleiss' kappa, character-level Krippendorff's alpha, bootstrap intervals; with three or more annotators also the pairwise matrix, each annotator against the others' majority and the documents with the least agreement |
| `spanagree confusion` | label confusion over spans whose boundaries match exactly |
| `spanagree convert` | JSON lines to column files and back, or one tag scheme to another |
| `spanagree check` | illegal tag transitions in column files (exit status 1 when any are found) |

Every command takes `--format md` (default) or `--format json`. Inputs are JSON lines files
(`.jsonl`, any number of annotators each, offsets in characters or, with `--unit token`, in
tokens) or CoNLL-style column files (one annotator each, named
after the file or by `--names a,b`). Column files take `--scheme iob1|iob2|bilou` (aliases `bio`,
`bioes`), `--tag-col` (default: the last column) and `--doc-mode` (split at `-DOCSTART-` lines, or
one document per sentence). The first annotator is the reference unless `--ref` and `--other` say
otherwise. `examples/compare_versions.py` shows the Python API.

## How it works

**Matching regimes.** The four regimes are those of the SemEval-2013 DDIExtraction evaluation
(Segura-Bedmar, Martínez and Herrero-Zazo 2013), which refers to the MUC-5 metrics (Chinchor and
Sundheim 1993) for the scoring:

| regime | a pair counts when | credit |
|---|---|---|
| strict | same start, same end, same label | 1 |
| exact | same start and end, any label | 1 |
| partial | the spans overlap, any label | 1 if the boundaries are identical, else 0.5 |
| type | the spans overlap and the labels agree | 1 |

With matched credit `c`, `n_ref` reference spans and `n_other` other spans, precision is
`c / n_other`, recall `c / n_ref` and F is `2c / (n_ref + n_other)`. The half credit for a partial
match follows the MUC-5 definition `recall = (correct + 0.5 * partial) / possible`. Per label,
recall uses the reference span's label and precision the other span's label (the two agree under
strict and type). Macro values are unweighted means over the labels seen in either set.

**Alignment.** Each span is paired with at most one span of the other set. Within each regime the
pairing is optimal, not greedy: among all one-to-one pairings of compatible spans it maximises the
number of pairs, then the number of pairs with identical boundaries, then the total overlap. It is
solved with `scipy.optimize.linear_sum_assignment` on each connected group of overlapping spans,
and the tests check it against brute-force enumeration on random small cases. A greedy
largest-overlap-first pairing can leave spans unpaired that the optimal pairing matches (a case is
in `tests/test_match.py`).

**Typed difference.** Spans of the two sets are grouped by overlap. A group with one reference span
and one other span is `identical`, `label` (same boundaries, other label), `boundary` (same label,
other boundaries) or `both`. A lone reference span is `missed` and a lone other span `extra`. One
reference span covered by several disjoint other spans, none with its boundaries, is `split`; the
mirror case is `merged`. Any other group is paired by the partial-regime alignment, its pairs typed
as above and its leftovers counted as missed or extra; such groups are also counted as "tangled".
Boundary shifts are `other.start - ref.start` (left side) and `other.end - ref.end` (right side) in
offset units, which are tokens for column files.

**Agreement.**

- *Pairwise F.* Hripcsak and Rothschild (2005) show that the average F-measure among pairs of
  annotators is numerically identical to the average positive specific agreement, and that kappa
  approaches these measures as the number of negative cases grows large. F is symmetric in the two
  sets, so it serves as an agreement measure for spans. It is reported under every regime.
- *Token-level kappa.* Every position gets the label of the span covering it, or `O`. Cohen's kappa
  (two annotators) and Fleiss' kappa (two or more) are computed over those position labels. The
  caveat about the outside class: most positions in span data are `O`, so kappa over all positions
  is dominated by agreement on `O`. spanagree reports it twice, over all positions and over the
  positions that at least one annotator marked (which drops agreement on `O` entirely). Neither is
  a chance-corrected agreement on the spans themselves.
- *Krippendorff's alpha.* Nominal alpha over character-level labels. Each position is weighted by
  its length in characters when the tokens are known, which equals alpha over every character of
  the tokens (separators excluded); with JSON lines character offsets each position is one
  character. This is plain nominal alpha over characters, not the unitizing alpha of Krippendorff
  (1995), which compares the units themselves on a continuum.
- *Bootstrap.* Every statistic is a function of per-document sums (matched credit and set sizes,
  confusion matrices, category counts, coincidence matrices), so the bootstrap resamples documents
  with replacement and recomputes each statistic from the resampled sums. The interval is the 95 %
  percentile interval; `--boot` sets the number of resamples and `--seed` fixes the draws.
- *More than two annotators.* The pairwise F matrix; each annotator scored against the spans that
  more than half of the other annotators marked with identical boundaries and label; and the
  documents ranked by mean pairwise F, lowest first.

Positions covered by several spans of one annotator take the label of the longest of them.

**Tag schemes.** IOB1 (`I-` inside every entity, `B-` only on the first word of an entity that
directly follows another of the same type, the format described for the original CoNLL-2003 files
by Tjong Kim Sang and De Meulder 2003), IOB2 (`B-` starts every chunk) and BILOU (`B-`, `I-`,
`L-`/`E-`, `U-`/`S-`). `spanagree check` counts illegal transitions with example positions: for
IOB2, `I-X` after `O` or after another type; for IOB1, `B-X` that does not follow an `X` chunk; for
BILOU, a chunk left open or an `I-`/`L-` that continues nothing. Every other command refuses a file
with illegal transitions unless `--repair` is passed. The repair starts a new chunk at a tag that
continues nothing and closes a chunk left open where the next chunk or `O` begins, and the number
of repairs is printed.

## Results on real public data

Two label versions of the CoNLL-2003 English test set (Tjong Kim Sang and De Meulder 2003): the
original labels, and the corrected labels of CoNLL++ (Wang et al. 2019, the CrossWeigh paper). Both
files hold the same 231 documents and 46,435 tokens, tagged in IOB2.

```
python scripts/fetch_conll.py data       # downloads both files and checks their SHA-256
sh scripts/conll_versions.sh data        # runs the commands below
```

The individual commands, as run by the script:

```
spanagree check data/test.txt data/conllpp_test.txt --scheme iob2 --examples 0
python scripts/changed_sentences.py data/test.txt data/conllpp_test.txt
spanagree score data/test.txt data/conllpp_test.txt --names original,corrected --regime all
spanagree diff data/test.txt data/conllpp_test.txt --names original,corrected --examples 0
spanagree agreement data/test.txt data/conllpp_test.txt --names original,corrected --boot 1000 --seed 0
spanagree confusion data/test.txt data/conllpp_test.txt --names original,corrected
```

`spanagree check` finds 0 illegal IOB2 transitions in either file, with 5,648 spans in the original
and 5,702 in the corrected version. 186 sentences carry at least one changed tag, which is the
number the CrossWeigh repository states for its corrected test set; 309 tags change and no token
does (`python scripts/changed_sentences.py data/test.txt data/conllpp_test.txt`).

**Differences.** 5,506 original spans are unchanged. There are 204 differences:

| kind | count |
|---|---|
| label | 80 |
| boundary | 21 |
| both (label and boundary) | 15 |
| missed (in the original, absent from the corrected version) | 9 |
| extra (added by the correction) | 70 |
| split (one original span became several) | 1 |
| merged (several original spans became one) | 8 |

Of the 36 boundary changes (boundary and both), 30 move the right edge and 6 the left edge; 21
extend the right edge by one token. The most frequent label changes (label and both) are LOC to ORG
(33), MISC to ORG (18) and LOC to MISC (18). No overlap group needed the tangled fallback.

**Agreement between the versions** (corrected scored against original; 95 % bootstrap intervals
over the 231 documents, 1,000 resamples, seed 0):

| regime | precision | recall | F [interval] |
|---|---|---|---|
| strict | 0.9656 | 0.9749 | 0.9702 [0.9636, 0.9762] |
| exact | 0.9797 | 0.9890 | 0.9843 [0.9806, 0.9878] |
| partial | 0.9836 | 0.9930 | 0.9883 [0.9854, 0.9909] |
| type | 0.9709 | 0.9802 | 0.9755 [0.9693, 0.9810] |

| statistic | value [interval] |
|---|---|
| Cohen's kappa, all 46,435 tokens | 0.9796 [0.9745, 0.9840] |
| Cohen's kappa, 8,273 tokens marked in either version | 0.9505 [0.9386, 0.9609] |
| Krippendorff's alpha, character labels | 0.9774 [0.9719, 0.9823] |

Strict F per label is 0.9904 for PER, 0.9716 for LOC, 0.9704 for ORG and 0.9207 for MISC. Of the
5,586 span pairs with identical boundaries, 5,506 keep their label; the largest off-diagonal cells
of `spanagree confusion` are LOC to ORG (28) and LOC to MISC (16).

**Data and licence.** `test.txt` comes from the pfliu-nlp/Named-Entity-Recognition-NER-Papers
repository on GitHub and `conllpp_test.txt` from the ZihanWangKi/CrossWeigh repository; neither
repository states a licence for the data files. The text is Reuters news. The Hugging Face card of
`conll2003` lists its licence as "other" and quotes the shared task page: only the annotations were
made available, "because of copyright reasons", and building the full data needs access to the
Reuters Corpus. The card of `conllpp` lists its licence as "unknown". This repository therefore
commits only the fetch script (with SHA-256 digests) and the counts above, and no rows of either
file. Reiss et al. (2020) published a separate set of corrections to the same corpus; it is not
used here.

## Limitations

- The token-level coefficients and the column-file writer assume flat spans. Nested or
  overlapping spans of one annotator are handled by the matching regimes and the typed difference,
  but for kappa and alpha each position takes the label of the longest covering span.
- Krippendorff's alpha here is nominal alpha over character labels, not the unitizing alpha of
  Krippendorff (1995). The gamma measure of Mathet, Widlöcher and Métivier (2015) is not
  implemented; pygamma-agreement computes it.
- The majority comparison uses exact span identity (boundaries and label) and a strict majority, so
  near-identical spans do not vote together.
- Bootstrap intervals resample whole documents and assume documents are independent. With few
  documents (the quickstart has two) they say little.
- Column files compared with each other must have the same documents and token counts; tokens are
  not re-aligned.
- Offsets in JSON lines are taken as given; nothing checks them against a text. Converting JSON
  lines to a column file needs token offsets; without `--tokens-from` the tokens are written as
  placeholders (`t0`, `t1`, ...) and each document becomes one sentence.
- Only the IOB1, IOB2 and BILOU families are read; IO and other schemes are not.

## Related tools

- [seqeval](https://github.com/chakki-works/seqeval): a testing framework for sequence labelling
  (span-level precision, recall and F).
- [nervaluate](https://github.com/MantisAI/nervaluate): NER evaluation with partial match scoring.
- [pygamma-agreement](https://pygamma-agreement.readthedocs.io/en/latest/): the gamma
  inter-annotator agreement measure and alignment.
- [krippendorff](https://github.com/pln-fing-udelar/fast-krippendorff): Krippendorff's alpha; used
  here as a test reference.
- [rubricops](https://github.com/vipul21435/rubricops): rubrics and inter-rater agreement for
  categorical ratings.
- [labelfuse](https://github.com/vipul21435/labelfuse): aggregation of categorical labels.
- [labelnoise](https://github.com/vipul21435/labelnoise): ranking of likely label errors.
- [evalstats](https://github.com/vipul21435/evalstats): error bars and paired comparisons for
  evaluations.

## References

- Chinchor, N. and Sundheim, B. M. (1993). MUC-5 Evaluation Metrics. Fifth Message Understanding
  Conference (MUC-5). https://aclanthology.org/M93-1007/
- Hripcsak, G. and Rothschild, A. S. (2005). Agreement, the F-measure, and reliability in
  information retrieval. Journal of the American Medical Informatics Association 12, 296-298.
  https://doi.org/10.1197/jamia.M1733
- Krippendorff, K. (1995). On the reliability of unitizing continuous data. Sociological
  Methodology 25. https://doi.org/10.2307/271061
- Mathet, Y., Widlöcher, A. and Métivier, J.-P. (2015). The Unified and Holistic Method Gamma (γ)
  for Inter-Annotator Agreement Measure and Alignment. Computational Linguistics.
  https://aclanthology.org/J15-3003/
- Reiss, F., Xu, H., Cutler, B., Muthuraman, K. and Eichenberger, Z. (2020). Identifying Incorrect
  Labels in the CoNLL-2003 Corpus. CoNLL 2020. https://aclanthology.org/2020.conll-1.16/
- Segura-Bedmar, I., Martínez, P. and Herrero-Zazo, M. (2013). Extraction of Drug-Drug
  Interactions from Biomedical Texts (DDIExtraction 2013). SemEval 2013.
  https://aclanthology.org/S13-2056/
- Tjong Kim Sang, E. and De Meulder, F. (2003). Introduction to the CoNLL-2003 Shared Task:
  Language-Independent Named Entity Recognition. CoNLL 2003. https://aclanthology.org/W03-0419/
- Wang, Z., Shang, J., Liu, L., Lu, L., Liu, J. and Han, J. (2019). CrossWeigh: Training Named
  Entity Tagger from Imperfect Annotations. EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1519/

## Licence

MIT. Copyright (c) 2026 Vipul Raj Jha.

Built by Vipul Raj Jha · https://vipul21435.github.io
