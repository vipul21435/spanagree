# Changelog

## 0.1.0

- Span records, a JSON lines reader and writer, and a reader for CoNLL-style column files in IOB1,
  IOB2 (BIO) and BILOU (BIOES), with illegal transitions counted and repaired only on request.
- Scoring under the strict, exact, partial and type regimes on an optimal one-to-one alignment:
  precision, recall and F, micro, macro and per label.
- Typed difference: identical, label, boundary, both, missed, extra, split and merged, with
  boundary shifts per side, label changes and example ids.
- Agreement: pairwise F, token-level Cohen's and Fleiss' kappa over all and over marked positions,
  nominal Krippendorff's alpha over characters, percentile bootstrap intervals over documents.
- Three or more annotators: pairwise F matrix, each annotator against the others' majority, the
  documents with the least agreement.
- Label confusion over boundary-matched spans.
- Commands `score`, `diff`, `agreement`, `confusion`, `convert` and `check`, in Markdown or JSON.
- Scripts that fetch the original and corrected CoNLL-2003 English test labels and compare them.
