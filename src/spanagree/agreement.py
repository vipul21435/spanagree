"""Agreement coefficients for span annotations, with bootstrap intervals over documents.

* Pairwise F as agreement (Hripcsak and Rothschild 2005): F of one set against the
  other, which is symmetric in the two sets, under any matching regime.
* Token-level Cohen's kappa (two annotators) and Fleiss' kappa (two or more):
  every position gets the label of the span covering it, or ``O``. Because most
  positions are ``O`` in typical span data, kappa over all positions is dominated
  by agreement on ``O``; it is therefore also reported over the positions that at
  least one annotator marked, which drops that agreement entirely. Neither number
  is a chance-corrected agreement on spans.
* Krippendorff's alpha (nominal) over character-level labels: each token
  position is weighted by its length in characters when the tokens are known
  (column files), so the value equals alpha computed over every character of the
  tokens; for JSON lines with character offsets each position is one character.

Positions covered by several spans of one annotator take the label of the
longest of them (the earliest on ties).

Every statistic is a function of per-document sums, so the bootstrap resamples
documents with replacement and recomputes it from the resampled sums.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from spanagree.match import REGIMES, align, credit
from spanagree.spans import Corpus

OUTSIDE = "O"


@dataclass
class Estimate:
    """A point estimate with a percentile bootstrap interval over documents."""

    value: float | None
    low: float | None = None
    high: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {"value": self.value, "low": self.low, "high": self.high}


def label_index(corpus: Corpus, annotators: Sequence[str]) -> list[str]:
    """``O`` followed by every label used by the given annotators, sorted."""
    labels = sorted({s.label for a in annotators for s in corpus.sets[a]} - {OUTSIDE})
    return [OUTSIDE, *labels]


def position_labels(corpus: Corpus, annotators: Sequence[str],
                    labels: Sequence[str]) -> dict[str, np.ndarray]:
    """Per document, an ``(annotators, positions)`` array of label indices."""
    idx = {lab: k for k, lab in enumerate(labels)}
    docs = corpus.documents()
    out = {d: np.zeros((len(annotators), corpus.length(d)), dtype=np.int64) for d in docs}
    for r, name in enumerate(annotators):
        # Paint shorter spans first so that the longest covering span wins.
        for s in sorted(corpus.sets[name], key=lambda s: (s.end - s.start, -s.start)):
            row = out[s.doc][r]
            if s.end > row.shape[0]:
                raise ValueError(f"span {s.key} ends past the document length {row.shape[0]}")
            row[s.start:s.end] = idx[s.label]
    return out


def position_weights(corpus: Corpus, doc: str) -> np.ndarray:
    """Characters per position: token lengths when tokens are known, else ones."""
    toks = corpus.tokens.get(doc)
    if toks:
        return np.array([len(t) for t in toks], dtype=np.float64)
    return np.ones(corpus.length(doc), dtype=np.float64)


# ---------------------------------------------------------------- coefficients

def cohen_kappa(conf: np.ndarray) -> float | None:
    """Cohen's kappa from a square confusion matrix of counts."""
    n = conf.sum()
    if n <= 0:
        return None
    po = np.trace(conf) / n
    pe = float(conf.sum(axis=1) @ conf.sum(axis=0)) / (n * n)
    if pe >= 1.0:
        return None
    return float((po - pe) / (1.0 - pe))


def fleiss_kappa(sum_sq: float, n_items: float, totals: np.ndarray, m: int) -> float | None:
    """Fleiss' kappa from ``sum_i sum_j n_ij**2``, the item count and column totals."""
    if n_items <= 0 or m < 2:
        return None
    p_bar = (sum_sq - n_items * m) / (n_items * m * (m - 1))
    p = totals / (n_items * m)
    pe = float(p @ p)
    if pe >= 1.0:
        return None
    return float((p_bar - pe) / (1.0 - pe))


def krippendorff_alpha_nominal(coincidence: np.ndarray) -> float | None:
    """Krippendorff's alpha for nominal data from a coincidence matrix."""
    n_c = coincidence.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return None
    observed = n - np.trace(coincidence)
    expected = (n * n - float(n_c @ n_c)) / (n - 1)
    if expected <= 0:
        return None
    return float(1.0 - observed / expected)


# ---------------------------------------------------------------- per-document sums

def pair_f_stats(corpus: Corpus, ref: str, other: str, regime: str) -> np.ndarray:
    """``(documents, 3)`` array: matched credit, reference spans, other spans."""
    docs = corpus.documents()
    pos = {d: k for k, d in enumerate(docs)}
    st = np.zeros((len(docs), 3))
    for a, b in align(corpus.sets[ref], corpus.sets[other], regime).pairs:
        st[pos[a.doc], 0] += credit(a, b, regime)
    for s in corpus.sets[ref]:
        st[pos[s.doc], 1] += 1
    for s in corpus.sets[other]:
        st[pos[s.doc], 2] += 1
    return st


def _f_from(sums: np.ndarray) -> np.ndarray:
    total = sums[..., 1] + sums[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, 2.0 * sums[..., 0] / total, np.nan)


def confusion_stats(arrays: dict[str, np.ndarray], docs: Sequence[str], i: int, j: int,
                    n_labels: int, marked_only: bool = False) -> np.ndarray:
    """``(documents, C*C)`` flattened token confusion matrices of annotators i and j."""
    out = np.zeros((len(docs), n_labels * n_labels))
    for k, d in enumerate(docs):
        a, b = arrays[d][i], arrays[d][j]
        conf = np.bincount(a * n_labels + b, minlength=n_labels * n_labels).astype(float)
        if marked_only:
            conf[0] = 0.0
        out[k] = conf
    return out


def fleiss_stats(arrays: dict[str, np.ndarray], docs: Sequence[str], n_labels: int,
                 marked_only: bool = False) -> np.ndarray:
    """``(documents, 2 + C)``: sum of squared category counts, items, column totals."""
    out = np.zeros((len(docs), 2 + n_labels))
    for k, d in enumerate(docs):
        arr = arrays[d]
        m = arr.shape[0]
        counts = np.zeros((arr.shape[1], n_labels))
        for r in range(m):
            counts[np.arange(arr.shape[1]), arr[r]] += 1
        if marked_only:
            counts = counts[counts[:, 0] < m]
        out[k, 0] = float((counts ** 2).sum())
        out[k, 1] = counts.shape[0]
        out[k, 2:] = counts.sum(axis=0)
    return out


def coincidence_stats(arrays: dict[str, np.ndarray], weights: dict[str, np.ndarray],
                      docs: Sequence[str], n_labels: int) -> np.ndarray:
    """``(documents, C*C)`` flattened weighted coincidence matrices."""
    out = np.zeros((len(docs), n_labels * n_labels))
    for k, d in enumerate(docs):
        arr = arrays[d]
        m = arr.shape[0]
        if m < 2 or arr.shape[1] == 0:
            continue
        counts = np.zeros((arr.shape[1], n_labels))
        for r in range(m):
            counts[np.arange(arr.shape[1]), arr[r]] += 1
        w = weights[d][:, None]
        o = (counts * w).T @ counts - np.diag((counts * w).sum(axis=0))
        out[k] = (o / (m - 1)).ravel()
    return out


# ---------------------------------------------------------------- bootstrap

def bootstrap(stats: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], n_boot: int = 1000,
              seed: int = 0, level: float = 0.95) -> Estimate:
    """Percentile interval of ``fn`` over document resamples.

    ``stats`` is ``(documents, k)``; ``fn`` maps an ``(r, k)`` array of summed
    statistics to ``r`` values (NaN where undefined).
    """
    point = fn(stats.sum(axis=0)[None, :])[0]
    value = None if np.isnan(point) else float(point)
    n_docs = stats.shape[0]
    if n_boot <= 0 or n_docs == 0 or value is None:
        return Estimate(value)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_docs, size=(n_boot, n_docs))
    weights = np.zeros((n_boot, n_docs))
    np.add.at(weights, (np.repeat(np.arange(n_boot), n_docs), draws.ravel()), 1.0)
    values = fn(weights @ stats)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return Estimate(value)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return Estimate(value, float(lo), float(hi))


def _vectorise(fn: Callable[[np.ndarray], float | None]) -> Callable[[np.ndarray], np.ndarray]:
    def wrapped(rows: np.ndarray) -> np.ndarray:
        out = np.empty(rows.shape[0])
        for i, row in enumerate(rows):
            v = fn(row)
            out[i] = np.nan if v is None else v
        return out
    return wrapped


# ---------------------------------------------------------------- reports

@dataclass
class PairAgreement:
    ref: str
    other: str
    f: dict[str, Estimate]
    kappa_all: Estimate
    kappa_marked: Estimate
    alpha: Estimate
    positions: int
    marked_positions: int

    def as_dict(self) -> dict[str, object]:
        return {
            "ref": self.ref, "other": self.other,
            "f": {r: e.as_dict() for r, e in self.f.items()},
            "cohen_kappa_all_positions": self.kappa_all.as_dict(),
            "cohen_kappa_marked_positions": self.kappa_marked.as_dict(),
            "krippendorff_alpha_characters": self.alpha.as_dict(),
            "positions": self.positions, "marked_positions": self.marked_positions,
        }


def pair_agreement(corpus: Corpus, ref: str, other: str, n_boot: int = 1000,
                   seed: int = 0, level: float = 0.95) -> PairAgreement:
    """F under every regime, token kappa (two variants) and character alpha."""
    docs = corpus.documents()
    f = {r: bootstrap(pair_f_stats(corpus, ref, other, r), _f_from, n_boot, seed, level)
         for r in REGIMES}
    labels = label_index(corpus, [ref, other])
    c = len(labels)
    arrays = position_labels(corpus, [ref, other], labels)
    conf_all = confusion_stats(arrays, docs, 0, 1, c)
    conf_marked = confusion_stats(arrays, docs, 0, 1, c, marked_only=True)
    kappa = _vectorise(lambda row: cohen_kappa(row.reshape(c, c)))
    weights = {d: position_weights(corpus, d) for d in docs}
    coin = coincidence_stats(arrays, weights, docs, c)
    alpha = _vectorise(lambda row: krippendorff_alpha_nominal(row.reshape(c, c)))
    return PairAgreement(
        ref, other, f,
        kappa_all=bootstrap(conf_all, kappa, n_boot, seed, level),
        kappa_marked=bootstrap(conf_marked, kappa, n_boot, seed, level),
        alpha=bootstrap(coin, alpha, n_boot, seed, level),
        positions=int(conf_all.sum()), marked_positions=int(conf_marked.sum()),
    )


@dataclass
class GroupAgreement:
    annotators: list[str]
    fleiss_all: Estimate
    fleiss_marked: Estimate
    alpha: Estimate

    def as_dict(self) -> dict[str, object]:
        return {"annotators": self.annotators,
                "fleiss_kappa_all_positions": self.fleiss_all.as_dict(),
                "fleiss_kappa_marked_positions": self.fleiss_marked.as_dict(),
                "krippendorff_alpha_characters": self.alpha.as_dict()}


def group_agreement(corpus: Corpus, annotators: Sequence[str] | None = None,
                    n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> GroupAgreement:
    """Fleiss' kappa (two variants) and character alpha over all annotators."""
    names = list(annotators or corpus.annotators)
    docs = corpus.documents()
    labels = label_index(corpus, names)
    c = len(labels)
    m = len(names)
    arrays = position_labels(corpus, names, labels)
    fl = _vectorise(lambda row: fleiss_kappa(row[0], row[1], row[2:], m))
    weights = {d: position_weights(corpus, d) for d in docs}
    alpha = _vectorise(lambda row: krippendorff_alpha_nominal(row.reshape(c, c)))
    return GroupAgreement(
        names,
        fleiss_all=bootstrap(fleiss_stats(arrays, docs, c), fl, n_boot, seed, level),
        fleiss_marked=bootstrap(fleiss_stats(arrays, docs, c, marked_only=True), fl,
                                n_boot, seed, level),
        alpha=bootstrap(coincidence_stats(arrays, weights, docs, c), alpha, n_boot, seed, level),
    )
