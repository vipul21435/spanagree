"""Command line interface: ``spanagree <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from spanagree import __version__
from spanagree.agreement import group_agreement, pair_agreement
from spanagree.bio import SchemeError, load_columns, scheme_name, write_columns
from spanagree.confusion import confusion
from spanagree.diff import CATEGORIES, diff
from spanagree.io import is_jsonl, load_inputs, pick_pair
from spanagree.match import REGIMES, score
from spanagree.multi import against_majority, least_agreement, pairwise_matrix
from spanagree.report import fmt, table
from spanagree.spans import Corpus, write_jsonl


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("inputs", nargs="+",
                   help="JSON lines files (.jsonl) or column files (one annotator each)")
    p.add_argument("--scheme", default="iob2",
                   help="tag scheme of column files: iob1, iob2 (bio), bilou (bioes)")
    p.add_argument("--repair", action="store_true",
                   help="accept column files with illegal transitions, using the repaired spans")
    p.add_argument("--tag-col", type=int, default=-1, help="tag column (default: last)")
    p.add_argument("--doc-mode", default="auto", choices=["auto", "docstart", "sentence"],
                   help="document boundaries in column files")
    p.add_argument("--names", help="comma-separated annotator names, one per input file")
    p.add_argument("--format", default="md", choices=["md", "json"])


def _add_pair_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ref", help="reference annotator (default: the first)")
    p.add_argument("--other", help="annotator compared with the reference (default: the second)")


def _load(args: argparse.Namespace):
    names = args.names.split(",") if args.names else None
    corpus, _ = load_inputs(args.inputs, scheme=args.scheme, repair=args.repair,
                            tag_col=args.tag_col, doc_mode=args.doc_mode, names=names)
    return corpus


def cmd_score(args: argparse.Namespace) -> str:
    corpus = _load(args)
    ref, other = pick_pair(corpus, args.ref, args.other)
    regimes = REGIMES if args.regime == "all" else (args.regime,)
    results = {r: score(corpus.sets[ref], corpus.sets[other], r) for r in regimes}
    if args.format == "json":
        return json.dumps({"ref": ref, "other": other,
                           "regimes": {r: s.as_dict() for r, s in results.items()}}, indent=2)
    out = [f"# {other} against {ref}", "",
           f"{len(corpus.sets[ref])} reference spans, {len(corpus.sets[other])} other spans, "
           f"{len(corpus.documents())} documents.", ""]
    rows = [(r, s.micro.precision, s.micro.recall, s.micro.f, s.macro_f)
            for r, s in results.items()]
    out.append(table(["regime", "precision", "recall", "F (micro)", "F (macro)"], rows))
    for r, s in results.items():
        out += ["", f"## Per label, {r}", ""]
        rows = [(lab, p.n_ref, p.n_other, p.precision, p.recall, p.f)
                for lab, p in sorted(s.per_label.items())]
        out.append(table(["label", "ref", "other", "precision", "recall", "F"], rows))
    return "\n".join(out)


def cmd_check(args: argparse.Namespace) -> str:
    reports = []
    for path in args.inputs:
        _, rep = load_columns(path, scheme=args.scheme, tag_col=args.tag_col,
                              doc_mode=args.doc_mode, repair=True)
        reports.append((path, rep))
    args.exit_code = 1 if any(r.violations for _, r in reports) else 0
    if args.format == "json":
        return json.dumps({str(p): r.as_dict(args.examples) for p, r in reports}, indent=2)
    out = []
    for path, rep in reports:
        d = rep.as_dict(args.examples)
        out += [f"# {path}", "",
                f"scheme {rep.scheme}: {d['tags']} tags, {d['chunks']} chunks, "
                f"{d['illegal_transitions']} illegal transitions", ""]
        if rep.violations:
            rows = [(k, v["count"], ", ".join(v["examples"])) for k, v in d["by_kind"].items()]
            out += [table(["transition", "count", "examples (doc:token)"], rows), ""]
    return "\n".join(out).rstrip()


def cmd_diff(args: argparse.Namespace) -> str:
    corpus = _load(args)
    ref, other = pick_pair(corpus, args.ref, args.other)
    res = diff(corpus.sets[ref], corpus.sets[other])
    if args.format == "json":
        return json.dumps({"ref": ref, "other": other,
                           **res.as_dict(args.examples, items=args.items)}, indent=2)
    counts = res.counts()
    out = [f"# Differences of {other} against {ref}", "",
           f"{len(corpus.sets[ref])} reference spans, {len(corpus.sets[other])} other spans; "
           f"{res.n_differences} differences.", ""]
    out.append(table(["kind", "count"], [(k, counts[k]) for k in CATEGORIES]))
    if res.tangled_groups:
        out += ["", f"{res.tangled_groups} overlap groups were neither one-to-one, split nor "
                "merged; their spans were paired by the partial-regime alignment."]
    sides = res.boundary_sides()
    if sum(sides.values()):
        out += ["", "## Boundary changes (boundary and both)", ""]
        out.append(table(["side", "count"], list(sides.items())))
        sh = res.shifts()
        rows = [(side, k, n) for side in ("left", "right") for k, n in sh[side].items()]
        out += ["", table(["side", "shift", "count"], rows)]
    changes = res.label_changes()
    if changes:
        out += ["", "## Label changes (label and both)", ""]
        out.append(table(["ref label", "other label", "count"],
                         [(a, b, n) for (a, b), n in changes.most_common()]))
    if args.examples:
        out += ["", "## Examples", ""]
        for k in CATEGORIES:
            if k == "identical":
                continue
            for ex in res.examples(k, args.examples):
                out.append(f"- {k}: {ex}")
    return "\n".join(out)


def _est(e) -> str:
    if e.value is None:
        return "-"
    if e.low is None:
        return fmt(e.value)
    return f"{fmt(e.value)} [{fmt(e.low)}, {fmt(e.high)}]"


def cmd_agreement(args: argparse.Namespace) -> str:
    corpus = _load(args)
    names = corpus.annotators
    if len(names) < 2:
        raise ValueError(f"need two annotators, found {len(names)}")
    if len(names) == 2 or args.ref or args.other:
        ref, other = pick_pair(corpus, args.ref, args.other)
        pa = pair_agreement(corpus, ref, other, n_boot=args.boot, seed=args.seed)
        if args.format == "json":
            return json.dumps({"pair": pa.as_dict(), "bootstrap": args.boot, "seed": args.seed},
                              indent=2)
        level = f"95 % bootstrap interval over {len(corpus.documents())} documents, " \
                f"{args.boot} resamples, seed {args.seed}"
        rows = [(f"F, {r} (Hripcsak and Rothschild)", _est(e)) for r, e in pa.f.items()]
        rows += [(f"Cohen's kappa, all {pa.positions} positions", _est(pa.kappa_all)),
                 (f"Cohen's kappa, {pa.marked_positions} positions marked by either",
                  _est(pa.kappa_marked)),
                 ("Krippendorff's alpha, character labels", _est(pa.alpha))]
        return "\n".join([f"# Agreement between {ref} and {other}", "", level, "",
                          table(["statistic", "value [interval]"], rows), "",
                          "Kappa over all positions counts agreement on O (outside) and is "
                          "inflated by it; over marked positions it drops that agreement "
                          "entirely. Read both."])
    ga = group_agreement(corpus, names, n_boot=args.boot, seed=args.seed)
    matrix = pairwise_matrix(corpus, names, args.regime)
    maj = against_majority(corpus, names, args.regime)
    worst = least_agreement(corpus, names, args.regime, top=args.top)
    if args.format == "json":
        return json.dumps({"group": ga.as_dict(), "regime": args.regime,
                           "pairwise_f": matrix,
                           "against_majority": [m.as_dict() for m in maj],
                           "least_agreement": [{"document": d, "mean_f": f, "spans": n}
                                               for d, f, n in worst],
                           "bootstrap": args.boot, "seed": args.seed}, indent=2)
    out = [f"# Agreement among {len(names)} annotators", "",
           f"95 % bootstrap interval over {len(corpus.documents())} documents, "
           f"{args.boot} resamples, seed {args.seed}", ""]
    out.append(table(["statistic", "value [interval]"], [
        ("Fleiss' kappa, all positions", _est(ga.fleiss_all)),
        ("Fleiss' kappa, positions marked by anyone", _est(ga.fleiss_marked)),
        ("Krippendorff's alpha, character labels", _est(ga.alpha))]))
    out += ["", f"## Pairwise F ({args.regime})", ""]
    out.append(table(["", *names], [(a, *(matrix[a][b] for b in names)) for a in names]))
    out += ["", f"## Each against the majority of the others ({args.regime})", ""]
    out.append(table(["annotator", "majority spans", "precision", "recall", "F"],
                     [(m.annotator, m.majority_spans, m.result.precision, m.result.recall,
                       m.result.f) for m in maj]))
    out += ["", f"## Documents with the least agreement ({args.regime})", ""]
    out.append(table(["document", "mean pairwise F", "spans"], worst))
    return "\n".join(out)


def cmd_confusion(args: argparse.Namespace) -> str:
    corpus = _load(args)
    ref, other = pick_pair(corpus, args.ref, args.other)
    c = confusion(corpus.sets[ref], corpus.sets[other])
    if args.format == "json":
        return json.dumps({"ref": ref, "other": other, **c.as_dict()}, indent=2)
    labs = c.labels
    out = [f"# Label confusion, {ref} (rows) against {other} (columns)", "",
           f"{c.matched} span pairs with identical boundaries, {c.agreeing} with the same label; "
           f"{c.unmatched_ref} reference and {c.unmatched_other} other spans have no "
           "boundary match and are left out.", ""]
    out.append(table(["ref \\ other", *labs], [(a, *row) for a, row in zip(labs, c.matrix(),
                                                                         strict=True)]))
    return "\n".join(out)


def cmd_convert(args: argparse.Namespace) -> str:
    names = args.names.split(",") if args.names else None
    corpus, reports = load_inputs([args.input], scheme=args.scheme, repair=args.repair,
                                  tag_col=args.tag_col, doc_mode=args.doc_mode, names=names)
    repaired = sum(len(r.violations) for r in reports)
    to = args.to or ("jsonl" if is_jsonl(args.output) else "conll")
    if to == "jsonl":
        write_jsonl(corpus, args.output)
        written = sum(len(v) for v in corpus.sets.values())
    else:
        if args.tokens_from:
            ref, _ = load_columns(args.tokens_from, scheme=args.scheme, repair=True,
                                  tag_col=args.tag_col, doc_mode=args.doc_mode)
            corpus = Corpus(sets=corpus.sets, lengths=corpus.lengths, tokens=ref.tokens,
                            breaks=ref.breaks, unit="token")
        annotator = args.annotator or corpus.annotators[0]
        if annotator not in corpus.sets:
            raise ValueError(f"no annotator {annotator!r}; found {', '.join(corpus.annotators)}")
        write_columns(corpus, annotator, args.output, scheme=scheme_name(args.to_scheme))
        written = len(corpus.sets[annotator])
    msg = f"wrote {written} spans to {args.output}"
    if repaired:
        msg += f" ({repaired} illegal transitions repaired on request)"
    return msg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spanagree",
        description="Agreement and typed error counts for labelled text spans.")
    parser.add_argument("--version", action="version", version=f"spanagree {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("score", help="precision, recall and F under the matching regimes")
    _add_input_args(p)
    _add_pair_args(p)
    p.add_argument("--regime", default="all", choices=["all", *REGIMES])
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("diff", help="typed differences: label, boundary, missed, extra, ...")
    _add_input_args(p)
    _add_pair_args(p)
    p.add_argument("--examples", type=int, default=5, help="example ids per kind")
    p.add_argument("--items", action="store_true", help="list every item in the JSON output")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("agreement", help="pairwise F, kappa and alpha with bootstrap intervals")
    _add_input_args(p)
    _add_pair_args(p)
    p.add_argument("--boot", type=int, default=1000, help="bootstrap resamples (0: none)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regime", default="strict", choices=REGIMES,
                   help="regime for the multi-annotator tables")
    p.add_argument("--top", type=int, default=10, help="documents with the least agreement")
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("confusion", help="label confusion over boundary-matched spans")
    _add_input_args(p)
    _add_pair_args(p)
    p.set_defaults(func=cmd_confusion)

    p = sub.add_parser("convert", help="convert between JSON lines and column files")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--to", choices=["jsonl", "conll"],
                   help="output format (default: from the output suffix)")
    p.add_argument("--scheme", default="iob2", help="tag scheme of a column-file input")
    p.add_argument("--to-scheme", default="iob2", help="tag scheme of a column-file output")
    p.add_argument("--repair", action="store_true",
                   help="accept and repair illegal transitions in the input")
    p.add_argument("--tag-col", type=int, default=-1)
    p.add_argument("--doc-mode", default="auto", choices=["auto", "docstart", "sentence"])
    p.add_argument("--names", help="annotator name for a column-file input")
    p.add_argument("--annotator", help="annotator to write to a column file")
    p.add_argument("--tokens-from",
                   help="column file supplying tokens and sentence breaks for a JSON lines input")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("check", help="count illegal tag transitions in column files")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--scheme", default="iob2")
    p.add_argument("--tag-col", type=int, default=-1)
    p.add_argument("--doc-mode", default="auto", choices=["auto", "docstart", "sentence"])
    p.add_argument("--examples", type=int, default=5, help="example positions per kind")
    p.add_argument("--format", default="md", choices=["md", "json"])
    p.set_defaults(func=cmd_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        text = args.func(args)
    except (SchemeError, ValueError, OSError) as exc:
        print(f"spanagree: {exc}", file=sys.stderr)
        return 2
    print(text)
    return int(getattr(args, "exit_code", 0))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
