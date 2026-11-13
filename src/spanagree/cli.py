"""Command line interface: ``spanagree <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from spanagree import __version__
from spanagree.bio import SchemeError, load_columns
from spanagree.io import load_inputs, pick_pair
from spanagree.match import REGIMES, score
from spanagree.report import table


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
