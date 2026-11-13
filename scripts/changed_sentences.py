"""Count sentences, tags and tokens that differ between two aligned column files.

Usage: python scripts/changed_sentences.py A B
"""

from __future__ import annotations

import sys

from spanagree.bio import read_columns


def main(a_path: str, b_path: str) -> int:
    a, b = read_columns(a_path), read_columns(b_path)
    if len(a.docs) != len(b.docs):
        print("the files hold different numbers of documents", file=sys.stderr)
        return 1
    sentences = tags = tokens = 0
    for (_, sa), (_, sb) in zip(a.docs, b.docs, strict=True):
        if len(sa) != len(sb):
            print("the files split a document into different sentences", file=sys.stderr)
            return 1
        for x, y in zip(sa, sb, strict=True):
            if len(x) != len(y):
                print("a sentence has different lengths in the two files", file=sys.stderr)
                return 1
            changed = sum(p[1] != q[1] for p, q in zip(x, y, strict=True))
            tags += changed
            sentences += changed > 0
            tokens += sum(p[0] != q[0] for p, q in zip(x, y, strict=True))
    print(f"sentences with a changed tag: {sentences}")
    print(f"tags changed: {tags}")
    print(f"tokens changed: {tokens}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
