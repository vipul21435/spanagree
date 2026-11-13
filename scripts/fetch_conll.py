"""Download the CoNLL-2003 English test set in two label versions.

* ``test.txt``: the original labels (IOB2), as redistributed in the
  pfliu-nlp/Named-Entity-Recognition-NER-Papers repository.
* ``conllpp_test.txt``: the corrected labels of CoNLL++ (Wang et al. 2019), from
  the ZihanWangKi/CrossWeigh repository.

Neither file states a licence for the data (the text is Reuters news), so this
repository commits only this script and the numbers derived from the files.
Files are checked against the SHA-256 digests recorded below.

Usage: python scripts/fetch_conll.py DEST_DIR
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

FILES = {
    "test.txt": (
        "https://raw.githubusercontent.com/pfliu-nlp/Named-Entity-Recognition-NER-Papers/"
        "master/ner_dataset/CoNLL2003/test.txt",
        "dc4ac7f3d6d01a6bec59261814561da3b287b6b493b97d1f69c580b52ec1831b",
    ),
    "conllpp_test.txt": (
        "https://raw.githubusercontent.com/ZihanWangKi/CrossWeigh/master/data/conllpp_test.txt",
        "4eb3b5114787d0df24087e9c9dfc956abf4cfd5e201bc1056b48ff4f7b37c0e8",
    ),
}


def main(dest: str) -> int:
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in FILES.items():
        path = out / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as resp:
                path.write_bytes(resp.read())
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != digest:
            print(f"{path}: sha256 {got} does not match {digest}", file=sys.stderr)
            return 1
        print(f"{path}: ok")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
