#!/bin/sh
# Compare the original and corrected CoNLL-2003 English test labels.
# Usage: sh scripts/conll_versions.sh DATA_DIR
set -eu
D=${1:?data directory}
python scripts/fetch_conll.py "$D"
A="$D/test.txt"
B="$D/conllpp_test.txt"
spanagree check "$A" "$B" --scheme iob2 --examples 0
spanagree score "$A" "$B" --names original,corrected --regime all
spanagree diff "$A" "$B" --names original,corrected --examples 0
spanagree agreement "$A" "$B" --names original,corrected --boot 1000 --seed 0
spanagree confusion "$A" "$B" --names original,corrected
