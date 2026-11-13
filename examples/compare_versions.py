"""Compare two span sets with the Python API.

Run from the repository root: python examples/compare_versions.py
"""

from pathlib import Path

from spanagree.agreement import pair_agreement
from spanagree.diff import diff
from spanagree.match import REGIMES, score
from spanagree.spans import load_jsonl

corpus = load_jsonl(Path(__file__).parent / "data" / "two_versions.jsonl")
ref, other = corpus.sets["first"], corpus.sets["second"]

for regime in REGIMES:
    s = score(ref, other, regime)
    print(f"{regime:8s} F = {s.micro.f:.4f}  macro F = {s.macro_f:.4f}")

d = diff(ref, other)
print("differences:", d.n_differences, d.counts())
for item in d.items:
    if item.kind != "identical":
        print(f"  {item.kind:9s} {item.example_id}")

agr = pair_agreement(corpus, "first", "second", n_boot=500, seed=1)
print(f"Cohen's kappa over all positions: {agr.kappa_all.value:.4f}")
print(f"Cohen's kappa over marked positions: {agr.kappa_marked.value:.4f}")
print(f"Krippendorff's alpha over characters: {agr.alpha.value:.4f}")
