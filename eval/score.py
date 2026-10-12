"""Score hypothesis files against a manifest: WER/CER on all utterances and on the digit-free subset.

usage: score.py MANIFEST HYPS.jsonl [HYPS2.jsonl ...]
"""
import json, sys

import jiwer

from textnorm import has_digit, normalize


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def load_failures(hyps):
    """Share of hypotheses whose audio failed to load (evaluators write an "error" field instead of crashing)."""
    return sum("error" in x for x in hyps) / max(1, len(hyps))


def score(rows, hyps):
    h = {x["id"]: x["hyp"] for x in hyps}
    missing = [r["id"] for r in rows if r["id"] not in h]
    out = {}
    for name, keep in (("all", lambda r: True), ("no-digit", lambda r: not has_digit(r["ref"]))):
        sel = [r for r in rows if keep(r) and r["id"] in h]
        refs = [normalize(r["ref"]) for r in sel]
        hyp = [normalize(h[r["id"]]) for r in sel]
        pairs = [(a, b) for a, b in zip(refs, hyp) if a]
        refs, hyp = [a for a, _ in pairs], [b for _, b in pairs]
        out[name] = (len(refs), 100 * jiwer.wer(refs, hyp), 100 * jiwer.cer(refs, hyp))
    return out, missing


if __name__ == "__main__":
    rows = load(sys.argv[1])
    print(f"{'hyps':58} {'n':>4} {'WER':>6} {'CER':>6} | {'n':>4} {'WER':>6} {'CER':>6}  (all | no-digit)")
    for hp in sys.argv[2:]:
        hy = load(hp)
        if load_failures(hy) > 0.01:  # never compare a system whose audio didn't load
            print(f"{hp[-58:]:58} INVALID: {100 * load_failures(hy):.0f}% of {len(hy)} utterances failed to load -> rerun")
            continue
        s, missing = score(rows, hy)
        a, d = s["all"], s["no-digit"]
        note = f"  MISSING {len(missing)}" if missing else ""
        print(f"{hp[-58:]:58} {a[0]:4d} {a[1]:6.2f} {a[2]:6.2f} | {d[0]:4d} {d[1]:6.2f} {d[2]:6.2f}{note}")
