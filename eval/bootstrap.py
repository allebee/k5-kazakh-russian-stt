"""WER with a bootstrap 95% confidence interval (utterance-level resampling, 2000 draws), per hyps file.
For small sets (31 code-switch clips, 175 KSC2 code-switched) a point WER alone misleads.
usage: bootstrap.py MANIFEST HYPS.jsonl [HYPS2.jsonl ...]"""
import json, random, sys

import jiwer

from textnorm import has_digit, normalize


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def per_utt(rows, hyps):
    h = {x["id"]: x["hyp"] for x in hyps}
    out = []
    for r in rows:
        ref, hyp = normalize(r["ref"]), normalize(h.get(r["id"], ""))
        if ref:
            o = jiwer.process_words(ref, hyp)
            out.append((o.substitutions + o.deletions + o.insertions, len(ref.split()), has_digit(r["ref"])))
    return out


def ci(u, n=2000, seed=0):
    rng = random.Random(seed)
    point = 100 * sum(e for e, _, _ in u) / max(1, sum(w for _, w, _ in u))
    draws = []
    for _ in range(n):
        s = [u[rng.randrange(len(u))] for _ in u]
        draws.append(100 * sum(e for e, _, _ in s) / max(1, sum(w for _, w, _ in s)))
    draws.sort()
    return point, draws[int(0.025 * n)], draws[int(0.975 * n) - 1]


if __name__ == "__main__":
    rows = load(sys.argv[1])
    print(f"{'hyps':60} {'n':>4}  WER [95% CI]          | no-digit WER [95% CI]")
    for hp in sys.argv[2:]:
        u = per_utt(rows, load(hp))
        a = ci(u); d = ci([x for x in u if not x[2]]) if any(not x[2] for x in u) else (float("nan"),) * 3
        print(f"{hp[-60:]:60} {len(u):4d}  {a[0]:5.2f} [{a[1]:5.2f}, {a[2]:5.2f}] | {d[0]:5.2f} [{d[1]:5.2f}, {d[2]:5.2f}]")
