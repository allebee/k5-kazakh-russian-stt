"""Bootstrap 95% CIs (2000 resamples over utterances) on a bench subset for selected systems, plus paired
differences vs the first system. Metrics: all (std), no-digit (std), ITN-fair (light vs raw refs).
usage: bench_ci.py SET TAG:spoken|written [TAG:...]   (first TAG is the reference for the paired differences)"""
import json, os, random, re, sys
from pathlib import Path
STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "kk_tn")]
import jiwer
from score import load
from textnorm import normalize, has_digit
from kk_tn.itn import inverse_normalize
import unicodedata


def light(t):  # = bench_score.light / itn_score.light (both run their report at import time)
    t = unicodedata.normalize("NFC", t).lower().replace("ё", "е")
    t = re.sub(r"(?<=\d)[\s\u00a0\u202f](?=\d{3}(?!\d))", "", t)
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)
    t = re.sub(r"[\u2010-\u2015\-_]", " ", t)
    t = re.sub(r"(?<!\d)[.:]|[.:](?!\d)", " ", t)
    t = re.sub(r"[^\w\s.:]", " ", t)
    return " ".join(t.split())

s, specs = sys.argv[1], [a.rsplit(":", 1) for a in sys.argv[2:]]
rows = load(STT / f"eval/manifests/bench_{s}.jsonl")
per = {}
for tag, kind in specs:
    f = STT / f"eval/hyps/{s}.{tag}.jsonl"
    if not f.exists() and s.endswith("200"):
        f = STT / f"eval/hyps/{s[:-3]}.{tag}.jsonl"
    hy = {x["id"]: x["hyp"] for x in load(f)}
    m = {"all": [], "no-digit": [], "ITN-fair": []}
    for r in rows:
        ref, h = normalize(r["ref"]), normalize(hy[r["id"]])
        e = jiwer.process_words(ref, h) if ref else None
        m["all"].append((e.substitutions + e.deletions + e.insertions, len(ref.split())) if e else (0, 0))
        m["no-digit"].append(m["all"][-1] if not has_digit(r["ref"]) else (0, 0))
        lr = light(r["ref"]); lh = light(inverse_normalize(hy[r["id"]]) if kind == "spoken" else hy[r["id"]])
        e2 = jiwer.process_words(lr, lh) if lr else None
        m["ITN-fair"].append((e2.substitutions + e2.deletions + e2.insertions, len(lr.split())) if e2 else (0, 0))
    per[tag] = m
rng = random.Random(0); n = len(rows)
idx = [[rng.randrange(n) for _ in range(n)] for _ in range(2000)]
w = lambda xs, ii: 100 * sum(xs[i][0] for i in ii) / max(1, sum(xs[i][1] for i in ii))
ref_tag = specs[0][0]
print(f"{s}, n={n}: WER [95% CI]; paired difference vs {ref_tag} [95% CI]")
for metric in ("all", "no-digit", "ITN-fair"):
    print(f"\n{metric}:")
    for tag, _ in specs:
        xs = per[tag][metric]
        bs = sorted(w(xs, ii) for ii in idx)
        line = f"  {tag}: {w(xs, range(n)):.2f} [{bs[49]:.2f}, {bs[1949]:.2f}]"
        if tag != ref_tag:
            ys = per[ref_tag][metric]
            ds = sorted(w(xs, ii) - w(ys, ii) for ii in idx)
            line += f" | diff {w(xs, range(n)) - w(ys, range(n)):+.2f} [{ds[49]:+.2f}, {ds[1949]:+.2f}]"
        print(line)
