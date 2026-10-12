"""Endpointing grid summary: final WER vs offline on the same calls, with a paired bootstrap 95% CI over calls of the
difference (final - offline), and final latency. usage: ep_summary.py 'eval/ep_*.json'"""
import glob, json, random, sys
import jiwer


def wer_counts(refs, hyps):
    m = jiwer.process_words(refs, hyps)
    return m.substitutions + m.deletions + m.insertions, sum(len(r.split()) for r in refs)


print("config | calls | words | final WER | offline WER | diff [95% CI] | final latency mean / p90 ms")
for f in sorted(glob.glob(sys.argv[1])):
    d = json.load(open(f)); pc = d["per_call"]
    per = [(wer_counts([c["ref"]], [c["final"]])[0], wer_counts([c["ref"]], [c["offline"]])[0], len(c["ref"].split())) for c in pc]
    rng, diffs = random.Random(0), []
    for _ in range(2000):
        smp = [per[rng.randrange(len(per))] for _ in per]
        n = sum(x[2] for x in smp)
        diffs.append(100 * (sum(x[0] for x in smp) - sum(x[1] for x in smp)) / n)
    diffs.sort()
    n = sum(x[2] for x in per)
    fl = d["final_latency_after_speech_end_ms"]
    print(f"{f.split('/')[-1][:-5]} | {len(pc)} | {n} | {100 * sum(x[0] for x in per) / n:.2f} | {100 * sum(x[1] for x in per) / n:.2f} | "
          f"{100 * (sum(x[0] for x in per) - sum(x[1] for x in per)) / n:+.2f} [{diffs[50]:+.2f}, {diffs[1949]:+.2f}] | {fl['mean']:.0f} / {fl['p90']:.0f}")
